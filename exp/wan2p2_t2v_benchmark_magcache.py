import os
# Crank it up to 3 (FATAL only) to be absolutely sure
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['XLA_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from datetime import datetime
import functools
import math
import re
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.experimental.pallas.ops.tpu import splash_attention

import torch
import numpy as np
from diffusers import WanPipeline, MagCacheConfig
from diffusers.utils import export_to_video
from diffusers.models.autoencoders import vae as diffusers_vae
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models import modeling_outputs as diffusers_modeling_outputs

from transformers import modeling_outputs

import torchax
from torchax.ops import jaten
from torchax.ops import jtorch
from torchax.ops import ops_registry

# Local file
import custom_splash_attention_updated as custom_splash_attention

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
}

SUPPORTED_SIZES = {
    "t2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "i2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "ti2v-5B": ("704*1280", "1280*704"),
    "s2v-14B": (
        "720*1280",
        "1280*720",
        "480*832",
        "832*480",
        "1024*704",
        "704*1024",
        "704*1280",
        "1280*704",
    ),
    "animate-14B": ("720*1280", "1280*720"),
}

DEFAULT_PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
DEFAULT_NEG_PROMPT = "Overly vibrant colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, washed out, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless scene, cluttered background, three legs, crowded background, walking backwards"
DEFAULT_PROFILE_OUT_PATH = "/tmp/wan_prof"

# fmt: off
TEXT_ENCODER_SHARDINGS = {
    'shared.weight': (), 
    'encoder.block.*.layer.*.SelfAttention.q.weight': ('tp',), 
    'encoder.block.*.layer.*.SelfAttention.k.weight': ('tp',), 
    'encoder.block.*.layer.*.SelfAttention.v.weight': ('tp',), 
    'encoder.block.*.layer.*.SelfAttention.o.weight': (None, 'tp',), 
    'encoder.block.*.layer.*.DenseReluDense.wi_0.weight': ('tp',), 
    'encoder.block.*.layer.*.DenseReluDense.wi_1.weight': ('tp',), 
    'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, 'tp',), 
}

TRANSFORMER_SHARDINGS = {
    # ---------------------------------------------------------
    # 1. Condition Embedders (Time & Text)
    # ---------------------------------------------------------
    # Up-projections (Column Parallel: Shard out_features)
    'transformer.condition_embedder.time_embedder.linear_1.weight': ('tp',),
    'transformer.condition_embedder.time_embedder.linear_1.bias': ('tp',),
    'transformer.condition_embedder.text_embedder.linear_1.weight': ('tp',),
    'transformer.condition_embedder.text_embedder.linear_1.bias': ('tp',),
    
    # Down-projections (Row Parallel: Shard in_features)
    'transformer.condition_embedder.time_embedder.linear_2.weight': (None, 'tp',),
    'transformer.condition_embedder.time_embedder.linear_2.bias': (None,), # Replicated after All-Reduce
    'transformer.condition_embedder.text_embedder.linear_2.weight': (None, 'tp',),
    'transformer.condition_embedder.text_embedder.linear_2.bias': (None,), # Replicated after All-Reduce

    # ---------------------------------------------------------
    # 2. Self Attention (attn1) & Cross Attention (attn2)
    # ---------------------------------------------------------
    # Q, K, V Projections (Column Parallel: Shard out_features/heads)
    'transformer.blocks.*.attn1.to_q.weight': ('tp',),
    'transformer.blocks.*.attn1.to_q.bias': ('tp',),
    'transformer.blocks.*.attn1.to_k.weight': ('tp',),
    'transformer.blocks.*.attn1.to_k.bias': ('tp',),
    'transformer.blocks.*.attn1.to_v.weight': ('tp',),
    'transformer.blocks.*.attn1.to_v.bias': ('tp',),
    
    'transformer.blocks.*.attn2.to_q.weight': ('tp',),
    'transformer.blocks.*.attn2.to_q.bias': ('tp',),
    'transformer.blocks.*.attn2.to_k.weight': ('tp',),
    'transformer.blocks.*.attn2.to_k.bias': ('tp',),
    'transformer.blocks.*.attn2.to_v.weight': ('tp',),
    'transformer.blocks.*.attn2.to_v.bias': ('tp',),

    # Output Projections (Row Parallel: Shard in_features)
    'transformer.blocks.*.attn1.to_out.0.weight': (None, 'tp',),
    'transformer.blocks.*.attn1.to_out.0.bias': (None,), # Replicated: added AFTER the ICI All-Reduce
    'transformer.blocks.*.attn2.to_out.0.weight': (None, 'tp',),
    'transformer.blocks.*.attn2.to_out.0.bias': (None,), 

    # ---------------------------------------------------------
    # 3. Feed-Forward Network (FFN) / MoE
    # ---------------------------------------------------------
    # Gate/Up Projections (Column Parallel: Shard out_features)
    'transformer.blocks.*.ffn.net.0.proj.weight': ('tp',),
    'transformer.blocks.*.ffn.net.0.proj.bias': ('tp',),
    
    # Down Projections (Row Parallel: Shard in_features)
    'transformer.blocks.*.ffn.net.2.weight': (None, 'tp',),
    'transformer.blocks.*.ffn.net.2.bias': (None,), # Replicated: added AFTER the ICI All-Reduce
}

VAE_ENCODER_SHARDINGS = {}
VAE_DECODER_SHARDINGS = {}
# fmt: on

BQSIZE = 3328
BKVSIZE = 2816
BKVCOMPUTESIZE = 256
BKVCOMPUTEINSIZE = 256

@contextmanager
def perf_time(name: str):
    print(f"{name} start")
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start: .6f}s")


def _print_weights(module):
    def make_key(name):
        return re.sub(r"\.\d+\.", ".*.", name)

    all_buffers = dict(module.named_parameters())
    all_buffers.update(module.named_buffers())
    result = {}
    for k, v in all_buffers.items():
        result[make_key(k)] = (v.shape, v.dtype)
    print("{")
    for k, v in result.items():
        print(f"'{k}': (), # {v}")
    print("}")


def _torch_conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, env
):
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
    return env.j2t_iso(res)


def _overide_op_definition(env, op_to_override, op_impl):
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        op_impl,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )


def _shard_weight_dict(weight_dict, sharding_dict, mesh):
    result = {}
    for k, v in weight_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.to("jax")
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                break
        else:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))

        result[k] = v
    return result


def _move_module(env, module):
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


### Flash Attention
def pad_to_multiple(x, multiple, axis):
    seq_len = x.shape[axis]
    pad_len = (multiple - seq_len % multiple) % multiple
    if pad_len == 0:
        return x, seq_len
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(x, pad_width), seq_len


def _tpu_custom_attention(query, key, value, mesh, scale=None):
    def _attention_on_slices(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        _LOG2_E = 1.44269504
        q = q * scale_factor * _LOG2_E

        def kernel_3d(q_3d, k_3d, v_3d):
            q_seq_len = q_3d.shape[1]
            kv_seq_len = k_3d.shape[1]
            num_heads_on_device = q_3d.shape[0]
            block_sizes = custom_splash_attention._BlockSizes(
                block_q=min(BQSIZE, q_seq_len),
                block_kv=min(BKVSIZE, kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, kv_seq_len),
            )
            splash_kernel = custom_splash_attention.make_splash_mha(
                block_sizes=block_sizes,
                bkv_compute_in=BKVCOMPUTEINSIZE,
                heads_per_tile=HEADS_PER_TILE
            )
            out = splash_kernel(q_3d, k_3d, v_3d).astype(q_3d.dtype)
            out = jnp.swapaxes(out, 1, 2)
            return out

        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    if args.FLAG and key.shape[0] > 1:
        dp_mesh_key = "dp"
        remain_mesh_key = ("tp",)
    else:
        dp_mesh_key = None
        remain_mesh_key = ("dp", "tp")
    
    remain_devices_prod = 1
    for d in remain_mesh_key:
        remain_devices_prod *= mesh.axis_sizes[mesh.axis_names.index(d)]

    q_num_head = query.shape[1]
    q_seq_len = query.shape[2]
    kv_num_head = key.shape[1]
    kv_seq_len = key.shape[2]
    
    if (
        kv_seq_len > 10000
        and kv_num_head % remain_devices_prod == 0
        and q_num_head % remain_devices_prod == 0
    ):
        q_partition_spec = P(dp_mesh_key, remain_mesh_key, None, None)
        kv_partition_spec = P(dp_mesh_key, remain_mesh_key, None, None)
    else:
        if q_seq_len % remain_devices_prod != 0:
            query, _ = pad_to_multiple(query, remain_devices_prod, axis=2)

        q_partition_spec = P(dp_mesh_key, None, remain_mesh_key, None)
        kv_partition_spec = P(dp_mesh_key, None, None, None)
    
    sharded_fn = jax.shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_vma=False,
    )
    query = jax.lax.with_sharding_constraint(query, q_partition_spec)
    key = jax.lax.with_sharding_constraint(key, kv_partition_spec)
    value = jax.lax.with_sharding_constraint(value, kv_partition_spec)

    out = sharded_fn(query, key, value)
    out = out[:, :, :q_seq_len, :]
    out = jax.lax.with_sharding_constraint(out, P(dp_mesh_key, None, remain_mesh_key, None))
    return out


def _scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    *,
    env,
    mesh,
) -> torch.Tensor:
    if key.shape[2] > 20000:
        assert attn_mask is None
        assert dropout_p == 0.0
        assert is_causal is False
        assert enable_gqa is False
        assert scale is None
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        res = _tpu_custom_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)

    return jtorch._sdpa_reference(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )


# register non-jax type
def _flatten_model_output(output):
    return tuple(output.values()), (type(output), tuple(output.keys()))

def _unflatten_model_output(aux, children):
    cls, keys = aux
    obj = cls.__new__(cls)
    import collections
    collections.OrderedDict.__init__(obj)
    for k, v in zip(keys, children):
        object.__setattr__(obj, k, v)
        obj[k] = v
    return obj

jax.tree_util.register_pytree_node(
    modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
    _flatten_model_output,
    _unflatten_model_output,
)

jax.tree_util.register_pytree_node(
    diffusers_vae.DecoderOutput,
    _flatten_model_output,
    _unflatten_model_output,
)

jax.tree_util.register_pytree_node(
    diffusers_modeling_outputs.AutoencoderKLOutput,
    _flatten_model_output,
    _unflatten_model_output,
)

def _flatten_diagonal_gaussian_distribution(
    obj: diffusers_vae.DiagonalGaussianDistribution,
):
    return (
        obj.parameters, obj.mean, obj.logvar,
        obj.deterministic, obj.std, obj.var,
    ), None

def _unflatten_diagonal_gaussian_distribution(
    aux, children
) -> diffusers_vae.DiagonalGaussianDistribution:
    obj = object.__new__(diffusers_vae.DiagonalGaussianDistribution)
    obj.parameters = children[0]
    obj.mean = children[1]
    obj.logvar = children[2]
    obj.deterministic = children[3]
    obj.std = children[4]
    obj.var = children[5]
    return obj

jax.tree_util.register_pytree_node(
    diffusers_vae.DiagonalGaussianDistribution,
    _flatten_diagonal_gaussian_distribution,
    _unflatten_diagonal_gaussian_distribution,
)


class Args(argparse.Namespace):
    size: str
    frame_num: int
    prompt: str
    base_seed: int
    sample_steps: int
    print_weights: bool
    FLAG: bool
    profile: str
    profile_output_path: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using Wan T2V"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="720*1280",
        choices=list(SIZE_CONFIGS.keys()),
        help="The (width*height) dimensions of the generated video.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of videos to generate in a single batch."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames of video are generated. The number should be 4n+1",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="The prompt to generate the video from.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="The seed to use for generating the video. Need to specify for multi-host sync.",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=40, help="The sampling steps."
    )
    parser.add_argument(
        "--print_weights", action="store_true", help="print weights in models"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="no",
        choices=["no", "dit", "all"],
        help="no for no profile, dit for dit only 3 steps, all including vae",
    )
    parser.add_argument(
        "--profile_output_path",
        type=str,
        default=DEFAULT_PROFILE_OUT_PATH,
        help="path to save profile output",
    )
    parser.add_argument(
        "--dp",
        type=int,
        default=2,
        help="Data parallelism for positive prompt and negative prompt.",
    )
    
    parser.add_argument("--bq", type=int, default=2048, help="Query block size for Splash Attention")
    parser.add_argument("--bkv", type=int, default=2048, help="KV block size for Splash Attention")
    parser.add_argument("--bkv_compute", type=int, default=1024, help="Compute block size for Splash Attention")
    parser.add_argument("--bkv_compute_in", type=int, default=1024, help="Input block size for Splash Attention")
    parser.add_argument("--FLAG", action="store_true", help="Sets head sharding")
    parser.add_argument("--heads_per_tile", type=int, default=1, help="MHPT: Heads per tile for MXU/VPU overlap")

    return parser.parse_args(namespace=Args())

from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

class WanVAEDecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        # Nest the necessary modules so torchax can trace them
        self.decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv
        self.patch_size = vae.config.patch_size
        
        # Pre-count the causal convs to initialize the correct cache size
        self.conv_num = sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())

    def forward(self, z):
        x = self.post_quant_conv(z)
        feat_map = [None] * self.conv_num
        
        # CHUNK 1: Prime the cache inside the XLA graph
        out_0, feat_map = self.decoder(
            x[:, :, 0:1, :, :], 
            feat_cache=feat_map, 
            first_chunk=True
        )
        
        # CHUNK 2: Process the rest (uses the cache directly from TPU memory!)
        out_rest, _ = self.decoder(
            x[:, :, 1:, :, :], 
            feat_cache=feat_map, 
            first_chunk=False
        )
        
        out = torch.cat([out_0, out_rest], dim=2)
        
        # Unpatchify directly in the compiled graph
        if self.patch_size is not None:
            patch_size = self.patch_size
            batch_size, c_patches, frames, height, width = out.shape
            channels = c_patches // (patch_size * patch_size)
            out = out.view(batch_size, channels, patch_size, patch_size, frames, height, width)
            out = out.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
            out = out.view(batch_size, channels, frames, height * patch_size, width * patch_size)

        return torch.clamp(out, min=-1.0, max=1.0)


class MicroMagCacheWrapper:
    def __init__(self, c_fused, env, threshold=0.04, max_skips=2, warmup_steps=7):
        self.c_fused = c_fused
        self.env = env
        self.threshold = threshold
        self.max_skips = max_skips
        self.warmup_steps = warmup_steps
        
        self.mag_ratios = []
        self.calibrate_mode = False
        self.mag_log = []
        self._reset()

    def _reset(self):
        self._step = 0
        self._prev_residual = None 
        self.skip_sequence = None

    def reset_for_new_generation(self):
        self._reset()

    def _precalculate_skips(self, num_steps=100):
        skips = []
        error_budget = [0.0, 0.0]
        accumulated_ratio = [1.0, 1.0]
        skip_count = 0
        
        for step in range(num_steps):
            skip_forward = False
            ratio_idx = max(0, step - 1)
            ratio_idx = min(ratio_idx, len(self.mag_ratios) - 1)
            mag_ratio = self.mag_ratios[ratio_idx] if self.mag_ratios else (1.0, 1.0)
            
            if step >= self.warmup_steps and step > 0:
                accumulated_ratio[0] *= mag_ratio[0]
                accumulated_ratio[1] *= mag_ratio[1]
                
                error_budget[0] += abs(1.0 - accumulated_ratio[0])
                error_budget[1] += abs(1.0 - accumulated_ratio[1])
                
                if max(error_budget) < self.threshold and skip_count < self.max_skips:
                    skip_forward = True

            if skip_forward:
                skip_count += 1
                skips.append(True)
            else:
                error_budget = [0.0, 0.0]
                accumulated_ratio = [1.0, 1.0]
                skip_count = 0
                skips.append(False)
        return skips

    def __call__(self, hidden_states, timestep, encoder_hidden_states, guidance_scale, rotary_emb=None, projected_text=False, cross_attn_kv_cache=None, encoder_hidden_states_image=None, **kwargs):
        _, _, num_frames, height, width = hidden_states.shape
        
        # Create a completely safe Torchax Dummy Tensor for Step 0!
        if self._prev_residual is not None:
            prev_res_input = self._prev_residual
        else:
            raw_zeros = torch.zeros(1, dtype=torch.bfloat16, device="cpu")
            prev_res_input = self.env.j2t_iso(self.env.t2j_iso(raw_zeros))
        
        if self.calibrate_mode:
            out, new_res = self.c_fused(
                hidden_states, timestep, encoder_hidden_states, rotary_emb, cross_attn_kv_cache,
                prev_res_input, guidance_scale=guidance_scale, num_frames=num_frames, height=height, width=width,
                encoder_hidden_states_image=encoder_hidden_states_image, projected_text=projected_text, skip_this_step=False
            )
            
            jresidual = self.env.t2j_iso(new_res)
            if jresidual.shape[0] == 2:
                self.mag_log.append((
                    float(jnp.linalg.norm(jresidual[0]).item()), 
                    float(jnp.linalg.norm(jresidual[1]).item())
                ))
            else:
                norm = float(jnp.linalg.norm(jresidual[0]).item())
                self.mag_log.append((norm, norm))
            
            self._prev_residual = new_res
            return out
            
        else:
            if self._step == 0 or self.skip_sequence is None:
                self.skip_sequence = self._precalculate_skips(num_steps=100)
                
            skip = self.skip_sequence[self._step]
            
            out, new_res = self.c_fused(
                hidden_states, timestep, encoder_hidden_states, rotary_emb, cross_attn_kv_cache,
                prev_res_input, guidance_scale=guidance_scale, num_frames=num_frames, height=height, width=width,
                encoder_hidden_states_image=encoder_hidden_states_image, projected_text=projected_text, skip_this_step=skip
            )
            
            self._prev_residual = new_res
            self._step += 1
            return out

class WanFusedStep(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # Explicitly extract submodules to bypass hidden/unregistered tensors
        self.transformer = torch.nn.Module()
        self.transformer.patch_embedding = transformer.patch_embedding
        self.transformer.condition_embedder = transformer.condition_embedder
        self.transformer.blocks = transformer.blocks
        self.transformer.norm_out = transformer.norm_out
        self.transformer.proj_out = transformer.proj_out
        self.transformer.scale_shift_table = transformer.scale_shift_table
        
        self.patch_size = transformer.config.patch_size
        
    def forward(self, hidden_states, timestep, encoder_hidden_states, rotary_emb, cross_attn_kv_cache, prev_residual, guidance_scale, num_frames, height, width, encoder_hidden_states_image=None, projected_text=False, skip_this_step=False):
        
        # --- 1. PRE-BLOCKS ---
        hs = self.transformer.patch_embedding(hidden_states)
        hs = hs.flatten(2).transpose(1, 2)
        
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, ts_proj, enc_hs, _ = self.transformer.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image=encoder_hidden_states_image, 
            timestep_seq_len=ts_seq_len, projected_text=projected_text, projected_image=False
        )
        
        if ts_seq_len is not None:
            ts_proj = ts_proj.unflatten(2, (6, -1))
        else:
            ts_proj = ts_proj.unflatten(1, (6, -1))
            
        if encoder_hidden_states_image is not None:
            enc_hs = torch.concat([encoder_hidden_states_image, enc_hs], dim=1)
            
        # --- 2. CONDITIONAL BLOCKS ---
        if skip_this_step:
            hs = hs + prev_residual
            new_residual = prev_residual
        else:
            ori_x = hs
            for i, block in enumerate(self.transformer.blocks):
                layer_cache = cross_attn_kv_cache[i] if cross_attn_kv_cache is not None else None
                hs = block(hs, enc_hs, ts_proj, rotary_emb, cross_attn_kv_cache=layer_cache)
            new_residual = hs - ori_x
            
        # --- 3. POST-BLOCKS & CFG ---
        if temb.ndim == 3:
            shift, scale = (self.transformer.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
            
        shift = shift.to(hs.device)
        scale = scale.to(hs.device)
        
        hs = (self.transformer.norm_out(hs.float()) * (1 + scale) + shift).type_as(hs)
        hs = self.transformer.proj_out(hs)
        
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        
        hs = hs.reshape(
            hs.shape[0], post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hs = hs.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        
        if guidance_scale > 1.0:
            noise_pred, noise_uncond = output.chunk(2)
            output = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            
        return output, new_residual

# =========================================================================
# NEW: Cross-Attention KV Cache Generator
# =========================================================================
class WanCrossAttnCacheGenerator(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        if hasattr(transformer, 'original_module'):
            self.transformer = transformer.original_module
        else:
            self.transformer = transformer 

    def forward(self, encoder_hidden_states):
        kv_cache = []
        for block in self.transformer.blocks:
            attn = block.attn2
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            k = attn.norm_k(k)
            
            k = k.unflatten(2, (attn.heads, -1))
            v = v.unflatten(2, (attn.heads, -1))
            
            kv_cache.append(k)
            kv_cache.append(v)
            
        return tuple(kv_cache)
# =========================================================================


class WanSchedulerStepWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample, noise_pred, sigma, sigma_next):
        # Pure Flow Matching / Euler discrete step math
        # XLA will fuse this into a single lightning-fast operation
        dt = sigma_next - sigma
        prev_sample = sample + noise_pred * dt
        return prev_sample

# Create the Sharding Wrapper to manage the XLA boundary
class SchedulerShardingWrapper:
    def __init__(self, compiled_sched, env, mesh):
        self.compiled_sched = compiled_sched
        self.mesh = mesh
        self.env = env
        self.dp_dim = mesh.axis_sizes[mesh.axis_names.index('dp')]

    def __call__(self, sample, noise_pred, sigma, sigma_next):
        # Safely convert to tensor without the warning
        t_sigma = torch.as_tensor(sigma, dtype=sample.dtype)
        t_sigma_next = torch.as_tensor(sigma_next, dtype=sample.dtype)
        
        j_sample, j_noise_pred, j_sigma, j_sigma_next = self.env.t2j_iso(
            (sample, noise_pred, t_sigma, t_sigma_next)
        )
        
        # Enforce Data Parallelism on the massive latents
        if j_sample.shape[0] > 1 and j_sample.shape[0] % self.dp_dim == 0:
            sharding = NamedSharding(self.mesh, P('dp', *([None] * (j_sample.ndim - 1))))
            j_sample = jax.device_put(j_sample, sharding)
            j_noise_pred = jax.device_put(j_noise_pred, sharding)
            
        t_sample, t_noise_pred, t_sigma, t_sigma_next = self.env.j2t_iso(
            (j_sample, j_noise_pred, j_sigma, j_sigma_next)
        )
        
        return self.compiled_sched(t_sample, t_noise_pred, t_sigma, t_sigma_next)

class TextEncoderShardingWrapper:
    def __init__(self, compiled_encoder, env, mesh):
        self.compiled_encoder = compiled_encoder
        self.dtype = compiled_encoder.dtype
        self.env = env
        self.mesh = mesh
        self.dp_dim = mesh.axis_sizes[mesh.axis_names.index('dp')]

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        j_input_ids, j_attention_mask = self.env.t2j_iso((input_ids, attention_mask))
        
        if j_input_ids.shape[0] > 1 and j_input_ids.shape[0] % self.dp_dim == 0:
            sharding = NamedSharding(self.mesh, P('dp', None))
            j_input_ids = jax.device_put(j_input_ids, sharding)
            if j_attention_mask is not None:
                j_attention_mask = jax.device_put(j_attention_mask, sharding)
                
        t_input_ids, t_attention_mask = self.env.j2t_iso((j_input_ids, j_attention_mask))
        return self.compiled_encoder(t_input_ids, attention_mask=t_attention_mask, **kwargs)

class TransformerShardingWrapper:
    def __init__(self, compiled_transformer, original_module, env, mesh):
        self.compiled_transformer = compiled_transformer
        self.original_module = original_module
        self.env = env
        self.mesh = mesh
        self.dtype = compiled_transformer.dtype
        self.config = compiled_transformer.config
        self.dp_dim = mesh.axis_sizes[mesh.axis_names.index('dp')]
    
    @property
    def rope(self):
        return self.original_module.rope

    @property
    def condition_embedder(self):
        return self.original_module.condition_embedder
    
    @property
    def blocks(self):
        return self.original_module.blocks

    def __call__(self, **kwargs):
        # FIREWALL: Recursively wrap ANY raw PyTorch tensor into a Torchax tensor
        def _force_torchax(obj):
            if isinstance(obj, torch.Tensor):
                return self.env.j2t_iso(self.env.t2j_iso(obj))
            elif isinstance(obj, tuple):
                return tuple(_force_torchax(x) for x in obj)
            elif isinstance(obj, list):
                return list(_force_torchax(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: _force_torchax(v) for k, v in obj.items()}
            return obj

        new_kwargs = kwargs.copy()
        
        # 1. Pop and explicitly shard the primary dimensions
        hidden_states = new_kwargs.pop('hidden_states', None)
        timestep = new_kwargs.pop('timestep', None)
        encoder_hidden_states = new_kwargs.pop('encoder_hidden_states', None)
        guidance_scale = new_kwargs.pop('guidance_scale', 5.0) 
        
        j_hidden_states, j_timestep, j_encoder_hidden_states = self.env.t2j_iso((hidden_states, timestep, encoder_hidden_states))
        
        if j_hidden_states is not None and j_hidden_states.shape[0] > 1 and j_hidden_states.shape[0] % self.dp_dim == 0:
            sharding = NamedSharding(self.mesh, P('dp', *([None] * (j_hidden_states.ndim - 1))))
            j_hidden_states = jax.device_put(j_hidden_states, sharding)
            
            sharding_ts = NamedSharding(self.mesh, P('dp', *([None] * (j_timestep.ndim - 1))))
            j_timestep = jax.device_put(j_timestep, sharding_ts)
            
            sharding_enc = NamedSharding(self.mesh, P('dp', *([None] * (j_encoder_hidden_states.ndim - 1))))
            j_encoder_hidden_states = jax.device_put(j_encoder_hidden_states, sharding_enc)
                
        t_hidden_states, t_timestep, t_encoder_hidden_states = self.env.j2t_iso((j_hidden_states, j_timestep, j_encoder_hidden_states))
        
        # 2. Push absolutely EVERYTHING else in kwargs through the firewall
        new_kwargs = _force_torchax(new_kwargs)

        # 3. Add the primary tensors back
        if hidden_states is not None: new_kwargs['hidden_states'] = t_hidden_states
        if timestep is not None: new_kwargs['timestep'] = t_timestep
        if encoder_hidden_states is not None: new_kwargs['encoder_hidden_states'] = t_encoder_hidden_states
        new_kwargs['guidance_scale'] = float(guidance_scale)
        
        return self.compiled_transformer(**new_kwargs)

def main(args: Args):
    global BQSIZE, BKVSIZE, BKVCOMPUTESIZE, BKVCOMPUTEINSIZE, HEADS_PER_TILE
    BQSIZE = args.bq
    BKVSIZE = args.bkv
    BKVCOMPUTESIZE = args.bkv_compute
    BKVCOMPUTEINSIZE = args.bkv_compute_in
    HEADS_PER_TILE = args.heads_per_tile
    torch.set_default_dtype(torch.bfloat16)

    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers" 
    dtype = torch.bfloat16

    with perf_time("load pipe"):
        pipe = WanPipeline.from_pretrained(model_id, torch_dtype=dtype, boundary_ratio=0.875)

    if args.print_weights:
        print("text_encoder_shardings = ", end="")
        _print_weights(pipe.text_encoder)
        print()
        print("transformer_shardings = ", end="")
        _print_weights(pipe.transformer)
        print()
        print("vae_decoder_shardings = ", end="")
        _print_weights(pipe.vae.decoder)
        print()

    torchax.enable_globally()
    env = torchax.default_env()
    assert isinstance(env, torchax.tensor.Environment)

    dp_dim = args.dp
    assert len(jax.devices()) % dp_dim == 0
    tp_dim = len(jax.devices()) // dp_dim
    mesh_devices = mesh_utils.create_device_mesh(
        (dp_dim, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    print(f"{mesh=}")
    pipe.mesh = mesh

    _overide_op_definition(
        env, torch.nn.functional.conv2d, functools.partial(_torch_conv2d, env=env)
    )
    _overide_op_definition(
        env,
        torch.nn.functional.scaled_dot_product_attention,
        functools.partial(_scaled_dot_product_attention, env=env, mesh=mesh),
    )

    def compile_micro_magcache(transformer_module, env, mesh, opts, max_skips=2, warmup_steps=0):
        fused_step = WanFusedStep(transformer_module)
        _move_module(env, fused_step)
        
        # We MUST declare skip_this_step as static for XLA to generate the two optimal graphs!
        fused_opts = torchax.CompileOptions(
            jax_jit_kwargs={"static_argnames": ("projected_text", "skip_this_step", "guidance_scale", "num_frames", "height", "width")}
        )
        
        c_fused = torchax.compile(fused_step, fused_opts)
        
        # The internal parameters are exactly the same, so shardings apply perfectly
        c_fused.params = _shard_weight_dict(c_fused.params, TRANSFORMER_SHARDINGS, mesh)
        
        wrapper = MicroMagCacheWrapper(c_fused, env, threshold=0.04, max_skips=max_skips, warmup_steps=warmup_steps)
        wrapper.dtype = transformer_module.dtype
        wrapper.config = transformer_module.config
        
        return wrapper

    with perf_time("Move model to tpu"):
        with perf_time("  Move text encoder"):
            _move_module(env, pipe.text_encoder)
            compiled_text_encoder = torchax.compile(pipe.text_encoder)
            compiled_text_encoder.params = _shard_weight_dict(
                compiled_text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
            )
            compiled_text_encoder.buffers = _shard_weight_dict(
                compiled_text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
            )

            pipe.text_encoder = TextEncoderShardingWrapper(compiled_text_encoder, env, mesh)
            
        transformer_options = torchax.CompileOptions(
            jax_jit_kwargs={"static_argnames": ("return_dict", "projected_text")}
        )

        with perf_time("  Move transformer"):
            # Strip Dropout for Identity to clean the XLA graph
            for block in pipe.transformer.blocks:
                block.attn1.to_out[1] = torch.nn.Identity()
                block.attn2.to_out[1] = torch.nn.Identity()

            micro_wrapper_1 = compile_micro_magcache(pipe.transformer, env, mesh, transformer_options, max_skips=2, warmup_steps=5)
            pipe.transformer = TransformerShardingWrapper(micro_wrapper_1, pipe.transformer, env, mesh)

            wrapped_cache_gen = WanCrossAttnCacheGenerator(pipe.transformer)
            _move_module(env, wrapped_cache_gen)
            compiled_cache_gen = torchax.compile(wrapped_cache_gen)
            compiled_cache_gen.params = _shard_weight_dict(compiled_cache_gen.params, TRANSFORMER_SHARDINGS, mesh)
            compiled_cache_gen.buffers = _shard_weight_dict(compiled_cache_gen.buffers, TRANSFORMER_SHARDINGS, mesh)
            pipe.compiled_cache_generator = compiled_cache_gen

        with perf_time("  Move transformer2"):
            if getattr(pipe, "transformer_2", None) is not None:
                # Strip Dropout for Identity
                for block in pipe.transformer_2.blocks:
                    block.attn1.to_out[1] = torch.nn.Identity()
                    block.attn2.to_out[1] = torch.nn.Identity()

                micro_wrapper_2 = compile_micro_magcache(pipe.transformer_2, env, mesh, transformer_options, max_skips=3, warmup_steps=7)
                pipe.transformer_2 = TransformerShardingWrapper(micro_wrapper_2, pipe.transformer_2, env, mesh)

                # >>> NEW: Compile the second KV cache generator for transformer_2 <<<
                wrapped_cache_gen_2 = WanCrossAttnCacheGenerator(pipe.transformer_2)
                _move_module(env, wrapped_cache_gen_2)
                compiled_cache_gen_2 = torchax.compile(wrapped_cache_gen_2)
                compiled_cache_gen_2.params = _shard_weight_dict(compiled_cache_gen_2.params, TRANSFORMER_SHARDINGS, mesh)
                compiled_cache_gen_2.buffers = _shard_weight_dict(compiled_cache_gen_2.buffers, TRANSFORMER_SHARDINGS, mesh)
                pipe.compiled_cache_generator_2 = compiled_cache_gen_2
            else:
                pipe.compiled_cache_generator_2 = None

        with perf_time("  Move scheduler"):
            # 1. Instantiate our new wrapper
            wrapped_scheduler = WanSchedulerStepWrapper()
            _move_module(env, wrapped_scheduler)
            
            # 2. Compile it
            compiled_scheduler = torchax.compile(wrapped_scheduler)
            
            # 3. Handle empty parameter dictionaries for the compiler API
            compiled_scheduler.params = _shard_weight_dict(compiled_scheduler.params, {}, mesh)
            compiled_scheduler.buffers = _shard_weight_dict(compiled_scheduler.buffers, {}, mesh)
                    
            # Attach it directly to the pipeline object
            pipe.compiled_scheduler_step = SchedulerShardingWrapper(compiled_scheduler, env, mesh)

        with perf_time("  Move vae"):
            _move_module(env, pipe.vae)

            # 1. Wrap the VAE decoder logic
            wrapped_vae_decoder = WanVAEDecodeWrapper(pipe.vae)
            
            # 2. Compile the whole chunking process as one XLA graph
            compiled_vae_decoder = torchax.compile(wrapped_vae_decoder)
            
            # 3. Shard as normal (Defaults to P() since VAE_DECODER_SHARDINGS is empty)
            compiled_vae_decoder.params = _shard_weight_dict(compiled_vae_decoder.params, VAE_DECODER_SHARDINGS, mesh)
            compiled_vae_decoder.buffers = _shard_weight_dict(compiled_vae_decoder.buffers, VAE_DECODER_SHARDINGS, mesh)
            
            # 4. Override the pipeline's decode method to use our compiled graph directly!
            def custom_vae_decode(z, return_dict=False):
                out = compiled_vae_decoder(z)
                if not return_dict:
                    return (out,)
                return DecoderOutput(sample=out)
                
            pipe.vae.decode = custom_vae_decode

    # === CALIBRATION SETUP ===
    pipe.transformer.compiled_transformer.calibrate_mode = True
    if getattr(pipe, "transformer_2", None) is not None:
        pipe.transformer_2.compiled_transformer.calibrate_mode = True
    # === END CALIBRATION SETUP ===
    
    raw_width, raw_height = SIZE_CONFIGS[args.size] 

    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = (raw_height // mod_value) * mod_value
    width = (raw_width // mod_value) * mod_value

    prompt = args.prompt
    negative_prompt = DEFAULT_NEG_PROMPT
    generator = torch.Generator().manual_seed(args.base_seed)
    
    guidance_low = 3
    guidance_high = 4
    with mesh:
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        with perf_time("Warmup and output video"):
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=args.frame_num,
                guidance_scale=guidance_high,
                guidance_scale_2=guidance_low,
                num_inference_steps=args.sample_steps,
                generator=generator,
                max_sequence_length=256,
            ).frames[0]
            file_name = f"{current_datetime}_batch_0.mp4"
            export_to_video(output, file_name, fps=16)
            print(f"output video done. {file_name}")

        # Fetch logs, compute ratios, and disable calibration!
        log_1 = pipe.transformer.compiled_transformer.mag_log
        # Create tuples of (cond_ratio, uncond_ratio)
        mag_ratios_1 = [(log_1[i][0] / log_1[i-1][0], log_1[i][1] / log_1[i-1][1]) for i in range(1, len(log_1))]
        
        pipe.transformer.compiled_transformer.mag_ratios = mag_ratios_1
        pipe.transformer.compiled_transformer.threshold = 0.04 # Alibaba default
        pipe.transformer.compiled_transformer.calibrate_mode = False
        print("Mag Ratio 1:", mag_ratios_1)

        if getattr(pipe, "transformer_2", None) is not None:
            log_2 = pipe.transformer_2.compiled_transformer.mag_log
            # Create tuples of (cond_ratio, uncond_ratio)
            mag_ratios_2 = [(log_2[i][0] / log_2[i-1][0], log_2[i][1] / log_2[i-1][1]) for i in range(1, len(log_2))]
            
            pipe.transformer_2.compiled_transformer.mag_ratios = mag_ratios_2
            pipe.transformer_2.compiled_transformer.threshold = 0.08 # Alibaba default
            pipe.transformer_2.compiled_transformer.calibrate_mode = False
            print("Mag Ratio 2:", mag_ratios_2)

        if args.profile != "no":
            with perf_time("Profile"):
                output_type = "latent" if args.profile == "dit" else "np"
                pipe.transformer.compiled_transformer.reset_for_new_generation()
                if getattr(pipe, "transformer_2", None) is not None:
                    pipe.transformer_2.compiled_transformer.reset_for_new_generation()
                
                generator = torch.Generator().manual_seed(args.base_seed)
                with jax.profiler.trace(args.profile_output_path):
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=args.frame_num,
                        guidance_scale=guidance_high,
                        guidance_scale_2=guidance_low,
                        num_inference_steps=3,
                        generator=generator,
                        max_sequence_length=256,
                        output_type=output_type,
                    ).frames[0]

        with perf_time("Benchmark"):
            pipe.transformer.compiled_transformer.reset_for_new_generation()
            if getattr(pipe, "transformer_2", None) is not None:
                pipe.transformer_2.compiled_transformer.reset_for_new_generation()
                
            generator = torch.Generator().manual_seed(args.base_seed)
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=args.frame_num,
                guidance_scale=guidance_high,
                guidance_scale_2=guidance_low,
                num_inference_steps=args.sample_steps,
                generator=generator,
                max_sequence_length=256,
            ).frames[0]

            file_name = f"{current_datetime}_batch_benchmark.mp4"
            export_to_video(output, file_name, fps=16)
            print(f"Benchmark video done. {file_name}")
    print("Done")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
