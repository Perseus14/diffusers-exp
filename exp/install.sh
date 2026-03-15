pip install -e .. && \
pip install transformers accelerate && \
pip install torch --index-url https://download.pytorch.org/whl/cpu && \
pip install -U jax[tpu] && \
pip install torchax && \
pip install flax && \
pip install ftfy imageio imageio-ffmpeg
