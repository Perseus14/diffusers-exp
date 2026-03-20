pip install -e .. && \
pip install transformers accelerate && \
pip install torch --index-url https://download.pytorch.org/whl/cpu && \
pip install -U jax[tpu]==0.9.1 && \
pip install torchax && \
pip install flax && \
pip install ftfy imageio imageio-ffmpeg
