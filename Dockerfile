# Hopper: Qwen/Qwen3.5-4B at a pinned revision + the LoRA adapter (merged at load) + the calibration
# map, behind POST /v1/systemone and POST /run.
#
#   docker build -t hopper .
#   docker run --gpus all -p 8080:8080 -v hf:/models hopper
FROM python:3.12-slim

ENV HF_HOME=/models TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Triton (under flash-linear-attention) compiles a small C helper the first time it runs a kernel;
# the slim image has no C compiler, and without one every fast kernel fails at the first request.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
# The PyPI torch 2.8.0 wheel bundles its CUDA 12 runtime; the host needs only a driver and the
# NVIDIA container toolkit. Without flash-linear-attention and causal-conv1d, transformers silently
# falls back to a slow reference path for Qwen3.5's linear-attention layers.
RUN pip install --no-cache-dir torch==2.8.0 transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 \
    flash-linear-attention==0.5.2 \
 && pip install --no-cache-dir "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY hopper_decisions ./hopper_decisions
# --no-deps: the pinned dependencies are installed above, exactly.
RUN pip install --no-cache-dir --no-deps .

# ADAPTER is a Hugging Face repo id or a mounted directory; the calibration map ships in the package.
ENV ADAPTER=HopitAI/hopper NAME=hopper
EXPOSE 8080
CMD ["sh", "-c", "exec hopper-serve --adapter \"$ADAPTER\" --name \"$NAME\" --port 8080"]
