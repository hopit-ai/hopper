# Hopper

A JevBench decision server. Hopper is a LoRA adapter on
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, merged into the bf16 weights at load. Each decision
takes one forward pass, with thinking off. The answer is read as a softmax over the option letters
(no text is generated), and a small calibration map, shipped in the package, then rescales that
distribution. The map never changes an answer.

The server speaks the JevBench `/v1/systemone` wire format, so the harness's existing `typesafe`
adapter runs unchanged. It also answers `remote_inproc`'s `POST /run`.

Code and adapter weights: Apache-2.0, the same licence as the base model. See `LICENSE` and `NOTICE`.

## The adapter

- Hugging Face: [`HF_ORG/hopper`](https://huggingface.co/HF_ORG/hopper). The server downloads it
  on first start.
- Or the GitHub release `v1.0.0` of this repository, which has the same files and a `CHECKSUMS.txt`:

  ```sh
  gh release download v1.0.0 -R hopit-ai/hopper -D hopper-adapter
  (cd hopper-adapter && shasum -a 256 -c CHECKSUMS.txt)
  hopper-serve --adapter ./hopper-adapter --port 8080
  ```

The adapter is `adapter_config.json` and `adapter_model.safetensors` (rank 16, 130 MB).
`hopper.json` is the calibration map, and the same file ships inside the package.

## Install and serve (RunPod, as tested)

Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (RunPod's "Runpod Pytorch 2.8.0"
template: Ubuntu 24.04, Python 3.12, CUDA 12.8.1, torch 2.8.0+cu128, `uv`). Blackwell cards need
CUDA 12.8 or newer.

```sh
git clone https://github.com/hopit-ai/hopper && cd hopper && git checkout v1.0.0
uv pip install --system --break-system-packages torch==2.8.0 transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 flash-linear-attention==0.5.2 "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install --system --break-system-packages --no-deps -e .
hopper-serve --adapter HF_ORG/hopper --port 8080
```

On any other Linux host with Python 3.12 and a CUDA 12.8+ driver, run the same two installs with
`pip install` in a virtualenv (drop `uv` and `--system --break-system-packages`). You also need a
C compiler (`gcc`), because Triton builds a small helper the first time it runs a kernel.

`hopper-serve` is the same as `python -m hopper_decisions.server`. Its defaults are the packaged
calibration map, `--name hopper` (the `model` field of every reply) and port 8080. `--no-map`
serves raw probabilities.

## Docker

```sh
docker build -t hopper .
docker run --gpus all -p 8080:8080 -v hf:/models hopper
# or with a local adapter: -v /path/to/hopper-adapter:/adapter -e ADAPTER=/adapter
```

## Running the benchmark

```sh
python -m jevbench.cli run --tasks datasets/public/original.jsonl --adapter typesafe \
    --endpoint http://127.0.0.1:8080 --key-env '' --model hopper \
    --results RESULTS.jsonl --cost-basis self_hosted_gpu --reserve-usd 0
```

The server is single-threaded on purpose, because the harness runs serially. It needs no API key
and ignores any `Authorization` header.

## GPU and memory

You need one CUDA GPU with 16 GB or more. The bf16 weights take about 9 GB, and the adapter is
merged into them at load. The longest public item we ran was 3,708 tokens. We tested on an A10G,
an H100 NVL and an RTX PRO 4500 Blackwell.

## Start-up: fast-kernel guard and warm-up

Qwen3.5's linear-attention layers need `flash-linear-attention` and `causal-conv1d`. Without them,
transformers falls back to a PyTorch reference path that is more than 10x slower, and it only logs
a warning. At start-up the server runs real forwards, records which implementation of each op
actually ran, and **refuses to start** if any of them is the slow path or if the forward fails on
this GPU. The error names the package and the pinned install command. The log shows the GPU,
compute capability, CUDA and torch versions, and one line per op, then `fast-kernel check passed`.
`--allow-slow-kernels` starts anyway, for debugging only. Never time or submit such a run.

The same start-up forwards (39, 1,948, 3,996 and 6,044 tokens) compile and autotune every Triton
kernel before the first request. Otherwise the first request past each 2,048-token band would
pay about 10 s once. After them the server also runs 16 forwards at lengths spread between 64 and
2,048 tokens. This absorbs per-length first-use costs seen on an H100, and it cannot change any
answer. `--no-length-warmup` turns it off. Start-up takes about a minute once the weights are local.

## Measured latency

We used the harness's own `typesafe` adapter over localhost HTTP, serially, on our 115-item public
development half. Numbers are standard-tier p50 / p95 in ms. Pass 1 is every item once, straight
after start-up. Pass 2 is the same items again.

| GPU | pass 1 | pass 2 | top answers identical to our evaluation |
| --- | --- | --- | ---: |
| H100 NVL (RunPod, recipe above) | 51.6 / 80.1 | 44.0 / 45.2 | 115 / 115 |
| RTX PRO 4500 Blackwell (RunPod, recipe above) | 45.6 / 46.8 | 45.4 / 46.5 | 115 / 115 |
| A10G (Docker image) | 53.7 / 56.7 | 53.5 / 55.0 | 114 / 115 |
| A10G (this repository's Dockerfile, length warm-up on) | 54.0 / 57.3 | 53.6 / 56.8 | 114 / 115 |

The first three rows were measured before the length warm-up was added. The last row is this
release, built from this repository alone. On the A10G, the one item that differs is a yes/no item
whose saved prediction was an exact tie (0.5 / 0.5); the served run breaks the tie the other way. At these lengths the forward pass is bound
by kernel launches, so the host CPU matters about as much as the GPU. No reply was invalid under
the harness's 1e-3 tolerance.

## Tests

```sh
pip install -e ".[test]" && pytest     # no GPU, no network
```
