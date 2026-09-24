# Hopper

> **Research and demo use only.** This adapter is published for research and demonstration. Its training data included passages from RACE (via the `cais/mmlu` auxiliary set), which its authors release for non-commercial research only and whose terms extend to derived data. Do not use this adapter commercially. A version trained without these passages is in development.

A JevBench decision server. Hopper is a LoRA adapter on
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, merged into the bf16 weights at load. Each decision
takes one forward pass, with thinking off. The answer is read as a softmax over the option letters
(no text is generated), and a small calibration map, shipped in the package, then rescales that
distribution by one temperature per answer type (choice 0.790, noul 0.753, score 0.900). The map
never changes an answer.

The server speaks the JevBench `/v1/systemone` wire format, so the harness's existing `typesafe`
adapter runs unchanged. It also answers `remote_inproc`'s `POST /run`.

Code: Apache-2.0, the same licence as the base model. See `LICENSE` and `NOTICE`. The adapter
weights are offered for research and demo use only, because of the RACE training-data terms above
(see `MODEL_CARD.md`, "Training data").

## The adapter

- Hugging Face: [`HopitAI/hopper`](https://huggingface.co/HopitAI/hopper). The server downloads it
  on first start.
- Or the GitHub release `v1.1.0` of this repository, which has the same files and a `CHECKSUMS.txt`:

  ```sh
  gh release download v1.1.0 -R hopit-ai/hopper -D hopper-adapter
  (cd hopper-adapter && shasum -a 256 -c CHECKSUMS.txt)
  hopper-serve --adapter ./hopper-adapter --port 8080
  ```

The adapter is `adapter_config.json` and `adapter_model.safetensors` (rank 16, 130 MB).
`hopper.json` is the calibration map, and the same file ships inside the package.

The adapter weights are identical in 1.0.0 and 1.1.0; only the calibration map changed (see
`CHANGELOG.md`). The 1.0.0 map is kept in the package as
`hopper_decisions/maps/hopper-v1.0-linear.json`, so 1.0.0 can still be reproduced from this
repository by passing it to `hopper-serve --map`.

## Install and serve (RunPod, as tested)

Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (RunPod's "Runpod Pytorch 2.8.0"
template: Ubuntu 24.04, Python 3.12, CUDA 12.8.1, torch 2.8.0+cu128, `uv`). Blackwell cards need
CUDA 12.8 or newer.

```sh
git clone https://github.com/hopit-ai/hopper && cd hopper && git checkout v1.1.0
uv pip install --system --break-system-packages torch==2.8.0 transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 flash-linear-attention==0.5.2 "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install --system --break-system-packages --no-deps -e .
hopper-serve --adapter HopitAI/hopper --port 8080
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

### In-process route (optional, faster)

The harness can also import Hopper instead of posting to it, the way its `semif_direct` adapter
works. Same weights, same calibration map, same forward pass, same answers — the HTTP hop is simply
gone.

There is no PyPI release: install the package from a clone, exactly as the recipe above does, and
then apply the patch to the harness.

```sh
git clone https://github.com/hopit-ai/hopper && cd hopper && git checkout v1.1.0
uv pip install --system --break-system-packages torch==2.8.0 transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 flash-linear-attention==0.5.2 "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install --system --break-system-packages --no-deps -e .

cd <jevbench> && git checkout v1.3.0
git apply <hopper>/jevbench_patch/hopper_direct-v1.3.0.patch

JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run \
    --tasks datasets/public/original.jsonl --adapter hopper_direct --endpoint HopitAI/hopper \
    --model hopper --cost-basis self_hosted_gpu --reserve-usd 0 --results RESULTS.jsonl
```

The patch is a `git diff` against tag `v1.3.0` and touches two files: in `cli.py` the import,
the `kinds` entry, the `choices=` list on `--adapter`, and the two option tuples. It copies no code
into the harness: the adapter is `hopper_decisions/jevbench_adapter.py`, imported lazily from the
installed package, so a host without `hopper-decisions` runs every other adapter unchanged.
`--endpoint` is the LoRA adapter — a Hub repo id or a local directory, the same value
`hopper-serve --adapter` takes — and may be omitted.

`JEVBENCH_WARM_LOAD=1` is the harness's own convention for in-process entrants: it calls `load()`
before starting the clock, so the weights, the fast-kernel check and the length warm-up all happen
outside the measurement, as they do for a server that is already up. Without it the first decision
carries the whole minute of start-up.

Out-of-contract inputs — a question type we do not serve, a malformed question, a prompt past the
model's context, and more than 26 options if the shortlist is turned off (see "Long menus") — come
back as `status = 422`, which the runner counts as a wrong answer and
exempts from its three-consecutive-failure abort, so one bad item can never stop a run. A load
failure or a CUDA fault comes back with no status and does count. `jevbench_patch/README.md` has
the details.

Measured on an A10G through the harness's own `cli.py` at v1.3.0 with this patch, over the same
115-item development half: standard-tier p50 / p95 **52.6 / 55.7 ms** and all-items 55.3 / 460.6 ms,
against 53.4 / 55.6 and 56.2 / 472.0 for the HTTP route on the same image and GPU. 114 of 115 top
answers identical to our evaluation — the same exact tie as in the table below — with no invalid
replies and no failures, and the probabilities identical to the served ones. Over a localhost
loopback the hop costs about a millisecond, so the gain here is small; how large it is on your own
host is yours to measure.

Re-measured with the 1.1.0 calibration map on a second A10G: same answers, same probabilities,
same `model` and `usage` fields, no failures. That container was a uniformly slower host —
standard-tier p50 / p95 67.3 / 71.5 ms in pass 1 and 63.7 / 65.5 in pass 2 — which is the box, not
the map: post-processing, where the map is applied, is 0.03 ms per decision.

The HTTP route above remains the primary one; this is the same system with one less hop.

## Long menus (more than 26 options)

The answer is read as a softmax over one letter per option, so one forward pass can weigh at most
26 options. Up to 1.1.0 a choice question with more options was refused (HTTP 400, or 422 on the
in-process route). Routing and retrieval menus are longer than that: 77 intents, 150 intents,
hundreds of tools. Hopper now answers them in two stages:

1. **A first stage cuts the menu to k options.** By default this is a *tournament*, which needs no
   extra model. The menu is dealt round-robin into `ceil(n / 26)` chunks of nearly equal size, in an
   order shuffled with a fixed seed, and each chunk is read by the ordinary single pass. An option's
   score is its probability within its own chunk.
2. **The ordinary single pass decides among the k best**, shown in the menu's own order, with the
   same prompt layout, the same readout and the same calibration map as every other decision.

A 77-option menu costs 3 + 1 = 4 forward passes and a 150-option menu 6 + 1 = 7. Every pass is
counted in `usage.input_tokens`. Nothing changes for a question with 26 options or fewer, which is
every JevBench item: it takes the same single eager pass as in 1.1.0 and gets the same reply, byte
for byte, and no setting sends it to the shortlist. Each chunk and the final pass is checked against
the model's context before it runs; a menu whose prompt does not fit is refused (400, or 422
in-process).

**Every option gets a probability, not just the k survivors.** The reply is the final pass's
(calibrated) distribution mixed with a small uniform distribution over the whole menu:

```
p(option) = (1 - r) * final(option) + r / n     if the option reached the final pass
p(option) =                           r / n     if the first stage eliminated it
```

with `r = 0.05` by default and `n` the number of options in the request. The probabilities sum to 1
and every label the request sent is in the reply, with positive mass. The mixture adds the same
amount to every label, so it does not change which option wins. In floating point, two survivors one
representable step apart can round to the same value, and the harness would then break the tie on
the label; so the reply is checked exactly as it is normalised and, only in that case, the final
pass's winner is raised by that one step. The answer is always the final pass's own answer. Together the eliminated options
get `r (n - k) / n`, which is meant to be the probability that the first stage threw away the right
answer. Zero would declare that impossible, and any proper score (log loss above all) punishes that
without bound when it happens. Too much would take confidence away from every right answer that
survived. 0.05 comes from the one measurement we have, below: with k = 10 the tournament dropped the
right answer on 4.0 % of CLINC-150 items and 15.2 % of Banking77 items, and 0.05 puts 4.7 % and
4.4 % on the eliminated options there. It is the low end of what was measured, and it is to be
refitted as `1 - recall@k` on held-out long menus with the adapter, never tuned on a benchmark.
`--shortlist-residual` takes any value from 0 to 0.5, so the final pass always carries at least
half the mass. `--shortlist-residual 0` gives the eliminated options exactly 0.0; they are still
listed. Labels must be distinct strings: a repeated label is refused rather than collapsed into one
probability key.

The calibration map was fitted on single-pass distributions over at most 26 options, so it is
applied to the final pass, before the mixture, and never to the whole menu.

**The embedding first stage (off by default).** `--shortlist embedding` replaces
the tournament with an embedding model: each option (the exact line the prompt shows for it) and
the request's state are embedded, and the k options with the highest dot product go to the final
pass. The model is [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
(Apache-2.0) at revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, used the way its model card
says: last-token pooling, normalised vectors, and a one-line instruction on the query side only.
Option vectors are cached (least recently used, up to 20,000 texts), so a fixed menu is embedded
once and each later request costs one embedding of its state plus one decision pass. It is
downloaded from the Hub on the first start with this flag and never otherwise, and it adds about
1.2 GB of weights on the GPU. Its default k is 26: it is a different model from the decider, so its
misses are not the decider's misses, and the final pass should see as many candidates as it can.
Measured against the tournament (below), it was level on accuracy and much faster once a menu is
cached, but it adds a second model, so it stays off by default.

| Flag | Default | |
| --- | --- | --- |
| `--shortlist` | `tournament` | `tournament`, `embedding`, or `off` (refuse menus over 26 options, as 1.1.0 did) |
| `--shortlist-k` | 10 (tournament), 26 (embedding) | options in the final pass, 2 to 26 |
| `--shortlist-residual` | 0.05 | `r` above, 0 to 0.5 |
| `--shortlist-seed` | 0 | how the tournament deals its chunks |
| `--embedding-model`, `--embedding-revision` | as above | the embedding model, for `--shortlist embedding` |

The in-process adapter takes the same settings as `HopperDirectAdapter(shortlist=...)`, a
`hopper_decisions.shortlist.Config`, or `None` to refuse long menus. For a shortlisted item, its
`raw.runtime` names the strategy, k, the options kept, the passes and the residual. Everything is
deterministic: the same request gets the same answer. Nothing is sent anywhere.

**Measured so far.** The tournament was measured on our own samples of CLINC-150 (150 intents,
150 items) and Banking77 (77 intents, 125 items), with the **frozen base model — not Hopper's
adapter** — and no calibration map:

| | k | top-1 accuracy | right answer kept by the first stage (recall@k) | passes |
| --- | ---: | ---: | ---: | ---: |
| CLINC-150 | 10 | 0.860 | 0.960 | 7 |
| CLINC-150 | 26 | 0.833 | 0.973 | 7 |
| Banking77 | 10 | 0.704 | 0.848 | 4 |
| Banking77 | 26 | 0.696 | 0.952 | 4 |

k = 10 was at least as good as k = 26 on both and costs the same passes, so it is the default. The
decision model's own mean-pooled hidden states, tried as an embedding stage in the same runs, kept the right
answer in the top 26 only 39 % of the time on CLINC-150; that is why the embedding stage uses a
dedicated embedding model instead.

**Measured with Hopper's adapter and map (1.1.1, 2026-09-24).** One A10G, eager, through the HTTP
server, default settings, 300 items each from the public test splits of BANKING77 (77 intents)
and CLINC150 (150 intents, out-of-scope dropped):

| | BANKING77 | CLINC150 |
| --- | ---: | ---: |
| top-1 accuracy (default: tournament, k = 10) | 0.673 | 0.863 |
| right answer kept by the first stage (recall@10) | 0.887 | 0.977 |
| accuracy with the embedding first stage, k = 10 | 0.667 | 0.860 |
| latency p50 / p95, default | 321 / 332 ms | 569 / 577 ms |
| latency p50 / p95, embedding stage, menu cached | 79 / 85 ms | 79 / 84 ms |

Standard error about 0.027 and 0.020. The adapter was trained on menus of at most six options; on
these long menus it is level with the frozen base model within sampling error. Repeated requests
got identical replies (50 of 50). With the embedding stage loaded, the server starts and serves
long menus on a card limited to 16 GB (peak 11.1 GiB). On BANKING77 the served top choice is
overconfident (mean top probability 0.78 against 0.67 accuracy), and `r` is still the default 0.05:
it under-covers BANKING77's first-stage misses and over-covers CLINC150's; refitting it on held-out
long menus is open. Not measured: CLINC150's out-of-scope class, and other long-menu tasks.

## GPU and memory

You need one CUDA GPU with 16 GB or more. The bf16 weights take about 9 GB, and the adapter is
merged into them at load. The CUDA graphs (below, opt-in) add at most 1.3 GiB. On an A10G the server used
10.8 GiB of device memory in total after serving every item. With the card limited to what a
16 GB card reports, peak use was 11.4 GiB. The optional embedding stage for long menus adds about
1.2 GB. The longest public item we ran was 3,708 tokens. We tested on an A10G,
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
answer. `--no-length-warmup` turns it off. Start-up takes about a minute once the weights are local,
plus about 25 s for the CUDA graphs if `--cuda-graphs` is given.

## CUDA graphs (opt-in: `--cuda-graphs`)

**Off by default in 1.1.1.** This release is a compatibility release whose replies to questions of
26 options or fewer are byte-identical to 1.1.0's, and a graph-replayed forward is not: the padded
shape changes the GPU's kernel tiling, so probabilities move in the last digits (top answers were
265 / 265 identical in the measurement below, but only 81 of the 115 development-half replies were
bitwise identical, and the largest probability change was 0.031). Turn graphs on with
`hopper-serve --cuda-graphs`, `Decider(cuda_graphs=True)` or `HopperDirectAdapter(cuda_graphs=True)`
when latency matters more than bit-for-bit agreement with 1.1.0.

With `--cuda-graphs`, after the guard and the warm-up, the server captures the forward pass into one CUDA graph per
length bucket: 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536,
2048, 3072 and 4096 tokens. A request is right-padded to the next bucket and the graph is
replayed, which removes the host's kernel-launch time. The padding cannot change an answer: the
model is causal, the answer is read at the last real token, and the linear-attention chunks are
counted from the first token, so they do not move. Requests over 4,096 tokens run eager.

A graph only helps a forward whose time goes into kernel launches, and padded tokens are real
GPU work. So at start-up each bucket's replay is timed against eager forwards at both ends of its
range, and the graph serves only the request lengths where it measured at least 2 % faster.
Every other request runs eager, exactly as without `--cuda-graphs`. On an A10G that means graphs
up to about 500 tokens (plus a sliver at 1,249-1,280). Above that the forward is bound by GPU
work, and the replay is no faster than eager. A faster GPU stays launch-bound to longer prompts
and keeps more buckets. The start-up log prints which buckets are in use, from which length, and
why any other bucket is not (not faster here, not enough memory, or a failed capture, which falls
back to eager).

All graphs share one memory pool. A bucket is captured only if it fits while enough memory stays
free for an eager forward of about 6,000 tokens. On a smaller card the ladder therefore stops
early rather than failing, and the log says where.

Measured on an A10G over our 115-item development half and 150 of our own judge-length items
(405-1,419 tokens, mean 663). Numbers are p50 / p95 in ms, in-process:

| | eager (the default) | CUDA graphs (`--cuda-graphs`) |
| --- | ---: | ---: |
| standard and easy items (mean 191 tokens) | 54.3 / 55.0 | 41.7 / 44.8 |
| judge-length items (mean 663 tokens) | 113.0 / 188.8 | 113.3 / 190.1 |

Answers: 265 / 265 top answers identical with and without graphs; 81 of the 115 development-half
replies and 146 of the 150 judge-length replies were bitwise identical. The largest probability
change is 0.031, which is the size of the difference between the serving path and our evaluation
path with no graphs at all. The same holds with the card limited to 16 GB. Capture takes 23.5 s on
the A10G. The in-process adapter takes `cuda_graphs=True` to turn graphs on.

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

## Changes

`CHANGELOG.md`. 1.1.1 answers choice questions with more than 26
options through a disclosed shortlist and leaves every question of 26 options or fewer exactly as
1.1.0 answers it, with the same weights and calibration map; CUDA graphs are available but opt-in.
It is for research and demo use (see `NOTICE`). 1.1.0 adds the in-process route above and replaces the calibration map with one
temperature per answer type; the adapter weights are unchanged from 1.0.0, and so is every answer.
