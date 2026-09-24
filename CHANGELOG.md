# Changes

## Unreleased (branch `inference-next`)

Same adapter weights and calibration map as 1.1.0. Same prompt and same wire format. Every question
with 26 options or fewer — every JevBench item — is answered exactly as in 1.1.0.

- **Long menus: choice questions with more than 26 options are answered, through a shortlist.**
  Until now they were refused (HTTP 400, or 422 in-process). A first stage cuts the menu to k
  options and the ordinary single pass decides among them, with the same prompt, readout and
  calibration map. The default first stage is a tournament: the menu is dealt into
  `ceil(n / 26)` chunks with a fixed seed, each chunk is read by the ordinary single pass, and the
  10 options that score best in their chunks go to the final pass (`ceil(n / 26) + 1` passes in
  all). The reply carries a probability for every option in the request: the final distribution
  mixed with a uniform distribution over the whole menu at weight 0.05, so an eliminated option gets
  0.05 / n and the winner can never change. An embedding first stage (`Qwen/Qwen3-Embedding-0.6B`,
  Apache-2.0, pinned revision, option embeddings cached) is available behind
  `--shortlist embedding` and is off by default until measured. `--shortlist off` (or
  `shortlist=None`) restores the 1.1.0 refusal. Accuracy with the adapter and latency on long menus
  are not yet measured; see `README.md`, "Long menus". New modules: `shortlist.py`, `embedder.py`,
  and `pipeline.py`, which now holds the request path `Decider.score` runs, so it is tested without
  a GPU.

- **CUDA graphs, on by default.** At start-up, after the fast-kernel guard and the length
  warm-up, the forward pass is captured into one CUDA graph per length bucket, from 128 to 4,096
  tokens. Requests are right-padded to the next bucket and replayed. Each bucket is timed against
  eager at start-up and used only for the request lengths where it measured faster. The graphs
  share one memory pool, and a bucket that would not leave room for a long eager request is
  skipped. `hopper-serve --no-cuda-graphs` (or `Decider(cuda_graphs=False)`,
  `HopperDirectAdapter(cuda_graphs=False)`) restores the 1.1.0 behaviour.
  On an A10G: standard-length p50 / p95 54.3 / 55.0 → 41.7 / 44.8 ms, and judge-length items
  unchanged (113.0 / 188.8 → 113.3 / 190.1 ms), because at those lengths the forward is bound by
  GPU work, not kernel launches. 265 / 265 top answers identical to eager, max |Δp| 0.031.
  Start-up adds 23.5 s, and graph memory is at most 1.3 GiB; 16 GB cards still work. Details in
  `README.md`, "CUDA graphs".

## 1.1.0

Same adapter weights as 1.0.0, byte for byte. Two changes, neither of which touches the prompt,
the forward pass or the wire format.

- **In-process JevBench route.** `hopper_decisions/jevbench_adapter.py` (`HopperDirectAdapter`,
  registered as `hopper_direct` by `jevbench_patch/hopper_direct-v1.3.0.patch`) lets the harness
  import Hopper instead of posting to it, as it does for `semif_direct`. It is not a second code
  path: `Decider` is built with the same arguments `hopper-serve` builds it with, and every
  decision goes through the same `Decider.score` and `request.harness_probs` the server's
  `POST /run` route uses. The HTTP route remains the primary one.
- **Calibration map replaced: one temperature per answer type.** `maps/hopper.json` is now a
  `per_kind` map — choice 0.790, noul 0.753, score 0.900 — instead of the 1.0.0 seven-coefficient
  `linear` map over option count, state length, JSON-or-not, answer type and output entropy. It is
  fitted on exactly the same held-out data as the 1.0.0 map: our own JevBench-style look-alike
  items, hard tier, 66 leakage-flagged items dropped, and never a JevBench item. A
  leave-one-source-out ablation showed the 1.0.0 map was worth +0.1 Calibration over applying no
  map on the two sources its fitting set had never seen, while the per-answer-type map is +4.5 and
  +2.9 there. **Answers are unchanged**: every map here divides log-probabilities by a positive
  temperature, so it can rescale a distribution but never reorder it. Only the confidences move.
  The +2.9 fold is 55 items, and a binned ECE over 55 items moves by more than that between two
  runs of the same weights on the same GPU, so the case for the change is the 517-item fold; see
  `MODEL_CARD.md`.
  The 1.0.0 map stays in the package as `maps/hopper-v1.0-linear.json`, so 1.0.0 can be reproduced
  from this repository with `hopper-serve --map .../maps/hopper-v1.0-linear.json`.

## 1.0.0

First release: serving code (request handling, prompt, one-pass option-letter readout, calibration
map and loader, fast-kernel guard, `/v1/systemone` and `/run` server, start-up warm-up), the
calibration map as package data, pinned `pyproject.toml`, Dockerfile, tests, README and model card.
