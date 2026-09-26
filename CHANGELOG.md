# Changes

## Hopper (G) 1.2 (tag g-1.2.0, 2026-09-26)

A separate, general-purpose line. The serving code is unchanged from 1.1.1; only the adapter weights differ, published
at `HopitAI/hopper-g` (revision `d60a1d6`). On our local run of the Decision Index 0.2 suite (40 benchmarks, same
serving code): balanced raw 53.50 vs 52.74 for Hopper 1.1.1 (paired bootstrap +0.76, 95 % interval +0.55 to +0.98);
an independently trained second seed confirms (53.44). Research and demo use (see NOTICE). Hopper 1.1.1 is unchanged.

## 1.1.1 (2026-09-24)

1.1.1: long menus (> 26 options) via a disclosed two-stage shortlist; ≤ 26 options
unchanged; same weights and calibration map as 1.1.0; research and demo use (see NOTICE).

A serving-only compatibility release. Same adapter weights and calibration map as 1.1.0, byte for
byte, and the same prompt and wire format. **Every valid question with 26 options or fewer — every
JevBench item — gets the reply 1.1.0 gives it, byte for byte:** the same single eager forward pass,
the same readout, the same map. No setting routes such a question anywhere else.

- **Long menus: choice questions with more than 26 options are answered, through a disclosed
  shortlist.** Until now they were refused (HTTP 400, or 422 in-process). A first stage cuts the
  menu to k options and the ordinary single pass decides among them, with the same prompt, readout
  and calibration map. The default first stage is a tournament: the menu is dealt into
  `ceil(n / 26)` chunks with a fixed seed, each chunk is read by the ordinary single pass, and the
  10 options that score best in their chunks go to the final pass (`ceil(n / 26) + 1` passes in
  all). The reply carries a probability for every option in the request: the final distribution
  mixed with a uniform distribution over the whole menu at weight 0.05, so an eliminated option gets
  0.05 / n. The reply's answer is always the final pass's own answer, even where floating-point
  rounding would otherwise tie two survivors one representable step apart (the winner is then
  raised by that step). `--shortlist-residual` takes 0 to 0.5. Every prompt a long menu builds —
  each chunk and the final pass — is checked against the model's context before its forward pass,
  and refused (400, or 422 in-process) if it does not fit. An embedding first stage
  (`Qwen/Qwen3-Embedding-0.6B`, Apache-2.0, pinned revision, option embeddings cached) is available
  behind `--shortlist embedding` and is off by default until measured. `--shortlist off` (or
  `shortlist=None`) restores the 1.1.0 refusal. Measured on an A10G through the server with the
  shipped adapter and map, 300 public test items each: BANKING77 (77 options) top-1 0.673,
  recall@10 0.887, p50 321 ms; CLINC150 in-scope (150 options) top-1 0.863, recall@10 0.977, p50
  569 ms; see `README.md`, "Long menus". New modules: `shortlist.py`, `embedder.py`,
  and `pipeline.py`, which now holds the request path `Decider.score` runs, so it is tested without
  a GPU.

- **Stricter checks on malformed requests.** The request and its question must be JSON objects,
  choice criteria an object, score criteria a list, and labels a list of distinct strings. A
  repeated label used to collapse two options into one probability key, and a body such as
  `{"question": []}` used to crash the handler instead of returning 400. These now get a 400 over
  HTTP and a 422 in-process, as other malformed requests do; the in-process adapter also returns
  422, instead of raising, for a task record it cannot read at all (no `type` or `instructions`).
  Only requests that were malformed are affected.

- **CUDA graphs, opt-in (`--cuda-graphs`).** At start-up, after the fast-kernel guard and the length
  warm-up, the forward pass can be captured into one CUDA graph per length bucket, from 128 to
  4,096 tokens, and replayed where each bucket measured faster than eager. It is off by default in
  this release, because a graph-replayed forward is not bit-identical to the eager one: on our
  115-item development half, top answers were 115 / 115 identical but only 81 of 115 replies were
  bitwise identical, with probabilities moving by up to 0.031. `hopper-serve --cuda-graphs`,
  `Decider(cuda_graphs=True)` or `HopperDirectAdapter(cuda_graphs=True)` turns it on. Details in
  `README.md`, "CUDA graphs".

- **Research and demo use.** The NOTICE, model card and README now say that the adapter weights
  are offered for research and demo use only, because of the RACE training-data terms (see
  `NOTICE` and `MODEL_CARD.md`), as on the main branch.

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
