# Changes

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
