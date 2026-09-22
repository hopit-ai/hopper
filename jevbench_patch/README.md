# Registering `hopper_direct` with the harness

`hopper_direct-v1.3.0.patch` is a `git diff` against JevBench tag **v1.3.0**. It registers the in-process
adapter that ships in this package (`hopper_decisions/jevbench_adapter.py`) at the five points the
harness needs, and adds nothing else. It does not copy any code into the harness: the adapter is
imported from the installed package, lazily, so a host without `hopper-decisions` runs every other
adapter exactly as before.

```sh
cd <jevbench>
git checkout v1.3.0
git apply /path/to/hopper/jevbench_patch/hopper_direct-v1.3.0.patch
```

The five points in `jevbench/cli.py`, plus the lazy factory that takes the place of copying a file
into `jevbench/adapters/`:

| where | what |
| --- | --- |
| `jevbench/adapters/__init__.py` | a lazy `HopperDirectAdapter(**kw)` factory, beside the existing `NeedleLocalAdapter` one |
| `cli.py` import | `HopperDirectAdapter` added to the `from .adapters import (...)` list |
| `cli.py` `kinds` | `"hopper_direct": HopperDirectAdapter` |
| `cli.py` no-endpoint tuple | `--endpoint` becomes optional; it defaults to `HopitAI/hopper` |
| `cli.py` revision tuple | `--revision` is forwarded, to repin the base model if you ever need to |
| `cli.py` `choices=` on `--adapter` | argparse validates this list separately, so the name is rejected before anything loads without it |

## Installing the package

There is no PyPI release. Install from a clone of this repository, with the same pinned runtime the
main README's recipe installs:

```sh
git clone https://github.com/hopit-ai/hopper && cd hopper && git checkout v1.1.0
uv pip install --system --break-system-packages torch==2.8.0 transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 flash-linear-attention==0.5.2 "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install --system --break-system-packages --no-deps -e .
```

On a host without `uv`, run the same two lines with `pip install` in a virtualenv (drop `uv` and
`--system --break-system-packages`). The versions are not optional: the server's fast-kernel guard
runs here too and refuses to start on the slow reference path, so a misinstalled stack fails loudly
instead of producing a slow number.

## Running it

```sh
JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run \
    --tasks datasets/public/original.jsonl \
    --adapter hopper_direct --endpoint HopitAI/hopper \
    --model hopper --cost-basis self_hosted_gpu --reserve-usd 0 \
    --results RESULTS.jsonl
```

`--endpoint` is the LoRA adapter, a Hub repo id or a local directory: the same value
`hopper-serve --adapter` takes. It may be omitted, in which case it is `HopitAI/hopper`.

**`JEVBENCH_WARM_LOAD=1` matters.** It is `cli.py`'s own convention for in-process entrants: the
harness calls `load()` before it starts the clock, so the weights, the merged adapter, the
fast-kernel check and the length warm-up all happen outside the measurement, as they would for a
server that is already up. Without it the first decision carries the whole start-up (about a minute)
and the run's latency is not a number worth publishing.

## What it does and does not change

The adapter builds `Decider` exactly as `hopper_decisions.server` builds it -- the calibration map
shipped in the package, the fast-kernel guard on, the length warm-up on -- and answers each decision
through the same `Decider.score` and `request.harness_probs` the server's `POST /run` route uses.
The probabilities are therefore identical to the served ones, float for float; only the HTTP hop is
gone.

Out-of-contract inputs (more than 26 options, a question type we do not serve, a prompt past the
model's context) come back as `status = 422` with `error` set, which `runner.py` records as
unprocessable and exempts from its three-consecutive-failure abort, so one bad item can never stop a
run. A load failure or a CUDA fault comes back with no status and does count toward that rule,
because three of those in a row is a real fault.

`usage` is `{"input_tokens": n, "output_tokens": 0}`, where `n` is the prompt length in tokens --
the same count the HTTP route reports. Nothing is generated, so there are no output tokens.
`cost_basis` is `self_hosted_gpu` and `reserve_estimate` is 0: there is no billable account.
