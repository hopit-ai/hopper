"""In-process JevBench adapter: the harness imports Hopper instead of posting to it.

Registered as `hopper_direct` (see `jevbench_patch/`), this runs inside the harness process, so a
decision costs one forward pass and no HTTP hop:

  JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run --tasks ... \\
      --adapter hopper_direct --endpoint HopitAI/hopper --cost-basis self_hosted_gpu

It is the same code path as `hopper-serve`, not a second one: `Decider` is built with the shipped
calibration map, the fast-kernel guard on, the length warm-up on and CUDA graphs on, exactly as
`hopper_decisions.server` builds it, and every decision goes through `Decider.score` and
`request.harness_probs`, exactly as the server's `POST /run` route does. The answers are therefore
identical to the served ones, down to the float.

`JEVBENCH_WARM_LOAD=1` makes the harness call `load()` before it starts the clock (jevbench/cli.py),
which is the leaderboard's convention for in-process entrants: weights loaded and kernels warm, like
a server that is already up. Without it the weights load inside the first decision's measurement,
which is not a number anyone should publish.

Failure policy, following the runner's stop rule (jevbench/runner.py): anything the contract cannot
represent -- more than 26 options, a question type we do not serve, a prompt longer than the model's
context -- comes back as `status = 422`, which the runner records as unprocessable and exempts from
the three-consecutive-failure abort. A load failure or a CUDA fault comes back with no status and
does count, because three of those in a row is a real fault and stopping is right.

Standard library and this package only. No network beyond the adapter download `Decider` already
does, and nothing is reported anywhere.
"""

from __future__ import annotations

import time

from hopper_decisions import HF_REPO, MAP, NAME, request

_DecisionResult = None


def _result(**fields):
    """The harness's own `DecisionResult` (jevbench/adapters/base.py), imported on first use so that
    this module imports -- and its contract tests run -- without the harness checked out."""
    global _DecisionResult
    if _DecisionResult is None:
        from jevbench.adapters.base import DecisionResult
        _DecisionResult = DecisionResult
    return _DecisionResult(**fields)


class HopperDirectAdapter:
    name = "hopper_direct"
    cost_basis = "self_hosted_gpu"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, revision=None,
                 calibration_map=MAP, allow_slow_kernels=False, cuda_graphs=True):
        # `endpoint` is the LoRA adapter: a Hugging Face repo id or a local directory, the same
        # value `hopper-serve --adapter` takes. `revision` optionally repins the base model.
        self.endpoint = endpoint or HF_REPO
        self.model = model or NAME
        self.revision = revision
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.calibration_map = calibration_map
        self.allow_slow_kernels = allow_slow_kernels
        self.cuda_graphs = cuda_graphs
        self.load_s = None
        self.fast_kernels = None
        self._decider = None

    def load(self):
        """Build the decider the server builds. Idempotent; called before the clock when
        JEVBENCH_WARM_LOAD=1, and otherwise by the first decision."""
        if self._decider is None:
            from hopper_decisions.model import Decider
            started = time.perf_counter()
            self._decider = Decider(adapter=self.endpoint, calibration_map=self.calibration_map,
                                    name=self.model, allow_slow_kernels=self.allow_slow_kernels,
                                    cuda_graphs=self.cuda_graphs,
                                    **({"revision": self.revision} if self.revision else {}))
            self.load_s = time.perf_counter() - started
            self._report(self._decider)
        return self._decider

    def _report(self, decider):
        """What `hopper-serve` prints at start-up, on the run's own log: which GPU, whether the
        linear-attention kernels are the fast ones, and how long loading and warming took. An
        in-process route has no server log, and a number is only worth publishing if this says so."""
        self.fast_kernels = not decider.slow_kernels
        gpu, (count, warm) = decider.gpu, decider.warm_seconds
        plan = getattr(decider, "graph_plan", None) or {}
        graphs = (f"cuda graphs {len(plan)} buckets, {min(plan)} to {max(plan)} tokens" if plan
                  else "cuda graphs off" if not getattr(decider, "graph_status", None) else "cuda graphs none in use")
        kernels = ", ".join(f"{op}={info['implementation']}{'' if info['fast'] else ' (SLOW)'}"
                            for op, info in decider.kernels.items())
        print(f"[hopper] {decider.name} on {gpu.get('name')} (compute capability {gpu.get('capability')}, "
              f"CUDA {gpu.get('cuda')}, torch {gpu.get('torch')}); "
              f"{'fast-kernel check passed' if self.fast_kernels else 'WARNING: SLOW reference path, do not time this run'}"
              f"{'; ' + kernels if kernels else ''}; loaded in {self.load_s:.1f} s, "
              f"{count} lengths warmed in {warm:.1f} s; {graphs}", flush=True)

    def build_request(self, task):
        """The canonical record as `hopper_decisions.request.parse` reads it. The question is built
        the way the harness's own `adapters/base.build_question` builds it, and `labels` carries the
        record's exact label strings, which are what we answer over."""
        question = {"type": task.question["type"], "instructions": task.question["instructions"]}
        if task.question.get("criteria") is not None:
            question["criteria"] = task.question["criteria"]
        return {"state": task.state, "question": question, "labels": list(task.labels)}

    def run(self, task):
        res = _result(adapter=self.name, ok=False, probs_source="native", model=self.model)
        body = self.build_request(task)
        res.request_body = {"id": task.id, "question": body["question"], "labels": body["labels"]}
        try:
            decider = self.load()
        except Exception as error:  # noqa: BLE001 - no weights is an infrastructure fault, not a 422
            res.error = f"load failed: {type(error).__name__}: {str(error)[:250]}"
            return res
        started = time.perf_counter()
        try:
            out = decider.score(body)
        except Exception as error:  # noqa: BLE001
            res.latency_s = time.perf_counter() - started
            res.status = self._refusal_status(decider, body, error)
            res.error = f"{type(error).__name__}: {str(error)[:300]}"
            return res
        res.latency_s = time.perf_counter() - started
        reply, kind = out["response"], out["example"]["kind"]
        (_, answer), = reply["answers"].items()  # one question per request, as the server's /run route reads it
        res.probs = request.harness_probs(kind, answer)
        res.usage = reply["usage"]          # the very count request.response() reports
        res.model = reply["model"]
        parse_s, forward_s, post_s = out["seconds"]
        res.raw = {"answer": {"probabilities": res.probs, "uncalibrated": out["raw"],
                              "input_tokens": out["tokens"], "question_type": kind},
                   "runtime": {"model": reply["model"], "readout": "option-letter softmax, one forward pass",
                               "probability_origin": "native-option-letter-softmax-then-calibration-map",
                               "fast_kernels": self.fast_kernels,
                               "seconds": {"tokenise": parse_s, "forward": forward_s, "post": post_s}}}
        res.ok = True
        return res

    def _refusal_status(self, decider, body, error):
        """422 when the input is one the contract cannot represent, so the runner counts it as a
        wrong answer and not toward its abort rule; None when it looks like a fault of ours, which
        should be allowed to stop the run. Only ever reached on a failure, so its cost is free."""
        if isinstance(error, (ValueError, KeyError, TypeError)):
            return 422  # request.parse refused it: bad type, missing criteria, more than 26 options
        try:
            example, _ = request.parse(body)
            return 422 if len(decider.encode(example)) > self.context_limit(decider) else None
        except Exception:  # noqa: BLE001 - the diagnosis must never replace the real error
            return None

    @staticmethod
    def context_limit(decider):
        return getattr(decider.model.config, "max_position_embeddings", 0) or float("inf")

    def reserve_estimate(self, task):
        return 0.0  # our own GPU: the harness's ledger has nothing to bill
