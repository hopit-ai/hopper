"""The in-process adapter's side of the harness contract, with a fake decider. No GPU, no network.

The harness is not a dependency of this package, so the two pieces of it the adapter touches are
rebuilt here from jevbench/adapters/base.py and jevbench/tasks.py at v1.3.0: `DecisionResult`'s
fields and a task record's attributes. Everything else -- the request the adapter builds, the
probabilities, the usage counts, the 422 rule and the warm-load convention -- is checked against a
`Decider` stand-in that runs the real `hopper_decisions.request` code without any weights.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from hopper_decisions import NAME, calibration, jevbench_adapter, pipeline, request
from hopper_decisions.jevbench_adapter import HopperDirectAdapter
from hopper_decisions.shortlist import DEFAULT as SHORTLIST


# --- the harness's DecisionResult (jevbench/adapters/base.py, v1.3.0), field for field ----------
@dataclass
class DecisionResult:
    adapter: str
    ok: bool
    probs: Optional[dict] = None
    probs_source: str = "unknown"
    model: str = ""
    status: Optional[int] = None
    error: Optional[str] = None
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    raw: Optional[Any] = None
    request_body: Optional[Any] = None
    label: Optional[str] = None


@dataclass
class Task:  # jevbench/tasks.py, only the attributes an adapter reads
    id: str
    state: Any
    question: dict
    labels: list
    family: str = "trap"
    split: str = "public"


@pytest.fixture(autouse=True)
def harness(monkeypatch):
    """Put `jevbench.adapters.base` on the import path, as it is inside the harness process."""
    base = types.ModuleType("jevbench.adapters.base")
    base.DecisionResult = DecisionResult
    package = types.ModuleType("jevbench")
    adapters = types.ModuleType("jevbench.adapters")
    monkeypatch.setitem(sys.modules, "jevbench", package)
    monkeypatch.setitem(sys.modules, "jevbench.adapters", adapters)
    monkeypatch.setitem(sys.modules, "jevbench.adapters.base", base)
    monkeypatch.setattr(jevbench_adapter, "_DecisionResult", None)
    yield
    jevbench_adapter._DecisionResult = None


class FakeDecider:
    """`Decider` without weights: `score` is the real request path (`pipeline.score`, which
    `model.Decider.score` is), with a fixed distribution in place of the forward pass: 0.7 on the
    first option a pass shows, the rest shared evenly."""

    def __init__(self, name=NAME, fail=None, context=1024, slow=(), shortlist=SHORTLIST):
        self.name, self.map, self.calls, self.fail = name, calibration.Constant(1.0), [], fail
        self.shortlist, self.embedder, self.shown = shortlist, None, []
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=context))
        self.gpu = {"name": "NVIDIA A10G", "capability": "8.6", "cuda": "12.8", "torch": "2.8.0"}
        self.kernels = {"causal_conv1d_fn": {"implementation": "causal_conv1d.causal_conv1d_fn",
                                             "fast": True, "ran": True}}
        self.slow_kernels, self.warm_seconds = list(slow), (16, 3.2)

    def encode(self, example):
        self.shown.append(request.names(example))
        return list(range(4 * len(example["document"].split()) + 8))

    def letter_probs(self, ids, count):
        assert count == len(self.shown[-1]) <= 26
        return [0.7 if i == 0 else 0.3 / (count - 1) for i in range(count)]

    def score(self, req):
        self.calls.append(req)
        if self.fail is not None:
            raise self.fail
        return pipeline.score(self, req)


def adapter(decider=None, **kw):
    a = HopperDirectAdapter(**kw)
    a._decider = decider if decider is not None else FakeDecider()
    return a


def choice_task(n=3):
    names = [f"opt{i}" for i in range(n)]
    return Task(id="c-1", state="The customer wants a refund.", labels=names,
                question={"type": "choice", "instructions": "Route it.",
                          "criteria": {k: f"about {k}" for k in names}})


NOUL = Task(id="n-1", state="Order 12 arrived late.", labels=["no", "yes"],
            question={"type": "noul", "instructions": "Is it late?",
                      "criteria": {"true": "it is late", "false": "it is not"}})
SCORE = Task(id="s-1", state={"amount": 12, "note": "e"}, labels=["0", "1", "2"],
             question={"type": "score", "instructions": "How urgent?",
                       "criteria": ["not urgent", "somewhat", "very"]})


def test_name_cost_basis_and_model():
    a = HopperDirectAdapter()
    assert (a.name, a.cost_basis, a.model) == ("hopper_direct", "self_hosted_gpu", "hopper")
    assert a.endpoint == "HopitAI/hopper"          # the published adapter, when --endpoint is omitted
    assert a.reserve_estimate(choice_task()) == 0.0
    assert a.price_input_per_m is None and a.price_output_per_m is None


def test_constructor_takes_every_kwarg_the_cli_passes():
    a = HopperDirectAdapter(endpoint="/adapters/local", model="hopper", key_env="",
                            price_input_per_m=None, price_output_per_m=None, revision="abc123")
    assert (a.endpoint, a.model, a.revision) == ("/adapters/local", "hopper", "abc123")


@pytest.mark.parametrize("task", [choice_task(), NOUL, SCORE])
def test_build_request_matches_the_harness_question(task):
    body = HopperDirectAdapter().build_request(task)
    assert body == {"state": task.state, "labels": task.labels,
                    "question": {"type": task.question["type"],
                                 "instructions": task.question["instructions"],
                                 "criteria": task.question["criteria"]}}
    request.parse(body)  # the serving path accepts it unchanged


def test_build_request_omits_absent_criteria():
    task = Task(id="x", state="s", labels=["no", "yes"],
                question={"type": "noul", "instructions": "q", "criteria": None})
    assert "criteria" not in HopperDirectAdapter().build_request(task)["question"]


def test_choice_returns_the_full_probability_dict_over_the_exact_labels():
    task = choice_task(4)
    res = adapter().run(task)
    assert res.ok and res.status is None and res.error is None
    assert set(res.probs) == set(task.labels)
    assert res.probs_source == "native" and res.adapter == "hopper_direct"
    assert abs(sum(res.probs.values()) - 1.0) < 1e-9
    assert max(res.probs, key=res.probs.get) == "opt0"


def test_noul_is_reported_over_yes_and_no():
    res = adapter().run(NOUL)
    assert res.ok and set(res.probs) == {"yes", "no"}
    assert abs(sum(res.probs.values()) - 1.0) < 1e-9


def test_score_is_reported_over_the_level_indices():
    res = adapter().run(SCORE)
    assert res.ok and set(res.probs) == {"0", "1", "2"}


def test_usage_is_the_count_the_response_reports():
    decider = FakeDecider()
    res = adapter(decider).run(choice_task())
    reply = decider.score(decider.calls[0])["response"]
    assert res.usage == {"input_tokens": reply["usage"]["input_tokens"], "output_tokens": 0}
    assert sorted(res.usage) == ["input_tokens", "output_tokens"]
    assert res.usage["input_tokens"] == res.raw["answer"]["input_tokens"] > 0


def test_model_is_hopper_and_follows_the_reply():
    res = adapter(FakeDecider(name="hopper")).run(choice_task())
    assert res.model == "hopper"


def test_raw_keeps_the_uncalibrated_distribution_and_no_state():
    res = adapter().run(choice_task())
    assert set(res.raw["answer"]["uncalibrated"]) == {"opt0", "opt1", "opt2"}
    assert res.raw["runtime"]["model"] == "hopper"
    assert "state" not in res.request_body and res.request_body["id"] == "c-1"


def test_latency_excludes_nothing_but_is_measured_around_score():
    res = adapter().run(choice_task())
    assert 0.0 <= res.latency_s < 1.0


# --- long menus: the shortlist ------------------------------------------------------------------
@pytest.mark.parametrize("n", [27, 77, 150])
def test_a_long_menu_is_answered_over_every_label(n):
    task, decider = choice_task(n), FakeDecider()
    res = adapter(decider).run(task)
    assert res.ok and res.status is None and res.error is None
    assert set(res.probs) == set(task.labels) and abs(sum(res.probs.values()) - 1.0) < 1e-9
    record = res.raw["runtime"]["shortlist"]
    assert record["strategy"] == "tournament" and record["k"] == 10 and record["options"] == n
    assert record["passes"] == -(-n // 26) + 1 == len(decider.shown)
    dropped = [label for label in task.labels if label not in record["kept"]]
    assert all(res.probs[label] == pytest.approx(0.05 / n) for label in dropped)
    assert "shortlist to 10 of" in res.raw["runtime"]["readout"]
    assert res.raw["runtime"]["probability_origin"].endswith("uniform-residual-over-all-labels")
    assert set(res.raw["answer"]["uncalibrated"]) == set(task.labels)
    assert res.usage["input_tokens"] == res.raw["answer"]["input_tokens"] == 28 * len(decider.shown)  # 28 tokens a pass, every pass billed


def test_exactly_26_options_is_one_pass_with_no_shortlist_record():
    decider = FakeDecider()
    res = adapter(decider).run(choice_task(26))
    assert res.ok and len(decider.shown) == 1 and "shortlist" not in res.raw["runtime"]
    assert res.raw["runtime"]["readout"] == "option-letter softmax, one forward pass"
    off = adapter(FakeDecider(shortlist=None)).run(choice_task(26))
    assert off.probs == res.probs


# --- the 422 rule (jevbench/runner.py: a 422 never counts toward the abort) ----------------------
def test_with_the_shortlist_off_more_than_26_options_is_a_422_not_an_abort():
    res = adapter(FakeDecider(shortlist=None)).run(choice_task(27))
    assert res.ok is False and res.status == 422
    assert "27 options" in res.error and res.probs is None


@pytest.mark.parametrize("labels", [[f"opt{i}" for i in range(29)], [f"opt{i}" for i in range(30)] + ["extra"]])
def test_a_malformed_long_menu_is_still_a_422(labels):
    task = choice_task(30)
    task.labels = labels                   # labels and criteria disagree
    decider = FakeDecider()
    res = adapter(decider).run(task)
    assert res.ok is False and res.status == 422 and "differ" in res.error and decider.shown == []


def test_a_score_question_with_more_than_26_levels_is_still_a_422():
    task = Task(id="s-30", state="s", labels=[str(i) for i in range(30)],
                question={"type": "score", "instructions": "q", "criteria": ["level"] * 30})
    res = adapter().run(task)
    assert res.ok is False and res.status == 422 and "30 options" in res.error


def test_a_question_type_we_do_not_serve_is_a_422():
    task = Task(id="r-1", state="s", labels=["a", "b"],
                question={"type": "rank", "instructions": "q", "criteria": {"a": "x", "b": "y"}})
    res = adapter().run(task)
    assert res.ok is False and res.status == 422 and "rank" in res.error


def test_an_over_context_prompt_is_a_422():
    long = choice_task()
    long.state = "word " * 4000
    res = adapter(FakeDecider(fail=RuntimeError("index out of range"), context=64)).run(long)
    assert res.ok is False and res.status == 422
    assert "RuntimeError" in res.error       # the real error is reported, not the diagnosis


def test_a_cuda_fault_has_no_status_so_three_in_a_row_stop_the_run():
    res = adapter(FakeDecider(fail=RuntimeError("CUDA error: device-side assert"))).run(choice_task())
    assert res.ok is False and res.status is None and "CUDA" in res.error


def test_a_load_failure_has_no_status():
    a = HopperDirectAdapter()
    a.load = lambda: (_ for _ in ()).throw(OSError("no such adapter"))
    res = a.run(choice_task())
    assert res.ok is False and res.status is None and res.error.startswith("load failed: OSError")


def test_the_runner_never_aborts_on_a_run_of_out_of_contract_items():
    """runner.run_all: errors resets to 0 on any result whose status_code is 422."""
    a, errors = adapter(FakeDecider(shortlist=None)), 0
    for _ in range(5):
        r = a.run(choice_task(30))
        errors = errors + 1 if not r.ok and r.status != 422 else 0
    assert errors == 0


# --- loading -------------------------------------------------------------------------------------
def test_load_is_lazy_and_idempotent(monkeypatch):
    built = []

    def fake_decider(**kw):
        built.append(kw)
        return FakeDecider()

    monkeypatch.setitem(sys.modules, "hopper_decisions.model",
                        types.SimpleNamespace(Decider=fake_decider))
    a = HopperDirectAdapter(endpoint="/adapters/x")
    assert built == []                    # nothing is built until a decision or an explicit load()
    a.run(choice_task())
    assert len(built) == 1 and a.load_s >= 0
    a.load()
    a.run(choice_task())
    assert len(built) == 1                # built once, reused


def test_load_builds_the_decider_the_server_builds(monkeypatch):
    from hopper_decisions import MAP
    built = []
    monkeypatch.setitem(sys.modules, "hopper_decisions.model",
                        types.SimpleNamespace(Decider=lambda **kw: built.append(kw) or FakeDecider()))
    HopperDirectAdapter(endpoint="/adapters/x").load()
    assert built[0] == {"adapter": "/adapters/x", "calibration_map": MAP, "name": "hopper",
                        "allow_slow_kernels": False, "cuda_graphs": True,
                        "shortlist": SHORTLIST}  # map shipped, guard, warm-up, graphs and the shortlist on
    built.clear()
    HopperDirectAdapter(shortlist=None).load()
    assert built[0]["shortlist"] is None     # long menus refused, as 1.1.0 did
    built.clear()
    HopperDirectAdapter(revision="deadbeef").load()
    assert built[0]["revision"] == "deadbeef"


def test_warm_load_env_is_honoured_by_exposing_load(monkeypatch):
    """jevbench/cli.py calls adapter.load() when JEVBENCH_WARM_LOAD=1, before the clock starts."""
    monkeypatch.setenv("JEVBENCH_WARM_LOAD", "1")
    built = []
    monkeypatch.setitem(sys.modules, "hopper_decisions.model",
                        types.SimpleNamespace(Decider=lambda **kw: built.append(kw) or FakeDecider()))
    a = HopperDirectAdapter()
    assert hasattr(a, "load")
    if __import__("os").environ.get("JEVBENCH_WARM_LOAD") == "1":
        a.load()
    assert len(built) == 1 and a.load_s is not None
    before = a.load_s
    a.run(choice_task())                  # the decision does not reload
    assert len(built) == 1 and a.load_s == before


def test_load_reports_the_kernel_check_the_server_prints(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "hopper_decisions.model",
                        types.SimpleNamespace(Decider=lambda **kw: FakeDecider()))
    a = HopperDirectAdapter()
    a.load()
    line = capsys.readouterr().out
    assert a.fast_kernels is True
    assert "NVIDIA A10G" in line and "fast-kernel check passed" in line and "16 lengths warmed" in line
    assert a.run(choice_task()).raw["runtime"]["fast_kernels"] is True


def test_a_slow_reference_path_is_said_out_loud(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "hopper_decisions.model",
                        types.SimpleNamespace(Decider=lambda **kw: FakeDecider(slow=["fla missing"])))
    a = HopperDirectAdapter()
    a.load()
    assert a.fast_kernels is False and "do not time this run" in capsys.readouterr().out


def test_the_adapter_has_no_prepare_hook():
    """runner.run_task calls prepare() per task when it exists; ours has no per-item setup."""
    assert not hasattr(HopperDirectAdapter(), "prepare")
