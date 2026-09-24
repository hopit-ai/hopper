import json
import math

import pytest

from hopper_decisions import MAP, NAME, calibration, prompt, request, speed
from hopper_decisions.calibration import Constant, Linear, PerKind, TempMix


def task(kind="choice", state="The customer wants a refund.", **extra):
    question = {"noul": {"type": "noul", "instructions": "Is it a refund?",
                         "criteria": {"true": "asks for money back", "false": "anything else"}},
                "choice": {"type": "choice", "instructions": "Route it.",
                           "criteria": {"refund": "money back", "track": "where is it", "other": "other"}},
                "score": {"type": "score", "instructions": "How urgent?",
                          "criteria": ["not urgent", "somewhat", "very"]}}[kind]
    labels = {"noul": ["no", "yes"], "choice": ["refund", "track", "other"], "score": ["0", "1", "2"]}[kind]
    return {"id": f"t-{kind}", "family": "trap", "state": state, "question": question, "labels": labels,
            "expected": {"noul": "yes", "choice": "refund", "score": 2}[kind], "split": "public",
            "group": None, "provenance": {}, **extra}


def systemone(record):
    return {"state": record["state"], "model": "x", "questions": {"decision": record["question"]}}


@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
@pytest.mark.parametrize("state", ["plain text", {"amount": 12, "note": "é"}])
def test_both_request_shapes_build_the_same_example(kind, state):
    record = task(kind, state)
    (a, key_a), (b, key_b) = request.parse(record), request.parse(systemone(record))
    assert a == b and key_a == key_b == "decision"
    assert a["document"] == (state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False))
    assert a["question"] == record["question"]["instructions"]
    if kind == "noul":
        assert a["options"] is None
        assert a["policy"] == f"{request.POLICY}\ntrue: asks for money back\nfalse: anything else"
    else:
        assert a["policy"] == request.POLICY
        described = record["question"]["criteria"]
        assert a["options"] == [{"name": label, "description": described[i if kind == "score" else label]}
                                for i, label in enumerate(record["labels"])]


def test_the_prompt_is_the_evaluated_json_layout():
    example, _ = request.parse(task("choice", "Order 12 arrived late."))
    shown = prompt.option_lines(example)
    assert shown[0] == ("refund", "refund: money back")
    msgs = prompt.messages(example, example["question"], shown)
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"] == prompt.SYSTEM
    user = json.loads(msgs[1]["content"])
    assert user == {"evidence": "Order 12 arrived late.", "criterion": f"{request.POLICY}\n\nRoute it.",
                    "options": [{"letter": "A", "description": "refund: money back"},
                                {"letter": "B", "description": "track: where is it"},
                                {"letter": "C", "description": "other"}]}


def test_systemone_keeps_the_question_key_and_rejects_two_questions():
    body = systemone(task())
    body["questions"] = {"route": body["questions"]["decision"]}
    assert request.parse(body)[1] == "route"
    body["questions"]["other"] = body["questions"]["route"]
    with pytest.raises(ValueError):
        request.parse(body)


@pytest.mark.parametrize("bad", [{"question": {"type": "rank"}, "state": "s"},
                                 {"state": "s"},
                                 {"question": {"type": "choice", "instructions": "q"}, "state": "s"},
                                 {"question": {"type": "choice", "criteria": {"a": "x"}}, "labels": ["b"], "state": "s"},
                                 {"question": {"type": "noul"}}])
def test_malformed_requests_are_refused(bad):
    with pytest.raises(ValueError):
        request.parse(bad)


def test_labels_order_the_options_when_given():
    record = task()
    record["labels"] = ["other", "refund", "track"]
    assert request.names(request.parse(record)[0]) == ["other", "refund", "track"]
    assert request.names(request.parse(systemone(record))[0]) == ["refund", "track", "other"]


def harness_validate(probs, labels, tolerance=1e-3):
    """jevbench/scoring.validate_probs, restated: exact keys, values in [0, 1], sum within 1e-3."""
    assert set(probs) == set(labels)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in probs.values())
    assert abs(sum(probs.values()) - 1.0) <= tolerance


@pytest.mark.parametrize("kind,probs", [("noul", {"true": 0.7, "false": 0.3}),
                                        ("choice", {"refund": 0.2, "track": 0.5, "other": 0.30001}),
                                        ("score", {"0": 0.1, "1": 0.1, "2": 0.8})])
def test_answers_have_the_typesafe_adapters_shape(kind, probs):
    reply = request.response("decision", kind, probs, "m", 42)
    assert reply["model"] == "m" and reply["usage"] == {"input_tokens": 42, "output_tokens": 0}
    answer = reply["answers"]["decision"]
    assert answer["type"] == kind
    if kind == "noul":
        assert set(answer) == {"type", "noul"} and answer["noul"] == pytest.approx(0.7)
    else:
        harness_validate(answer["probabilities"], list(probs))
    if kind == "choice":
        assert answer["choice"] == "track"
    harness_validate(request.harness_probs(kind, answer), task(kind)["labels"])


def test_choice_ties_break_like_the_harness():
    assert request.answer("choice", {"b": 0.5, "a": 0.5})["choice"] == "a"


def test_normalise_clips_and_sums_to_one():
    out = request.normalise({"a": 1.2, "b": -0.1, "c": 0.3})
    assert out["b"] == 0.0 and math.isclose(sum(out.values()), 1.0)


LINEAR = Linear(("intercept", "log_options", "log_words", "json_state", "is_noul", "is_score", "entropy"),
                (-0.5, -0.1, 0.02, -0.06, -0.4, 0.04, 0.2), (0.0, 1.2, 5.6, 0.4, 0.3, 0.06, 0.6),
                (1.0, 0.37, 0.79, 0.49, 0.47, 0.24, 0.2))


def by_hand(probs, temperature):
    logs = {k: math.log(p) / temperature for k, p in probs.items()}
    total = sum(math.exp(v) for v in logs.values())
    return {k: math.exp(v) / total for k, v in logs.items()}


@pytest.mark.parametrize("mapping", [Constant(0.7), PerKind({"choice": 0.6, "noul": 0.5}), TempMix(0.8, 0.02), LINEAR])
def test_a_map_survives_the_file_and_never_moves_the_answer(mapping, tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(calibration.to_dict(mapping)))
    loaded = calibration.read(path)
    example, _ = request.parse(task("choice", {"a": 1}))
    probs = {"refund": 0.5, "track": 0.3, "other": 0.2}
    served = calibration.apply(loaded, example, probs)
    assert served == pytest.approx(calibration.apply(mapping, example, probs), abs=1e-12)
    assert math.isclose(sum(served.values()), 1.0) and max(served, key=served.get) == "refund"


def test_the_linear_map_is_a_bounded_temperature():
    example, _ = request.parse(task("choice", "one two three"))
    probs = {"refund": 0.5, "track": 0.3, "other": 0.2}
    raw = calibration.features(example, {"probs": probs})
    assert raw["log_options"] == pytest.approx(math.log(3)) and raw["log_words"] == pytest.approx(math.log(4))
    vector = [(raw[n] - m) / d for n, m, d in zip(LINEAR.names, LINEAR.means, LINEAR.deviations)]
    t = math.exp(math.log(3) * math.tanh(sum(w * x for w, x in zip(LINEAR.weights, vector))))
    assert 1 / 3 <= t <= 3
    assert calibration.apply(LINEAR, example, probs) == pytest.approx(by_hand(probs, t), abs=1e-12)


def test_no_map_is_the_identity_and_answerable_maps_are_refused():
    probs = {"true": 0.6, "false": 0.4}
    assert calibration.apply(calibration.read(None), {"kind": "noul", "document": "x"}, probs) == \
        pytest.approx(probs)
    with pytest.raises(ValueError):
        calibration.from_dict({"kind": "linear", "names": ["intercept", "answerable"], "weights": [0, 0],
                               "means": [0, 0], "deviations": [1, 1]})


PER_KIND = {"choice": 0.790, "noul": 0.753, "score": 0.900}   # the 1.1.0 map, to three digits


def test_the_shipped_map_is_the_per_kind_map():
    assert MAP.name == f"{NAME}.json" and MAP.is_file()
    spec = json.loads(MAP.read_text())
    assert set(spec) == {"kind", "temperatures", "fitted_on"}
    assert spec["kind"] == "per_kind" and set(spec["temperatures"]) == {"choice", "noul", "score"}
    for kind, temperature in PER_KIND.items():
        assert spec["temperatures"][kind] == pytest.approx(temperature, abs=5e-4)
    assert isinstance(calibration.read(MAP), PerKind)
    # Provenance: our own fitting set, the 66 leakage-flagged items dropped, and no data file named.
    assert spec["fitted_on"]["dropped_leakage_flagged"] == 66 and spec["fitted_on"]["tier"] == "hard"
    assert ".jsonl" not in json.dumps(spec["fitted_on"])


def test_the_v1_0_linear_map_still_ships_beside_it():
    """1.0.0 stays reproducible from this package: same weights, the map it was measured with."""
    old = MAP.parent / f"{NAME}-v1.0-linear.json"
    assert old.is_file()
    spec = json.loads(old.read_text())
    assert spec["kind"] == "linear" and spec["weights"][0] == pytest.approx(-0.19086, abs=1e-5)
    assert isinstance(calibration.read(old), Linear)


@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
def test_each_answer_type_gets_its_own_temperature(kind):
    """The map reaches the answer type through a real request, and cannot move the answer."""
    mapping = calibration.read(MAP)
    example, _ = request.parse(systemone(task(kind)))
    assert example["kind"] == kind
    probs = {"noul": {"true": 0.62, "false": 0.38},
             "choice": {"refund": 0.5, "track": 0.3, "other": 0.2},
             "score": {"0": 0.2, "1": 0.3, "2": 0.5}}[kind]
    exact = json.loads(MAP.read_text())["temperatures"][kind]
    served = calibration.apply(mapping, example, probs)
    assert served == pytest.approx(by_hand(probs, exact), abs=1e-12)
    assert served == pytest.approx(by_hand(probs, PER_KIND[kind]), abs=1e-3)
    assert max(served, key=served.get) == max(probs, key=probs.get)
    assert math.isclose(sum(served.values()), 1.0)


def test_speed_axis_matches_the_published_rows():
    assert speed.point(0.1) == 100 and speed.point(1.0) == 80 and speed.point(100) == 40 and speed.point(1e6) == 0
    # SemIf's published row: raw p50 0.198 s, p95 0.315 s, self-hosted GPU, Speed 83.7.
    assert speed.speed(0.19796114787459373, 0.3153164997696876, "gpu") == pytest.approx(83.7, abs=0.05)
    # djev, a production API scored raw: 91.4.
    assert speed.speed(0.2370578795671463, 0.30865143015980717, "api") == pytest.approx(91.4, abs=0.05)


def test_percentile_interpolates_like_the_harness():
    assert speed.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert speed.percentile([5], 0.95) == 5
    assert speed.percentile(list(range(101)), 0.95) == 95


def test_the_warmup_lengths_span_64_to_2048_tokens():
    pytest.importorskip("torch")
    from hopper_decisions.model import WARM_LENGTHS
    assert len(WARM_LENGTHS) == 16 and WARM_LENGTHS[0] == 64 and WARM_LENGTHS[-1] == 2048
    assert list(WARM_LENGTHS) == sorted(set(WARM_LENGTHS))


def graphs_decider(status, plan=None, timing=None):
    import types
    return types.SimpleNamespace(
        graph_status=status, graph_timing=timing or {}, graph_seconds=12.3, graph_bytes=0.5 * 2**30,
        graph_plan=plan if plan is not None else {b: 1 for b, s in status.items() if s == "captured"})


def test_cuda_graphs_are_opt_in_so_replies_stay_those_of_1_1_0():
    import inspect

    from hopper_decisions import server
    from hopper_decisions.jevbench_adapter import HopperDirectAdapter
    assert server.build_parser().parse_args([]).cuda_graphs is False
    assert server.build_parser().parse_args(["--cuda-graphs"]).cuda_graphs is True
    with pytest.raises(SystemExit):
        server.build_parser().parse_args(["--no-cuda-graphs"])
    assert inspect.signature(HopperDirectAdapter).parameters["cuda_graphs"].default is False
    # model.py imports torch, which the tests do without: read Decider's default from its source
    import ast
    from pathlib import Path
    tree = ast.parse((Path(server.__file__).parent / "model.py").read_text())
    init, = [f for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Decider"
             for f in c.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"]
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults))
    assert ast.literal_eval(defaults["cuda_graphs"]) is False


def test_the_start_up_log_names_the_buckets_in_use_and_every_fallback():
    from hopper_decisions.server import graph_lines
    status = {128: "captured", 256: "captured", 512: "RuntimeError: CUDA error during capture",
              1024: "not used: replay 166.0 ms is not faster than eager (150.0 ms at 513, 165.2 ms at 1024 tokens)",
              2048: "skipped: needs 1.10 GiB and 1.50 GiB is free, of which 1.00 GiB is kept for longer requests",
              4096: "not attempted: a shorter bucket did not fit in memory"}
    lines = graph_lines(graphs_decider(status))
    assert lines[0] == ("cuda graphs: 2 of 6 buckets in use, 128 to 256 tokens (captured in 12.3 s, graph memory "
                        "0.50 GiB); requests over 256 tokens run eager")
    fallback, = [line for line in lines if "NOT captured" in line]
    assert "512 tokens" in fallback and "CUDA error during capture" in fallback
    assert any(line.startswith("cuda graph 1024 tokens: eager, not used: replay 166.0 ms") for line in lines)
    assert any(line.startswith("cuda graph 2048 tokens: eager, skipped: needs 1.10 GiB") for line in lines)
    assert any(line.startswith("cuda graph 4096 tokens: eager, not attempted") for line in lines)


def test_the_start_up_log_says_where_a_bucket_serves_only_part_of_its_range():
    from hopper_decisions.server import graph_lines
    lines = graph_lines(graphs_decider({448: "captured", 512: "captured"}, {448: 385, 512: 470},
                                       {448: {"from": 385}, 512: {"from": 449}}))
    assert "cuda graph 512 tokens: serves 470-512; 449-469 run eager, where the graph is not faster" in lines
    assert not any(line.startswith("cuda graph 448 tokens") for line in lines)


def test_the_start_up_log_is_silent_about_graphs_that_were_never_asked_for():
    from hopper_decisions.server import graph_lines
    assert graph_lines(graphs_decider({})) == []
    lines = graph_lines(graphs_decider({128: "not used: replay 50.0 ms is not faster than eager"}))
    assert lines[0].startswith("cuda graphs: none in use")


def test_cuda_graphs_and_prefix_cache_cannot_both_be_asked_for():
    pytest.importorskip("torch")
    from hopper_decisions.model import Decider
    with pytest.raises(ValueError):
        Decider(cuda_graphs=True, prefix_cache=True)
