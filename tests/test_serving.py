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


def test_the_shipped_map_is_the_submitted_linear_map():
    assert MAP.name == f"{NAME}.json" and MAP.is_file()
    spec = json.loads(MAP.read_text())
    assert set(spec) == {"kind", "bound", "names", "weights", "means", "deviations"}
    assert spec["kind"] == "linear" and spec["weights"][0] == pytest.approx(-0.19086, abs=1e-5)
    assert isinstance(calibration.read(MAP), Linear)


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
