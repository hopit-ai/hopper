"""JevBench requests in, JevBench answers out. Pure Python, no ML framework, so it is tested without a GPU.

Two request shapes are accepted, the two a maintainer's adapter can send:

  a canonical task record (jevbench/tasks.py), as an in-process adapter gets it:
      {"state": ..., "question": {"type", "instructions", "criteria"}, "labels": [...], ...}
  a `/v1/systemone` body, as the `typesafe` adapter posts it (jevbench/adapters/typesafe.py):
      {"state": ..., "model": ..., "questions": {"decision": {"type", "instructions", "criteria"}}}

The answer is always the `/v1/systemone` response the `typesafe` adapter parses:
  {"model": ..., "usage": {...}, "answers": {"decision": answer}} with, per type,
  noul   {"type": "noul", "noul": P(yes)}
  choice {"type": "choice", "choice": label, "probabilities": {label: p}}
  score  {"type": "score", "probabilities": {"0": p, ...}}

The example built here is exactly the one the adapter was evaluated on.
"""

from __future__ import annotations

import json

from hopper_decisions.prompt import LETTERS

KINDS = ("noul", "choice", "score")
POLICY = "Decide the case using only what the document states. Exactly one option is correct."


def document(state):
    """Structured states are rendered as JSON, which is what the harness sends."""
    return state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)


def options_for(question, labels):
    """None for noul; otherwise one option per label, in the record's own order."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        return None
    if kind == "score":  # levels are described positionally, and `labels` is ["0", "1", ...]
        return [{"name": label, "description": criteria[i]} for i, label in enumerate(labels)]
    return [{"name": label, "description": criteria[label]} for label in labels]


def parse(request, large_choice=False):
    """(example, question key). Raises ValueError on a request we cannot answer.

    More than 26 options is one: the readout has one letter per option. `large_choice=True` lets a
    choice question past that limit through, for the shortlist (`shortlist.py`) to cut down before
    the readout; every other check is the same, and a score question is still held to 26 levels.

    The shapes are checked before anything is read from them: the request and its question are
    objects, choice criteria an object and score criteria a list, and labels a list of distinct
    strings. A repeated label would collapse two options into one probability key."""
    if not isinstance(request, dict):
        raise ValueError(f"a request is a JSON object, not {type(request).__name__}")
    if "questions" in request:
        questions = request["questions"]
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError("exactly one question per request")
        (key, question), = questions.items()
        labels = None
    elif "question" in request:
        key, question, labels = "decision", request["question"], request.get("labels")
    else:
        raise ValueError("request has neither 'questions' nor 'question'")
    if not isinstance(question, dict):
        raise ValueError(f"a question is a JSON object, not {type(question).__name__}")
    kind, criteria = question.get("type"), question.get("criteria")
    if kind not in KINDS:
        raise ValueError(f"unsupported question type {kind!r}")
    if "state" not in request:
        raise ValueError("request has no state")
    policy, options = POLICY, None
    if kind == "noul":  # the rubric has no option to sit on, so it goes in the policy
        criteria = criteria or {}
        if not isinstance(criteria, dict):
            raise ValueError("noul criteria are an object with 'true' and 'false'")
        policy = f"{POLICY}\ntrue: {criteria.get('true', 'yes')}\nfalse: {criteria.get('false', 'no')}"
    else:
        if not criteria:
            raise ValueError(f"a {kind} question needs criteria")
        if kind == "choice" and not isinstance(criteria, dict):
            raise ValueError("choice criteria are an object of label: description")
        if kind == "score" and not isinstance(criteria, list):
            raise ValueError("score criteria are a list of level descriptions")
        if labels is None:  # a /v1/systemone body carries no labels: they are the criteria's keys or levels
            labels = list(criteria) if kind == "choice" else [str(i) for i in range(len(criteria))]
        if not isinstance(labels, (list, tuple)) or not all(isinstance(label, str) for label in labels):
            raise ValueError("labels are a list of strings")
        if len(set(labels)) != len(labels):
            raise ValueError(f"labels repeat: {len(labels)} labels, {len(set(labels))} distinct")
        if kind == "choice" and set(labels) != set(criteria):
            raise ValueError("choice labels and criteria keys differ")
        if kind == "score" and len(labels) > len(criteria):
            raise ValueError(f"{len(labels)} score labels for {len(criteria)} levels")
        if len(labels) > len(LETTERS) and not (large_choice and kind == "choice"):
            raise ValueError(f"{len(labels)} options, more than {len(LETTERS)} letters")
        options = options_for(question, list(labels))
    return {"kind": kind, "policy": policy, "document": document(request["state"]),
            "question": question.get("instructions", ""), "options": options}, key


def names(example):
    """Our option names, in prompt order: true/false for noul, the exact labels otherwise."""
    return ["true", "false"] if example["kind"] == "noul" else [o["name"] for o in example["options"]]


def normalise(probs):
    """Floats in [0, 1] summing to 1 to rounding, well inside the harness's strict 1e-3."""
    clipped = {k: min(max(float(v), 0.0), 1.0) for k, v in probs.items()}
    total = sum(clipped.values())
    return {k: v / total for k, v in clipped.items()}


def argmax(probs):
    """The harness's own tie-break (jevbench/scoring.py): the lexicographically smallest label."""
    return max(sorted(probs), key=lambda k: probs[k])


def answer(kind, probs):
    probs = normalise(probs)
    if kind == "noul":
        return {"type": "noul", "noul": probs["true"]}
    if kind == "choice":
        return {"type": "choice", "choice": argmax(probs), "probabilities": probs}
    return {"type": "score", "probabilities": probs}


def response(key, kind, probs, model, input_tokens):
    return {"model": model, "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "answers": {key: answer(kind, probs)}}


def harness_probs(kind, reply):
    """What the harness's `typesafe` adapter makes of an answer: probabilities over the exact labels."""
    if kind == "noul":
        return {"yes": reply["noul"], "no": 1.0 - reply["noul"]}
    return reply["probabilities"]
