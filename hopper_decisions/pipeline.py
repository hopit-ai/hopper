"""The request path around the forward pass: parse the request, make the decision, apply the
calibration map, shape the reply. `Decider.score` is `score` below. Imports no ML framework, so the
whole path is tested with a stand-in decider that has no weights.

A request the ordinary path can read -- any noul or score question, and a choice question with at
most 26 options -- takes one forward pass, exactly as in 1.1.0; no setting changes that. A choice
question with more options goes to the shortlist (`shortlist.py`) when one is configured, and is
refused as before when `shortlist=None`.
"""

from __future__ import annotations

import time

from hopper_decisions import calibration, request, shortlist


def score(decider, req):
    """Everything one decision produces: the pre-map probabilities, the response, the token count,
    and seconds in tokenisation / forward pass / post-processing. A shortlisted decision also
    carries a `shortlist` record (strategy, k, the options kept, passes, residual)."""
    t0 = time.perf_counter()
    config = getattr(decider, "shortlist", None)
    example, key = request.parse(req, large_choice=config is not None)
    if shortlist.applies(config, example):
        return shortlist.decide(decider, example, key, config, started=t0)
    ids = decider.encode(example)
    t1 = time.perf_counter()
    labels = request.names(example)
    raw = dict(zip(labels, decider.letter_probs(ids, len(labels))))
    t2 = time.perf_counter()
    reply = request.response(key, example["kind"], calibration.apply(decider.map, example, raw), decider.name, len(ids))
    t3 = time.perf_counter()
    return {"example": example, "raw": raw, "response": reply, "tokens": len(ids),
            "seconds": (t1 - t0, t2 - t1, t3 - t2)}
