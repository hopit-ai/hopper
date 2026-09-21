"""The calibration map, a small JSON file, and the arithmetic that applies it.

Every map divides log-probabilities by a positive temperature (and optionally mixes in the uniform
distribution), so no map can move an answer. The shipped map is `linear`: log T is a bounded
linear function of what the request shows (log option count, log(1 + state length in words),
JSON state or not, answer type, and the normalised entropy of the model's own distribution),
squashed through BOUND * tanh so T stays in [1/3, 3].
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

BOUND = math.log(3.0)  # |log T| <= this, so T is in [1/3, 3]


def rescale(probs, temperature):
    logs = {k: math.log(max(p, 1e-12)) / temperature for k, p in probs.items()}
    top = max(logs.values())
    total = sum(math.exp(v - top) for v in logs.values())
    return {k: math.exp(v - top) / total for k, v in logs.items()}


def entropy(probs):
    """Shannon entropy in nats, divided by log(k) so it means the same for 2 and 6 options."""
    values = [p for p in probs.values() if p > 0]
    total = -sum(p * math.log(p) for p in values)
    return total / math.log(len(probs)) if len(probs) > 1 else 0.0


def features(example, prediction):
    """The observable features of one item, before standardisation; nothing here needs a label."""
    probs = prediction["probs"]
    return {"intercept": 1.0,
            "log_options": math.log(max(len(probs), 2)),
            "log_words": math.log(1 + len(example["document"].split())),
            "json_state": float(example["document"].lstrip().startswith("{")),
            "is_noul": float(example["kind"] == "noul"),
            "is_score": float(example["kind"] == "score"),
            "entropy": entropy(probs)}


def mix_uniform(probs, weight):
    """(1 - w) p + w uniform. Adds the same mass to every label, so the argmax is untouched."""
    share = weight / len(probs)
    return {k: (1 - weight) * p + share for k, p in probs.items()}


def temperature(weights, vector):
    return math.exp(BOUND * math.tanh(sum(w * x for w, x in zip(weights, vector))))


@dataclass(frozen=True)
class Constant:
    temperature: float = 1.0

    def apply(self, probs, row):
        return rescale(probs, self.temperature)


@dataclass(frozen=True)
class PerKind:
    temperatures: dict

    def apply(self, probs, row):
        return rescale(probs, self.temperatures.get(row["kind"], 1.0))


@dataclass(frozen=True)
class TempMix:
    temperature: float
    weight: float

    def apply(self, probs, row):
        return mix_uniform(rescale(probs, self.temperature), self.weight)


@dataclass(frozen=True)
class Linear:
    """log T = BOUND * tanh(w . x), x standardised by the fitting set's own means and deviations."""

    names: tuple
    weights: tuple
    means: tuple
    deviations: tuple

    def vector(self, example, prediction):
        raw = features(example, prediction)
        return tuple((raw[n] - m) / d for n, m, d in zip(self.names, self.means, self.deviations))

    def temperature_for(self, example, prediction):
        return temperature(self.weights, self.vector(example, prediction))

    def apply(self, probs, row):
        return rescale(probs, self.temperature_for(row["example"], row["prediction"]))


def to_dict(mapping):
    if isinstance(mapping, Constant):
        return {"kind": "global", "temperature": mapping.temperature}
    if isinstance(mapping, PerKind):
        return {"kind": "per_kind", "temperatures": dict(mapping.temperatures)}
    if isinstance(mapping, TempMix):
        return {"kind": "temp_mix", "temperature": mapping.temperature, "weight": mapping.weight}
    if isinstance(mapping, Linear):
        return {"kind": "linear", "bound": BOUND, "names": list(mapping.names), "weights": list(mapping.weights),
                "means": list(mapping.means), "deviations": list(mapping.deviations)}
    raise ValueError(f"cannot export {type(mapping).__name__}")


def from_dict(spec):
    kind = (spec or {"kind": "none"})["kind"]
    if kind == "none":
        return Constant(1.0)
    if kind == "global":
        return Constant(spec["temperature"])
    if kind == "per_kind":
        return PerKind(dict(spec["temperatures"]))
    if kind == "temp_mix":
        return TempMix(spec["temperature"], spec["weight"])
    if kind == "linear":
        if "answerable" in spec["names"]:
            raise ValueError("serving asks no answerability question, so the map cannot use it")
        if abs(spec.get("bound", BOUND) - BOUND) > 1e-12:
            raise ValueError("the map was fitted with a different temperature bound")
        return Linear(tuple(spec["names"]), tuple(spec["weights"]), tuple(spec["means"]),
                      tuple(spec["deviations"]))
    raise ValueError(f"unknown map kind {kind!r}")


def read(path):
    return from_dict(json.loads(Path(path).read_text())) if path else Constant(1.0)


def apply(mapping, example, probs):
    """The recalibrated distribution. `example` needs only `kind` and `document`, which a request carries."""
    return mapping.apply(probs, {"kind": example["kind"], "example": example, "prediction": {"probs": probs}})
