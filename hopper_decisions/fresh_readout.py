"""Load and apply a sealed ``readout-bias/readout-seal-v2`` map."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


VERSION = "readout-bias/readout-seal-v2"
VARIANT = "sealed-d+k2"


def _softmax_logs(probs, names, temperature=1.0, offsets=None):
    offsets = offsets or [0.0] * len(names)
    logits = [math.log(max(float(probs[name]), 1e-12)) / temperature + offset
              for name, offset in zip(names, offsets)]
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return {name: value / total for name, value in zip(names, weights)}


class FreshReadout:
    def __init__(self, variant, parameters):
        if variant != VARIANT or not isinstance(parameters, dict):
            raise ValueError("unknown or malformed readout map")
        self.variant = variant
        try:
            self.temperature = float(parameters["temperature"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed readout temperature") from error
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("readout temperature must be positive and finite")
        source, checked = parameters.get("position_priors"), {}
        if not isinstance(source, dict):
            raise ValueError("readout position_priors must be an object")
        for size, fitted in source.items():
            try:
                count, n_k = int(size), int(fitted["n_k"])
                ratios = tuple(float(value) for value in fitted["ratio"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"malformed position prior {size!r}") from error
            if (str(count) != str(size) or not 1 <= count <= 26 or len(ratios) != count or n_k < 0
                    or any(not math.isfinite(value) or value <= 0 for value in ratios)):
                raise ValueError(f"malformed position prior {size!r}")
            checked[str(count)] = {"n_k": n_k, "ratio": ratios}
        self.position_priors = checked

    @classmethod
    def from_dict(cls, seal):
        if not isinstance(seal, dict) or seal.get("version") != VERSION:
            raise ValueError(f"readout map must be a {VERSION} object")
        body = {key: value for key, value in seal.items() if key != "seal_sha256"}
        try:
            encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()
        except (TypeError, ValueError) as error:
            raise ValueError("malformed readout map") from error
        digest = hashlib.sha256(encoded).hexdigest()
        if seal.get("seal_sha256") != digest:
            raise ValueError("readout seal self-hash differs")
        return cls(seal.get("variant"), seal.get("parameters"))

    @classmethod
    def read(cls, path):
        if path is None:
            raise ValueError("the Gemma base loader requires a serialized readout seal")
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"missing or malformed readout seal: {path}") from error

    def apply(self, probs, example):
        names, corrected = list(probs), dict(probs)
        if example.get("kind") == "choice":
            fitted = self.position_priors.get(str(len(names)))
            if fitted:
                corrected = _softmax_logs(corrected, names,
                                          offsets=[math.log(value) for value in fitted["ratio"]])
        return _softmax_logs(corrected, names, self.temperature)

    def apply_shortlist(self, probs):
        return _softmax_logs(probs, list(probs), self.temperature)


__all__ = ["FreshReadout", "VERSION", "VARIANT"]
