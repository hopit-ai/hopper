"""Latency summaries and the leaderboard's Speed axis, as jevbench/composite_v12.py and
jevbench/metrics.py define them (MIT, github.com/fstandhartinger/jevbench): Speed is the mean of
score(p50) and score(p95), score(s) = clamp(100 - 20 log10(s / 0.1), 0, 100). A system the
maintainer runs on his own rented GPU is endpoint kind "gpu", and its latency is taken as
2 x measured + 0.15 s before scoring; production APIs are scored raw."""

from __future__ import annotations

import math

LOAD_FACTOR, OWN_SERVER_ADD_S = 2.0, 0.15


def percentile(values, q):
    """Linear interpolation between order statistics, the harness's rule."""
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    low, high = math.floor(k), math.ceil(k)
    return ordered[low] if low == high else ordered[low] * (high - k) + ordered[high] * (k - low)


def point(seconds):
    return max(0.0, min(100.0, 100 - 20 * math.log10(seconds / 0.1)))


def adjusted(seconds, kind="gpu"):
    if kind == "api":
        return seconds
    return seconds * LOAD_FACTOR + (OWN_SERVER_ADD_S if kind in ("gpu", "cpu") else 0.0)


def speed(p50, p95, kind="api"):
    """kind="api" scores the raw numbers; "gpu" applies the self-hosted adjustment."""
    return (point(adjusted(p50, kind)) + point(adjusted(p95, kind))) / 2


def summary(seconds):
    return {"n": len(seconds), "p50_ms": 1000 * percentile(seconds, 0.5),
            "p95_ms": 1000 * percentile(seconds, 0.95), "mean_ms": 1000 * sum(seconds) / len(seconds)}
