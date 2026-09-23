"""Refuse to serve Qwen3.5 on the slow reference path of its linear-attention layers, and hold the
pure part of the CUDA-graph fast path (length buckets, padding and which requests use a graph).

transformers 5.17 picks each Gated-DeltaNet op once, when `modeling_qwen3_5` is imported
(`integrations/hub_kernels.py`, `use_kernel_func_from_hub_with_fallback`): it tries to import the
op from `fla` or `causal_conv1d` and, if that fails for any reason (not installed, a wheel built
for another torch or CUDA, a Triton that cannot load), keeps its own PyTorch reference function.
That is correct but more than 10x slower for the chunked delta rule, and it only logs a warning.

The choice is held in each wrapper's closure (`implementation`) and the layers look the op up by
its module-level name on every call. So the check reads the choice from the closures and runs one
short forward with those names wrapped, recording which implementation actually ran on this GPU.
"""

from __future__ import annotations

import importlib
import math
import sys
from contextlib import contextmanager

# op name in the modeling module -> the package transformers takes the fast version from
OPS = {"torch_chunk_gated_delta_rule": "fla", "torch_recurrent_gated_delta_rule": "fla",
       "causal_conv1d_fn": "causal_conv1d", "causal_conv1d_update": "causal_conv1d"}
FLA_VERSION = "0.5.2"
CONV1D_WHEEL = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
                "causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
INSTALL = {"fla": f"pip install flash-linear-attention=={FLA_VERSION}",
           "causal_conv1d": f'pip install "{CONV1D_WHEEL}"  (needs torch 2.8, CUDA 12, CPython 3.12)'}
OVERRIDE = "--allow-slow-kernels"
# A CUDA graph needs one static shape, so a request is right-padded to the next bucket and the graph
# for that bucket is replayed. The graph removes the host's kernel-launch time (~12-20 ms of a ~53 ms
# forward on an A10G), but every padded token is real GPU work, and once a forward is long enough to
# be GPU-bound the launches already overlap the GPU work and there is nothing left to remove: on an
# A10G that happens above ~500 tokens. Where it happens depends on the card, so the ladder covers
# 128-4,096 tokens (64-token steps up to 512, the delta-rule chunk size, coarser above; eager beyond)
# and capture measures, per bucket, the replay against eager forwards at both ends of its range,
# using the graph only for request lengths where it is measurably faster (`threshold`). One memory
# pool is shared by all buckets: 1.3 GiB for the whole ladder on an A10G (README, "CUDA graphs").
BUCKETS = (128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048, 3072, 4096)
MARGIN = 0.02  # a graph must be at least this much faster than eager, or the request stays eager


class SlowKernels(RuntimeError):
    pass


def bucket_for(n, buckets=BUCKETS):
    """The smallest captured length that fits n tokens, or None: such a request runs eager."""
    return next((b for b in sorted(buckets) if b >= n), None)


def threshold(lo, hi, eager_lo_ms, eager_hi_ms, replay_ms, margin=MARGIN):
    """The shortest request, in lo..hi tokens, for which replaying the hi-token graph beats an eager
    forward by `margin`, or None if no length in the range does. Eager time is measured at lo and at
    hi and taken as linear in between; the replay costs the same for every length it serves."""
    target = replay_ms / (1 - margin)
    if eager_lo_ms >= target:
        return lo
    if eager_hi_ms < target or hi <= lo:
        return None
    return min(hi, lo + math.ceil((target - eager_lo_ms) / (eager_hi_ms - eager_lo_ms) * (hi - lo)))


def route(n, plan):
    """The bucket whose graph serves an n-token request, or None for the eager path. `plan` maps each
    bucket in use to the shortest request it serves; anything shorter than that in its range, and
    anything longer than the largest bucket, runs eager."""
    bucket = bucket_for(n, plan)
    return bucket if bucket is not None and n >= plan[bucket] else None


def fits(need, free, reserve):
    """Capture a bucket only if its activations fit while `reserve` bytes stay free for eager
    requests longer than the ladder."""
    return need + reserve <= free


def padded(ids, bucket, pad):
    """Right-pad to the bucket length. The model is causal (softmax attention is masked, the
    Gated-DeltaNet recurrence and the causal conv run forward in time) and we read position
    len(ids)-1, so nothing appended after it can change the answer. Chunk boundaries inside the
    delta-rule kernels are counted from token 0 and therefore do not move either."""
    if not 0 < len(ids) <= bucket:
        raise ValueError(f"{len(ids)} tokens do not fit a bucket of {bucket}")
    return list(ids) + [pad] * (bucket - len(ids))


def implementation(fn):
    """The function a transformers fallback wrapper dispatches to, found in its closure (or in the
    closure of anything it wraps). A plain function is its own implementation."""
    seen, queue = set(), [fn]
    while queue:
        f = queue.pop(0)
        if id(f) in seen:
            continue
        seen.add(id(f))
        code, cells = getattr(f, "__code__", None), getattr(f, "__closure__", None) or ()
        free = dict(zip(code.co_freevars, (c.cell_contents for c in cells))) if code else {}
        if callable(free.get("implementation")):
            return free["implementation"]
        queue += [v for v in free.values() if callable(v)] + [getattr(f, "__wrapped__", None)] * hasattr(f, "__wrapped__")
    return fn


def describe(fn):
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', '?'))}"


def import_error(package):
    """Why a package does not import, or None if it does."""
    try:
        importlib.import_module(package)
        return None
    except Exception as error:  # noqa: BLE001 - whatever stops transformers stops us
        return f"{type(error).__name__}: {str(error).splitlines()[0][:200] if str(error) else ''}"


def verdict(chosen, ran, errors, failure=None):
    """Pure decision. chosen: op -> implementation description; ran: ops a real forward called;
    errors: package -> import error or None; failure: the test forward's exception, if it raised
    (a kernel that refuses this GPU, e.g. fla's Triton-version check on H100, or has no build for it).
    Returns (report, problems)."""
    report = {op: {"implementation": impl, "fast": not impl.startswith("transformers."), "ran": op in ran}
              for op, impl in chosen.items()}
    problems = []
    if failure is not None:
        problems.append(f"the test forward failed on this GPU ({type(failure).__name__}: "
                        f"{str(failure).splitlines()[0][:300] if str(failure) else ''}). If this is a kernel error, the fast "
                        f"kernels cannot run on this GPU with these versions (torch, triton, "
                        f"flash-linear-attention, causal-conv1d)")
    elif not ran:
        problems.append("the test forward called no linear-attention op, so the fast path could not be confirmed")
    for package in sorted({OPS[op] for op, r in report.items() if r["ran"] and not r["fast"]}):
        ops = ", ".join(op for op, r in report.items() if OPS[op] == package and r["ran"] and not r["fast"])
        why = errors.get(package) or "imported, but transformers did not select it"
        problems.append(f"{ops} ran on transformers' slow PyTorch reference path because `{package}` is "
                        f"unusable ({why}). Install it: {INSTALL[package]}")
    return report, problems


def message(problems, gpu):
    lines = [f"Refusing to start: Qwen3.5's linear-attention layers would not use the fast kernels on "
             f"{gpu.get('name', 'this device')} (compute capability {gpu.get('capability', '?')}). Answers "
             f"would be correct, but far slower than the published timings."]
    lines += [f"  - {p}" for p in problems]
    lines.append(f"Pass {OVERRIDE} (allow_slow_kernels=True) only to debug a missing package; never time or "
                 f"submit such a run. It does not override a forward that fails.")
    return "\n".join(lines)


def gpu_info(torch, device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {"name": str(device), "capability": None, "cuda": getattr(torch.version, "cuda", None)}
    index = torch.device(device).index or 0
    major, minor = torch.cuda.get_device_capability(index)
    return {"name": torch.cuda.get_device_name(index), "capability": f"{major}.{minor}",
            "cuda": torch.version.cuda, "torch": torch.__version__}


def linear_modules(model):
    """The modeling modules that define the model's Gated-DeltaNet layers."""
    names = {type(m).__module__ for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")}
    return [sys.modules[n] for n in sorted(names)]


@contextmanager
def recording(modules, ran):
    """Wrap each op's module-level name so a forward records the ops it calls."""
    saved = []
    for module in modules:
        for op in OPS:
            if hasattr(module, op):
                original = getattr(module, op)

                def spy(*args, _op=op, _f=original, **kwargs):
                    ran.add(_op)
                    return _f(*args, **kwargs)
                saved.append((module, op, original))
                setattr(module, op, spy)
    try:
        yield
    finally:
        for module, op, original in saved:
            setattr(module, op, original)


def check(model, forward):
    """(report, problems) for a loaded model; `forward()` runs one short real forward."""
    modules = linear_modules(model)
    if not modules:
        return {}, []  # no linear-attention layers: nothing to guard
    chosen = {op: describe(implementation(getattr(m, op))) for m in modules for op in OPS if hasattr(m, op)}
    ran, failure = set(), None
    with recording(modules, ran):
        try:
            forward()
        except Exception as error:  # noqa: BLE001 - reported as a refusal, never served around
            failure = error
    report, problems = verdict(chosen, ran, {p: import_error(p) for p in set(OPS.values())}, failure)
    report["_failed"] = failure is not None
    return report, problems
