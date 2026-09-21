"""Refuse to serve Qwen3.5 on the slow reference path of its linear-attention layers.

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


class SlowKernels(RuntimeError):
    pass


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
