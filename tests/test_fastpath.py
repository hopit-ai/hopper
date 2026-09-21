import functools
import sys
import types

from hopper_decisions import fastpath


def fallback_wrapper(torch_function, implementation):
    """The shape of transformers' use_kernel_func_from_hub_with_fallback wrapper."""
    @functools.wraps(torch_function)
    def wrapped(*args, **kwargs):
        return implementation(*args, **kwargs)
    return wrapped


def reference(x):
    return ("reference", x)


def fast(x):
    return ("fast", x)


reference.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"
fast.__module__ = "fla.ops.gated_delta_rule.chunk"


def test_implementation_is_read_from_the_closure():
    assert fastpath.implementation(fallback_wrapper(reference, fast)) is fast
    assert fastpath.implementation(fallback_wrapper(reference, reference)) is reference
    assert fastpath.implementation(fast) is fast


def test_implementation_through_an_outer_wrapper():
    inner = fallback_wrapper(reference, fast)
    outer = functools.wraps(inner)(lambda *a, **k: inner(*a, **k))
    assert fastpath.implementation(outer) is fast


def test_verdict_passes_when_every_op_that_ran_is_fast():
    chosen = {"torch_chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
              "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn",
              "torch_recurrent_gated_delta_rule": "transformers.models.qwen3_5.modeling_qwen3_5.x"}
    report, problems = fastpath.verdict(chosen, {"torch_chunk_gated_delta_rule", "causal_conv1d_fn"}, {})
    assert problems == []
    assert report["torch_chunk_gated_delta_rule"] == {"implementation": chosen["torch_chunk_gated_delta_rule"],
                                                      "fast": True, "ran": True}
    assert report["torch_recurrent_gated_delta_rule"]["fast"] is False  # slow, but never called here


def test_verdict_names_the_missing_package_and_how_to_install_it():
    chosen = {"torch_chunk_gated_delta_rule": "transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule",
              "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn"}
    _, problems = fastpath.verdict(chosen, set(chosen), {"fla": "ModuleNotFoundError: No module named 'fla'"})
    assert len(problems) == 1
    assert "torch_chunk_gated_delta_rule" in problems[0] and "`fla`" in problems[0]
    assert "No module named 'fla'" in problems[0] and "pip install flash-linear-attention==" in problems[0]


def test_verdict_reports_both_packages():
    ref = "transformers.models.qwen3_5.modeling_qwen3_5.f"
    chosen = {op: ref for op in fastpath.OPS}
    _, problems = fastpath.verdict(chosen, {"torch_chunk_gated_delta_rule", "causal_conv1d_fn"},
                                   {"fla": None, "causal_conv1d": "ImportError: undefined symbol"})
    assert [p.split("`")[1] for p in problems] == ["causal_conv1d", "fla"]
    assert "imported, but transformers did not select it" in problems[1]
    assert "causal_conv1d-1.7.0" in problems[0]


def test_verdict_refuses_when_nothing_ran():
    _, problems = fastpath.verdict({"causal_conv1d_fn": "causal_conv1d.x"}, set(), {})
    assert problems and "no linear-attention op" in problems[0]


def test_message_names_gpu_and_override():
    text = fastpath.message(["x is slow"], {"name": "NVIDIA RTX PRO 4500 Blackwell", "capability": "12.0"})
    assert text.startswith("Refusing to start") and "12.0" in text and "RTX PRO 4500" in text
    assert "  - x is slow" in text and fastpath.OVERRIDE in text


def test_check_records_what_a_forward_actually_calls(monkeypatch):
    module = types.ModuleType("transformers.models.fake.modeling_fake")
    module.torch_chunk_gated_delta_rule = fallback_wrapper(reference, fast)
    module.causal_conv1d_fn = fallback_wrapper(reference, reference)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    layer_cls = type("FakeGatedDeltaNet", (), {"__module__": module.__name__})
    model = types.SimpleNamespace(modules=lambda: [object(), layer_cls()])
    originals = (module.torch_chunk_gated_delta_rule, module.causal_conv1d_fn)

    report, problems = fastpath.check(model, lambda: module.torch_chunk_gated_delta_rule(1))
    assert problems == [] and report["torch_chunk_gated_delta_rule"]["ran"]
    assert not report["causal_conv1d_fn"]["ran"]
    assert (module.torch_chunk_gated_delta_rule, module.causal_conv1d_fn) == originals  # restored

    report, problems = fastpath.check(model, lambda: (module.torch_chunk_gated_delta_rule(1), module.causal_conv1d_fn(2)))
    assert len(problems) == 1 and "causal_conv1d_fn" in problems[0]


def test_check_ignores_models_without_linear_attention():
    model = types.SimpleNamespace(modules=lambda: [object()])
    assert fastpath.check(model, lambda: None) == ({}, [])


def test_gpu_info_off_cuda():
    torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda=None),
                                  cuda=types.SimpleNamespace(is_available=lambda: False))
    assert fastpath.gpu_info(torch, "cpu") == {"name": "cpu", "capability": None, "cuda": None}


def test_verdict_refuses_a_forward_that_fails_on_this_gpu():
    error = RuntimeError("Triton >= 3.4.0 and < 3.7.1 on this GPU produces incorrect results\nmore")
    _, problems = fastpath.verdict({"causal_conv1d_fn": "causal_conv1d.x"}, set(), {}, error)
    assert len(problems) == 1 and "failed on this GPU" in problems[0] and "Triton >= 3.4.0" in problems[0]


def test_check_catches_a_failing_forward(monkeypatch):
    module = types.ModuleType("transformers.models.fake2.modeling_fake")
    module.torch_chunk_gated_delta_rule = fallback_wrapper(reference, fast)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    layer_cls = type("FakeGatedDeltaNet", (), {"__module__": module.__name__})
    model = types.SimpleNamespace(modules=lambda: [layer_cls()])

    def boom():
        module.torch_chunk_gated_delta_rule(1)
        raise RuntimeError("no kernel image is available for execution on the device")
    report, problems = fastpath.check(model, boom)
    assert report["_failed"] and "no kernel image" in problems[0]
