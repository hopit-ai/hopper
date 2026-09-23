import functools
import math
import sys
import types

import pytest

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


@pytest.mark.parametrize("n,bucket", [(1, 128), (128, 128), (129, 160), (157, 160), (194, 224),
                                      (224, 224), (225, 256), (236, 256)])
def test_bucket_for_takes_the_smallest_length_that_fits(n, bucket):
    assert fastpath.bucket_for(n) == bucket


def test_nothing_fits_above_the_largest_bucket():
    assert fastpath.bucket_for(4097) is None  # past the ladder: eager
    assert fastpath.bucket_for(200, (512,)) == 512 and fastpath.bucket_for(600, (512,)) is None


def test_bucket_for_does_not_depend_on_the_order_given():
    assert fastpath.bucket_for(200, (1024, 128, 256)) == 256


def test_padded_right_pads_and_leaves_the_prompt_where_it_was():
    ids = [7, 8, 9]
    assert fastpath.padded(ids, 8, 0) == [7, 8, 9, 0, 0, 0, 0, 0]
    assert fastpath.padded(ids, 8, 0)[len(ids) - 1] == ids[-1]  # the position we read
    assert fastpath.padded(ids, 3, 0) == ids and ids == [7, 8, 9]  # no padding, input untouched


@pytest.mark.parametrize("ids,bucket", [([], 8), ([1, 2, 3], 2)])
def test_padded_refuses_what_does_not_fit(ids, bucket):
    with pytest.raises(ValueError):
        fastpath.padded(ids, bucket, 0)


def test_the_ladder_reaches_4096_in_64_token_steps_where_graphs_pay():
    assert fastpath.BUCKETS == tuple(sorted(set(fastpath.BUCKETS)))
    assert fastpath.BUCKETS[0] == 128 and fastpath.BUCKETS[-1] == 4096
    short = [b for b in fastpath.BUCKETS if b <= 512]
    assert all(b - a <= 64 for a, b in zip(short, short[1:]))
    assert fastpath.bucket_for(4096) == 4096 and fastpath.bucket_for(4097) is None


@pytest.mark.parametrize("n,bucket", [(236, 256), (257, 320), (660, 768), (1500, 1536), (3708, 4096)])
def test_judge_length_requests_have_a_bucket(n, bucket):
    assert fastpath.bucket_for(n) == bucket


def test_threshold_serves_the_whole_range_when_even_its_shortest_request_is_slower_eager():
    # A10G, launch-bound: eager ~53 ms at every length here, replay 44 ms at 256
    assert fastpath.threshold(225, 256, 53.0, 53.6, 44.3) == 225


def test_threshold_never_serves_when_the_graph_is_not_faster_even_at_its_own_length():
    # GPU-bound: the replay pays for the padded length and eager at the bucket length is no slower
    assert fastpath.threshold(897, 1024, 150.0, 165.2, 166.0) is None
    assert fastpath.threshold(897, 1024, 150.0, 166.0, 166.0) is None  # a tie is not a win
    assert fastpath.threshold(1024, 1024, 170.0, 170.0, 166.0) == 1024  # one-length range that wins


def test_threshold_cuts_the_range_where_eager_overtakes_the_replay():
    first = fastpath.threshold(449, 512, 80.0, 96.0, 85.0, margin=0.0)
    assert first == 449 + math.ceil((85.0 - 80.0) / 16.0 * 63)
    assert 449 < first < 512
    # with the default margin the graph must win by 2 %, so it serves fewer lengths
    assert fastpath.threshold(449, 512, 80.0, 96.0, 85.0) > first


def test_route_uses_a_graph_only_where_its_plan_says_it_is_faster():
    plan = {128: 1, 256: 129, 512: 470}
    assert fastpath.route(1, plan) == 128 and fastpath.route(128, plan) == 128
    assert fastpath.route(129, plan) == 256
    assert fastpath.route(469, plan) is None  # in 512's range, but eager is faster there
    assert fastpath.route(470, plan) == 512 and fastpath.route(512, plan) == 512
    assert fastpath.route(513, plan) is None  # beyond the ladder: eager
    assert fastpath.route(100, {}) is None  # graphs off, or nothing captured


def test_route_never_pads_past_a_bucket_that_was_dropped():
    # 320 failed or never paid: a 300-token request must not be padded into 384, whose threshold
    # was measured from 321 tokens up
    plan = {256: 1, 384: 321}
    assert fastpath.route(300, plan) is None and fastpath.route(321, plan) == 384


def test_fits_keeps_memory_for_longer_eager_requests():
    gib = 2**30
    assert fastpath.fits(1 * gib, 3 * gib, 2 * gib)
    assert not fastpath.fits(1 * gib + 1, 3 * gib, 2 * gib)
