from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import sys
import types

import pytest

from hopper_decisions import HF_REPO, MAP, NAME, request, server, shortlist
from hopper_decisions.fresh_readout import FreshReadout, VARIANT, VERSION
from hopper_decisions.frozen import BASE, K, RESIDUAL, REVISION, SEED, FrozenDecider, _adapter_spec, _load_adapter


def seal(parameters, **extra):
    body = {"version": VERSION, "variant": VARIANT, "parameters": parameters, **extra}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    return {**body, "seal_sha256": digest}


def test_the_shipped_readout_is_the_locked_v2_artifact():
    path = MAP.parent / "hopper-12b-readout.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        "ff324cacd5398808cb58387de63047fe7f41ac2cea37bd90804f32efe7acc4f6"
    mapping = FreshReadout.read(path)
    assert mapping.variant == VARIANT
    assert mapping.temperature == 4.28
    assert set(mapping.position_priors) == {"2", "4"}


def test_the_readout_rejects_a_changed_body_with_the_old_self_hash(tmp_path):
    spec = seal({"position_priors": {}, "temperature": 2.0})
    spec["parameters"]["temperature"] = 3.0
    path = tmp_path / "readout.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="self-hash"):
        FreshReadout.read(path)


def test_position_priors_are_applied_before_the_temperature():
    mapping = FreshReadout.from_dict(seal({
        "position_priors": {"2": {"n_k": 40, "ratio": [2.0, 0.5]}},
        "temperature": 2.0,
    }))
    probs = {"left": 0.4, "right": 0.6}
    out = mapping.apply(probs, {"kind": "choice"})
    weights = {"left": math.sqrt(0.4 * 2.0), "right": math.sqrt(0.6 * 0.5)}
    total = sum(weights.values())
    assert out == pytest.approx({name: value / total for name, value in weights.items()})
    assert mapping.apply(probs, {"kind": "score"}) == pytest.approx({
        "left": math.sqrt(0.4) / (math.sqrt(0.4) + math.sqrt(0.6)),
        "right": math.sqrt(0.6) / (math.sqrt(0.4) + math.sqrt(0.6)),
    })


def test_position_priors_only_touch_original_two_and_four_option_choice_menus():
    mapping = FreshReadout.from_dict(seal({
        "position_priors": {
            "2": {"n_k": 40, "ratio": [2.0, 0.5]},
            "4": {"n_k": 40, "ratio": [2.0, 1.0, 1.0, 0.5]},
        },
        "temperature": 1.0,
    }))
    two = {"a": 0.5, "b": 0.5}
    three = {"a": 0.5, "b": 0.3, "c": 0.2}
    four = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    assert mapping.apply(two, {"kind": "choice"}) == pytest.approx({"a": 0.8, "b": 0.2})
    assert mapping.apply(four, {"kind": "choice"}) == pytest.approx(
        {"a": 4 / 9, "b": 2 / 9, "c": 2 / 9, "d": 1 / 9})
    assert mapping.apply(three, {"kind": "choice"}) == pytest.approx(three)
    assert mapping.apply(two, {"kind": "noul"}) == pytest.approx(two)
    assert mapping.apply_shortlist(two) == pytest.approx(two)


def test_frozen_base_loader_dispatches_adapter_but_not_graph_arguments(tmp_path):
    args = server.build_parser().parse_args([
        "--base-loader", "gemma-4-12b-it", "--readout-map", str(tmp_path / "readout.json"),
        "--adapter", "unused", "--cuda-graphs",
    ])
    sentinel, calls = object(), []

    def factory(path, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    assert server.load_decider(args, frozen_factory=factory) is sentinel
    assert calls == [(str(tmp_path / "readout.json"), {
        "name": "gemma-4-12b-it", "adapter": "unused"})]
    assert K == 10 and RESIDUAL == 0.05 and SEED == 0


def test_frozen_base_loader_without_adapter_keeps_the_original_factory_call(tmp_path):
    args = server.build_parser().parse_args([
        "--base-loader", "gemma-4-12b-it", "--readout-map", str(tmp_path / "readout.json"),
    ])
    calls = []

    def factory(path, **kwargs):
        calls.append((path, kwargs))
        return object()

    server.load_decider(args, frozen_factory=factory)
    assert calls == [(str(tmp_path / "readout.json"), {"name": "gemma-4-12b-it"})]


def test_frozen_base_loader_requires_a_readout_map():
    args = server.build_parser().parse_args(["--base-loader", "gemma-4-12b-it"])
    with pytest.raises(ValueError, match="--readout-map is required"):
        server.load_decider(args)


def test_default_dispatch_preserves_the_hopper_constructor_call(monkeypatch):
    import hopper_decisions.model as model

    sentinel, calls = object(), []

    def factory(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(model, "Decider", factory)
    args = server.build_parser().parse_args([])
    assert args.base_loader == "hopper"
    assert server.load_decider(args) is sentinel
    assert calls == [{
        "adapter": HF_REPO,
        "calibration_map": str(MAP),
        "allow_slow_kernels": False,
        "name": NAME,
        "cuda_graphs": False,
        "shortlist": shortlist.DEFAULT,
    }]


class TinyTokenizer:
    def __init__(self):
        self.templates = []

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False and len(text) == 1
        return [ord(text) - ord("A")]

    def apply_chat_template(self, messages, *, add_generation_prompt, enable_thinking):
        self.templates.append((messages, add_generation_prompt, enable_thinking))
        return [30, 31]


class TinyModel:
    device = "cpu"
    config = types.SimpleNamespace(max_position_embeddings=32, _commit_hash=REVISION)

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, *, input_ids, use_cache, return_dict, logits_to_keep=None):
        import torch

        # The decider must request only the last position (long Index prompts OOM otherwise).
        assert logits_to_keep == 1
        self.calls.append((tuple(input_ids.shape), use_cache, return_dict))
        logits = torch.zeros((1, 1, 26), dtype=torch.float32)
        logits[0, -1, :3] = torch.log(torch.tensor([0.7, 0.2, 0.1]))
        return types.SimpleNamespace(logits=logits)


def choice_request():
    return {"state": "document", "labels": ["a", "b", "c"],
            "question": {"type": "choice", "instructions": "choose",
                         "criteria": {"a": "first", "b": "second", "c": "third"}}}


def test_tiny_frozen_model_uses_batch_one_and_returns_pre_map_probabilities():
    mapping = FreshReadout.from_dict(seal({"position_priors": {}, "temperature": 2.0}))
    model, tokenizer = TinyModel(), TinyTokenizer()
    decider = FrozenDecider(model, tokenizer, mapping, device="cpu")
    out = decider.score(choice_request())
    assert out["raw"] == pytest.approx({"a": 0.7, "b": 0.2, "c": 0.1})
    assert request.harness_probs("choice", out["response"]["answers"]["decision"]) == pytest.approx(
        mapping.apply(out["raw"], out["example"]))
    assert model.calls == [((1, 2), False, True)]
    assert tokenizer.templates[0][1:] == (True, False)
    assert decider.graph_status == decider.graph_plan == {}


def test_no_adapter_response_is_the_12b_1_0_0_byte_snapshot():
    mapping = FreshReadout.from_dict(seal({"position_priors": {}, "temperature": 1.0}))
    decider = FrozenDecider(TinyModel(), TinyTokenizer(), mapping, device="cpu")
    encoded = json.dumps(decider.decide(choice_request()), sort_keys=True, separators=(",", ":")).encode()
    assert encoded == (
        b'{"answers":{"decision":{"choice":"a","probabilities":{"a":0.700000008940697,'
        b'"b":0.1999999940395354,"c":0.09999999701976765},"type":"choice"}},'
        b'"model":"gemma-4-12b-it","usage":{"input_tokens":2,"output_tokens":0}}'
    )


def test_run_route_returns_the_pre_map_distribution_without_a_socket():
    mapping = FreshReadout.from_dict(seal({"position_priors": {}, "temperature": 2.0}))
    decider = FrozenDecider(TinyModel(), TinyTokenizer(), mapping, device="cpu")
    payload = json.dumps({"task": choice_request()}).encode()
    handler = object.__new__(server.handler(decider))
    handler.path = "/run"
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile, handler.wfile = io.BytesIO(payload), io.BytesIO()
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda *_: None
    handler.end_headers = lambda: None
    handler.do_POST()
    body = json.loads(handler.wfile.getvalue())
    assert statuses == [200]
    assert body["raw"] == pytest.approx({"a": 0.7, "b": 0.2, "c": 0.1})
    assert body["probs"] != pytest.approx(body["raw"])


def test_pretrained_loader_pins_weights_dtype_attention_and_device(monkeypatch):
    import torch

    model, tokenizer, calls = TinyModel(), TinyTokenizer(), []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("model", name, kwargs))
            return model

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoModelForCausalLM=AutoModel, AutoTokenizer=AutoTokenizer))
    mapping = FreshReadout.from_dict(seal({"position_priors": {}, "temperature": 1.0}))
    loaded = FrozenDecider.from_pretrained(mapping, device="cpu")
    assert isinstance(loaded, FrozenDecider)
    assert calls == [
        ("tokenizer", BASE, {"revision": REVISION}),
        ("model", BASE, {"revision": REVISION, "dtype": torch.bfloat16, "device_map": "cpu",
                         "attn_implementation": "sdpa", "low_cpu_mem_usage": True}),
    ]


def test_adapter_spec_accepts_local_paths_and_pinned_hub_ids(tmp_path):
    local = tmp_path / "adapter@local"
    local.mkdir()
    assert _adapter_spec(local) == (str(local), None)
    assert _adapter_spec("owner/model") == ("owner/model", None)
    assert _adapter_spec("owner/model@abc123") == ("owner/model", "abc123")
    with pytest.raises(ValueError, match="adapter must be"):
        _adapter_spec("@abc123")


def test_tiny_seven_projection_lora_stays_unmerged_and_changes_logits(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")

    torch.manual_seed(7)
    config = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    base_dir = tmp_path / "tiny-base"
    config.save_pretrained(base_dir)
    config._name_or_path = str(base_dir)
    base = transformers.LlamaForCausalLM(config).eval()
    initial = copy.deepcopy(base.state_dict())
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    adapted = peft.get_peft_model(base, peft.LoraConfig(
        task_type="CAUSAL_LM", r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=target_modules,
    ))
    for module in adapted.modules():
        if hasattr(module, "lora_B") and "default" in module.lora_B:
            module.lora_B["default"].weight.data.fill_(0.125)
    adapter_dir = tmp_path / "tiny-adapter"
    adapted.save_pretrained(adapter_dir)

    fresh = transformers.LlamaForCausalLM(config).eval()
    fresh.load_state_dict(initial)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        before = fresh(input_ids=ids).logits
    loaded = _load_adapter(fresh, adapter_dir).eval()
    with torch.inference_mode():
        after = loaded(input_ids=ids, logits_to_keep=1).logits

    saved = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert saved["r"] == 32 and saved["lora_alpha"] == 64
    assert set(saved["target_modules"]) == set(target_modules)
    assert not torch.equal(before[:, -1:], after)
    assert "Peft" in type(loaded).__name__


def test_position_priors_fitted_on_few_menus_are_identity():
    few = FreshReadout.from_dict(seal({
        "position_priors": {"2": {"n_k": 19, "ratio": [2.0, 0.5]}},
        "temperature": 1.0,
    }))
    probs = {"left": 0.4, "right": 0.6}
    assert few.apply(probs, {"kind": "choice"}) == pytest.approx(probs)
