"""The Gemma loader never reads option letters from a forward pass whose length is 1 mod 32.

On some GPUs (seen on an L40S: bf16, SDPA, gemma-4-12B-it) such a forward returns a corrupted last row, the row
the letters are read from. CPU kernels are exact, so these tests inject that fault into the SDPA attention
function of a tiny random-weight Gemma-3 text model (sliding-window layers, final-logit softcapping), with and
without a LoRA adapter (the frozen and trained paths), and check that:

- letter reads at every other length are bit-identical to the 1.2.0 read (same forward, same row);
- reads at 1 mod 32 equal the clean read of the unpadded prompt, although the fault is active;
- no forward the decider runs, on the direct path or the long-menu tournament, has an affected length.
"""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from hopper_decisions import frozen  # noqa: E402
from hopper_decisions.fresh_readout import VARIANT, VERSION, FreshReadout  # noqa: E402
from hopper_decisions.frozen import LIMIT, FrozenDecider, unsafe_length  # noqa: E402

TOLERANCE = 1e-5  # float32 on CPU; a trailing pad changes only reduction order in the rows before it


class CharTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [min(ord(c), 250) + 3 for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=False):
        text = "".join(f"<{m['role']}>\n{m['content']}<end>\n" for m in messages)
        return [2] + self.encode(text + ("<model>\n" if add_generation_prompt else ""))


def tiny_model(adapter):
    torch.manual_seed(1234)
    config = transformers.Gemma3TextConfig(
        vocab_size=256, hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=16_384, initializer_range=0.05, num_hidden_layers=4, head_dim=8,
        sliding_window=16, query_pre_attn_scalar=8,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        final_logit_softcapping=30.0, attn_implementation="sdpa")
    model = transformers.Gemma3ForCausalLM(config).float().eval()
    if adapter:
        peft = pytest.importorskip("peft")
        model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, init_lora_weights=False,
                                                           target_modules=["q_proj", "v_proj", "down_proj"]))
        model.eval()
    return model


def readout():
    body = {"version": VERSION, "variant": VARIANT, "parameters": {"position_priors": {}, "temperature": 1.3}}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    return FreshReadout.from_dict({**body, "seal_sha256": digest})


def decider(adapter):
    return FrozenDecider(tiny_model(adapter), CharTokenizer(), readout(), name="tiny", device="cpu")


def faulty_sdpa(monkeypatch):
    """Swap SDPA for one that zeroes the last query row when the query length is 1 mod 32 (the GPU defect);
    returns the list of query lengths it saw."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    clean, seen = ALL_ATTENTION_FUNCTIONS["sdpa"], []

    def faulty(module, query, key, value, attention_mask, **kwargs):
        out, weights = clean(module, query, key, value, attention_mask, **kwargs)
        seen.append(query.shape[2])
        if query.shape[2] % frozen.QUERY_TILE == 1:
            out = out.clone()
            out[:, -1] = 0.0  # (batch, query, heads, head_dim): the last query row
        return out, weights
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "sdpa", faulty)
    return seen


def read_1_2_0(d, ids, count):
    """FrozenDecider.letter_probs as released in 1.2.0."""
    tensor = torch.tensor([list(ids)], dtype=torch.long, device=d.device)
    with torch.inference_mode():
        logits = d.model(input_ids=tensor, use_cache=False, return_dict=True, logits_to_keep=1).logits[
            0, -1, d.letter_ids[:count]].float()
        return torch.softmax(logits, dim=-1).tolist()


def ids_of(length, seed=7):
    return torch.randint(3, 250, (length,), generator=torch.Generator().manual_seed(seed)).tolist()


def close(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) <= TOLERANCE and len(a) == len(b)


def test_unsafe_lengths():
    assert [n for n in range(1, 130) if unsafe_length(n)] == [1, 33, 65, 97, 129]


@pytest.mark.parametrize("adapter", [False, True])
def test_injected_fault_is_seen_by_the_1_2_0_read(monkeypatch, adapter):
    """Power check: the injected fault moves the 1.2.0 read far beyond the tolerance."""
    d = decider(adapter)
    ids = ids_of(97)
    want = read_1_2_0(d, ids, 10)
    faulty_sdpa(monkeypatch)
    got = read_1_2_0(d, ids, 10)
    assert max(abs(x - y) for x, y in zip(got, want)) > 1e-3


@pytest.mark.parametrize("adapter", [False, True])
def test_letter_reads_avoid_affected_lengths_and_are_otherwise_bit_identical(monkeypatch, adapter):
    d = decider(adapter)
    lengths = range(1, 100)
    clean = {n: read_1_2_0(d, ids_of(n), 26) for n in lengths}  # no fault yet: the true distribution
    seen = faulty_sdpa(monkeypatch)
    for n in lengths:
        ids = ids_of(n)
        seen.clear()
        got = d.letter_probs(ids, 26)
        assert seen and not any(unsafe_length(q) for q in seen)
        if unsafe_length(n):
            assert seen[0] == n + 1  # one trailing pad
            assert close(got, clean[n])
        else:
            assert seen[0] == n
            assert got == clean[n]  # the 1.2.0 forward and row, bit for bit


def long_request(document, extra, count=30):
    """A 30-option menu whose option descriptions all have the same length, so a pass's prompt length depends
    only on how many options it shows, not on which."""
    labels = [f"opt{i:02d}" for i in range(count)]
    return {"state": document, "labels": labels,
            "question": {"type": "choice", "instructions": "Which option fits the document?",
                         "criteria": {label: f"choice number {i:02d}" + "z" * extra for i, label in enumerate(labels)}}}


def pass_lengths(d, req):
    """Prompt lengths of the tournament's group passes and of its concluding pass (any K options)."""
    from hopper_decisions import request
    example, _ = request.parse(req, large_choice=True)
    groups = frozen._groups(len(example["options"]))
    return ([len(d.encode(frozen._subset(example, group))) for group in groups],
            len(d.encode(frozen._subset(example, list(range(frozen.K))))))


def affected_long_request(d):
    """Padding such that both group passes and the concluding pass have lengths of 1 mod 32."""
    for extra in range(32):
        for pad in range(32):
            req = long_request("The record lists thirty options. " + "x" * pad, extra)
            groups, final = pass_lengths(d, req)
            if all(unsafe_length(n) for n in groups) and unsafe_length(final):
                return req, groups, final
    pytest.fail("no padding gave affected group and concluding lengths")


@pytest.mark.parametrize("adapter", [False, True])
def test_long_menu_tournament_never_reads_an_affected_pass(monkeypatch, adapter):
    """Two group passes and a concluding pass, all of affected lengths: with the fault active, the reply equals
    the one computed with no fault, and each pass equals the clean 1.2.0 read of the same prompt."""
    d = decider(adapter)
    req, groups, final = affected_long_request(d)
    assert len(groups) == 2 and 30 > LIMIT
    reference = d.score(req)  # no fault injected
    original, calls, faulted = d.letter_probs, [], {}
    seen = faulty_sdpa(monkeypatch)

    def checked(ids, count):
        calls.append(len(ids))
        faulted[tuple(ids)] = original(ids, count)
        return faulted[tuple(ids)]
    d.letter_probs = checked
    out = d.score(req)
    assert seen and not any(unsafe_length(q) for q in seen)
    assert calls == [*groups, final]
    assert out["shortlist"]["passes"] == 3 and out["shortlist"]["kept"] == reference["shortlist"]["kept"]
    assert out["response"] == reference["response"] and out["raw"] == reference["raw"]
    monkeypatch.undo()  # clean kernels again: the 1.2.0 reads of the same prompts
    for ids, got in faulted.items():
        assert close(got, read_1_2_0(d, ids, len(got)))


@pytest.mark.parametrize("adapter", [False, True])
def test_direct_choice_at_an_affected_length_matches_the_clean_read(monkeypatch, adapter):
    """A short menu (one forward) whose prompt length is 1 mod 32: the pre-map probabilities equal the clean
    1.2.0 read of the same prompt, with the fault active."""
    d = decider(adapter)
    from hopper_decisions import request
    for pad in range(64):
        req = {"state": "Two lamps in the hall. " + "y" * pad, "labels": ["a", "b", "c"],
               "question": {"type": "choice", "instructions": "Which lamp?",
                            "criteria": {"a": "first", "b": "second", "c": "third"}}}
        ids = d.encode(request.parse(req)[0])
        if unsafe_length(len(ids)):
            break
    clean = read_1_2_0(d, ids, 3)
    seen = faulty_sdpa(monkeypatch)
    out = d.score(req)
    assert seen == [len(ids) + 1] * len(seen)
    assert close([out["raw"][k] for k in ("a", "b", "c")], clean)


def test_the_pad_id_comes_from_the_tokenizer():
    class NoPad(CharTokenizer):
        pad_token_id = None
    model = tiny_model(False)
    assert FrozenDecider(model, CharTokenizer(), readout(), device="cpu").pad_id == 0
    assert FrozenDecider(model, NoPad(), readout(), device="cpu").pad_id == 0

    class Pad5(CharTokenizer):
        pad_token_id = 5
    assert FrozenDecider(model, Pad5(), readout(), device="cpu").pad_id == 5
