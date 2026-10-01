"""Gemma base loader behind Hopper's existing HTTP response contract."""

from __future__ import annotations

import inspect
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path

from hopper_decisions import fastpath, prompt, request
from hopper_decisions.fresh_readout import FreshReadout


BASE = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
NAME = "gemma-4-12b-it"
LIMIT, K, RESIDUAL, SEED = len(prompt.LETTERS), 10, 0.05, 0


def _adapter_spec(value):
    """Return a PEFT source and optional revision from ``path-or-repo[@revision]``."""
    source = str(value)
    if Path(source).exists() or "@" not in source:
        return source, None
    source, separator, revision = source.rpartition("@")
    if not separator or not source or not revision:
        raise ValueError("adapter must be a directory or repo id, optionally followed by @revision")
    return source, revision


def _merge_adapter(model, adapter):
    """Load a PEFT adapter and merge it once, matching Hopper's established 4B path."""
    from peft import PeftModel
    source, revision = _adapter_spec(adapter)
    kwargs = {"revision": revision} if revision is not None else {}
    return PeftModel.from_pretrained(model, source, **kwargs).merge_and_unload()


def _flat_ids(value):
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("the chat template returned a batch larger than one")
        value = value[0]
    return list(value)


def chat_ids(tokenizer, messages):
    kwargs = {"add_generation_prompt": True}
    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
        if ("enable_thinking" in signature.parameters
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())):
            kwargs["enable_thinking"] = False
    except (TypeError, ValueError):
        kwargs["enable_thinking"] = False
    try:
        return _flat_ids(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError as error:
        if "enable_thinking" not in kwargs or "enable_thinking" not in str(error):
            raise
        kwargs.pop("enable_thinking")
        return _flat_ids(tokenizer.apply_chat_template(messages, **kwargs))


def _groups(count):
    order = list(range(count))
    random.Random(SEED).shuffle(order)
    rounds = -(-count // LIMIT)
    return [order[i::rounds] for i in range(rounds)]


def _subset(example, indices):
    return {**example, "options": [example["options"][i] for i in indices]}


def _spread(names, kept, values):
    share = RESIDUAL / len(names)
    probs = {name: share for name in names}
    for index, value in zip(kept, values):
        probs[names[index]] += (1.0 - RESIDUAL) * value
    winner = request.argmax({names[index]: value for index, value in zip(kept, values)})
    for _ in range(64):
        if request.argmax(request.normalise(probs)) == winner:
            return probs
        probs[winner] = math.nextafter(probs[winner], math.inf)
    raise AssertionError("could not retain the concluding-pass winner")


class FrozenDecider:
    loader = "gemma-4-12b-it"

    def __init__(self, model, tokenizer, readout, *, name=NAME, device=None):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.map = readout if isinstance(readout, FreshReadout) else FreshReadout.read(readout)
        self.name = name
        self.device = device or getattr(model, "device", "cuda")
        self.attention = "sdpa"
        self.letter_ids = prompt.letter_token_ids(tokenizer)
        self.graph_status = self.graph_plan = self.graph_timing = {}
        self.graph_seconds, self.graph_bytes = 0.0, None
        self.kernels, self.slow_kernels, self.warm_seconds = {}, [], (0, 0.0)
        try:
            import torch
            self.gpu = fastpath.gpu_info(torch, self.device)
        except (ImportError, RuntimeError, AssertionError):
            self.gpu = {"name": str(self.device), "capability": None, "cuda": None, "torch": None}

    @classmethod
    def from_pretrained(cls, readout, *, name=NAME, device="cuda", adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, revision=REVISION, dtype=torch.bfloat16, device_map=device,
            attn_implementation="sdpa", low_cpu_mem_usage=True)
        resolved = getattr(model.config, "_commit_hash", None) or REVISION
        if resolved != REVISION:
            raise ValueError(f"model resolved to {resolved}, not pinned revision {REVISION}")
        if adapter is not None:
            model = _merge_adapter(model, adapter)
        return cls(model, tokenizer, readout, name=name, device=device)

    def encode(self, example):
        shown = prompt.option_lines(example)
        return chat_ids(self.tokenizer, prompt.messages(example, example["question"], shown))

    def reference_ids(self, example):
        return self.encode(example)

    def letter_probs(self, ids, count):
        if isinstance(count, bool) or not 1 <= count <= LIMIT:
            raise ValueError(f"letter readout needs 1..{LIMIT} options, not {count!r}")
        import torch
        tensor = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=tensor, use_cache=False, return_dict=True, logits_to_keep=1).logits[
                0, -1, self.letter_ids[:count]].float()
            return torch.softmax(logits, dim=-1).tolist()

    def _encode_checked(self, example):
        ids = self.encode(example)
        limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if limit is not None and len(ids) > limit:
            count = 2 if example["options"] is None else len(example["options"])
            raise ValueError(f"a {count}-option prompt of {len(ids)} tokens is longer than "
                             f"the model's context of {limit}")
        return ids

    def _long(self, example, key, started):
        names, scores, tokens, forward = request.names(example), [0.0] * len(example["options"]), 0, 0.0
        groups = _groups(len(names))
        for group in groups:
            ids = self._encode_checked(_subset(example, group))
            tokens += len(ids)
            before = time.perf_counter()
            values = self.letter_probs(ids, len(group))
            forward += time.perf_counter() - before
            for index, value in zip(group, values):
                scores[index] = value
        kept = sorted(sorted(range(len(names)), key=lambda i: (-scores[i], i))[:K])
        concluding = _subset(example, kept)
        ids = self._encode_checked(concluding)
        tokens += len(ids)
        before = time.perf_counter()
        raw_final = self.letter_probs(ids, len(kept))
        forward += time.perf_counter() - before
        kept_names = [names[i] for i in kept]
        mapped = self.map.apply_shortlist(dict(zip(kept_names, raw_final)))
        probs = _spread(names, kept, [mapped[name] for name in kept_names])
        raw = _spread(names, kept, raw_final)
        reply = request.response(key, "choice", probs, self.name, tokens)
        done = time.perf_counter()
        record = {"strategy": "tournament", "options": len(names), "k": K, "kept": kept_names,
                  "passes": len(groups) + 1, "residual": RESIDUAL,
                  "eliminated_mass": RESIDUAL * (len(names) - K) / len(names)}
        return {"example": example, "raw": raw, "response": reply, "tokens": tokens,
                "shortlist": record, "seconds": (max(0.0, done - started - forward), forward, 0.0)}

    def score(self, req):
        started = time.perf_counter()
        example, key = request.parse(req, large_choice=True)
        if example["kind"] == "choice" and len(example["options"]) > LIMIT:
            return self._long(example, key, started)
        ids = self._encode_checked(example)
        split = time.perf_counter()
        names = request.names(example)
        raw = dict(zip(names, self.letter_probs(ids, len(names))))
        forwarded = time.perf_counter()
        reply = request.response(key, example["kind"], self.map.apply(raw, example), self.name, len(ids))
        done = time.perf_counter()
        return {"example": example, "raw": raw, "response": reply, "tokens": len(ids),
                "seconds": (split - started, forwarded - split, done - forwarded)}

    def decide(self, req):
        return self.score(req)["response"]


__all__ = ["BASE", "REVISION", "NAME", "FrozenDecider", "chat_ids"]
