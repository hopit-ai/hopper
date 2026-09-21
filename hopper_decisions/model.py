"""The serving path: one JevBench decision in, one answer out, one forward pass.

  from hopper_decisions import HF_REPO, Decider
  decider = Decider(adapter=HF_REPO)   # the Hub repo id from __init__.py, or a local directory
  decider.decide({"state": ..., "questions": {"decision": {"type": "choice", ...}}})

Qwen3.5-4B at its pinned revision, the LoRA adapter merged into the weights at load, bf16, the
JSON prompt with thinking off, and a softmax over only the option letters' rows of the output
layer. The prompt and the model are exactly what was evaluated; everything here only changes how
fast the same numbers come out.
"""

from __future__ import annotations

import copy
import os
import time

import torch

from hopper_decisions import MAP, NAME, calibration, fastpath, request
from hopper_decisions.prompt import SYSTEM, body, chat_ids, letter_token_ids, messages, option_lines

BASE, REVISION = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# fla autotunes its q/k l2norm again whenever NB = ceil(tokens x 32 heads / 65536) changes, i.e. at
# every 2048 tokens, and that costs ~10 s once (measured: 9.5 s on the first 2k+ token request on an
# A10G). Warming 1..3 of those bands at start-up covers requests up to 6144 tokens.
WARM_TOKENS = (1948, 3996, 6044)
# On an H100 the first request at each new length was slower than later ones with no Triton work
# at all (per-shape first-use costs outside Triton), so also warm a geometric spread of lengths.
WARM_LENGTHS = tuple(round(64 * 32 ** (i / 15)) for i in range(16))  # 64 .. 2048 tokens
SENTINEL = "⁣USER⁣"  # invisible, never in a request, and no template rule touches it


class Decider:
    def __init__(self, adapter=None, calibration_map=MAP, base=BASE, revision=REVISION, name=NAME,
                 device="cuda", attention=None, prefix_cache=False, allow_slow_kernels=False,
                 warm_lengths=WARM_LENGTHS):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if attention is None:
            try:
                import flash_attn  # noqa: F401
                attention = "flash_attention_2"
            except ImportError:
                attention = "sdpa"  # with no mask and one sequence, SDPA takes its own flash kernel
        self.name, self.device, self.attention = name, device, attention
        self.tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16,
                                                     device_map=device, attn_implementation=attention)
        if adapter:  # merged once here, so no request pays for the LoRA matmuls
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval()
        self.body = model.model
        self.head = model.get_output_embeddings().weight[letter_token_ids(self.tokenizer)].detach().clone()
        self.map = calibration_map if not isinstance(calibration_map, (str, os.PathLike, type(None))) \
            else calibration.read(calibration_map)
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        self.encode_text = (lambda s: backend.encode(s, add_special_tokens=False).ids) if backend else \
            (lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"])
        # The chat template around the user turn never changes: render and tokenise it once. The
        # user turn starts with '{' after a newline and the tail starts with a special token, so
        # both joins are pre-tokeniser boundaries; `reference_ids` checks that on real requests.
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": SENTINEL}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        head, self.tail = rendered.split(SENTINEL)
        self.head_ids = self.encode_text(head)
        self.gpu = fastpath.gpu_info(torch, device)
        self.prefix = None  # the warm-up below runs the uncached request path
        filler = self.head_ids * (1 + max((*WARM_TOKENS, *warm_lengths)) // len(self.head_ids))
        # Real forwards on the request path (no cache) prove which linear-attention kernels run on
        # this GPU and compile and autotune their Triton kernels before the first request.
        self.kernels, problems = fastpath.check(model, lambda: [self.letter_probs(filler[:n], 2)
                                                                for n in (len(self.head_ids), *WARM_TOKENS)])
        if problems and (not allow_slow_kernels or self.kernels.pop("_failed", False)):
            raise fastpath.SlowKernels(fastpath.message(problems, self.gpu))
        self.kernels.pop("_failed", None)
        self.slow_kernels = problems
        started = time.perf_counter()
        for n in warm_lengths:  # stateless forwards: they cannot change any later answer
            self.letter_probs(filler[:n], 2)
        self.warm_seconds = (len(warm_lengths), time.perf_counter() - started)
        self.prefix = self.prefill(self.head_ids) if prefix_cache else None

    @torch.inference_mode()
    def prefill(self, ids):
        """The model state after the fixed system-prompt tokens: attention KV for the full-attention
        layers, conv and recurrent state for the linear ones. Every request continues from a copy."""
        return self.body(input_ids=torch.tensor([ids], device=self.device), use_cache=True).past_key_values

    def encode(self, example):
        return self.head_ids + self.encode_text(body(example, example["question"], option_lines(example))
                                                + self.tail)

    def reference_ids(self, example):
        """The same prompt tokenised whole through the chat template."""
        return chat_ids(self.tokenizer, messages(example, example["question"], option_lines(example)))

    @torch.inference_mode()
    def letter_probs(self, ids, count):
        if self.prefix is None:
            hidden = self.body(input_ids=torch.tensor([ids], device=self.device), use_cache=False).last_hidden_state
        else:  # the cache is updated in place, so each request gets its own copy
            suffix = torch.tensor([ids[len(self.head_ids):]], device=self.device)
            hidden = self.body(input_ids=suffix, past_key_values=copy.deepcopy(self.prefix),
                               use_cache=True).last_hidden_state
        logits = (self.head[:count] @ hidden[0, -1]).float()
        return torch.log_softmax(logits, -1).exp().tolist()  # as the evaluation computed it

    def score(self, req):
        """Everything one decision produces: the pre-map probabilities, the response, the token
        count, and seconds in tokenisation / forward pass / post-processing."""
        t0 = time.perf_counter()
        example, key = request.parse(req)
        ids = self.encode(example)
        t1 = time.perf_counter()
        labels = request.names(example)
        raw = dict(zip(labels, self.letter_probs(ids, len(labels))))
        t2 = time.perf_counter()
        reply = request.response(key, example["kind"], calibration.apply(self.map, example, raw), self.name, len(ids))
        t3 = time.perf_counter()
        return {"example": example, "raw": raw, "response": reply, "tokens": len(ids),
                "seconds": (t1 - t0, t2 - t1, t3 - t2)}

    def decide(self, req):
        return self.score(req)["response"]
