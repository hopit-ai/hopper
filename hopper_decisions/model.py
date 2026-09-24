"""The serving path: one JevBench decision in, one answer out, one forward pass.

  from hopper_decisions import HF_REPO, Decider
  decider = Decider(adapter=HF_REPO)   # the Hub repo id from __init__.py, or a local directory
  decider.decide({"state": ..., "questions": {"decision": {"type": "choice", ...}}})

Qwen3.5-4B at its pinned revision, the LoRA adapter merged into the weights at load, bf16, the
JSON prompt with thinking off, and a softmax over only the option letters' rows of the output
layer. The prompt and the model are exactly what was evaluated; everything here only changes how
fast the same numbers come out.

By default the forward is replayed from a CUDA graph captured per length bucket at start-up, which
removes the host's kernel-launch time; `cuda_graphs=False` (`hopper-serve --no-cuda-graphs`) keeps
every request on the eager path. Padding to a bucket cannot change an answer (`fastpath.padded`).

A choice question with more than 26 options is answered through a shortlist (`shortlist.py`): a
first stage keeps k options and the same single pass decides among them. Every other request is
the one forward pass above. `shortlist=None` refuses long menus, as 1.1.0 did.
"""

from __future__ import annotations

import copy
import os
import time

import torch

from hopper_decisions import MAP, NAME, calibration, fastpath, pipeline
from hopper_decisions.prompt import SYSTEM, body, chat_ids, letter_token_ids, messages, option_lines
from hopper_decisions.shortlist import DEFAULT as SHORTLIST

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
                 warm_lengths=WARM_LENGTHS, cuda_graphs=True, buckets=fastpath.BUCKETS, shortlist=SHORTLIST):
        if cuda_graphs and prefix_cache:
            raise ValueError("cuda_graphs and prefix_cache are exclusive: the graph replays a full forward")
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
        # Long choice menus (`shortlist.py`): None refuses them, as 1.1.0 did. The embedding first
        # stage loads its own model here, before graph capture measures what memory is left.
        self.shortlist, self.embedder = shortlist, None
        if shortlist is not None and shortlist.strategy == "embedding":
            from hopper_decisions.embedder import Embedder
            self.embedder = Embedder(shortlist.embedder, shortlist.embedder_revision, device=device)
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
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        self.buckets, self.graphs, self.graph_plan, self.graph_status = tuple(sorted(buckets)), {}, {}, {}
        self.graph_timing, self.graph_seconds, self.graph_bytes = {}, 0.0, None
        self.prefix = None  # the warm-up below runs the uncached request path
        cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        if cuda:  # the guard's longest forward sets how much memory graph capture must leave free
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
        filler = self.head_ids * (1 + max((*WARM_TOKENS, *warm_lengths)) // len(self.head_ids))
        # Real forwards on the request path (no cache) prove which linear-attention kernels run on
        # this GPU and compile and autotune their Triton kernels before the first request.
        self.kernels, problems = fastpath.check(model, lambda: [self.letter_probs(filler[:n], 2)
                                                                for n in (len(self.head_ids), *WARM_TOKENS)])
        if problems and (not allow_slow_kernels or self.kernels.pop("_failed", False)):
            raise fastpath.SlowKernels(fastpath.message(problems, self.gpu))
        self.kernels.pop("_failed", None)
        self.slow_kernels = problems
        self.eager_reserve = int(1.25 * (torch.cuda.max_memory_allocated() - resident)) if cuda else 0
        started = time.perf_counter()
        for n in warm_lengths:  # stateless forwards: they cannot change any later answer
            self.letter_probs(filler[:n], 2)
        self.warm_seconds = (len(warm_lengths), time.perf_counter() - started)
        self.prefix = self.prefill(self.head_ids) if prefix_cache else None
        if cuda_graphs and cuda:  # after the guard and the warm-up, so nothing compiles inside a capture
            self.capture()

    @torch.inference_mode()
    def capture(self, reps=3):
        """One CUDA graph per length bucket, so a request costs one replay instead of the thousands
        of small launches the forward is bound by at these lengths. Capture runs after the
        fast-kernel guard on purpose: every Triton kernel is compiled and autotuned by then, and
        autotuning inside a capture cannot work (it times candidates, which synchronises).

        Per bucket, shortest first: one warm-up forward off the default stream (as capture requires),
        eager forwards timed at both ends of the bucket's range, the capture, and timed replays.
        `fastpath.threshold` then decides which request lengths the graph serves; a graph that is
        never faster than eager is dropped. All graphs share one memory pool: a request reads its
        graph's output straight after replaying it and replays never overlap, so one graph reusing
        another's scratch memory is harmless, and the ladder costs about what its largest bucket
        does. A bucket whose activations would not fit while `eager_reserve` bytes stay free for
        longer (eager) requests is skipped, with every larger one. A bucket whose capture raises is
        recorded and left to the eager path."""
        self.graphs, self.graph_plan, self.graph_status, self.graph_timing = {}, {}, {}, {}
        started, pool, stream, previous = time.perf_counter(), torch.cuda.graph_pool_handle(), torch.cuda.Stream(), 0
        rest = "not attempted"
        for bucket in self.buckets:
            lo, previous, n = previous + 1, bucket, reps if bucket <= 1024 else 1  # long forwards time steadily
            ids = torch.full((1, bucket), self.pad_id, dtype=torch.long, device=self.device)
            try:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self.body(input_ids=ids, use_cache=False)  # first use of this shape, untimed
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    eager_hi = forward_ms(lambda: self.body(input_ids=ids, use_cache=False), n)
                    need = torch.cuda.max_memory_allocated() - base
                    eager_lo = forward_ms(lambda: self.body(input_ids=ids[:, :lo], use_cache=False), n)
                torch.cuda.current_stream().wait_stream(stream)
                free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                if not fastpath.fits(need, free, self.eager_reserve):
                    self.graph_status[bucket] = (f"skipped: needs {need / 2**30:.2f} GiB and {free / 2**30:.2f} GiB is "
                                                 f"free, of which {self.eager_reserve / 2**30:.2f} GiB is kept for "
                                                 f"longer requests")
                    rest = "not attempted: a shorter bucket did not fit in memory"
                    break
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    hidden = self.body(input_ids=ids, use_cache=False).last_hidden_state
                replay = forward_ms(graph.replay, n)
                first = fastpath.threshold(lo, bucket, eager_lo, eager_hi, replay)
                self.graph_timing[bucket] = {"from": lo, "eager_from_ms": eager_lo, "eager_ms": eager_hi,
                                             "replay_ms": replay, "need_bytes": need, "first": first}
                if first is None:
                    self.graph_status[bucket] = (f"not used: replay {replay:.1f} ms is not faster than eager "
                                                 f"({eager_lo:.1f} ms at {lo}, {eager_hi:.1f} ms at {bucket} tokens)")
                    del graph, hidden
                    continue
                self.graphs[bucket], self.graph_plan[bucket] = (graph, ids, hidden), first
                self.graph_status[bucket] = "captured"
            except Exception as error:  # noqa: BLE001 - one bucket falling back is not a serving fault
                torch.cuda.synchronize()
                self.graph_status[bucket] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
                if not self.usable():  # a capture that died mid-stream can poison the context
                    self.graphs, self.graph_plan = {}, {}
                    rest = "not attempted: an earlier capture left the CUDA context unusable"
                    break
        self.graph_status.update({b: rest for b in self.buckets if b not in self.graph_status})
        self.graph_seconds = time.perf_counter() - started
        self.graph_bytes = pool_bytes(pool)
        return self.graph_status

    @torch.inference_mode()
    def usable(self):
        """Can this process still run an ordinary forward? Checked after a capture fails."""
        try:
            self.body(input_ids=torch.full((1, 8), self.pad_id, dtype=torch.long, device=self.device),
                      use_cache=False)
            torch.cuda.synchronize()
            return True
        except Exception:  # noqa: BLE001
            return False

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
        captured = self.graphs.get(fastpath.route(len(ids), self.graph_plan)) if self.graphs else None
        if captured is not None:  # right-pad into the graph's static input and replay it
            graph, static_ids, static_hidden = captured
            static_ids[0].copy_(torch.tensor(fastpath.padded(ids, static_ids.shape[1], self.pad_id)))
            graph.replay()
            last = static_hidden[0, len(ids) - 1]
        elif self.prefix is None:
            last = self.body(input_ids=torch.tensor([ids], device=self.device),
                             use_cache=False).last_hidden_state[0, -1]
        else:  # the cache is updated in place, so each request gets its own copy
            suffix = torch.tensor([ids[len(self.head_ids):]], device=self.device)
            last = self.body(input_ids=suffix, past_key_values=copy.deepcopy(self.prefix),
                             use_cache=True).last_hidden_state[0, -1]
        logits = (self.head[:count] @ last).float()
        return torch.log_softmax(logits, -1).exp().tolist()  # as the evaluation computed it

    def score(self, req):
        """Everything one decision produces: the pre-map probabilities, the response, the token
        count, and seconds in tokenisation / forward pass / post-processing (`pipeline.score`)."""
        return pipeline.score(self, req)

    def decide(self, req):
        return self.score(req)["response"]


def forward_ms(fn, reps):
    """Median wall time of `reps` synchronised calls, in ms."""
    values = []
    for _ in range(reps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append(1000 * (time.perf_counter() - started))
    return sorted(values)[len(values) // 2]


def pool_bytes(pool):
    """Device memory held by the graphs' shared pool, or None where this torch does not say."""
    try:
        segments = torch.cuda.memory_snapshot()
    except Exception:  # noqa: BLE001 - a number for the log, never a reason not to serve
        return None
    if not any("segment_pool_id" in s for s in segments):
        return None
    return sum(s["total_size"] for s in segments if tuple(s.get("segment_pool_id") or ()) == tuple(pool))
