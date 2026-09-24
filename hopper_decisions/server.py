"""A one-request-at-a-time HTTP server around `Decider`, for either of the harness's remote routes.

  hopper-serve --port 8080            # same as: python -m hopper_decisions.server --port 8080

  POST /v1/systemone  the `typesafe` adapter's wire format (jevbench/adapters/typesafe.py)
  POST /run           the `remote_inproc` client's format (jevbench/adapters/remote_inproc.py):
                      {"task": <record>} -> {"ok", "probs" over the exact labels, "latency_s", ...}

Single-threaded on purpose: the harness runs serially, and one GPU should see one request at a time.

A choice question with more than 26 options is answered through a shortlist (`--shortlist`, see
`shortlist.py` and the README, "Long menus"); `--shortlist off` refuses it with a 400, as 1.1.0 did.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from hopper_decisions import HF_REPO, MAP, NAME, request, shortlist


def handler(decider):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            started = time.perf_counter()
            status, body = 200, None
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/v1/systemone":
                    body = decider.decide(payload)
                elif self.path == "/run":
                    task = payload["task"]
                    reply = decider.decide(task)
                    (key, answer), = reply["answers"].items()
                    body = {"ok": True, "probs": request.harness_probs(task["question"]["type"], answer),
                            "model": reply["model"], "usage": reply["usage"], "error": None,
                            "latency_s": time.perf_counter() - started, "raw": None}
                else:
                    status, body = 404, {"error": f"no route {self.path}"}
            except (ValueError, KeyError, TypeError) as error:  # a malformed request, not a server fault
                status, body = 400, {"ok": False, "error": f"{type(error).__name__}: {error}"}
            out = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    return Handler


def graph_lines(decider):
    """The start-up log's CUDA-graph lines: which request lengths replay a graph, and for every
    bucket that does not serve, why (slower than eager here, out of memory, or a failed capture)."""
    status, plan = decider.graph_status, decider.graph_plan
    if not status:
        return []
    size = f", graph memory {decider.graph_bytes / 2**30:.2f} GiB" if decider.graph_bytes is not None else ""
    if plan:
        lines = [f"cuda graphs: {len(plan)} of {len(status)} buckets in use, {min(plan)} to {max(plan)} tokens "
                 f"(captured in {decider.graph_seconds:.1f} s{size}); requests over {max(plan)} tokens run eager"]
    else:
        lines = [f"cuda graphs: none in use (tried in {decider.graph_seconds:.1f} s); every request runs eager"]
    for bucket, state in status.items():
        timing = decider.graph_timing.get(bucket, {})
        if bucket in plan and plan[bucket] > timing.get("from", plan[bucket]):
            lines.append(f"cuda graph {bucket} tokens: serves {plan[bucket]}-{bucket}; {timing['from']}-"
                         f"{plan[bucket] - 1} run eager, where the graph is not faster")
        elif state == "captured":
            continue
        elif state.startswith(("not used", "skipped", "not attempted")):
            lines.append(f"cuda graph {bucket} tokens: eager, {state}")
        else:
            lines.append(f"cuda graph {bucket} tokens: NOT captured, eager fallback ({state})")
    return lines


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default=HF_REPO, help=f"LoRA adapter directory or Hugging Face repo id (default {HF_REPO})")
    parser.add_argument("--map", default=str(MAP), help="calibration map JSON (default: the one shipped in the package)")
    parser.add_argument("--no-map", action="store_true", help="serve raw probabilities, without the calibration map")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--name", default=NAME, help=f"the `model` field every reply carries (default {NAME})")
    parser.add_argument("--no-length-warmup", action="store_true",
                        help="skip the start-up forwards over a spread of lengths (64 to 2,048 tokens)")
    parser.add_argument("--no-cuda-graphs", action="store_true",
                        help="run every request eagerly instead of replaying the CUDA graphs captured at "
                             "start-up (shorter start-up, same answers)")
    parser.add_argument("--allow-slow-kernels", action="store_true",
                        help="DEBUG ONLY: start even if the linear-attention layers would run on transformers' "
                             "slow PyTorch reference path; never time or submit such a run")
    long = parser.add_argument_group("long menus", "choice questions with more options than one pass can read")
    long.add_argument("--shortlist", choices=(*shortlist.STRATEGIES, "off"), default=shortlist.DEFAULT.strategy,
                      help="first stage for a long menu: 'tournament' (default, measured; no extra model), "
                           "'embedding' (unmeasured; downloads Qwen/Qwen3-Embedding-0.6B on first start), or "
                           "'off' (refuse menus over 26 options, as 1.1.0 did)")
    long.add_argument("--shortlist-k", type=int, default=None,
                      help=f"options kept for the final pass, 2 to {shortlist.LIMIT} (default "
                           + ", ".join(f"{k} for {s}" for s, k in shortlist.DEFAULT_K.items()) + ")")
    long.add_argument("--shortlist-threshold", type=int, default=shortlist.DEFAULT.threshold,
                      help=f"shortlist choice questions with more options than this, 1 to {shortlist.LIMIT} "
                           f"(default {shortlist.DEFAULT.threshold})")
    long.add_argument("--shortlist-residual", type=float, default=shortlist.DEFAULT.residual,
                      help="probability mass mixed in uniformly over the whole menu, so that every option, "
                           f"including the eliminated ones, keeps some (default {shortlist.DEFAULT.residual:g})")
    long.add_argument("--shortlist-seed", type=int, default=shortlist.DEFAULT.seed,
                      help="seed for dealing a menu into tournament chunks (default 0)")
    long.add_argument("--embedding-model", default=shortlist.EMBEDDER,
                      help=f"embedding model for --shortlist embedding (default {shortlist.EMBEDDER})")
    long.add_argument("--embedding-revision", default=shortlist.EMBEDDER_REVISION,
                      help="its Hub revision (default: the pinned one)")
    return parser


def shortlist_config(args):
    """The `shortlist.Config` the flags ask for, or None for --shortlist off. Raises ValueError."""
    if args.shortlist == "off":
        return None
    return shortlist.Config(strategy=args.shortlist, k=args.shortlist_k, threshold=args.shortlist_threshold,
                            residual=args.shortlist_residual, seed=args.shortlist_seed,
                            embedder=args.embedding_model, embedder_revision=args.embedding_revision)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = shortlist_config(args)
    except ValueError as error:
        parser.error(str(error))
    from hopper_decisions import fastpath
    from hopper_decisions.model import Decider
    try:
        decider = Decider(adapter=args.adapter, calibration_map=None if args.no_map else args.map,
                          allow_slow_kernels=args.allow_slow_kernels, name=args.name,
                          cuda_graphs=not args.no_cuda_graphs, shortlist=config,
                          **({"warm_lengths": ()} if args.no_length_warmup else {}))
    except fastpath.SlowKernels as error:
        sys.exit(str(error))  # exit status 1, the message on stderr
    gpu = decider.gpu
    print(f"gpu: {gpu['name']}, compute capability {gpu['capability']}, CUDA {gpu['cuda']}, torch {gpu.get('torch')}",
          flush=True)
    for op, info in decider.kernels.items():
        print(f"kernel {op}: {info['implementation']}{'' if info['fast'] else ' (SLOW reference path)'}"
              f"{' [ran]' if info['ran'] else ''}", flush=True)
    if decider.slow_kernels:
        print("WARNING: --allow-slow-kernels: serving on the slow reference path; do not time this run.", flush=True)
    else:
        print("fast-kernel check passed", flush=True)
    count, seconds = decider.warm_seconds
    print(f"length warm-up: {count} lengths in {seconds:.1f} s", flush=True)
    for line in graph_lines(decider) or ["cuda graphs: off (--no-cuda-graphs); every request runs eager"]:
        print(line, flush=True)
    print(config.describe() if config else "shortlist: off; choice questions over 26 options are refused",
          flush=True)
    print(f"serving {decider.name} on {args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), handler(decider)).serve_forever()


if __name__ == "__main__":
    main()
