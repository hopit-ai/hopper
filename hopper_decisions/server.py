"""A one-request-at-a-time HTTP server around `Decider`, for either of the harness's remote routes.

  hopper-serve --port 8080            # same as: python -m hopper_decisions.server --port 8080

  POST /v1/systemone  the `typesafe` adapter's wire format (jevbench/adapters/typesafe.py)
  POST /run           the `remote_inproc` client's format (jevbench/adapters/remote_inproc.py):
                      {"task": <record>} -> {"ok", "probs" over the exact labels, "latency_s", ...}

Single-threaded on purpose: the harness runs serially, and one GPU should see one request at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from hopper_decisions import HF_REPO, MAP, NAME, request


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default=HF_REPO, help=f"LoRA adapter directory or Hugging Face repo id (default {HF_REPO})")
    parser.add_argument("--map", default=str(MAP), help="calibration map JSON (default: the one shipped in the package)")
    parser.add_argument("--no-map", action="store_true", help="serve raw probabilities, without the calibration map")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--name", default=NAME, help=f"the `model` field every reply carries (default {NAME})")
    parser.add_argument("--no-length-warmup", action="store_true",
                        help="skip the start-up forwards over a spread of lengths (64 to 2,048 tokens)")
    parser.add_argument("--allow-slow-kernels", action="store_true",
                        help="DEBUG ONLY: start even if the linear-attention layers would run on transformers' "
                             "slow PyTorch reference path; never time or submit such a run")
    args = parser.parse_args()
    from hopper_decisions import fastpath
    from hopper_decisions.model import Decider
    try:
        decider = Decider(adapter=args.adapter, calibration_map=None if args.no_map else args.map,
                          allow_slow_kernels=args.allow_slow_kernels, name=args.name,
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
    print(f"serving {decider.name} on {args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), handler(decider)).serve_forever()


if __name__ == "__main__":
    main()
