"""Hopper, a JevBench decision server: Qwen/Qwen3.5-4B + a LoRA adapter, one forward pass per decision."""

from pathlib import Path

# HF_ORG is a placeholder until the adapter is published on the Hugging Face Hub.
NAME = "hopper"               # the `model` field of every reply
HF_REPO = "HF_ORG/hopper"     # the adapter on the Hugging Face Hub
MAP = Path(__file__).parent / "maps" / f"{NAME}.json"


def __getattr__(name):  # torch is only imported when a model is actually built
    if name == "Decider":
        from hopper_decisions.model import Decider
        return Decider
    raise AttributeError(name)
