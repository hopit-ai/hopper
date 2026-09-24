"""Choice questions with more than 26 options: a first stage cuts the menu to k candidates, and the
ordinary one-pass letter readout decides among those. Imports no ML framework.

The readout puts one option per letter, so one pass can weigh at most 26 options. Routing and
retrieval menus are longer (77 intents, 150 intents, hundreds of tools). A choice question with
more options than `Config.threshold` (26 by default) is therefore answered in two stages:

  1. First stage, one of two:
     tournament  no extra model. The menu is dealt round-robin into ceil(n / 26) chunks, and each
                 chunk is scored by the ordinary one-pass readout. An option's score is its
                 probability within its own chunk. ceil(n / 26) passes. The default.
     embedding   a separate embedding model (`embedder.py`) scores each option by the dot product
                 of its cached, normalised embedding with the request's. No letter passes; one
                 embedding of the request, plus one per option the first time it is seen. Off by
                 default until it is measured.
  2. The k best-scoring options, in the menu's own order, go through the ordinary single pass
     (same prompt layout, same readout, same calibration map), so the final decision is exactly the
     kind of decision everything else here was measured on.

The reply is a probability for EVERY option in the request, never only the k survivors:

    p(option) = (1 - residual) * final(option) + residual / n      for an option in the shortlist
    p(option) =                                  residual / n      for an eliminated option

that is, the final distribution mixed with `residual` of the uniform distribution over the whole
menu. It sums to 1, every label keeps positive mass, and the mixture adds the same amount to every
label, so it can never change which option wins. The eliminated options together get
residual * (n - k) / n, which is meant as an estimate of how often the first stage throws away the
right answer. The default, 0.05, is set from the one measurement we have (see the README, "Long
menus"), and is to be refitted as 1 - recall@k once the adapter is measured on long menus. Nothing
is renormalised away silently: `residual=0` is allowed, and then the eliminated options are still
listed, at 0.0.

Deterministic: the chunks are dealt with a fixed seed, and ties break on the option's position.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from hopper_decisions import calibration, request
from hopper_decisions.prompt import LETTERS, option_lines

LIMIT = len(LETTERS)                                  # options one readout pass can weigh
STRATEGIES = ("tournament", "embedding")
DEFAULT_K = {"tournament": 10, "embedding": 26}      # tournament k=10 is the measured best; see README
RESIDUAL = 0.05
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"                # Apache-2.0
EMBEDDER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


@dataclass(frozen=True)
class Config:
    """How a long menu is answered. `k=None` takes the strategy's default. `threshold`: a choice
    question with more options than this is shortlisted; at most 26, since a longer menu cannot be
    read in one pass."""

    strategy: str = "tournament"
    k: int | None = None
    threshold: int = LIMIT
    residual: float = RESIDUAL
    seed: int = 0
    embedder: str = EMBEDDER
    embedder_revision: str | None = EMBEDDER_REVISION

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown shortlist strategy {self.strategy!r}; one of {', '.join(STRATEGIES)}")
        if not 2 <= self.size <= LIMIT:
            raise ValueError(f"shortlist k must be 2 to {LIMIT}, not {self.size}")
        if not 1 <= self.threshold <= LIMIT:
            raise ValueError(f"shortlist threshold must be 1 to {LIMIT}, not {self.threshold}")
        if not 0.0 <= self.residual < 1.0:
            raise ValueError(f"residual must be in [0, 1), not {self.residual}")

    @property
    def size(self):
        return DEFAULT_K[self.strategy] if self.k is None else self.k

    def describe(self):
        return (f"shortlist: {self.strategy}, k={self.size}, for choice questions over {self.threshold} options, "
                f"residual {self.residual:g} spread over the whole menu"
                + (f", embedder {self.embedder}@{(self.embedder_revision or 'main')[:12]}"
                   if self.strategy == "embedding" else ""))


DEFAULT = Config()


def applies(config, example):
    """Is this parsed request one for the shortlist? Only a choice question longer than the
    threshold and longer than k; everything else is the ordinary single pass, unchanged."""
    if config is None or example["kind"] != "choice":
        return False
    count = len(example["options"])
    return count > config.threshold and count > config.size


def chunk_indices(count, size=LIMIT, rng=None):
    """Option indices in ceil(count / size) groups, dealt round-robin so the groups are within one
    option of each other: a stub group of one would hand its option a probability of 1.0 and walk it
    into the shortlist. `rng` shuffles first, so a menu sorted by anything is not chunked by it."""
    order = list(range(count))
    if rng is not None:
        rng.shuffle(order)
    groups = max(1, -(-count // size))
    return [order[i::groups] for i in range(groups)]


def top_indices(scores, k):
    """The k best-scoring option indices, ascending (the menu's own order). Ties break on the index."""
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(ranked[:k])


def spread(names, shortlist, final, residual):
    """Probabilities over every option: `final` (a distribution over the options at `shortlist`)
    mixed with `residual` of the uniform distribution over all of `names`. See the module docstring."""
    share = residual / len(names)
    probs = {name: share for name in names}
    for index, p in zip(shortlist, final):
        probs[names[index]] += (1.0 - residual) * p
    return probs


def subset(example, indices):
    """The same decision over some of the options, in the given order."""
    return {**example, "options": [example["options"][i] for i in indices]}


class Clock:
    """Adds up tokenisation, forward and embedding time over the passes of one decision."""

    def __init__(self):
        self.tokenise = self.forward = 0.0

    def encode(self, decider, example):
        started = time.perf_counter()
        ids = decider.encode(example)
        self.tokenise += time.perf_counter() - started
        return ids

    def run(self, fn, *args):
        started = time.perf_counter()
        out = fn(*args)
        self.forward += time.perf_counter() - started
        return out


def tournament_scores(decider, example, config, clock):
    """(score per option, passes, tokens): every option's probability within its own chunk of at most
    26, each chunk read by the ordinary single pass over the same prompt layout."""
    groups = chunk_indices(len(example["options"]), LIMIT, random.Random(config.seed))
    scores, tokens = [0.0] * len(example["options"]), 0
    for group in groups:
        ids = clock.encode(decider, subset(example, group))
        tokens += len(ids)
        for index, p in zip(group, clock.run(decider.letter_probs, ids, len(group))):
            scores[index] = p
    return scores, len(groups), tokens


def embedding_scores(decider, example, clock):
    """(score per option, 0 passes, 0 tokens): dot products of normalised embeddings. The embedder
    caches option embeddings, so a menu seen before costs one embedding of the request."""
    texts = [shown for _, shown in option_lines(example)]  # exactly the lines the prompt shows
    return clock.run(decider.embedder.similarities, example["document"], example["question"], texts), 0, 0


def decide(decider, example, key, config, started=None):
    """Answer a long choice menu with `decider`'s own `encode`, `letter_probs`, calibration map and
    name. Returns what `Decider.score` returns, plus a `shortlist` record of what was done."""
    entered = time.perf_counter()
    parse_s = 0.0 if started is None else entered - started  # request.parse, before this was called
    clock = Clock()
    names = [o["name"] for o in example["options"]]
    if config.strategy == "tournament":
        scores, passes, tokens = tournament_scores(decider, example, config, clock)
    else:
        scores, passes, tokens = embedding_scores(decider, example, clock)
    kept = top_indices(scores, min(config.size, len(names)))
    final = subset(example, kept)
    ids = clock.encode(decider, final)
    tokens += len(ids)
    raw_final = clock.run(decider.letter_probs, ids, len(kept))
    kept_names = [names[i] for i in kept]
    # The map was fitted on single-pass distributions over at most 26 options, which is what the
    # final pass is; it is applied there, before the residual, and never to the whole menu.
    mapped = calibration.apply(decider.map, final, dict(zip(kept_names, raw_final)))
    probs = spread(names, kept, [mapped[name] for name in kept_names], config.residual)
    reply = request.response(key, "choice", probs, decider.name, tokens)
    done = time.perf_counter()
    record = {"strategy": config.strategy, "options": len(names), "k": len(kept), "kept": kept_names,
              "passes": passes + 1, "residual": config.residual,
              "eliminated_mass": config.residual * (len(names) - len(kept)) / len(names)}
    tokenise = parse_s + clock.tokenise
    post = max(0.0, done - entered - clock.tokenise - clock.forward)  # choosing, mapping, spreading
    return {"example": example, "raw": spread(names, kept, raw_final, config.residual), "response": reply,
            "tokens": tokens, "shortlist": record, "seconds": (tokenise, clock.forward, post)}
