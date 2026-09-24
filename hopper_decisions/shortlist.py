"""Choice questions with more than 26 options: a first stage cuts the menu to k candidates, and the
ordinary one-pass letter readout decides among those. Imports no ML framework.

The readout puts one option per letter, so one pass can weigh at most 26 options. Routing and
retrieval menus are longer (77 intents, 150 intents, hundreds of tools). A choice question with
more than 26 options is therefore answered in two stages (a question with 26 or fewer never is,
whatever the configuration):

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
menu. It sums to 1 and every label keeps positive mass. The mixture adds the same amount to every
label, so in exact arithmetic it cannot change which option wins; in floating point two survivors
one representable step apart can round to the same value, so `spread` checks the reply exactly as
`request.response` will normalise it and, only if the final pass's own answer no longer wins, raises
that one probability by single representable steps until it does. The eliminated options together
get residual * (n - k) / n, which is meant as an estimate of how often the first stage throws away
the right answer. The default, 0.05, is set from the one measurement we have (see the README, "Long
menus"), and is to be refitted as 1 - recall@k once the adapter is measured on long menus; any
value from 0 to `RESIDUAL_MAX` (0.5) is accepted, so the final pass always carries at least half the
mass. Nothing is renormalised away silently: `residual=0` is allowed, and then the eliminated
options are still listed, at 0.0.

Every prompt this module builds -- each tournament chunk and the final pass -- is checked against
the model's context before its forward pass (`TooLong`, a `ValueError`), so a menu that cannot be
read is refused like any other malformed request (HTTP 400, or 422 in-process).

Deterministic: the chunks are dealt with a fixed seed, and ties break on the option's position.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from hopper_decisions import calibration, request
from hopper_decisions.prompt import LETTERS, option_lines

LIMIT = len(LETTERS)                                  # options one readout pass can weigh
STRATEGIES = ("tournament", "embedding")
DEFAULT_K = {"tournament": 10, "embedding": 26}      # tournament k=10 is the measured best; see README
RESIDUAL = 0.05
RESIDUAL_MAX = 0.5                                    # the final pass always carries at least half the mass
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"                # Apache-2.0
EMBEDDER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


class TooLong(ValueError):
    """A prompt of a shortlisted decision is longer than the model's context."""


@dataclass(frozen=True)
class Config:
    """How a long menu is answered. `k=None` takes the strategy's default. Only a choice question
    with more than 26 options is ever shortlisted (`applies`); there is no setting that sends a
    shorter menu here, so every question of 26 options or fewer takes the ordinary single pass."""

    strategy: str = "tournament"
    k: int | None = None
    residual: float = RESIDUAL
    seed: int = 0
    embedder: str = EMBEDDER
    embedder_revision: str | None = EMBEDDER_REVISION

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown shortlist strategy {self.strategy!r}; one of {', '.join(STRATEGIES)}")
        if not 2 <= self.size <= LIMIT:
            raise ValueError(f"shortlist k must be 2 to {LIMIT}, not {self.size}")
        if isinstance(self.residual, bool) or not isinstance(self.residual, (int, float)) \
                or not 0.0 <= self.residual <= RESIDUAL_MAX:
            raise ValueError(f"residual must be from 0 to {RESIDUAL_MAX}, not {self.residual!r}")

    @property
    def size(self):
        return DEFAULT_K[self.strategy] if self.k is None else self.k

    def describe(self):
        return (f"shortlist: {self.strategy}, k={self.size}, for choice questions over {LIMIT} options, "
                f"residual {self.residual:g} spread over the whole menu"
                + (f", embedder {self.embedder}@{(self.embedder_revision or 'main')[:12]}"
                   if self.strategy == "embedding" else ""))


DEFAULT = Config()


def applies(config, example):
    """Is this parsed request one for the shortlist? Only a choice question with more than 26
    options, which one pass cannot read; everything else is the ordinary single pass, unchanged.
    (k is at most 26, so a shortlisted menu is always longer than k.)"""
    return config is not None and example["kind"] == "choice" and len(example["options"]) > LIMIT


def context_limit(decider):
    """The model's context in tokens, or None where the decider does not say."""
    config = getattr(getattr(decider, "model", None), "config", None)
    return getattr(config, "max_position_embeddings", None) or None


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
    mixed with `residual` of the uniform distribution over all of `names`. See the module docstring.
    The reply's answer is always the one the final pass alone gives (`keep_winner`)."""
    if len(set(names)) != len(names):
        raise ValueError("option names repeat")
    share = residual / len(names)
    probs = {name: share for name in names}
    for index, p in zip(shortlist, final):
        probs[names[index]] += (1.0 - residual) * p
    alone = {names[index]: p for index, p in zip(shortlist, final)}
    return keep_winner(probs, request.argmax(request.normalise(alone)))


def keep_winner(probs, winner, steps=64):
    """`probs`, with `winner` raised by single representable steps only if that is needed for the
    reply -- normalised and tie-broken exactly as `request.response` and the harness do it -- to
    name `winner`. The mixture is monotone, so it can only ever create a tie, never reverse an
    order; one or two steps (about 1e-16) break it. Leaves every other value untouched."""
    for _ in range(steps):
        if request.argmax(request.normalise(probs)) == winner:
            return probs
        probs[winner] = math.nextafter(probs[winner], math.inf)
    raise AssertionError(f"could not keep {winner!r} the answer")  # unreachable for a distribution


def subset(example, indices):
    """The same decision over some of the options, in the given order."""
    return {**example, "options": [example["options"][i] for i in indices]}


class Clock:
    """Adds up tokenisation, forward and embedding time over the passes of one decision, and
    refuses a prompt longer than `limit` tokens before its forward pass."""

    def __init__(self, limit=None):
        self.tokenise = self.forward = 0.0
        self.limit = limit

    def encode(self, decider, example):
        started = time.perf_counter()
        ids = decider.encode(example)
        self.tokenise += time.perf_counter() - started
        if self.limit is not None and len(ids) > self.limit:
            raise TooLong(f"a {len(example['options'])}-option prompt of {len(ids)} tokens is longer than "
                          f"the model's context of {self.limit}")
        return ids

    def run(self, fn, *args):
        started = time.perf_counter()
        out = fn(*args)
        self.forward += time.perf_counter() - started
        return out


def tournament_groups(example, config):
    """The chunks a tournament deals this menu into, exactly as `tournament_scores` deals them."""
    return chunk_indices(len(example["options"]), LIMIT, random.Random(config.seed))


def tournament_scores(decider, example, config, clock):
    """(score per option, passes, tokens): every option's probability within its own chunk of at most
    26, each chunk read by the ordinary single pass over the same prompt layout."""
    groups = tournament_groups(example, config)
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
    clock = Clock(context_limit(decider))
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
