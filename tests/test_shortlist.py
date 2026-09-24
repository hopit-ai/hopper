"""Long choice menus: the shortlist in the request path, with a stand-in decider. No GPU, no network.

`FakeDecider` has no weights. Its `encode` records which options a pass shows and its
`letter_probs` returns a fixed distribution over them, so every pass the real `pipeline.score` and
`shortlist.decide` make is visible here.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from hopper_decisions import calibration, embedder, pipeline, request, server, shortlist
from hopper_decisions.shortlist import DEFAULT, LIMIT, Config


class FakeDecider:
    """`model.Decider` without weights. The option named `prefer` gets 0.7 in any pass that shows it
    (the first option does otherwise) and the rest share 0.3; `score` is the real request path."""

    def __init__(self, config=DEFAULT, prefer=None, mapping=None, embed=None):
        self.name, self.map = "hopper", mapping or calibration.Constant(1.0)
        self.shortlist, self.embedder, self.prefer = config, embed, prefer
        self.passes = []

    def encode(self, example):
        self.passes.append([o["name"] for o in example["options"]] if example["options"] else ["true", "false"])
        return list(range(3 * len(self.passes[-1]) + 5))

    def letter_probs(self, ids, count):
        shown = self.passes[-1]
        assert count == len(shown) and len(ids) == 3 * count + 5 and count <= LIMIT
        if count == 1:
            return [1.0]
        win = shown.index(self.prefer) if self.prefer in shown else 0
        return [0.7 if i == win else 0.3 / (count - 1) for i in range(count)]

    def score(self, req):
        return pipeline.score(self, req)

    def decide(self, req):
        return self.score(req)["response"]


class FakeEmbedder:
    """Scores an option by a fixed table; records every call."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def similarities(self, document, question, texts):
        self.calls.append((document, question, list(texts)))
        return [self.table.get(t, 0.0) for t in texts]


def names(n):
    return [f"intent {i:03d}" for i in range(n)]


def systemone(n, state="I lost my card yesterday", kind="choice"):
    labels = names(n)
    return {"state": state, "model": "x",
            "questions": {"decision": {"type": kind, "instructions": "Which intent is this?",
                                       "criteria": {label: label for label in labels}}}}


def record(n, labels=None):
    labels = labels or names(n)
    return {"state": "I lost my card", "labels": labels,
            "question": {"type": "choice", "instructions": "Route it.",
                         "criteria": {label: f"about {label}" for label in names(n)}}}


def probs_of(out):
    return out["response"]["answers"]["decision"]["probabilities"]


# --- configuration ---------------------------------------------------------------------------------
def test_the_default_is_the_measured_tournament_at_k_10_over_26_options():
    assert (DEFAULT.strategy, DEFAULT.size, DEFAULT.threshold, DEFAULT.residual) == ("tournament", 10, 26, 0.05)
    assert Config("embedding").size == 26       # the embedding stage is opt-in; its own default k


@pytest.mark.parametrize("bad", [{"k": 1}, {"k": 27}, {"k": 0}, {"threshold": 0}, {"threshold": 27},
                                 {"residual": -0.01}, {"residual": 1.0}, {"strategy": "bm25"}])
def test_a_config_the_readout_cannot_serve_is_refused(bad):
    with pytest.raises(ValueError):
        Config(**bad)


@pytest.mark.parametrize("k", [2, 26])
def test_k_may_be_anything_from_2_to_26(k):
    assert Config(k=k).size == k


def test_the_embedder_is_pinned_and_permissively_licensed():
    assert shortlist.EMBEDDER == "Qwen/Qwen3-Embedding-0.6B"
    assert len(shortlist.EMBEDDER_REVISION) == 40 and int(shortlist.EMBEDDER_REVISION, 16) >= 0


# --- which requests take the shortlist ---------------------------------------------------------------
@pytest.mark.parametrize("n,expected", [(2, False), (9, False), (26, False), (27, True), (77, True), (150, True)])
def test_only_a_choice_menu_over_26_options_is_shortlisted_by_default(n, expected):
    example, _ = request.parse(systemone(n), large_choice=True)
    assert shortlist.applies(DEFAULT, example) is expected
    assert shortlist.applies(None, example) is False


def test_noul_and_score_questions_never_take_the_shortlist():
    noul = {"state": "s", "question": {"type": "noul", "criteria": {"true": "a", "false": "b"}}}
    score = {"state": "s", "question": {"type": "score", "criteria": ["x"] * 5}}
    for req in (noul, score):
        example, _ = request.parse(req, large_choice=True)
        assert not shortlist.applies(Config(threshold=1, k=2), example)


def test_a_lower_threshold_shortlists_shorter_menus_but_never_one_no_longer_than_k():
    example, _ = request.parse(systemone(20))
    assert shortlist.applies(Config(threshold=10, k=5), example)
    assert not shortlist.applies(Config(threshold=10, k=20), example)  # keeping all 20 is the single pass
    assert not shortlist.applies(Config(threshold=20, k=5), example)


# --- the pure parts ------------------------------------------------------------------------------------
@pytest.mark.parametrize("count", [1, 26, 27, 52, 53, 77, 150, 255])
def test_chunks_cover_every_option_once_balanced_and_within_the_letters(count):
    groups = shortlist.chunk_indices(count)
    assert sorted(i for g in groups for i in g) == list(range(count))
    assert len(groups) == math.ceil(count / LIMIT)
    sizes = [len(g) for g in groups]
    assert max(sizes) <= LIMIT and max(sizes) - min(sizes) <= 1


def test_chunks_are_seeded_so_a_menu_is_always_dealt_the_same_way():
    import random
    a = shortlist.chunk_indices(77, rng=random.Random(0))
    assert a == shortlist.chunk_indices(77, rng=random.Random(0)) != shortlist.chunk_indices(77)


def test_top_indices_are_the_best_k_in_menu_order_with_ties_on_position():
    assert shortlist.top_indices([0.1, 0.9, 0.5, 0.7], 2) == [1, 3]
    assert shortlist.top_indices([0.5, 0.5, 0.5, 0.5], 3) == [0, 1, 2]


def test_spread_lists_every_label_sums_to_one_and_gives_each_eliminated_label_residual_over_n():
    labels = ["a", "b", "c", "d", "e"]
    probs = shortlist.spread(labels, [1, 3], [0.25, 0.75], 0.05)
    assert list(probs) == labels and sum(probs.values()) == pytest.approx(1.0, abs=1e-12)
    assert probs["a"] == probs["c"] == probs["e"] == pytest.approx(0.01)
    assert probs["b"] == pytest.approx(0.95 * 0.25 + 0.01) and probs["d"] == pytest.approx(0.95 * 0.75 + 0.01)


def test_spread_with_no_residual_keeps_the_eliminated_labels_at_zero_rather_than_dropping_them():
    probs = shortlist.spread(["a", "b", "c"], [0], [1.0], 0.0)
    assert probs == {"a": 1.0, "b": 0.0, "c": 0.0}


@pytest.mark.parametrize("residual", [0.0, 0.05, 0.3, 0.9])
def test_the_residual_can_never_make_an_eliminated_label_the_answer(residual):
    labels = names(27)
    probs = shortlist.spread(labels, list(range(26)), [1 / 26] * 26, residual)  # the flattest final pass
    top = max(probs.values())
    assert probs[labels[26]] < top and request.argmax(probs) in labels[:26]


# --- the request path: long menus -----------------------------------------------------------------------
def test_27_options_take_two_chunks_and_a_final_pass_of_10():
    decider = FakeDecider(prefer="intent 026")
    out = decider.score(systemone(27))
    answer = out["response"]["answers"]["decision"]
    assert [len(p) for p in decider.passes] == [14, 13, 10]            # 27 dealt into 14 + 13, then k = 10
    assert all(len(p) <= LIMIT for p in decider.passes)
    assert answer["choice"] == "intent 026"
    assert set(answer["probabilities"]) == set(names(27))
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-9)
    assert out["shortlist"]["passes"] == 3 and out["shortlist"]["k"] == 10 and len(out["shortlist"]["kept"]) == 10


def test_exactly_26_options_is_the_ordinary_single_pass_and_unchanged():
    on, off = FakeDecider(prefer="intent 020"), FakeDecider(config=None, prefer="intent 020")
    a, b = on.score(systemone(26)), off.score(systemone(26))
    assert on.passes == off.passes == [names(26)]                       # one pass over the whole menu
    assert a["response"] == b["response"] and a["raw"] == b["raw"] and "shortlist" not in a


@pytest.mark.parametrize("n,chunks", [(27, 2), (52, 2), (53, 3), (77, 3), (150, 6), (255, 10)])
def test_the_tournament_costs_one_pass_per_chunk_plus_one(n, chunks):
    decider = FakeDecider(prefer=names(n)[n // 2])
    out = decider.score(systemone(n))
    assert len(decider.passes) == chunks + 1 == out["shortlist"]["passes"]
    assert sorted(o for p in decider.passes[:-1] for o in p) == names(n)  # every option scored once
    assert out["response"]["answers"]["decision"]["choice"] == names(n)[n // 2]
    assert len(probs_of(out)) == n and sum(probs_of(out).values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("k", [2, 10, 25, 26])
def test_the_final_pass_shows_exactly_k_options(k):
    decider = FakeDecider(config=Config(k=k), prefer="intent 040")
    out = decider.score(systemone(77))
    assert len(decider.passes[-1]) == k == out["shortlist"]["k"]
    assert decider.passes[-1] == sorted(decider.passes[-1])            # in the menu's own order
    assert "intent 040" in decider.passes[-1] and out["response"]["answers"]["decision"]["choice"] == "intent 040"


def test_each_eliminated_option_gets_exactly_residual_over_n():
    for residual in (0.05, 0.2):
        decider = FakeDecider(config=Config(residual=residual))
        out = decider.score(systemone(77))
        probs, kept = probs_of(out), set(out["shortlist"]["kept"])
        dropped = [label for label in probs if label not in kept]
        assert len(dropped) == 67
        assert all(probs[label] == pytest.approx(residual / 77, rel=1e-9) for label in dropped)
        assert sum(probs[label] for label in dropped) == pytest.approx(out["shortlist"]["eliminated_mass"])
        assert out["shortlist"]["eliminated_mass"] == pytest.approx(residual * 67 / 77)


def test_the_calibration_map_is_applied_to_the_final_pass_only():
    mapping = calibration.PerKind({"choice": 2.0})
    decider = FakeDecider(mapping=mapping, prefer="intent 005")
    out = decider.score(systemone(40))
    kept = out["shortlist"]["kept"]
    final = dict(zip(kept, [0.7 if name == "intent 005" else 0.3 / 9 for name in kept]))
    mapped = calibration.rescale(final, 2.0)
    probs = probs_of(out)
    for name in kept:
        assert probs[name] == pytest.approx(0.95 * mapped[name] + 0.05 / 40)
    assert out["raw"]["intent 005"] == pytest.approx(0.95 * 0.7 + 0.05 / 40)  # pre-map, same mixture
    assert set(out["raw"]) == set(names(40))


def test_usage_counts_the_tokens_of_every_pass():
    decider = FakeDecider()
    out = decider.score(systemone(77))
    expected = sum(3 * len(p) + 5 for p in decider.passes)
    assert out["tokens"] == out["response"]["usage"]["input_tokens"] == expected
    assert out["response"]["usage"]["output_tokens"] == 0


def test_a_long_menu_is_answered_the_same_way_every_time():
    first, second = FakeDecider(), FakeDecider()
    assert first.score(systemone(150))["response"] == second.score(systemone(150))["response"]
    assert first.passes == second.passes


def test_the_seed_changes_how_the_menu_is_dealt_not_whether_it_is_deterministic():
    a, b, c = FakeDecider(config=Config(seed=1)), FakeDecider(config=Config(seed=1)), FakeDecider()
    for decider in (a, b, c):
        decider.score(systemone(77))
    assert a.passes == b.passes and a.passes[0] != c.passes[0]


def test_a_lower_threshold_runs_a_one_chunk_qualifier_then_the_final_pass():
    decider = FakeDecider(config=Config(threshold=10, k=5), prefer="intent 017")
    out = decider.score(systemone(20))
    assert [len(p) for p in decider.passes] == [20, 5]
    assert out["response"]["answers"]["decision"]["choice"] == "intent 017" and len(probs_of(out)) == 20


def test_the_record_route_answers_over_its_own_label_order():
    labels = list(reversed(names(30)))
    decider = FakeDecider(prefer="intent 003")
    out = decider.score(record(30, labels))
    assert [o["name"] for o in out["example"]["options"]] == labels
    assert decider.passes[-1] == [name for name in labels if name in set(decider.passes[-1])]
    assert list(probs_of(out)) == labels


def test_the_embedding_stage_takes_the_nearest_options_in_one_pass():
    table = {"intent 007": 0.9, "intent 031": 0.8, "intent 012": 0.1}
    fake = FakeEmbedder(table)
    decider = FakeDecider(config=Config("embedding", k=3), prefer="intent 031", embed=fake)
    out = decider.score(systemone(40))
    (document, question, texts), = fake.calls
    assert document == "I lost my card yesterday" and question == "Which intent is this?"
    assert texts == names(40)                                        # the prompt's own option lines
    assert decider.passes == [["intent 007", "intent 012", "intent 031"]]
    assert out["shortlist"]["passes"] == 1 and out["response"]["answers"]["decision"]["choice"] == "intent 031"
    assert len(probs_of(out)) == 40 and sum(probs_of(out).values()) == pytest.approx(1.0, abs=1e-9)


def test_the_embedder_sees_name_and_description_as_the_prompt_shows_them():
    fake = FakeEmbedder({})
    FakeDecider(config=Config("embedding"), embed=fake).score(record(30))
    assert fake.calls[0][2][0] == "intent 000: about intent 000"


def test_the_embedding_query_follows_the_models_instruction_format():
    assert embedder.query_text("I lost my card", "Which intent?") == (
        "Instruct: Given a document and a question about it, retrieve the option that answers the question. "
        "Question: Which intent?\nQuery:I lost my card")
    assert embedder.query_text("", "Which intent?").endswith("\nQuery:Which intent?")


# --- refusals stay refusals --------------------------------------------------------------------------
def test_with_the_shortlist_off_a_long_menu_is_refused_as_before():
    with pytest.raises(ValueError, match="27 options, more than 26 letters"):
        FakeDecider(config=None).score(systemone(27))


@pytest.mark.parametrize("bad", [
    {**record(30), "labels": names(29)},                                           # labels != criteria
    {k: v for k, v in record(30).items() if k != "state"},                         # no state
    {"state": "s", "question": {"type": "choice", "instructions": "q"}},           # no criteria
    {"state": "s", "question": {"type": "score", "criteria": ["level"] * 30}},     # 30 score levels
    {"state": "s", "question": {"type": "rank", "criteria": {n: n for n in names(30)}}},
    {"state": "s", "questions": {"a": systemone(30)["questions"]["decision"],
                                 "b": systemone(30)["questions"]["decision"]}},
])
def test_malformed_long_requests_are_still_refused_with_the_shortlist_on(bad):
    decider = FakeDecider()
    with pytest.raises((ValueError, KeyError, TypeError)):
        decider.score(bad)
    assert decider.passes == []                                     # refused before any forward pass


# --- the HTTP server -----------------------------------------------------------------------------------
@pytest.fixture
def served():
    decider = FakeDecider(prefer="intent 044")
    httpd = HTTPServer(("127.0.0.1", 0), server.handler(decider))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", decider
    httpd.shutdown()
    httpd.server_close()


def post(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_server_answers_a_77_option_menu_over_every_label(served):
    url, _ = served
    status, body = post(f"{url}/v1/systemone", systemone(77))
    answer = body["answers"]["decision"]
    assert status == 200 and answer["choice"] == "intent 044" and set(answer["probabilities"]) == set(names(77))
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-3                 # the harness's own tolerance
    status, body = post(f"{url}/run", {"task": record(77)})
    assert status == 200 and body["ok"] and set(body["probs"]) == set(names(77))


def test_the_server_still_answers_malformed_requests_with_a_400(served):
    url, decider = served
    for bad in ({**record(30), "labels": names(29)}, {"state": "s", "question": {"type": "rank"}}):
        status, body = post(f"{url}/v1/systemone", bad)
        assert status == 400 and body["ok"] is False and body["error"].startswith("ValueError")
        status, body = post(f"{url}/run", {"task": bad})
        assert status == 400 and body["ok"] is False
    assert decider.passes == []


def test_the_flags_default_to_the_tournament_and_off_restores_the_refusal():
    args = server.build_parser().parse_args([])
    assert server.shortlist_config(args) == DEFAULT
    assert server.shortlist_config(server.build_parser().parse_args(["--shortlist", "off"])) is None
    args = server.build_parser().parse_args(["--shortlist", "embedding", "--shortlist-k", "20",
                                             "--shortlist-threshold", "12", "--shortlist-residual", "0.02"])
    assert server.shortlist_config(args) == Config("embedding", k=20, threshold=12, residual=0.02)
    with pytest.raises(ValueError):
        server.shortlist_config(server.build_parser().parse_args(["--shortlist-k", "30"]))
