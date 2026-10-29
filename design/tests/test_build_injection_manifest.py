import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the design/ directory
import build_injection_manifest as bim  # noqa: E402


def docs(n, seed=0):
    rng = random.Random(seed)
    return [{"doc_id": f"d{i:03d}", "n_tokens": rng.randint(1000, 9000), "bpb_pre": rng.random()} for i in range(n)]


def test_pairs_are_balanced_and_deterministic():
    a = bim.make_pairs(docs(40), random.Random(1), "M", "s", [0])
    b = bim.make_pairs(docs(40), random.Random(1), "M", "s", [0])
    assert [(r["doc_id"], r["arm"]) for r in a] == [(r["doc_id"], r["arm"]) for r in b]
    arms = Counter(r["arm"] for r in a)
    assert arms == {"inject": 20, "holdout": 20}
    by_pair = Counter(r["pair_id"] for r in a)
    assert set(by_pair.values()) == {2}


def test_pairs_match_on_length_bin():
    rows = bim.make_pairs(docs(60), random.Random(2), "M", "s", [0])
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair_id"], []).append(r)
    for p in pairs.values():
        assert p[0]["length_bin"] == p[1]["length_bin"] or abs(p[0]["length_bin"] - p[1]["length_bin"]) == 1


def test_coin_flip_changes_with_rng():
    a = bim.make_pairs(docs(40), random.Random(1), "M", "s", [0])
    b = bim.make_pairs(docs(40), random.Random(2), "M", "s", [0])
    arm_a = {r["doc_id"]: r["arm"] for r in a}
    arm_b = {r["doc_id"]: r["arm"] for r in b}
    assert arm_a != arm_b                       # different flips
    assert {r["pair_id"] for r in a} == {r["pair_id"] for r in b}  # same pairs


def test_odd_document_is_dropped():
    rows = bim.make_pairs(docs(7), random.Random(0), "M", "s", [0])
    assert len(rows) == 6


class FakeTok:
    """1 token per whitespace word."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_truncate_at_paragraph_boundary_nearest_target():
    paras = [" ".join(["w"] * 100) for _ in range(20)]      # 20 paragraphs x 100 tokens
    text = "\n\n".join(paras)
    out = bim.truncate_paragraph(text, FakeTok(), 512)
    n = len(out.split())
    assert n in (500, 600)
    assert out.endswith("\n")
    assert text.startswith(out.rstrip("\n").rstrip())


def test_truncate_longer_than_text_returns_whole():
    text = "a b c\n\nd e f"
    assert bim.truncate_paragraph(text, FakeTok(), 10_000).split() == text.split()


def test_shuffles_preserve_multiset():
    text = "The cat sat. The dog ran! A bird flew? Fish swim."
    ws = bim.word_shuffle(text, random.Random(0))
    assert Counter(ws.split()) == Counter(text.split())
    ss = bim.sentence_shuffle(text, random.Random(0))
    assert Counter(ss.split()) == Counter(text.split())
    assert ss.split() != text.split() or ws.split() != text.split()


def test_sub_rng_is_named_and_stable():
    assert bim.sub_rng(1, "M").random() == bim.sub_rng(1, "M").random()
    assert bim.sub_rng(1, "M").random() != bim.sub_rng(1, "D").random()
