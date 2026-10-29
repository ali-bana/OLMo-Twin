import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_edited_wikipedia_group as bw  # noqa: E402


ANC = "The quick brown fox jumps over the lazy dog every single morning before breakfast is served"


def test_identical_text_is_fully_shared():
    w = bw.word_masks(ANC, ANC)
    assert w["n_words_novel"] == 0
    assert w["frac_ancestor_retained"] == 1.0
    assert w["longest_shared_run_words"] == w["n_folded"]


def test_unrelated_text_is_fully_novel():
    w = bw.word_masks("completely different words here with no overlap whatsoever at all in any place", ANC)
    assert w["n_words_novel"] == w["n_words"]
    assert w["frac_ancestor_retained"] == 0.0


def test_inserted_sentence_is_novel_and_context_shared():
    edited = ANC.replace("every single morning", "every single morning, and this new sentence was added later by an editor, then continued")
    w = bw.word_masks(edited, ANC)
    words = [edited[s:e] for s, e in w["spans"]]
    novel = {wd for wd, sh in zip(words, w["word_shared"]) if not sh}
    assert {"new", "sentence", "added", "later", "editor"} <= novel
    # words far from the edit remain covered by shared 8-grams
    assert all(sh for wd, sh in zip(words, w["word_shared"]) if wd in ("quick", "brown", "fox", "jumps"))
    assert 0 < w["n_words_novel"] < w["n_words"]


def test_short_shared_fragments_count_as_novel():
    # 5 shared words is below the 8-gram unit: not evidence of copying
    w = bw.word_masks("the quick brown fox jumps in an entirely different unrelated story about cats", ANC)
    assert w["n_words_novel"] == w["n_words"]


def test_case_and_punctuation_fold_like_the_scan():
    edited = ANC.upper().replace(" ", ", ")
    w = bw.word_masks(edited, ANC)
    assert w["n_words_novel"] == 0


def test_token_mask_lifts_word_mask_and_fills_gaps():
    spans = [(0, 3), (4, 9), (12, 15)]           # "the quick   fox"
    shared = np.array([True, False, True])
    offsets = [(0, 3), (3, 4), (4, 6), (6, 9), (9, 12), (12, 15)]
    m = bw.token_mask(offsets, spans, shared)
    # "the"->shared, " "->inherits next (novel), "qu","ick"->novel, "   "->inherits next (shared), "fox"->shared
    assert m.tolist() == [0, 1, 1, 1, 0, 0]


def test_token_mask_trailing_gap_inherits_previous():
    m = bw.token_mask([(0, 3), (3, 5)], [(0, 3)], np.array([False]))
    assert m.tolist() == [1, 1]


def test_strata_boundaries():
    assert bw.stratum(0.0) == "lt10"
    assert bw.stratum(0.0999) == "lt10"
    assert bw.stratum(0.10) == "10-25"
    assert bw.stratum(0.5) == "50-75"
    assert bw.stratum(0.75) == "ge75"
    assert bw.stratum(1.0) == "ge75"
