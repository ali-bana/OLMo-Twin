"""Tests for the exhaustive matcher that makes the membership decision.

Every case guards a property whose failure would be a false-negative path of
the overlap search: the 20-character span filter is the same function on the
candidate and the corpus side and measures to the end of the last word; the
longest contiguous run is measured correctly, also end to end on planted text;
the bitmap prefilter never drops a real fingerprint; a scan with zero shards or
unparseable records fails instead of reporting a clean result; and a scan
against a different manifest or fingerprint table is refused.
(test_olmo2_midtrain_membership.py covers the normalization and hashing
library.)
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


au = load_module("olmo2_nonmember_audit", "olmo2_nonmember_audit.py")
mm = au.mm

K = 8


def words(n: int, rng: random.Random, prefix: str = "w") -> list[str]:
    return [f"{prefix}{rng.randrange(10**6)}" for _ in range(n)]


def norm(text: str) -> bytes:
    return mm.normalize(text)


# --------------------------------------------------------------------------
# shingle spans (same filter on both sides; span includes the last word)
# --------------------------------------------------------------------------


class ShingleSpanTests(unittest.TestCase):
    def test_span_includes_last_word(self):
        """The old code measured to the START of the k-th word, silently
        excluding its characters. 8 five-letter words with single spaces span
        8*5 + 7 = 47 characters, not 42."""
        data = b" ".join([b"abcde"] * 8)
        h, spans = au.shingles_with_spans(data, K)
        self.assertEqual(len(spans), 1)
        self.assertEqual(int(spans[0]), 47)

    def test_span_matches_python_reference_on_random_text(self):
        rng = random.Random(3)
        for _ in range(40):
            n = rng.randrange(K, 60)
            ws = [rng.choice("abcdefghij") * rng.randrange(1, 9) for _ in range(n)]
            data = " ".join(ws).encode()
            _, spans = au.shingles_with_spans(data, K)
            starts, at = [], 0
            for w in ws:
                starts.append(at)
                at += len(w) + 1
            ends = [s + len(w) for s, w in zip(starts, ws)]
            ref = [ends[i + K - 1] - starts[i] for i in range(n - K + 1)]
            self.assertEqual(spans.tolist(), ref)

    def test_filter_is_symmetric_by_construction(self):
        """Both sides must call the SAME function with the SAME threshold.
        A candidate-only filter gave 'a b c d e f g h' one corpus shingle
        and zero candidate shingles."""
        data = b"a b c d e f g h i j"
        cand_h, cand_pos, _ = au.shingles_filtered(data, K, 20)
        corp_h, corp_pos, _ = au.shingles_filtered(data, K, 20)
        self.assertTrue(np.array_equal(cand_h, corp_h))
        self.assertTrue(np.array_equal(cand_pos, corp_pos))
        # and at this span every shingle is junk, so both are empty
        self.assertEqual(len(cand_h), 0)

    def test_filter_off_keeps_everything(self):
        data = b"a b c d e f g h i j"
        h, pos, kept = au.shingles_filtered(data, K, 0)
        self.assertEqual(len(h), 3)
        self.assertEqual(pos.tolist(), [0, 1, 2])
        self.assertTrue(kept.all())

    def test_hashes_agree_with_unfiltered_reference(self):
        rng = random.Random(9)
        data = " ".join(words(200, rng)).encode()
        ref = mm.shingle_hashes(data, K)
        h, pos, kept = au.shingles_filtered(data, K, 0)
        self.assertTrue(np.array_equal(h, ref))


# --------------------------------------------------------------------------
# contiguous-run measurement
# --------------------------------------------------------------------------


class LongestRunTests(unittest.TestCase):
    def test_single_contiguous_block(self):
        # matched shingle positions 10..29 inclusive -> 20 shingles -> 27 words
        pos = np.arange(10, 30)
        kept = np.ones(100, dtype=bool)
        run_sh, run_words, start = au.longest_run(pos, kept, K)
        self.assertEqual((run_sh, run_words, start), (20, 27, 10))

    def test_two_blocks_reports_longer(self):
        pos = np.array(list(range(0, 5)) + list(range(40, 52)))
        kept = np.ones(100, dtype=bool)
        run_sh, run_words, start = au.longest_run(pos, kept, K)
        self.assertEqual((run_sh, run_words, start), (12, 19, 40))

    def test_scattered_matches_are_not_a_run(self):
        pos = np.array([3, 17, 41, 88])
        kept = np.ones(100, dtype=bool)
        run_sh, run_words, _ = au.longest_run(pos, kept, K)
        self.assertEqual(run_sh, 1)
        self.assertEqual(run_words, K)

    def test_filtered_position_bridges_a_run(self):
        """A shingle the span filter removed is NOT evidence of a mismatch;
        the run must continue across it, or maths papers full of short tokens
        get their reused sections chopped into pieces below threshold."""
        pos = np.array([10, 11, 12, 14, 15, 16])   # 13 is missing ...
        kept = np.ones(100, dtype=bool)
        kept[13] = False                          # ... because it was filtered
        run_sh, run_words, start = au.longest_run(pos, kept, K)
        self.assertEqual(start, 10)
        self.assertEqual(run_words, 16 - 10 + K)  # first..last matched + k-1

    def test_unmatched_kept_position_breaks_a_run(self):
        pos = np.array([10, 11, 12, 14, 15, 16])
        kept = np.ones(100, dtype=bool)            # 13 was kept but did not match
        run_sh, run_words, start = au.longest_run(pos, kept, K)
        self.assertEqual(run_sh, 3)

    def test_empty(self):
        self.assertEqual(au.longest_run(np.array([], dtype=np.int64),
                                        np.ones(5, dtype=bool), K), (0, 0, -1))


# --------------------------------------------------------------------------
# end-to-end matching on planted text
# --------------------------------------------------------------------------


def make_table(docs: list[bytes], k: int, min_chars: int):
    fps, ows = [], []
    for i, d in enumerate(docs):
        h, _, _ = au.shingles_filtered(d, k, min_chars)
        h = np.unique(h)
        fps.append(h)
        ows.append(np.full(len(h), i, dtype=np.uint32))
    fp = np.concatenate(fps)
    ow = np.concatenate(ows)
    order = np.argsort(fp, kind="stable")
    return fp[order], ow[order]


class PlantedMatchTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(11)
        self.rng = rng
        self.cand_words = words(300, rng, "cand")
        self.other_words = words(300, rng, "other")
        self.cand = " ".join(self.cand_words).encode()
        self.other = " ".join(self.other_words).encode()
        self.fps, self.owners = make_table([self.cand, self.other], K, 20)
        self.bitmap, _ = au.build_bitmap(self.fps, bits=20)
        self.mask = np.uint64((1 << 20) - 1)

    def match(self, record: bytes, floor: int = 1):
        return au.match_record(
            record, K, 20, self.fps, self.owners, self.bitmap, self.mask, floor
        )

    def test_planted_block_gives_correct_run_length(self):
        block = self.cand_words[100:140]            # 40 words verbatim
        filler = words(200, self.rng, "x")
        record = " ".join(filler[:100] + block + filler[100:]).encode()
        res = self.match(record)
        self.assertIn(0, res)
        m = res[0]
        self.assertEqual(m["run_words"], 40)
        self.assertEqual(m["run_shingles"], 40 - K + 1)
        self.assertEqual(m["shingles"], 40 - K + 1)
        self.assertNotIn(1, res)                    # 'other' did not match

    def test_scattered_reuse_has_large_count_but_short_run(self):
        """This is the case a span-only rule misses and a count-only rule
        catches: many shared phrases, no long contiguous block."""
        pieces = []
        for s in range(0, 300, 20):
            pieces += self.cand_words[s:s + K]     # exactly one 8-gram each
            pieces += words(5, self.rng, "gap")
        record = " ".join(pieces).encode()
        res = self.match(record)
        m = res[0]
        self.assertGreaterEqual(m["shingles"], 10)
        self.assertEqual(m["run_shingles"], 1)

    def test_repeated_phrase_counts_once(self):
        """Distinct shared shingles, not matched positions: a corpus record
        that repeats one phrase 50 times shares ONE shingle set."""
        phrase = self.cand_words[0:K]
        record = " ".join(phrase * 50).encode()
        res = self.match(record)
        self.assertEqual(res[0]["shingles"], 1)

    def test_run_start_position_points_at_the_block(self):
        block = self.cand_words[50:80]
        record = " ".join(words(60, self.rng, "p") + block).encode()
        res = self.match(record)
        self.assertEqual(res[0]["word"], 60)

    def test_floor_suppresses_incidental_hits(self):
        record = " ".join(self.cand_words[0:K] + words(50, self.rng, "z")).encode()
        self.assertIn(0, self.match(record, floor=1))
        self.assertNotIn(0, self.match(record, floor=2))

    def test_bitmap_never_drops_a_real_fingerprint(self):
        for bits in (12, 16, 20):
            bm, _ = au.build_bitmap(self.fps, bits=bits)
            mask = np.uint64((1 << bits) - 1)
            keep = np.empty(len(self.fps), dtype=np.int64)
            n = au._bitmap_survivors(np.ascontiguousarray(self.fps), bm, mask, keep)
            self.assertEqual(n, len(self.fps), f"bits={bits}")


# --------------------------------------------------------------------------
# fail-open paths
# --------------------------------------------------------------------------


class FailOpenTests(unittest.TestCase):
    def test_zero_shards_is_an_error_not_a_clean_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data" / "dclm").mkdir(parents=True)
            with self.assertRaises(SystemExit) as cm:
                au.require_shards(mm.discover_shards(root, ["dclm"]), root, ["dclm"])
            self.assertIn("zero shards", str(cm.exception).lower())

    def test_record_counters_are_kept(self):
        lines = [
            json.dumps({"text": "the quick brown fox jumps over the lazy dog many times today"}),
            "not json at all {",
            json.dumps({"no_text_field": 1}),
            json.dumps({"text": 42}),
        ]
        stats = au.classify_records(lines)
        self.assertEqual(stats["rows"], 4)
        self.assertEqual(stats["parse_failed"], 1)
        self.assertEqual(stats["missing_text"], 2)
        self.assertEqual(stats["usable"], 1)

    def test_shard_that_parsed_nothing_is_flagged(self):
        stats = {"rows": 1000, "parse_failed": 400, "missing_text": 600, "usable": 0}
        self.assertTrue(au.shard_is_suspect(stats))
        ok = {"rows": 1000, "parse_failed": 2, "missing_text": 1, "usable": 997}
        self.assertFalse(au.shard_is_suspect(ok))


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


class ProvenanceTests(unittest.TestCase):
    def test_file_digest_is_stable_and_sensitive(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.csv.gz"
            with gzip.open(p, "wt") as fh:
                fh.write("doc_id\na\nb\n")
            d1 = au.file_digest(p)
            d2 = au.file_digest(p)
            self.assertEqual(d1, d2)
            self.assertEqual(len(d1), 64)
            with gzip.open(p, "wt") as fh:
                fh.write("doc_id\na\nc\n")
            self.assertNotEqual(d1, au.file_digest(p))

    def test_provenance_mismatch_is_refused(self):
        meta = {"candidates_sha256": "a" * 64, "fingerprints_sha256": "b" * 64}
        with self.assertRaises(SystemExit):
            au.check_provenance(meta, candidates_sha256="c" * 64,
                                fingerprints_sha256="b" * 64)
        # identical digests pass silently
        au.check_provenance(meta, candidates_sha256="a" * 64,
                            fingerprints_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
