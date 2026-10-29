from __future__ import annotations

import importlib.util
import random
import sys
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


mm = load_module(
    "olmo2_midtrain_membership", "olmo2_midtrain_membership.py"
)


class NormalizationTests(unittest.TestCase):
    def test_folds_case_punctuation_and_whitespace(self):
        self.assertEqual(
            mm.normalize("Hello,   WORLD!\n\tFoo-bar."), b"hello world foo bar"
        )

    def test_punctuation_variants_agree(self):
        # Cleaned arXiv text vs. the same passage as rendered on the web:
        # differing punctuation, spacing, and dashes must not hide a match.
        latex = "We show that alpha_i --- the rate --- is O(n log n)."
        web = "We show that alpha-i,  the  rate:  is O(n log n)!"
        self.assertEqual(mm.normalize(latex), mm.normalize(web))

    def test_accents_fold_without_joining_words(self):
        # "ö" must fold to "o" (combining mark removed), while the en dash must
        # become a separator. Deleting non-ASCII instead would yield
        # "schrodingerpoisson" and cause a false negative.
        self.assertEqual(mm.normalize("Schrödinger–Poisson"), b"schrodinger poisson")

    def test_non_latin_scripts_become_separators(self):
        self.assertEqual(mm.normalize("alpha α beta"), b"alpha beta")
        self.assertEqual(mm.normalize("data中文set"), b"data set")

    def test_empty_and_punctuation_only(self):
        self.assertEqual(mm.normalize(""), b"")
        self.assertEqual(mm.normalize("!!! ??? ..."), b"")

    def test_norm_hash_is_stable_and_case_insensitive(self):
        self.assertEqual(mm.norm_hash("Foo Bar"), mm.norm_hash("foo   bar!"))
        self.assertNotEqual(mm.norm_hash("foo bar"), mm.norm_hash("foo baz"))


class RollingHashTests(unittest.TestCase):
    def test_matches_independent_recomputation(self):
        data = b"the quick brown fox jumps over the lazy dog many times over"
        width = 8
        got = mm.window_hashes(data, width)
        self.assertEqual(len(got), len(data) - width + 1)
        for start in range(len(got)):
            expected = 0
            power = 1
            for offset in range(width):
                expected = (expected + data[start + offset] * power) & mm.MASK64
                power = (power * mm.HASH_BASE) & mm.MASK64
            self.assertEqual(int(got[start]), expected, f"mismatch at {start}")

    def test_identical_windows_hash_identically(self):
        data = b"abcdefgh" + b"xxxx" + b"abcdefgh"
        hashes = mm.window_hashes(data, 8)
        self.assertEqual(int(hashes[0]), int(hashes[12]))

    def test_short_input_returns_empty(self):
        self.assertEqual(len(mm.window_hashes(b"abc", 8)), 0)


class WinnowGuaranteeTests(unittest.TestCase):
    """Any shared run of >= width + winnow - 1 WORDS must share a fingerprint.

    This is the property the no-false-negative claim rests on, so it is tested
    against randomized inputs rather than one example.
    """

    @staticmethod
    def _words(rng, n):
        return " ".join(
            "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 9)))
            for _ in range(n)
        )

    def test_shared_long_word_run_always_shares_a_fingerprint(self):
        rng = random.Random(20260908)
        k, size = 5, 8
        threshold = k + size - 1
        for trial in range(80):
            shared = self._words(rng, threshold)
            a = mm.normalize(
                self._words(rng, rng.randint(0, 40)) + " " + shared + " " + self._words(rng, rng.randint(0, 40))
            )
            b = mm.normalize(
                self._words(rng, rng.randint(0, 40)) + " " + shared + " " + self._words(rng, rng.randint(0, 40))
            )
            fa = set(int(x) for x in mm.fingerprint_text(a, k, size))
            fb = set(int(x) for x in mm.fingerprint_text(b, k, size))
            self.assertTrue(fa & fb, f"trial {trial}: shared {threshold}-word run missed")

    def test_density_is_bounded(self):
        rng = random.Random(7)
        data = mm.normalize(self._words(rng, 20000))
        sel = mm.fingerprint_text(data, 13, 32)
        n_words = len(data.split(b" "))
        self.assertLess(len(sel) / n_words, 6.0 / (32 + 1))
        self.assertGreater(len(sel), 0)

    def test_unrelated_documents_do_not_collide(self):
        rng = random.Random(11)
        fa = set(int(x) for x in mm.fingerprint_text(mm.normalize(self._words(rng, 6000)), 13, 32))
        fb = set(int(x) for x in mm.fingerprint_text(mm.normalize(self._words(rng, 6000)), 13, 32))
        self.assertEqual(len(fa & fb), 0)

    def test_verbatim_paper_inside_a_larger_document(self):
        rng = random.Random(3)
        norm = mm.normalize(self._words(rng, 4000))
        corpus = mm.normalize("unrelated preamble text " * 60) + b" " + norm + b" tail"
        fp_paper = set(int(x) for x in mm.fingerprint_text(norm, 13, 32))
        fp_corpus = set(int(x) for x in mm.fingerprint_text(corpus, 13, 32))
        self.assertTrue(fp_paper & fp_corpus)
        # A verbatim full-paper copy must share most of its fingerprints.
        self.assertGreater(len(fp_paper & fp_corpus) / len(fp_paper), 0.9)

    def test_shingles_are_shift_invariant(self):
        # The same word run at different offsets must hash identically.
        rng = random.Random(5)
        run = self._words(rng, 30)
        a = mm.shingle_hashes(mm.normalize("alpha beta " + run), 13)
        b = mm.shingle_hashes(mm.normalize("gamma delta epsilon " + run), 13)
        self.assertTrue(set(int(x) for x in a) & set(int(x) for x in b))

    def test_too_few_words_returns_empty(self):
        self.assertEqual(len(mm.fingerprint_text(mm.normalize("only three words"), 13, 32)), 0)


class ArxivIdExtractionTests(unittest.TestCase):
    def test_new_style_ids(self):
        blob = b'see arXiv:2301.11760 and https://arxiv.org/abs/2201.00001v3 here'
        found = [m.group(1).decode() for m in mm.ARXIV_NEW_RE.finditer(blob)]
        self.assertIn("2301.11760", found)
        self.assertIn("2201.00001", found)

    def test_legacy_style_ids(self):
        blob = b"older work astro-ph/0701001v2 and hep-th/9901001"
        found = [m.group(1).decode() for m in mm.ARXIV_OLD_RE.finditer(blob)]
        self.assertIn("astro-ph/0701001", found)
        self.assertIn("hep-th/9901001", found)

    def test_does_not_match_version_numbers_or_decimals(self):
        blob = b"value 1234.5678901234 and v1.2345 and 12345.678"
        found = [m.group(1).decode() for m in mm.ARXIV_NEW_RE.finditer(blob)]
        self.assertEqual(found, [])


class MonthHelperTests(unittest.TestCase):
    def test_yymm_new_style(self):
        self.assertEqual(mm.yymm_of("2301.11760"), "2301")
        self.assertEqual(mm.yymm_of("2301.11760v2"), "2301")

    def test_yymm_legacy_style(self):
        self.assertEqual(mm.yymm_of("astro-ph/0701001"), "0701")

    def test_yymm_rejects_junk(self):
        self.assertIsNone(mm.yymm_of("not-an-id"))
        self.assertIsNone(mm.yymm_of("abc.defg"))

    def test_month_key_spans_the_century_boundary(self):
        self.assertEqual(mm.month_key("9108"), "1991-08")
        self.assertEqual(mm.month_key("0701"), "2007-01")
        self.assertEqual(mm.month_key("2302"), "2023-02")

    def test_month_key_ordering_is_lexicographic(self):
        keys = [mm.month_key(v) for v in ("9912", "0001", "2302")]
        self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()


class EndToEndTests(unittest.TestCase):
    """prepare is data-dependent, so this exercises fingerprints -> scan -> verify.

    Plants known matches of each kind in a synthetic corpus and asserts the
    verdicts, which is what pins down the no-false-negative behaviour.
    """

    def setUp(self):
        import csv as _csv
        import gzip as _gzip
        import json as _json
        import tempfile

        import zstandard as zstd

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        rng = random.Random(4242)
        letters = "abcdefghijklmnopqrstuvwxyz"

        def paper(n):
            # High-entropy tokens on purpose. A small shared vocabulary would
            # make every synthetic paper a near-duplicate of every other and
            # the fixture would test nothing.
            return " ".join(
                "".join(rng.choice(letters) for _ in range(7)) for _ in range(n)
            )

        self.papers = {}
        text_dir = root / "texts"
        text_dir.mkdir()
        rows = []
        cctx = zstd.ZstdCompressor()
        for i in range(6):
            doc_id = f"2301.{10000 + i}"
            body = paper(3000)
            norm = mm.normalize(body)
            path = text_dir / f"{doc_id}.zst"
            path.write_bytes(cctx.compress(norm))
            self.papers[doc_id] = norm
            rows.append(
                {
                    "doc_id": doc_id,
                    "yymm": "2301",
                    "month": "2023-01",
                    "primary_category": "cs.LG",
                    "title": f"A Study of {doc_id} Spectral Invariants In Lattices",
                    "abstract": f"We study {doc_id}. " + paper(80),
                    "source_path": "",
                    "text_path": str(path),
                    "parser_status": "ok",
                    "norm_bytes": len(norm),
                    "norm_sha256": "",
                }
            )
        self.rows = rows
        self.candidates = root / "candidates.csv.gz"
        with _gzip.open(self.candidates, "wt", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        ids = [r["doc_id"] for r in rows]
        comp = root / "corpus" / "data" / "synth"
        comp.mkdir(parents=True)
        recs = [
            {"id": "neg", "text": "unrelated content about gardening " * 300},
            {"id": "full-0", "text": "pre. " + self.papers[ids[0]].decode().upper() + " post."},
            {"id": "exc-1", "text": "pad " * 40 + self.papers[ids[1]].decode()[1000:1500] + " pad " * 40},
            {
                "id": "meta-2",
                "text": "unrelated body",
                "metadata": {"title": rows[2]["title"], "abstract": rows[2]["abstract"]},
            },
            {"id": "cite-3", "text": f"see arXiv:{ids[3]} for details. " * 10},
        ]
        with (comp / "synth-0000.jsonl").open("w") as fh:
            for r in recs:
                fh.write(_json.dumps(r) + "\n")
        self.ids = ids

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self):
        import argparse as ap

        fp_dir = self.root / "fp"
        mm.cmd_fingerprints(
            ap.Namespace(
                candidates=self.candidates, out_dir=fp_dir, window=13, winnow=8
            )
        )
        scan_dir = self.root / "scan"
        mm.cmd_scan(
            ap.Namespace(
                candidates=self.candidates,
                fingerprint_dir=fp_dir,
                corpus=self.root / "corpus",
                out_dir=scan_dir,
                components=None,
                workers=1,
                limit_shards=0,
                max_refs_per_hit=4,
                min_title_bytes=24,
                min_abstract_bytes=64,
                no_title_automaton=False,
                force=True,
            )
        )
        out_dir = self.root / "verdict"
        mm.cmd_verify(
            ap.Namespace(
                candidates=self.candidates,
                scan_dir=scan_dir,
                out_dir=out_dir,
                min_text_fingerprints=2,
                strong_text_fingerprints=10,
                probable_text_fingerprints=5,
            )
        )
        import csv as _csv
        import gzip as _gzip

        with _gzip.open(out_dir / "verdicts.csv.gz", "rt") as fh:
            return {r["doc_id"]: r for r in _csv.DictReader(fh)}

    def test_planted_matches_are_all_found(self):
        v = self._run()
        # Full verbatim copy, re-cased: many fingerprints in ONE corpus doc.
        self.assertEqual(v[self.ids[0]]["verdict"], "member_fulltext")
        # A 500-word excerpt far exceeds the 20-word guarantee used here.
        self.assertEqual(v[self.ids[1]]["verdict"], "member_fulltext")
        # Both structured fields matched one record.
        self.assertEqual(v[self.ids[2]]["verdict"], "member_abstract_only")
        # A bare citation is evidence of a reference, never of membership.
        self.assertEqual(v[self.ids[3]]["verdict"], "citation_only")

    def test_strength_uses_single_document_not_a_cross_document_sum(self):
        """One boilerplate phrase seen in many records must not add up to a
        membership call; strength is distinct fingerprints within one doc."""
        v = self._run()
        import json as _json
        ev = _json.loads(v[self.ids[0]]["evidence"])
        self.assertIn("best_single_doc_fingerprints", ev)
        self.assertGreaterEqual(ev["best_single_doc_fingerprints"], 10)
        # The summed figure is reported but must not be what decides.
        self.assertGreaterEqual(ev["summed_fingerprints"], ev["best_single_doc_fingerprints"])

    def test_absent_papers_report_no_evidence(self):
        v = self._run()
        for doc_id in self.ids[4:]:
            self.assertEqual(
                v[doc_id]["verdict"],
                "no_evidence",
                f"{doc_id} was never planted but was not reported absent",
            )

    def test_citation_is_not_counted_as_membership(self):
        v = self._run()
        self.assertNotIn("member", v[self.ids[3]]["verdict"])


class CombinedIdRegexTests(unittest.TestCase):
    """The scan uses ARXIV_ANY_RE for speed; it must find exactly what the two
    separate patterns find, or tier 1 silently loses recall."""

    def _combined(self, blob):
        out = []
        for m in mm.ARXIV_ANY_RE.finditer(blob):
            out.append((m.group(1) or m.group(2)).decode())
        return out

    def _separate(self, blob):
        return [m.group(1).decode() for m in mm.ARXIV_NEW_RE.finditer(blob)] + [
            m.group(1).decode() for m in mm.ARXIV_OLD_RE.finditer(blob)
        ]

    def test_agrees_with_separate_patterns(self):
        cases = [
            b"see arXiv:2301.11760v2 and astro-ph/0701001 plus 2201.00001",
            b"hep-th/9901001 alone",
            b"no identifiers here at all, just 12345.678 and v1.2",
            b'{"id":"http://arxiv.org/abs/2010.12345","text":"cf. math/0501001"}',
            b"boundary 0704.0001 and 1234.56789 and 9999.9999",
        ]
        for blob in cases:
            self.assertCountEqual(
                self._combined(blob), self._separate(blob), f"blob={blob!r}"
            )

    def test_finds_both_styles_in_one_pass(self):
        blob = b"mix of 2301.11760 and cond-mat/0211616 together"
        self.assertCountEqual(
            self._combined(blob), ["2301.11760", "cond-mat/0211616"]
        )


class NormalizeBackendEquivalenceTests(unittest.TestCase):
    """The numba path is the one that runs in production; the numpy fallback is
    what runs where numba is unavailable. They must agree exactly, or results
    would silently depend on the host."""

    def _both(self, text):
        original = mm.HAS_NUMBA
        try:
            mm.HAS_NUMBA = True
            fast = mm.normalize(text)
            mm.HAS_NUMBA = False
            slow = mm.normalize(text)
        finally:
            mm.HAS_NUMBA = original
        return fast, slow

    def test_agree_on_random_ascii(self):
        rng = random.Random(99)
        alphabet = "abcXYZ012 \t\n.,;:-_()[]{}/\\!?'\"*&^%$#@+=<>|~`"
        for _ in range(200):
            text = "".join(
                rng.choice(alphabet) for _ in range(rng.randint(0, 200))
            )
            fast, slow = self._both(text)
            self.assertEqual(fast, slow, f"disagreement on {text!r}")

    def test_agree_on_unicode_and_edge_cases(self):
        for text in [
            "",
            "   ",
            "!!!",
            "a",
            " leading and trailing ",
            "Schrödinger–Poisson",
            "alpha α beta",
            "data中文set",
            "MiXeD CaSe 123",
            "tabs\tand\nnewlines\r\nhere",
            "hyphen-ated  multiple   spaces",
        ]:
            fast, slow = self._both(text)
            self.assertEqual(fast, slow, f"disagreement on {text!r}")

    def test_no_leading_or_trailing_space_and_single_spacing(self):
        out = mm.normalize("  ...Hello,,,   World!!!  ")
        self.assertEqual(out, b"hello world")
        self.assertFalse(out.startswith(b" "))
        self.assertFalse(out.endswith(b" "))
        self.assertNotIn(b"  ", out)


class ArxivHintGateTests(unittest.TestCase):
    """Tier 1 only runs the identifier regex where the literal "arxiv" appears.
    Anything that gate would skip must not have been a real reference."""

    def test_gate_admits_real_reference_forms(self):
        for blob in [
            b"see arXiv:2301.11760 for details",
            b'{"id":"http://arxiv.org/abs/2301.11760"}',
            b"ARXIV:2301.11760",
            b"preprint at https://www.arxiv.org/pdf/2301.11760v2.pdf",
            b"published on ArXiv, id 2301.11760",
        ]:
            self.assertIsNotNone(mm.ARXIV_HINT_RE.search(blob), blob)
            found = [
                (m.group(1) or m.group(2)).decode()
                for m in mm.ARXIV_ANY_RE.finditer(blob)
            ]
            self.assertIn("2301.11760", found, blob)

    def test_gate_skips_bare_decimals(self):
        # These are exactly the records the gate is meant to skip: a bare
        # decimal is not evidence of an arXiv reference.
        for blob in [
            b'{"attributes":{"score":[[0,3558,2.125]]},"text":"price 2301.11760"}',
            b"measurement 2301.11760 units",
        ]:
            self.assertIsNone(mm.ARXIV_HINT_RE.search(blob), blob)


class MembershipKindTests(unittest.TestCase):
    """Abstract-only and full-text membership must not collapse together.

    pes2o carries title+abstract only, while candidate fingerprints cover the
    body (the cleaner strips everything before the first section). A pes2o
    match proves the model saw the abstract, not the body -- a different
    condition for any attack that scores body text.
    """

    def _verdict(self, kinds, best, strong=50, probable=10, minimum=2):
        has_title = "meta_title" in kinds
        has_abstract = "meta_abstract" in kinds
        strong_text = best >= strong
        meta_exact = has_title and has_abstract
        if meta_exact and strong_text:
            return "member_abstract_and_fulltext"
        if meta_exact:
            return "member_abstract_only"
        if strong_text:
            return "member_fulltext"
        if has_title or has_abstract:
            return "member_metadata_partial"
        if best >= probable:
            return "probable_fulltext"
        if best >= minimum:
            return "partial_text_overlap"
        if best >= 1:
            return "weak_text_overlap"
        if "title_in_text" in kinds:
            return "title_only"
        if "id" in kinds:
            return "citation_only"
        return "weak_other"

    def test_abstract_only_is_distinguished_from_fulltext(self):
        meta = {"meta_title": 1, "meta_abstract": 1}
        self.assertEqual(self._verdict(meta, 0), "member_abstract_only")
        self.assertEqual(self._verdict({}, 60), "member_fulltext")
        self.assertEqual(self._verdict(meta, 60), "member_abstract_and_fulltext")

    def test_ambiguous_band_gets_its_own_label(self):
        # 10-49 fingerprints was only ~30% corroborated on pass A, so it must
        # not be reported with the same confidence as the >=50 band.
        self.assertEqual(self._verdict({}, 10), "probable_fulltext")
        self.assertEqual(self._verdict({}, 49), "probable_fulltext")
        self.assertEqual(self._verdict({}, 50), "member_fulltext")

    def test_partial_metadata_is_weaker_than_exact(self):
        self.assertEqual(
            self._verdict({"meta_title": 1}, 0), "member_metadata_partial"
        )

    def test_citation_and_title_never_become_membership(self):
        self.assertEqual(self._verdict({"id": 3}, 0), "citation_only")
        self.assertEqual(self._verdict({"title_in_text": 2}, 0), "title_only")

    def test_text_grading_thresholds(self):
        self.assertEqual(self._verdict({}, 1), "weak_text_overlap")
        self.assertEqual(self._verdict({}, 5), "partial_text_overlap")
        self.assertEqual(self._verdict({}, 10), "probable_fulltext")
        self.assertEqual(self._verdict({}, 50), "member_fulltext")


class ReportFieldCompatTests(unittest.TestCase):
    """cmd_report read a field name verify no longer writes, which silently
    reported 0% recall. Guard both names so a rename cannot fail quietly."""

    def test_reads_current_and_legacy_field_names(self):
        import argparse as ap
        import csv as _csv
        import gzip as _gzip
        import json as _json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand = root / "c.csv.gz"
            with _gzip.open(cand, "wt", newline="") as fh:
                w = _csv.DictWriter(
                    fh,
                    fieldnames=["doc_id", "month", "primary_category",
                                "parser_status", "norm_bytes", "title",
                                "abstract", "text_path", "yymm", "source_path",
                                "norm_sha256"],
                )
                w.writeheader()
                for i in (0, 1):
                    w.writerow({"doc_id": f"2301.{i}", "month": "2023-01",
                                "primary_category": "cs.LG",
                                "parser_status": "ok", "norm_bytes": 1000,
                                "title": "t", "abstract": "a", "text_path": "",
                                "yymm": "2301", "source_path": "",
                                "norm_sha256": ""})
            verd = root / "v.csv.gz"
            with _gzip.open(verd, "wt", newline="") as fh:
                w = _csv.DictWriter(
                    fh, fieldnames=["doc_id", "month", "primary_category",
                                    "norm_bytes", "verdict", "evidence"])
                w.writeheader()
                # one row with the current field name, one with the legacy name
                w.writerow({"doc_id": "2301.0", "month": "2023-01",
                            "primary_category": "cs.LG", "norm_bytes": 1000,
                            "verdict": "member_abstract_and_fulltext",
                            "evidence": _json.dumps(
                                {"kinds": {"meta_title": 1, "meta_abstract": 1},
                                 "best_single_doc_fingerprints": 80})})
                w.writerow({"doc_id": "2301.1", "month": "2023-01",
                            "primary_category": "cs.LG", "norm_bytes": 1000,
                            "verdict": "member_abstract_only",
                            "evidence": _json.dumps(
                                {"kinds": {"meta_title": 1, "meta_abstract": 1},
                                 "text_fingerprints": 90})})
            out = root / "rep"
            mm.cmd_report(ap.Namespace(candidates=cand, verdicts=verd,
                                       out_dir=out, min_text_fingerprints=2,
                                       strong_text_fingerprints=50))
            summary = _json.loads((out / "report_summary.json").read_text())
            pc = summary["positive_control"]
            self.assertEqual(pc["known_positives"], 2)
            # both rows must count, whichever field name carried the value
            self.assertEqual(pc["found_by_text_tier"], 2)
            self.assertEqual(pc["found_at_threshold"], 2)
            self.assertEqual(pc["text_tier_recall"], 1.0)


class DegenerateShingleFilterTests(unittest.TestCase):
    """PDF-derived text carries toolchain artefacts that LaTeX source never had.

    Measured on the pes2o rescan: one corpus record was claimed by 387
    unrelated candidates at exactly 79 fingerprints each (shared text
    "l a t e x i t s h a 1 b a s e 6 4", from <latexit sha1_base64="...">
    markers) and another by 88 candidates at 1,525 each ("q q q q q ...").
    """

    def test_min_chars_zero_is_identical_to_unfiltered(self):
        data = b"the quick brown fox jumps over the lazy dog again and again now"
        self.assertTrue(
            np.array_equal(
                mm.fingerprint_text(data, 4, 3),
                mm.fingerprint_text_filtered(data, 4, 3, 0),
            )
        )

    def test_winnow_pick_indices_reproduce_winnow_output(self):
        rng = np.random.default_rng(7)
        for _ in range(60):
            n = int(rng.integers(1, 200))
            w = int(rng.integers(1, 25))
            h = rng.integers(0, 5, size=n, dtype=np.uint64)  # heavy ties
            picks = mm._winnow_picks_py(h, w)
            self.assertTrue(np.array_equal(mm.winnow(h, w), h[picks]))

    def test_measured_spans_separate_junk_from_prose(self):
        """Measured over 734,888 shingles of real PDF-derived candidate text:
        median span 71 chars, p1 = 32, p0.1 = 25. Junk runs sit at the floor --
        13 one-char words span exactly 25 (13 chars + 12 spaces), 13 two-char
        words span 38. Hence the 40-char default: it clears both junk forms and
        costs 3.13% of real shingles, which cannot flip a member whose match
        rests on hundreds of fingerprints.
        """
        self.assertEqual(mm.shingle_spans(b" ".join([b"q"] * 40), 13, 1)[0], 25)
        self.assertEqual(mm.shingle_spans(b" ".join([b"qq"] * 40), 13, 1)[0], 38)

    def test_latexit_marker_is_filtered_out(self):
        junk = " ".join("latexitsha1base64" * 4).encode()
        self.assertGreater(len(mm.fingerprint_text_filtered(junk, 13, 32, 0)), 0)
        self.assertEqual(len(mm.fingerprint_text_filtered(junk, 13, 32, 40)), 0)

    def test_glyph_run_is_filtered_out(self):
        self.assertEqual(
            len(mm.fingerprint_text_filtered(b" ".join([b"q"] * 80), 13, 32, 40)), 0
        )

    def test_two_char_junk_run_is_filtered_out(self):
        self.assertEqual(
            len(mm.fingerprint_text_filtered(b" ".join([b"qq"] * 80), 13, 32, 40)), 0
        )

    def test_real_prose_survives_the_filter(self):
        prose = (
            b"we present a new method for estimating the spectral density of a "
            b"stationary process under mild regularity conditions and show that "
            b"the resulting estimator converges at the optimal rate in probability"
        )
        self.assertGreater(len(mm.fingerprint_text_filtered(prose, 13, 32, 40)), 0)

    def test_filter_never_adds_fingerprints(self):
        data = b" ".join(b"word%d" % i for i in range(200))
        base = mm.fingerprint_text(data, 13, 32)
        filt = mm.fingerprint_text_filtered(data, 13, 32, 40)
        self.assertLessEqual(len(filt), len(base))
        self.assertTrue(set(filt.tolist()) <= set(base.tolist()))


class ShardSuffixTests(unittest.TestCase):
    """Stage-1 dclm ships as `.jsonl.zstd`; dolmino uses `.jsonl.zst`.

    Missing the four-letter form made discover_shards return an empty list,
    which a scan reports as "no matches" rather than an error -- a silent
    false negative over a 6.6 TB corpus.
    """

    def test_both_zstd_spellings_supported(self):
        for suf in (".jsonl.zst", ".jsonl.zstd", ".json.zstd", ".json.gz"):
            self.assertTrue(
                suf.endswith(mm.SUPPORTED_SUFFIXES), f"{suf} not discoverable"
            )

    def test_discover_finds_nested_zstd_shards(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            deep = root / "data" / "dclm" / "raw" / "a" / "b" / "c"
            deep.mkdir(parents=True)
            (deep / "shard_0001_processed.jsonl.zstd").write_bytes(b"x")
            (deep / "shard_0002_processed.jsonl.zst").write_bytes(b"x")
            (deep / "notes.txt").write_bytes(b"x")
            found = mm.discover_shards(root, ["dclm"])
            self.assertEqual(len(found), 2)
            self.assertTrue(all(s.component == "dclm" for s in found))

    def test_reader_dispatches_on_zstd(self):
        import inspect

        src = inspect.getsource(mm.iter_lines)
        self.assertIn('".zstd"', src)
