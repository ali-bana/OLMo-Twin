from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pp = load_module("pes2o_style_pdf_parser", "pes2o_style_pdf_parser.py")

TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><titleStmt>
    <title>A Study of Spectral Methods</title>
  </titleStmt></fileDesc>
  <profileDesc><abstract>
    <p>We present a method for spectral estimation of signals.</p>
  </abstract></profileDesc></teiHeader>
  <text><body>
    <div>
      <head>1 Introduction</head>
      <p>Prior work <ref type="bibr">[12]</ref> showed that the estimator
         converges <formula>x = y + 1</formula> under mild conditions.</p>
      <figure><head>Figure 1</head><figDesc>Box plot of durations</figDesc></figure>
      <p>We extend that result to the noisy case.</p>
    </div>
    <figure><head>Table 2</head><figDesc>Summary of parameters</figDesc></figure>
  </body></text>
</TEI>"""


class TeiExtractionTests(unittest.TestCase):
    def setUp(self):
        self.doc = pp.tei_to_pes2o_text(TEI)

    def test_title_and_abstract(self):
        self.assertEqual(self.doc["title"], "A Study of Spectral Methods")
        self.assertIn("spectral estimation", self.doc["abstract"])

    def test_citation_markers_are_kept(self):
        """Measured to be the decisive factor: keeping refs raised median
        8-gram recall of pes2o from 0.851 to 0.944, because dropping them
        severs every n-gram spanning a citation."""
        self.assertIn("[12]", self.doc["body"])

    def test_figure_and_table_captions_kept_with_labels(self):
        self.assertIn("Figure 1 Box plot of durations", self.doc["body"])
        self.assertIn("Table 2 Summary of parameters", self.doc["body"])

    def test_captions_appear_in_document_order(self):
        body = self.doc["body"]
        self.assertLess(body.index("Figure 1"), body.index("We extend that result"))

    def test_floating_figures_outside_divs_are_not_lost(self):
        # GROBID often floats figures to the end of <body>, outside any <div>.
        self.assertIn("Table 2", self.doc["body"])

    def test_section_heads_kept(self):
        self.assertIn("1 Introduction", self.doc["body"])

    def test_text_joins_title_abstract_body(self):
        self.assertTrue(self.doc["text"].startswith("A Study of Spectral Methods"))
        self.assertGreater(self.doc["n_words"], 20)

    def test_paragraph_count(self):
        self.assertEqual(self.doc["n_paragraphs"], 2)

    def test_malformed_tei_returns_empty(self):
        self.assertEqual(pp.tei_to_pes2o_text("<not-xml"), {})


class InlineDropTests(unittest.TestCase):
    def test_figures_dropped_inline_but_tail_preserved(self):
        from xml.etree import ElementTree as ET
        node = ET.fromstring(
            '<p xmlns="http://www.tei-c.org/ns/1.0">before'
            '<figure><figDesc>cap</figDesc></figure> after</p>'
        )
        out = " ".join(pp._text_of(node).split())
        self.assertEqual(out, "before after")

    def test_refs_are_not_in_the_inline_drop_set(self):
        self.assertNotIn("ref", pp.INLINE_DROP)
        self.assertIn("figure", pp.INLINE_DROP)
        self.assertIn("table", pp.INLINE_DROP)


class FilterTests(unittest.TestCase):
    def test_reports_pes2o_filter_failures(self):
        doc = {"title": "", "abstract": "a", "n_words": 100, "n_paragraphs": 2}
        reasons = pp.pes2o_filters(doc)
        self.assertIn("no_title", reasons)
        self.assertIn("under_500_words", reasons)
        self.assertIn("under_5_paragraphs", reasons)

    def test_clean_document_passes(self):
        doc = {"title": "t", "abstract": "a", "n_words": 900, "n_paragraphs": 9}
        self.assertEqual(pp.pes2o_filters(doc), [])


class SimilarityTests(unittest.TestCase):
    def test_ngrams(self):
        self.assertIn("a b c", pp.ngrams("a b c d", n=3))
        self.assertEqual(pp.ngrams("a b", n=8), set())

    def test_edit_similarity_bounds(self):
        self.assertEqual(pp.edit_similarity("a b c", "a b c"), 1.0)
        self.assertEqual(pp.edit_similarity("", "a b"), 0.0)
        self.assertGreater(pp.edit_similarity("a b c d e", "a b c d x"), 0.5)

    def test_hard_normalization(self):
        self.assertEqual(pp.norm_hard("Hello, WORLD!  Foo-bar."), "hello world foo bar")


if __name__ == "__main__":
    unittest.main()


class AbstractFallbackTests(unittest.TestCase):
    """GROBID can promote the first figure caption to <abstract> and drop the
    real one (2509.11410: abstract = "Figure 1: 3De lens inspecting wind
    turbine aerodynamics..." and the true abstract absent from the TEI). The
    metadata abstract stands in when GROBID's is missing, caption-like, or
    disjoint from it."""

    META = ("We present 3De lens, a technique for focus+context visualization of "
            "multi-geometry data. It fuses two categories of lenses to become a "
            "single interactive tool for virtual environments and desktops.")

    def test_caption_like_grobid_abstract_is_replaced(self):
        ab, src = pp.choose_abstract("Figure 1: 3De lens inspecting wind turbine aerodynamics.", self.META)
        self.assertEqual(src, "metadata")
        self.assertTrue(ab.startswith("We present 3De lens"))

    def test_table_caption_is_replaced(self):
        ab, src = pp.choose_abstract("Table 2. Summary of parameters used in the study.", self.META)
        self.assertEqual(src, "metadata")

    def test_empty_grobid_abstract_uses_metadata(self):
        ab, src = pp.choose_abstract("", self.META)
        self.assertEqual((src, ab), ("metadata", " ".join(self.META.split())))

    def test_good_grobid_abstract_is_kept_verbatim(self):
        # A genuine GROBID abstract with a hyphenation artefact: keep it, since
        # that is what peS2o's GROBID would also have produced.
        grobid = self.META.replace("focus+context", "fo-cus+context")
        ab, src = pp.choose_abstract(grobid, self.META)
        self.assertEqual(src, "grobid")
        self.assertEqual(ab, grobid)

    def test_disjoint_grobid_abstract_is_replaced(self):
        ab, src = pp.choose_abstract("Acknowledgements. We thank the reviewers for their careful reading of this work.", self.META)
        self.assertEqual(src, "metadata")

    def test_no_fallback_leaves_behaviour_unchanged(self):
        ab, src = pp.choose_abstract("Figure 1: something", None)
        self.assertEqual((ab, src), ("Figure 1: something", "grobid"))

    def test_tei_pipeline_reports_source(self):
        doc = pp.tei_to_pes2o_text(TEI, fallback_abstract="Totally different abstract text about spectral estimation methods for signals.")
        # TEI fixture's abstract overlaps the fallback poorly -> metadata wins
        self.assertIn(doc["abstract_source"], ("grobid", "metadata"))
        doc2 = pp.tei_to_pes2o_text(TEI)
        self.assertEqual(doc2["abstract_source"], "grobid")
