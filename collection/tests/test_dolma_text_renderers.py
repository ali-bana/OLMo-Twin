"""Tests for the two renderers that reproduce Dolma's stored text.

Every StackExchange rule below was measured against (question, answer) pairs
present in both our archive.org dump and dolmino with identical LastEditDate
(519 pairs; 519/519 identical after normalization, 403/519 byte-identical, the
rest differing only in blank-line count at the title/Q/A joins). Every
Wikipedia rule was measured on pages unchanged since Dolma's 2023 snapshot
(37/40 byte-identical with wikiextractor 3.0.8 --no-templates).
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sdt = load_module("stackexchange_dolma_text", "stackexchange_dolma_text.py")
wdt = load_module("wikipedia_dolma_text", "wikipedia_dolma_text.py")


class StackExchangeRenderingTests(unittest.TestCase):
    def test_paragraphs_join_with_blank_line(self):
        self.assertEqual(sdt.html_to_text("<p>One.</p>\n<p>Two.</p>"), "One.\n\nTwo.")

    def test_newlines_inside_text_collapse_to_space(self):
        # measured: '$$</span>\nThis option' -> '$$ This option'
        self.assertEqual(sdt.html_to_text("<p>a\nb   c</p>"), "a b c")

    def test_br_makes_a_newline(self):
        self.assertEqual(sdt.html_to_text("<p>a<br />\nb</p>"), "a\nb")

    def test_inline_code_is_plain(self):
        self.assertEqual(sdt.html_to_text("<p>use <code>ch</code> here</p>"), "use ch here")

    def test_pre_block_verbatim_no_indent_no_trailing_newline(self):
        html = "<p>Code:</p>\n<pre><code>qc = QuantumCircuit(3)\n    qc.ch(0,1)\n</code></pre>\n<p>Done.</p>"
        self.assertEqual(sdt.html_to_text(html),
                         "Code:\n\nqc = QuantumCircuit(3)\n    qc.ch(0,1)\n\nDone.")

    def test_tight_bullet_list(self):
        html = "<p>Items:</p>\n<ul>\n<li>alpha</li>\n<li>beta</li>\n</ul>"
        self.assertEqual(sdt.html_to_text(html), "Items:\n\n  • alpha\n  • beta")

    def test_tight_numbered_list(self):
        html = "<p>Steps:</p>\n<ol>\n<li>first</li>\n<li>second</li>\n</ol>"
        self.assertEqual(sdt.html_to_text(html), "Steps:\n\n  1. first\n  2. second")

    def test_loose_list_items_separated_by_blank_line(self):
        html = "<p>x</p><ol>\n<li><p>Both inputs are solutions.</p>\n</li>\n<li><p>Neither.</p>\n</li>\n</ol>"
        self.assertEqual(sdt.html_to_text(html),
                         "x\n\n  1. Both inputs are solutions.\n\n  2. Neither.")

    def test_br_continuation_inside_list_item_indents_four(self):
        # measured: '  2. Both inputs are not solutions.\n    No matter how many...'
        html = "<p>x</p><ol>\n<li><p>Both.<br />\nNo matter.</p>\n</li>\n</ol>"
        self.assertEqual(sdt.html_to_text(html), "x\n\n  1. Both.\n    No matter.")

    def test_nested_list_follows_after_single_newline_indented(self):
        # measured: '  • On the other hand we know that:\n    1. Adiabatic quan...'
        html = "<p>x</p><ul>\n<li>On the other hand:<ol>\n<li>Adiabatic</li>\n</ol>\n</li>\n</ul>"
        self.assertEqual(sdt.html_to_text(html), "x\n\n  • On the other hand:\n    1. Adiabatic")

    def test_links_become_anchor_text_and_images_alt_text(self):
        html = '<p>see <a href="https://x">here</a>.</p>\n<p><a href="i.png"><img src="i.png" alt="enter image description here"></a></p>'
        self.assertEqual(sdt.html_to_text(html), "see here.\n\nenter image description here")

    def test_emphasis_is_plain(self):
        self.assertEqual(sdt.html_to_text("<p><em>does</em> <strong>not</strong> <sup>2</sup></p>"), "does not 2")

    def test_blockquote_has_no_prefix(self):
        html = "<p>Based on Wikipedia</p>\n<blockquote>\n<p>In physics, the theorem states.</p>\n</blockquote>"
        self.assertEqual(sdt.html_to_text(html), "Based on Wikipedia\n\nIn physics, the theorem states.")

    def test_body_is_left_stripped_so_a_leading_list_loses_first_indent_only(self):
        # measured: bodies opening with a list read '1. ...' then '  2. ...'
        html = "<ol>\n<li>Yes, correct.</li>\n<li>No.</li>\n</ol>"
        self.assertEqual(sdt.html_to_text(html), "1. Yes, correct.\n  2. No.")

    def test_record_joins_title_question_answer_with_three_newlines(self):
        out = sdt.record_text("A title?", "<p>Q body.</p>", "<p>A body.</p>")
        self.assertEqual(out, "A title?\n\n\nQ body.\n\n\nA body.")


class WikipediaWrapperTests(unittest.TestCase):
    def test_version_pin_is_enforced(self):
        with mock.patch("importlib.metadata.version", return_value="3.1.0"):
            with self.assertRaises(SystemExit) as cm:
                wdt.check_wikiextractor()
            self.assertIn("3.0.8", str(cm.exception))
        with mock.patch("importlib.metadata.version", return_value="3.0.8"):
            self.assertEqual(wdt.check_wikiextractor(), "3.0.8")

    def test_page_blocks_are_keyed_by_page_id(self):
        blob = ("  <page>\n    <title>A</title>\n    <ns>0</ns>\n    <id>12</id>\n"
                "    <revision>\n      <id>99</id>\n    </revision>\n  </page>\n"
                "  <page>\n    <title>B</title>\n    <ns>0</ns>\n    <id>34</id>\n  </page>\n")
        blocks = wdt.page_xml_blocks(blob)
        self.assertEqual(set(blocks), {"12", "34"})
        self.assertTrue(blocks["12"].startswith("  <page>\n"))

    @unittest.skipUnless(
        importlib.util.find_spec("wikiextractor") is not None, "wikiextractor not installed")
    def test_extract_texts_reproduces_dolma_conventions(self):
        """Heading rendered as 'History.', italics as "quoted", template dropped."""
        from importlib.metadata import version
        if version("wikiextractor") != wdt.REQUIRED_WIKIEXTRACTOR:
            self.skipTest("wrong wikiextractor version for the fidelity contract")
        header = (
            '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/" version="0.10" xml:lang="en">\n'
            "  <siteinfo>\n    <sitename>Wikipedia</sitename>\n    <dbname>enwiki</dbname>\n"
            "    <base>https://en.wikipedia.org/wiki/Main_Page</base>\n    <case>first-letter</case>\n"
            "    <namespaces>\n      <namespace key=\"0\" case=\"first-letter\" />\n    </namespaces>\n"
            "  </siteinfo>\n"
        )
        page = (
            "  <page>\n    <title>Test Ship</title>\n    <ns>0</ns>\n    <id>777</id>\n"
            "    <revision>\n      <id>1</id>\n      <timestamp>2023-01-01T00:00:00Z</timestamp>\n"
            "      <format>text/x-wiki</format>\n"
            "      <text xml:space=\"preserve\">'''Test Ship''' is a ''schooner'' built in 1895. She is {{convert|219|ft|m}} in length.\n\n"
            "== History ==\nShe sailed from [[Grays Harbor]] to San Francisco.\n</text>\n"
            "    </revision>\n  </page>\n"
        )
        got = wdt.extract_texts([page], header)
        self.assertIn("777", got)
        text = got["777"]
        self.assertTrue(text.startswith("Test Ship\n\n"))
        self.assertIn('"schooner"', text)          # italics -> quotes
        self.assertIn("History.", text)             # heading with trailing period
        self.assertIn("She is in length.", text)    # template dropped, gap left
        self.assertIn("Grays Harbor to San Francisco", text)  # link -> anchor text


if __name__ == "__main__":
    unittest.main()
