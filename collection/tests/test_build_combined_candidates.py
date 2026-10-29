"""The combined manifest's row order is load-bearing: scan hit records refer to
candidates by integer index into the parser_status=="ok" rows, in order, and
index_ids.json is the persisted mapping. These tests pin that contract."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "build_combined_candidates.py"


def write_pool(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


class CombinedManifestTests(unittest.TestCase):
    def build(self, with_se: bool) -> tuple[list[dict], list[str], str]:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            write_pool(td / "arxiv.csv.gz", [
                {"doc_id": "2505.00001", "month": "2025-05", "category_group": "cs",
                 "title": "A", "abstract": "aa", "latex_status": "ok",
                 "latex_text_path": "/x/a.latex.zst", "pdf_status": "ok",
                 "pdf_text_path": "/x/a.pdf.zst"},
                {"doc_id": "2505.00002", "month": "2025-05", "category_group": "math",
                 "title": "B", "abstract": "bb", "latex_status": "ok",
                 "latex_text_path": "/x/b.latex.zst", "pdf_status": "grobid_empty",
                 "pdf_text_path": ""},
            ])
            write_pool(td / "wiki.csv.gz", [
                {"doc_id": "wikiA:11", "group": "A", "month": "2025-06", "title": "W1",
                 "text_path": "/x/w1.zst", "parser_status": "ok"},
                {"doc_id": "wikiB:22", "group": "B", "month": "2025-07", "title": "W2",
                 "text_path": "/x/w2.zst", "parser_status": "ok"},
            ])
            write_pool(td / "se.csv.gz", [
                {"doc_id": "se:math_stackexchange_com-1-2", "forum": "math_stackexchange_com",
                 "month": "2025-08", "text_path": "/x/s1.zst", "parser_status": "ok"},
                {"doc_id": "se:math_stackexchange_com-3-4", "forum": "math_stackexchange_com",
                 "month": "2025-08", "text_path": "/x/s2.zst", "parser_status": "failed"},
            ])
            cmd = [sys.executable, str(SCRIPT), "--arxiv-pool", str(td / "arxiv.csv.gz"),
                   "--wiki-pool", str(td / "wiki.csv.gz"), "--out-dir", str(td / "out")]
            if with_se:
                cmd += ["--se-pool", str(td / "se.csv.gz")]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            rows = list(csv.DictReader(gzip.open(td / "out" / "combined.csv.gz", "rt", newline="")))
            ids = json.loads((td / "out" / "index_ids.json").read_text())
            return rows, ids, proc.stdout

    def test_index_ids_match_rows_exactly_in_order(self):
        rows, ids, _ = self.build(with_se=True)
        self.assertEqual([r["doc_id"] for r in rows], ids)
        self.assertTrue(all(r["parser_status"] == "ok" for r in rows))

    def test_arxiv_gets_one_row_per_available_representation(self):
        rows, ids, _ = self.build(with_se=False)
        self.assertEqual(ids[:3], ["2505.00001#latex", "2505.00001#pdf", "2505.00002#latex"])
        self.assertNotIn("2505.00002#pdf", ids)  # GROBID failed -> no PDF row

    def test_wikipedia_rows_carry_provisional_group(self):
        rows, _, _ = self.build(with_se=False)
        wiki = [r for r in rows if r["source"] == "wikipedia"]
        self.assertEqual([r["group"] for r in wiki], ["wikiA", "wikiB"])

    def test_stackexchange_rows_optional_and_filtered_on_status(self):
        rows_no, ids_no, _ = self.build(with_se=False)
        rows_se, ids_se, out = self.build(with_se=True)
        self.assertEqual(len(ids_se), len(ids_no) + 1)      # one ok SE row, one failed dropped
        se = [r for r in rows_se if r["source"] == "stackexchange"]
        self.assertEqual(se[0]["doc_id"], "se:math_stackexchange_com-1-2")
        self.assertEqual(se[0]["group"], "se")
        self.assertEqual(se[0]["category_group"], "math_stackexchange_com")
        self.assertIn("stackexchange (Q,A) pairs:       1", out)


if __name__ == "__main__":
    unittest.main()
