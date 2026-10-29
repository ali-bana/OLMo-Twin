#!/usr/bin/env python3
"""Turn the post-training overlap hits into the analysis exclusion list.

posttrain_impact.py writes posttrain_overlap/posttrain_hits.csv.gz: every
document in use (injection manifest or null pool) that a record of the
post-training datasets flags at the Stage-G thresholds (>= 40 contiguous words
or >= 50 distinct 8-word shingles). The scan ran after the mid-training run had
finished, so these documents stayed in the trained set; instead they and the
pairs they belong to (both arms) are excluded from every analysis, so that one
document set is used at every stage. This script writes that list as
analysis/posttrain_exclusions.json, which
analysis/injection_v2_plotstyle.py (posttrain_exclusions()) reads.

Fields:
  exclude_pair_ids  sorted non-empty pair_id of the flagged documents
  exclude_doc_ids   sorted doc_id of the flagged documents
  by_cohort         {cohort: {docs, pairs, components}}; components counts the
                    `component` column (the post-training dataset with the
                    most shared shingles)
  source, audit_date, rationale

The paper's file was assembled by hand from the same hits file; this script
reproduces its pair and document lists and its by_cohort counts exactly (the
`source` string names the layout of this repository).
"""

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

DEFAULT_HITS = paths.POSTTRAIN_OVERLAP / "posttrain_hits.csv.gz"
SOURCE = "posttrain_overlap/posttrain_hits.csv.gz"
RATIONALE = (
    "Documents whose text was found in OLMo 2 0425 1B post-training data (Tulu 3 SFT / DPO) "
    "by the same 8-word matcher and fingerprint table used for the 13 pre/mid-training "
    "components. Excluded globally so that one document set is used at every stage."
)


def build(rows: list, audit_date: str, source: str) -> dict:
    by_cohort = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        sub = [r for r in rows if r["cohort"] == cohort]
        by_cohort[cohort] = {
            "docs": len({r["doc_id"] for r in sub}),
            "pairs": len({r["pair_id"] for r in sub if r["pair_id"]}),
            "components": dict(Counter(r["component"] for r in sub)),
        }
    return {
        "source": source,
        "audit_date": audit_date,
        "rationale": RATIONALE,
        "exclude_pair_ids": sorted({r["pair_id"] for r in rows if r["pair_id"]}),
        "exclude_doc_ids": sorted({r["doc_id"] for r in rows}),
        "by_cohort": by_cohort,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", type=Path, default=DEFAULT_HITS,
                    help="posttrain_hits.csv.gz (default: posttrain_overlap/ under the data root)")
    ap.add_argument("--out", type=Path, default=paths.ANALYSIS / "posttrain_exclusions.json",
                    help="output JSON (default: analysis/posttrain_exclusions.json under the data root)")
    ap.add_argument("--audit-date", default="2026-09-29", help="date of the post-training scan")
    args = ap.parse_args()

    with gzip.open(args.hits, "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{args.hits}: no flagged documents; refusing to write an empty exclusion list")
    missing = {"doc_id", "cohort", "pair_id", "component"} - set(rows[0])
    if missing:
        raise SystemExit(f"{args.hits}: missing columns {sorted(missing)}")
    source = SOURCE if args.hits == DEFAULT_HITS else str(args.hits)
    out = build(rows, args.audit_date, source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {len(out['exclude_doc_ids'])} documents, "
          f"{len(out['exclude_pair_ids'])} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
