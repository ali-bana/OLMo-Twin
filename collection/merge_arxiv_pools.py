#!/usr/bin/env python3
"""Merge the two parsed arXiv pools into one (between `parse` and `rawtext`).

The arXiv pool was sampled twice with build_postcutoff_arxiv_pool.py: the main
sample (postcutoff/, seed 20260915) and a 300-per-month oversample
(postcutoff_extra/, seed 20260916); both were parsed with `parse`. This writes
postcutoff/pool_parsed_merged.csv.gz: every row of the main pool, then the rows
of the oversample whose doc_id is not in the main pool (the main pool wins),
with the union of the columns (the main pool's columns first). The merged pool
is the input of `build_postcutoff_arxiv_pool.py rawtext`,
build_combined_candidates.py, cohortF_reparse.py and the design step.

The code is the inline merge step of the original collection driver, unchanged
except that the three paths are arguments.
"""
import argparse
import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-pool", type=Path, default=paths.COLLECTION / "postcutoff" / "pool_parsed.csv.gz",
                    help="parsed main pool; its rows win (default: collection/postcutoff/pool_parsed.csv.gz)")
    ap.add_argument("--extra-pool", type=Path,
                    default=paths.COLLECTION / "postcutoff_extra" / "pool_parsed.csv.gz",
                    help="parsed oversample (default: collection/postcutoff_extra/pool_parsed.csv.gz)")
    ap.add_argument("--out", type=Path, default=paths.COLLECTION / "postcutoff" / "pool_parsed_merged.csv.gz",
                    help="merged pool (default: collection/postcutoff/pool_parsed_merged.csv.gz)")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    a=list(csv.DictReader(gzip.open(args.main_pool,"rt",newline="")))
    b=list(csv.DictReader(gzip.open(args.extra_pool,"rt",newline="")))
    seen={r["doc_id"] for r in a}; add=[r for r in b if r["doc_id"] not in seen]
    cols=list(a[0].keys())
    for c in b[0].keys():
        if c not in cols: cols.append(c)
    with gzip.open(args.out,"wt",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=cols,extrasaction="ignore"); w.writeheader()
        for r in a+add: w.writerow(r)
    print(f"   merged pool: {len(a):,} + {len(add):,} new = {len(a)+len(add):,} papers (overlap dropped {len(b)-len(add)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
