#!/usr/bin/env python3
"""Tokenise the cohort-F 2x2 re-parse (sets A and B) for scoring.

Set A = PDF rendering of the 278 INJECTED papers (their LaTeX form was trained
on; this form was not). Set B = LaTeX rendering of the 278 HELD-OUT papers
(their PDF form is the existing holdout arm; this form was never trained on).

Together with the two already-scored cells -- LaTeX(A) is the existing F inject
arm, PDF(B) the existing F holdout arm -- these complete the 2x2 that lets the
format-transfer coefficient be measured with the surface form matched on both
sides of every comparison.

Control: before tokenising anything new, re-tokenise the 20 set C documents
(whose text is byte-identical to the shipped bundle text) and require the token
ids to match the bundle's stored .npy exactly. If the tokenisation convention
were wrong, every downstream number would be subtly wrong and nothing else
would reveal it.

Inputs: the re-parsed texts under <data root>/cohortF_reparse/ (subdirectories
pipeline_check/ = set C, pdf_of_injected/ = set A, latex_of_holdout/ = set B;
built by collection/cohortF_reparse.py), the bundle's tokens/F/*.npy for the
control, and the OLMo 2 tokenizer in <data root>/assets/hf_models/OLMo-2-0425-1B.
Output: <data root>/run/cohortF_2x2/ (tokens/<set>/<arxiv id>.npy and
evaluation_manifest.csv.gz), scored by scoring/run_scoring.sh F2x2-<model>.
The data root is $OLMO_TWIN_DATA (default <repository>/data); see
olmo_twin/paths.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

BUNDLE = paths.BUNDLE
REPARSE = paths.COHORTF_REPARSE
OUT = paths.RUN / "cohortF_2x2"
TOKENIZER = paths.ASSETS / "hf_models/OLMo-2-0425-1B"
EOS = 100257

SETS = [
    # directory, doc_id prefix, arm label, what it is
    ("pdf_of_injected", "Fpdf", "inject", "pes2o_style_pdf"),
    ("latex_of_holdout", "Flatex", "holdout", "redpajama_latex"),
]


def encode(tok, text: str) -> np.ndarray:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return np.asarray(list(ids) + [EOS], dtype=np.int32)


def main() -> int:
    # No options: the script always writes run/cohortF_2x2/ under the data root.
    # Parsing the (empty) command line makes -h/--help print this docstring and
    # rejects stray arguments instead of starting a run that overwrites the output.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))

    # ---------------- control: reproduce the bundle's own token ids
    ok = bad = 0
    for p in sorted((REPARSE / "pipeline_check").glob("*.npy" if False else "*.txt")):
        ref = BUNDLE / "tokens" / "F" / f"{p.stem}.npy"
        if not ref.exists():
            continue
        mine = encode(tok, p.read_text(encoding="utf-8"))
        theirs = np.load(ref)
        if mine.shape == theirs.shape and (mine == theirs).all():
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"  MISMATCH {p.stem}: mine {mine.shape} theirs {theirs.shape}")
                n = min(len(mine), len(theirs))
                d = np.flatnonzero(mine[:n] != theirs[:n])
                print(f"    first differing position: {d[0] if d.size else 'n/a'}")
    print(f"tokenisation control (set C vs bundle tokens): {ok} identical, {bad} differing")
    if bad or ok == 0:
        print("ABORT: tokenisation convention does not reproduce the bundle")
        return 1

    # ---------------- tokenise A and B
    rows = []
    for sub, prefix, arm, form in SETS:
        d = OUT / "tokens" / sub
        d.mkdir(parents=True, exist_ok=True)
        files = sorted((REPARSE / sub).glob("*.txt"))
        for p in files:
            text = p.read_text(encoding="utf-8")
            ids = encode(tok, text)
            np.save(d / f"{p.stem}.npy", ids)
            rows.append({
                "doc_id": f"{prefix}:{p.stem}",
                "cohort": "F2x2",
                "stratum": form,
                "arm": arm,
                "pair_id": "",          # filled at analysis time from the F manifest
                "split": "holdout",
                "r": "1" if arm == "inject" else "0",
                "group": "postcutoff",
                "tokens_relpath": f"tokens/{sub}/{p.stem}.npy",
                "n_tokens": str(int(ids.size)),
                "n_bytes": str(len(text.encode("utf-8"))),
            })
        tot = sum(int(r["n_tokens"]) for r in rows if r["stratum"] == form)
        print(f"{sub:18s} {len(files):4d} documents, {tot:>10,} tokens "
              f"(median {int(np.median([int(r['n_tokens']) for r in rows if r['stratum']==form])):,})")

    man = OUT / "evaluation_manifest.csv.gz"
    with gzip.open(man, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {man}")
    print(f"total {len(rows)} documents, "
          f"{sum(int(r['n_tokens']) for r in rows):,} tokens to score per checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
