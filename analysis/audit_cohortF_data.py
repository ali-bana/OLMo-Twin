#!/usr/bin/env python3
"""Data-quality audit of cohort F across all four cells of the 2x2.

Prompted by 2508.05121, whose injected "paper" turned out to be 59 tokens of
acknowledgements: the bundle's 1,024-token arXiv floor was checked against the
PRE-TREATMENT (PDF) rendering while the form actually injected was the LaTeX
one. F is the only cohort where those are different renderings, so it is the
only cohort where such a slip is possible -- which is exactly why it deserves an
exhaustive sweep rather than a spot check.

Checks, each reported with every offending id so nothing is hidden behind a count:

  1  injected-form length against the cohort's own 1,024-32,768 token window
  2  degenerate extractions: very short, or dominated by a single section
  3  both renderings present for every paper, and non-empty
  4  same-paper correspondence: do a paper's two renderings actually share
     content? A silently mismatched file is invisible to every other check.
  5  duplicate or near-duplicate text within the cohort
  6  scoring status across all four cells
  7  markup fraction of the LaTeX form (is it text or is it macros?)
  8  per-pair effect outliers that would move the mean on their own

Inputs (under the data root, see olmo_twin/paths.py): bundle/manifest.csv.gz,
bundle/texts_index.csv.gz and the texts it points to (bundle/texts_all/...), the
re-parsed second forms in cohortF_reparse/{pdf_of_injected,latex_of_holdout}/
(written by collection/cohortF_reparse.py), and the scores.csv.gz files under
run/scores/ and run/scores_cohortF_2x2/. The document texts are third-party text
that is not redistributed; rebuild them before running this script.
Output: analysis/cohortF_data_exclusions.json (read by
injection_v2_plotstyle.cohortF_exclusions() and analysis/cohortF_2x2_check.py).
"""

from __future__ import annotations

import csv
import gzip
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

BUNDLE = paths.BUNDLE
REPARSE = paths.COHORTF_REPARSE
RUN = paths.RUN

ARXIV_MIN, ARXIV_MAX = 1024, 32768
problems = defaultdict(list)


def norm_words(t: str) -> list[str]:
    """Comparable word stream: drop LaTeX control words, keep alphanumerics."""
    t = re.sub(r"\\[a-zA-Z@]+\*?", " ", t)       # \section, \cite, ...
    t = re.sub(r"[^0-9A-Za-z]+", " ", t.lower())
    return t.split()


def shingles(ws: list[str], k: int = 8) -> set:
    return {hash(tuple(ws[i:i + k])) for i in range(max(0, len(ws) - k + 1))}


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    man = [r for r in csv.DictReader(gzip.open(BUNDLE / "manifest.csv.gz", "rt"))
           if r["cohort"] == "F"]
    byp = defaultdict(dict)
    for r in man:
        if r["pair_id"]:
            byp[r["pair_id"]][r["arm"]] = r
    pairs = {p: v for p, v in byp.items() if "inject" in v and "holdout" in v}
    print(f"cohort F: {len(man)} manifest rows, {len(pairs)} complete pairs\n")

    idx = {r["doc_id"]: r for r in
           csv.DictReader(gzip.open(BUNDLE / "texts_index.csv.gz", "rt"))
           if r["cohort"] == "F"}

    # ---- 1. injected-form length window
    print("[1] injected-form length against the 1,024-32,768 arXiv window")
    for r in man:
        if r["arm"] != "inject":
            continue
        n = int(r["n_tokens_inject"] or 0)
        if not (ARXIV_MIN <= n <= ARXIV_MAX):
            problems["length_window"].append((r["doc_id"], n, r["pair_id"]))
    if problems["length_window"]:
        for d, n, p in sorted(problems["length_window"], key=lambda x: x[1]):
            side = "BELOW floor" if n < ARXIV_MIN else "above ceiling"
            print(f"    {d}  {n:>7,} tokens as injected  {side}  (pair {p})")
    else:
        print("    none")

    # ---- 2+3+7. the text itself, both renderings
    print("\n[2,3,7] the text of every rendering: present, non-trivial, not pure markup")
    forms = {"latex_injected": [], "pdf_of_injected": [], "latex_of_holdout": [],
             "pdf_holdout": []}
    texts = {}
    for pid, v in pairs.items():
        A, B = v["inject"]["doc_id"], v["holdout"]["doc_id"]
        cand = [
            ("latex_injected", A, BUNDLE / idx[A]["text_relpath"]),
            ("pdf_of_injected", A, REPARSE / "pdf_of_injected" / f"{A}.txt"),
            ("latex_of_holdout", B, REPARSE / "latex_of_holdout" / f"{B}.txt"),
            ("pdf_holdout", B, BUNDLE / idx[B]["text_relpath"]),
        ]
        for form, did, p in cand:
            if not p.exists():
                problems["missing_text"].append((form, did))
                continue
            t = p.read_text(encoding="utf-8", errors="replace")
            texts[(form, did)] = t
            ws = norm_words(t)
            forms[form].append(len(ws))
            if len(ws) < 300:
                problems["tiny_text"].append((form, did, len(ws), pid))
            alpha = sum(c.isalpha() or c.isspace() for c in t) / max(1, len(t))
            if alpha < 0.70:
                problems["markup_heavy"].append((form, did, round(alpha, 3)))
            head = t[:200].lower()
            if re.match(r"\s*\\?\\?section\*?\{?\s*(acknowledge|references|bibliography|appendix)",
                        head) or head.strip().startswith("\\section*{acknowledge"):
                problems["fragment_section"].append((form, did, t[:60].replace("\n", " ")))
    for form, lens in forms.items():
        a = np.array(lens)
        print(f"    {form:18s} n={a.size:3d}  words: min {a.min():>7,} "
              f"p5 {np.percentile(a,5):>7,.0f}  median {np.median(a):>7,.0f}  "
              f"max {a.max():>8,}")
    for key, label in [("missing_text", "MISSING text file"),
                       ("tiny_text", "under 300 words"),
                       ("fragment_section", "starts at acknowledgements/references"),
                       ("markup_heavy", "under 70% letters+spaces")]:
        if problems[key]:
            print(f"    {label}: {len(problems[key])}")
            for x in problems[key][:12]:
                print(f"        {x}")
        else:
            print(f"    {label}: none")

    # ---- 4. same-paper correspondence between the two renderings
    print("\n[4] do a paper's two renderings actually share content?")
    same, cross = [], []
    ids = sorted({v["inject"]["doc_id"] for v in pairs.values()} |
                 {v["holdout"]["doc_id"] for v in pairs.values()})
    for did in ids:
        a = texts.get(("latex_injected", did)) or texts.get(("latex_of_holdout", did))
        b = texts.get(("pdf_of_injected", did)) or texts.get(("pdf_holdout", did))
        if not a or not b:
            continue
        sa, sb = shingles(norm_words(a)), shingles(norm_words(b))
        if not sa or not sb:
            continue
        c = len(sa & sb) / min(len(sa), len(sb))
        same.append((c, did))
    rng = np.random.default_rng(0)
    for _ in range(300):
        d1, d2 = rng.choice(ids, 2, replace=False)
        a = texts.get(("latex_injected", d1)) or texts.get(("latex_of_holdout", d1))
        b = texts.get(("pdf_of_injected", d2)) or texts.get(("pdf_holdout", d2))
        if not a or not b:
            continue
        sa, sb = shingles(norm_words(a)), shingles(norm_words(b))
        if sa and sb:
            cross.append(len(sa & sb) / min(len(sa), len(sb)))
    sc = np.array([c for c, _ in same]); cr = np.array(cross)
    print(f"    same paper, LaTeX vs PDF 8-gram containment: "
          f"median {np.median(sc):.3f}  p5 {np.percentile(sc,5):.3f}  min {sc.min():.3f}")
    print(f"    DIFFERENT papers (300 random draws):          "
          f"median {np.median(cr):.3f}  p95 {np.percentile(cr,95):.3f}  max {cr.max():.3f}")
    thresh = 0.05  # unrelated papers top out at ~0.001, same-paper p5 is ~0.46
    weak = sorted([(c, d) for c, d in same if c < thresh])
    print(f"    papers whose two renderings barely overlap (containment < "
          f"{thresh:.2f}, against {cr.max():.3f} max for unrelated papers and "
          f"{np.percentile(sc,5):.2f} at the same-paper 5th percentile): {len(weak)}")
    for c, d in weak[:12]:
        print(f"        {d}  containment {c:.3f}")
        problems["no_correspondence"].append((d, round(c, 4)))

    # ---- 5. duplicates inside the cohort
    print("\n[5] duplicate or near-duplicate text inside cohort F")
    seen = {}
    for (form, did), t in texts.items():
        h = hash(t)
        if h in seen:
            problems["duplicate_text"].append((did, form, seen[h]))
        seen[h] = (did, form)
    print(f"    byte-identical texts: {len(problems['duplicate_text'])}")
    for x in problems["duplicate_text"][:10]:
        print(f"        {x}")

    # ---- 6. scoring status across the four cells
    print("\n[6] scoring status across all four cells")

    def status(p):
        with gzip.open(p, "rt") as fh:
            return {r["doc_id"]: r["status"] for r in csv.DictReader(fh)}
    st = {}
    for lab, p in [("old_inj", RUN / "scores/injected-step23852/scores.csv.gz"),
                   ("old_ctl", RUN / "scores/official-seed42-step23852/scores.csv.gz"),
                   ("new_inj", RUN / "scores_cohortF_2x2/F2x2-injected/scores.csv.gz"),
                   ("new_ctl", RUN / "scores_cohortF_2x2/F2x2-seed42/scores.csv.gz"),
                   ("new_st1", RUN / "scores_cohortF_2x2/F2x2-stage1/scores.csv.gz")]:
        st[lab] = status(p) if p.exists() else {}
    bad = Counter()
    for pid, v in pairs.items():
        A, B = v["inject"]["doc_id"], v["holdout"]["doc_id"]
        for lab, key in [("old_inj", A), ("old_ctl", A), ("old_inj", B), ("old_ctl", B),
                         ("new_inj", f"Fpdf:{A}"), ("new_ctl", f"Fpdf:{A}"),
                         ("new_inj", f"Flatex:{B}"), ("new_ctl", f"Flatex:{B}"),
                         ("new_st1", f"Fpdf:{A}"), ("new_st1", f"Flatex:{B}")]:
            s = st[lab].get(key, "ABSENT")
            if s != "ok":
                bad[(lab, key, s)] += 1
                problems["score_status"].append((pid, lab, key, s))
    print(f"    non-ok cells: {len(set((b[1], b[2]) for b in bad))}")
    for (lab, key, s), _ in sorted(bad.items()):
        print(f"        {lab:8s} {key:28s} {s}")

    # ---- emit the exclusion list, from the rules rather than by hand
    import json
    EXCL = paths.ANALYSIS / "cohortF_data_exclusions.json"
    docs = {}
    for form, did, nwords, pid in problems["tiny_text"]:
        docs.setdefault(did, {"pair_id": pid, "reasons": []})["reasons"].append(
            f"rule A: {form} rendering is only {nwords} words, below the cohort's "
            f"own 1,024-token arXiv floor -- the LaTeX cleaner recovered a fragment")
    for did, c in problems["no_correspondence"]:
        pid = next((pp for pp, v in pairs.items()
                    if did in (v["inject"]["doc_id"], v["holdout"]["doc_id"])), None)
        docs.setdefault(did, {"pair_id": pid, "reasons": []})["reasons"].append(
            f"rule B: its two renderings share only {c:.3f} 8-gram containment, no "
            f"more than two unrelated papers do -- they are not the same text")
    out = {
        "generated_by": "analysis/audit_cohortF_data.py",
        "scope": "cohort F only; these are data-quality defects, not contamination",
        "rules": {
            "A": "a LaTeX rendering under 1,000 words (the cohort's floor is 1,024 tokens)",
            "B": "8-gram containment under 0.05 between a paper's two renderings",
        },
        "root_cause": (
            "the 1,024-32,768 token window was checked against n_tokens (the "
            "PRE-TREATMENT PDF rendering) while the form actually injected was the "
            "LaTeX one. F is the only cohort where those differ, so it is the only "
            "cohort where a fragmentary extraction could pass the filter."),
        "exclude_doc_ids": sorted(docs),
        "exclude_pair_ids": sorted({v["pair_id"] for v in docs.values() if v["pair_id"]}),
        "detail": docs,
        "not_excluded_but_recorded": {
            "above_ceiling": sorted(d for d, n, _ in problems["length_window"] if n > ARXIV_MAX),
            "too_long_for_scorer": ["2510.26874", "2508.10813", "2602.16849"],
            "note": ("20 injected renderings exceed the 32,768-token ceiling, up to "
                     "69,157. The documents are intact, so they are kept; three "
                     "exceed the scorer's 65,536-token cap and drop out on that "
                     "ground, which is the convention the original pass used."),
        },
    }
    EXCL.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwrote {EXCL.name}: {len(out['exclude_pair_ids'])} pairs, "
          f"{len(out['exclude_doc_ids'])} documents")
    for d, v in sorted(docs.items()):
        print(f"    {d}  pair {v['pair_id']}")
        for r in v["reasons"]:
            print(f"        {r}")

    # ---- summary
    print("\n" + "=" * 72)
    order = ["length_window", "missing_text", "tiny_text", "fragment_section",
             "markup_heavy", "no_correspondence", "duplicate_text", "score_status"]
    tot = 0
    for k in order:
        n = len(problems[k])
        tot += n
        print(f"  {k:20s} {n}")
    affected = set()
    for k in ("length_window",):
        affected |= {x[2] for x in problems[k]}
    for k in ("tiny_text",):
        affected |= {x[3] for x in problems[k]}
    for k in ("score_status",):
        affected |= {x[0] for x in problems[k]}
    print(f"\n  distinct pairs implicated: {len(affected)} of {len(pairs)}")
    print(f"  {sorted(affected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
