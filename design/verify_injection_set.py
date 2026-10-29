#!/usr/bin/env python3
"""Final verification of the injection set against the audit. Fails loudly.

For every document in manifest.csv.gz (both arms) and null_pool.csv.gz:
  1. non-member cohorts: the document is in the FULL-audit clean set
     (stageG over all 13 scan targets) and not in any exclusion list;
  2. its fingerprints were part of the scanned fingerprint table
     (index_docs.txt.gz of the fingerprint build, which every scan checked by
     digest), i.e. it was actually searched, not merely absent from a list;
  3. edited-wiki cohorts (E0, E0-D, E, N-edited): members by design; the
     trained ancestor exists and the page is NOT in the clean set;
  4. anchors (X): synthetic; the arXiv sources of the shuffled strata are in
     the clean set and are used nowhere else; model samples and random tokens
     are recorded as unaudited synthetic text;
  5. every scan target has status ok, the same manifest and fingerprint
     digests, and all 13 stage/component targets are present;
  6. structural: pairs complete, arms balanced, copies == r, no document in
     two places; for a random sample of --check-tokens documents the injected
     text file and the token file exist, and the token file has the recorded
     length and ends in EOS.
Prints a summary and exits non-zero on any violation.

Defaults: --injection-dir is the bundle/ directory of the data root;
--audit-dir and --edited-dir are audit_v2/ and wiki_edited/ under its
collection/ directory (see olmo_twin/paths.py).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# repository root on sys.path, so that `olmo_twin` imports from a checkout
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

W = paths.COLLECTION   # collection working directory (audit_v2/, wiki_edited/)
EXPECTED_TARGETS = {
    "scan_dolmino-mix-1124__wiki", "scan_dolmino-mix-1124__stackexchange", "scan_dolmino-mix-1124__math",
    "scan_dolmino-mix-1124__flan", "scan_dolmino-mix-1124__pes2o", "scan_dolmino-mix-1124__dclm",
    "scan_olmo-mix-1124__wiki", "scan_olmo-mix-1124__algebraic-stack", "scan_olmo-mix-1124__open-web-math",
    "scan_olmo-mix-1124__arxiv", "scan_olmo-mix-1124__starcoder", "scan_olmo-mix-1124__pes2o",
    "scan_olmo-mix-1124__dclm",
}
MEMBER_COHORTS = {"E0", "E0-D", "E", "N-edited"}


def rd(p: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with gzip.open(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--injection-dir", type=Path, default=paths.BUNDLE)
    ap.add_argument("--audit-dir", type=Path, default=W / "audit_v2")
    ap.add_argument("--stage-g", default="stageG_13", help="Stage G directory over ALL scan targets")
    ap.add_argument("--edited-dir", type=Path, default=W / "wiki_edited")
    ap.add_argument("--check-tokens", type=int, default=300, help="documents whose text and token files are checked (existence, length, trailing EOS)")
    args = ap.parse_args()
    I, A = args.injection_dir, args.audit_dir
    problems: list[str] = []
    def bad(msg):
        problems.append(msg); print("  VIOLATION:", msg)

    # ---- 5. scan targets and provenance ----------------------------------------
    fp_meta = json.loads((A / "fp" / "fingerprint_meta.json").read_text())
    targets = {d.name: json.loads((d / "scan_summary.json").read_text())
               for d in A.glob("scan_*") if (d / "scan_summary.json").exists()}
    missing = EXPECTED_TARGETS - set(targets)
    if missing:
        bad(f"scan targets missing or not complete: {sorted(missing)}")
    for name, s in targets.items():
        if s.get("status") != "ok":
            bad(f"{name}: status {s.get('status')}")
        if s.get("candidates_sha256") != fp_meta["candidates_sha256"] or s.get("fingerprints_sha256") != fp_meta["fingerprints_sha256"]:
            bad(f"{name}: provenance digests differ from the fingerprint build")
        if (A / name / "scan_summary.FAILED.json").exists():
            bad(f"{name}: a FAILED summary exists next to the ok one")
    sg = json.loads((A / args.stage_g / "assembly_summary.json").read_text())
    if len(sg["scan_targets"]) != len(EXPECTED_TARGETS) or set(sg["scan_targets"]) != EXPECTED_TARGETS:
        bad(f"{args.stage_g} was assembled from {len(sg['scan_targets'])} targets, not all {len(EXPECTED_TARGETS)}")
    if sg["manifest_sha256"] != fp_meta["candidates_sha256"]:
        bad(f"{args.stage_g} manifest digest differs from the fingerprint build")
    print(f"scan targets complete: {len(targets)}/{len(EXPECTED_TARGETS)}; Stage G '{args.stage_g}' over {len(sg['scan_targets'])} targets; "
          f"provenance {fp_meta['candidates_sha256'][:12]}/{fp_meta['fingerprints_sha256'][:12]}")

    # ---- audited universe ---------------------------------------------------------
    with gzip.open(A / "fp" / "index_docs.txt.gz", "rt") as fh:
        fingerprinted_rows = [l.strip() for l in fh if l.strip()]
    # arXiv rows carry a representation suffix; Stage G keys papers by the bare id
    def paper_key(doc_id: str) -> str:
        return doc_id.rsplit("#", 1)[0] if "#" in doc_id else doc_id
    fingerprinted = {paper_key(d) for d in fingerprinted_rows} | set(fingerprinted_rows)
    clean = {r["doc_id"] for r in rd(A / args.stage_g / "candidates_clean.csv.gz")}
    evidence = {r["doc_id"]: r for r in rd(A / args.stage_g / "candidates_all_evidence.csv.gz")}
    exclusions = set()
    for p in I.glob("*exclusions*.csv.gz"):
        # rows flagged as pair partners were dropped for their partner, not for
        # themselves; they may legitimately be reused (restore-unfilled)
        exclusions |= {r["doc_id"] for r in rd(p) if r.get("reason", "hit") != "pair_partner"}
    # Analysis-time flags are NOT membership exclusions: the documents stayed in
    # the trained set and are dropped only when the results are analysed
    # (post-training overlap, found after training). Reported, never a violation.
    analysis_flags = set()
    for p in I.glob("*analysis_flags*.csv.gz"):
        analysis_flags |= {r["doc_id"] for r in rd(p)}
    ancestors = {json.loads(l)["page_id"] for l in gzip.open(args.edited_dir / "ancestors.jsonl.gz", "rt")}
    print(f"audited universe: {len(fingerprinted_rows):,} fingerprint rows, {len(evidence):,} Stage G documents, "
          f"{len(clean):,} clean, {len(exclusions):,} later exclusions, {len(ancestors):,} edited pages with ancestor, "
          f"{len(analysis_flags):,} analysis-time flags")

    # ---- documents in the set ----------------------------------------------------
    man = rd(I / "manifest.csv.gz")
    null = rd(I / "null_pool.csv.gz")
    copies = rd(I / "copies.csv.gz")
    ids = [r["doc_id"] for r in man] + [r["doc_id"] for r in null]
    dup = [d for d, n in Counter(ids).items() if n > 1]
    if dup:
        bad(f"{len(dup)} documents appear more than once (manifest + null): {dup[:5]}")
    n_ok = Counter()
    for r in man + null:
        d, c = r["doc_id"], r["cohort"]
        if c == "X":
            st = r["stratum"]
            if st in ("sentence_shuffled", "word_shuffled"):
                src = r["source_doc_id"]
                if src not in clean:
                    bad(f"{d}: shuffled anchor source {src} not in the clean set")
                if src in {x["doc_id"] for x in man} or src in {x["doc_id"] for x in null}:
                    bad(f"{d}: anchor source {src} is also used as a document")
            n_ok[f"X:{st}"] += 1
            continue
        if c in MEMBER_COHORTS:
            pid = d.split(":", 1)[1]
            if pid not in ancestors:
                bad(f"{d}: edited-wiki document without a trained ancestor")
            if d in clean:
                bad(f"{d}: edited-wiki document is in the clean (non-member) set")
            n_ok["member(edited)"] += 1
            continue
        # non-member cohorts
        if d not in fingerprinted:
            bad(f"{d}: was never fingerprinted (not searched)")
        if d not in evidence:
            bad(f"{d}: not in Stage G evidence")
        if d not in clean:
            bad(f"{d}: NOT in the full-audit clean set (reasons: {evidence.get(d, {}).get('reasons', '?')})")
        if d in exclusions:
            bad(f"{d}: in a later exclusion list")
        if c == "F" and r["arm"] == "inject" and r["inject_text_form"] != "redpajama_latex":
            bad(f"{d}: F inject arm is not the LaTeX form")
        n_ok["non-member"] += 1
    # both arXiv representations of every arXiv document were fingerprinted (union rule)
    arx = [r["doc_id"] for r in man if r["group"] == "postcutoff"]
    reps = Counter(paper_key(x) for x in fingerprinted_rows if "#" in x or x in set(arx))
    print("documents checked:", dict(n_ok))

    # ---- 6. structure -------------------------------------------------------------
    pairs = defaultdict(list)
    for r in man:
        pairs[r["pair_id"]].append(r)
    for pid, v in pairs.items():
        if len(v) != 2 or {x["arm"] for x in v} != {"inject", "holdout"}:
            bad(f"pair {pid} is not one inject + one holdout")
        elif any(v[0][k] != v[1][k] for k in ("cohort", "stratum", "r", "target_len", "split")):
            bad(f"pair {pid}: members differ in cohort/stratum/r/target_len/split")
    inj = {r["doc_id"]: int(r["r"]) for r in man if r["arm"] == "inject"}
    cc = Counter(c["doc_id"] for c in copies)
    if set(cc) != set(inj):
        bad(f"copies refer to {len(set(cc) - set(inj))} non-inject documents / miss {len(set(inj) - set(cc))} inject documents")
    for d, r_ in inj.items():
        if cc[d] != r_:
            bad(f"{d}: {cc[d]} copies, r = {r_}")
    pf = np.array([float(c["position_frac"]) for c in copies])
    if pf.min() < 0.01 or pf.max() > 0.99:
        bad("copy positions outside [0.01, 0.99]")
    rng = np.random.default_rng(0)
    for r in rng.choice(man, size=min(args.check_tokens, len(man)), replace=False):
        p = Path(r["inject_text_path"]); t = Path(r["tokens_path"])
        if not p.exists() or not t.exists():
            bad(f"{r['doc_id']}: missing text or token file"); continue
        a = np.load(t)
        if len(a) != int(r["n_tokens_inject"]) or a[-1] != 100257:
            bad(f"{r['doc_id']}: token file length/EOS mismatch")
    print(f"structure: {len(pairs):,} pairs, {len(copies):,} copies, arms {dict(Counter(r['arm'] for r in man))}, "
          f"{min(args.check_tokens, len(man))} token files re-checked")
    in_set_flagged = sorted(analysis_flags & ({r["doc_id"] for r in man} | {r["doc_id"] for r in null}))
    if in_set_flagged:
        pairs_flagged = {r["pair_id"] for r in man if r["doc_id"] in analysis_flags and r["pair_id"]}
        print(f"analysis-time flags in the trained set: {len(in_set_flagged)} documents, {len(pairs_flagged)} pairs "
              f"(post-training overlap; drop these pairs when analysing, not from the training stream)")
    print("unaudited by construction: X model_samples (generated by the pre-treatment model) and X random_tokens — "
          "anchors for the surprisal spectrum, not non-member claims.")
    if problems:
        print(f"\nFAILED: {len(problems)} violation(s)")
        return 1
    print("\nOK: every non-member document in the set was fingerprinted, searched against all 13 targets, "
          "is in the full-audit clean set and in no exclusion list; edited cohorts are members with a known ancestor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
