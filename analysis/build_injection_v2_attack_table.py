#!/usr/bin/env python3
"""Build the tidy per-document attack table for the injection-v2 experiment.

Reads the per-token score files produced by scoring/score_injection_v2{,_extended}.py
under one or more model revisions and computes, for every evaluation
document, the standard membership-inference statistics from the literature
plus the design metadata needed to condition on cohort, dose, timing, length
and novelty.

ORIENTATION CONVENTION: every score is signed so that HIGHER = MORE
MEMBER-LIKE. An attack that works has AUC > 0.5 when ranking the inject arm
against the holdout arm.

Single-model attacks (what a realistic attacker has):
  loss            mean log p(x_t)                        (Yeom et al. 2018)
  ppl             -perplexity
  bpb             -bits per byte                          (length/encoding normalised)
  mink_K          mean of the K% SMALLEST log p(x_t)      (Shi et al. 2024)
  minkpp_K        mean of the K% smallest (log p - mu)/sd (Zhang et al. 2024)
  maxk_K          mean of the K% LARGEST log p(x_t)       (ablation/contrast)
  surprise        mean (log p(x_t) - mu_t)                (model-relative surprise)
  zlib            -nll / zlib_compressed_bytes            (Carlini et al. 2021)

Reference-based attacks (need a second model):
  ratio_stage1    loss(target) - loss(stage-1 pre-treatment)   realistic: AI2
                  published the stage-1 checkpoint our run starts from
  ratio_oracle    loss(target) - loss(official seed-42)        the EXACT
                  counterfactual: an upper bound no real attacker attains
  ratio_sibling   loss(target) - loss(ingredient1, seed 42069)  same recipe and
                  same data mixture as the control, different random order --
                  the reference an adversary can build by retraining
  mink_ratio_K    mink_K(target) - mink_K(reference)

Also emits per-token aggregates used by the position/decay figures.

All input and output locations are command-line arguments; analysis/run_analysis.sh
passes the ones used for the paper (the bundle files and the four score
directories injected, seed42, stage1 and sibling under the run directory).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

K_VALUES = (5, 10, 20, 30, 40, 50)
POS_BINS = 10


def read_csv_gz(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with gzip.open(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def safe(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace(":", "_")


def mink(values: np.ndarray, k: int) -> float:
    """Mean of the k% smallest entries (Min-K%: members lack very-low-prob tokens)."""
    if values.size == 0:
        return float("nan")
    n = max(1, int(round(values.size * k / 100.0)))
    return float(np.partition(values, n - 1)[:n].mean())


def maxk(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return float("nan")
    n = max(1, int(round(values.size * k / 100.0)))
    return float(np.partition(values, -n)[-n:].mean())


def load_tokens(path: Path):
    with np.load(path) as z:
        lp = z["logprobs"].astype(np.float64)
        mu = z["mu"].astype(np.float64) if "mu" in z.files else None
        sd = z["sigma"].astype(np.float64) if "sigma" in z.files else None
        ids = z["token_ids"]
    return ids, lp, mu, sd


def process_doc(job):
    doc_id, meta, dirs, text_path = job
    row = dict(meta)
    per_model = {}
    pos_profile = {}

    for label, d in dirs.items():
        f = Path(d) / "tokens" / f"{safe(doc_id)}.npz"
        if not f.exists():
            continue
        try:
            ids, lp, mu, sd = load_tokens(f)
        except Exception:
            continue
        valid = ~np.isnan(lp)
        v = lp[valid]
        if v.size == 0:
            continue
        stats = {
            "loss": float(v.mean()),
            "nll_sum": float(-v.sum()),
            "ppl": float(-math.exp(min(-v.mean(), 700))),
            "n_scored": int(v.size),
        }
        for k in K_VALUES:
            stats[f"mink_{k}"] = mink(v, k)
            stats[f"maxk_{k}"] = maxk(v, k)
        if mu is not None and sd is not None:
            m, s = mu[valid], sd[valid]
            good = np.isfinite(m) & np.isfinite(s) & (s > 1e-8)
            if good.any():
                z = (v[good] - m[good]) / s[good]
                for k in K_VALUES:
                    stats[f"minkpp_{k}"] = mink(z, k)
                stats["surprise"] = float((v[good] - m[good]).mean())
                stats["mean_entropy"] = float(-m[good].mean())
        per_model[label] = stats
        # position profile: mean logprob in each decile of the document
        if v.size >= POS_BINS:
            idx = np.linspace(0, v.size, POS_BINS + 1).astype(int)
            pos_profile[label] = [float(v[idx[i]:idx[i + 1]].mean()) for i in range(POS_BINS)]

    if not per_model:
        return None

    for label, stats in per_model.items():
        for k, val in stats.items():
            row[f"{label}__{k}"] = val

    # zlib ratio on the exact text (higher = more member-like)
    zlib_bytes = None
    if text_path and Path(text_path).exists():
        try:
            raw = Path(text_path).read_bytes()
            zlib_bytes = len(zlib.compress(raw, 9))
            row["zlib_bytes"] = zlib_bytes
        except Exception:
            pass
    tgt = per_model.get("injected")
    if tgt and zlib_bytes:
        row["zlib"] = -tgt["nll_sum"] / zlib_bytes
    if tgt and row.get("n_bytes"):
        try:
            row["bpb"] = -tgt["nll_sum"] / math.log(2) / int(row["n_bytes"])
        except Exception:
            pass

    # reference-based
    # "sibling" is ingredient1: the same stage-2 recipe and the same 576 data
    # paths as the control, trained from seed 42069 instead of 42. It is the
    # reference an adversary could build by retraining the published recipe,
    # and sits between ratio_stage1 (realistic, weak) and ratio_oracle (exact
    # counterfactual, unobtainable).
    for ref, tag in (("stage1", "ratio_stage1"), ("seed42", "ratio_oracle"),
                     ("sibling", "ratio_sibling")):
        if tgt and ref in per_model:
            row[tag] = tgt["loss"] - per_model[ref]["loss"]
            for k in K_VALUES:
                row[f"mink_{k}_{tag}"] = tgt[f"mink_{k}"] - per_model[ref][f"mink_{k}"]
            if f"minkpp_{K_VALUES[0]}" in tgt and f"minkpp_{K_VALUES[0]}" in per_model[ref]:
                for k in K_VALUES:
                    row[f"minkpp_{k}_{tag}"] = tgt[f"minkpp_{k}"] - per_model[ref][f"minkpp_{k}"]

    # per-token position profile of the ORACLE delta
    if "injected" in pos_profile and "seed42" in pos_profile:
        for i, (a, b) in enumerate(zip(pos_profile["injected"], pos_profile["seed42"])):
            row[f"posdelta_{i}"] = a - b
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluation-manifest", type=Path, required=True)
    ap.add_argument("--design-manifest", type=Path, default=None)
    ap.add_argument("--placement", type=Path, default=None, help="stream_plan/placement.jsonl.gz")
    ap.add_argument("--texts-index", type=Path, default=None)
    ap.add_argument("--bundle", type=Path, default=None)
    ap.add_argument("--scores", action="append", required=True,
                    metavar="LABEL=DIR", help="repeatable, e.g. injected=/path/to/scores_ext/ext-injected")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    dirs = {}
    for spec in args.scores:
        label, _, d = spec.partition("=")
        dirs[label] = d
    print(f"score dirs: {dirs}", flush=True)

    manifest = read_csv_gz(args.evaluation_manifest)
    design = {r["doc_id"]: r for r in read_csv_gz(args.design_manifest)} if args.design_manifest else {}

    placement = defaultdict(list)
    if args.placement and args.placement.exists():
        with gzip.open(args.placement, "rt") as fh:
            for line in fh:
                if line.strip():
                    j = json.loads(line)
                    placement[j["doc_id"]].append(j)

    texts = {}
    if args.texts_index and args.texts_index.exists() and args.bundle:
        for r in read_csv_gz(args.texts_index):
            texts[r["doc_id"]] = str(args.bundle / r["text_relpath"])

    jobs = []
    for m in manifest:
        doc_id = m["doc_id"]
        d = design.get(doc_id, {})
        cps = placement.get(doc_id, [])
        meta = {
            "doc_id": doc_id, "cohort": m["cohort"], "stratum": m["stratum"], "arm": m["arm"],
            "pair_id": m["pair_id"], "split": m["split"], "r": int(m["r"] or 0),
            "group": m["group"], "n_bytes": m.get("n_bytes", ""),
            "bpb_pre": d.get("bpb_pre", ""), "novelty_stratum": d.get("novelty_stratum", ""),
            "frac_novel_tokens": d.get("frac_novel_tokens", ""),
            "burst": d.get("burst", ""), "target_len": d.get("target_len", ""),
            "month": d.get("month", ""), "category_group": d.get("category_group", ""),
            "n_copies_placed": len(cps),
            "first_update": min((c["first_update"] for c in cps), default=""),
            "last_update": max((c["last_update"] for c in cps), default=""),
            "mean_update": (sum(c["first_update"] for c in cps) / len(cps)) if cps else "",
            "n_tokens_injected": cps[0]["n_tokens_with_eos"] if cps else "",
        }
        jobs.append((doc_id, meta, dirs, texts.get(doc_id)))

    print(f"processing {len(jobs):,} documents with {args.workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_doc, jobs, chunksize=32), 1):
            if r is not None:
                rows.append(r)
            if i % 2000 == 0:
                print(f"  {i:,}/{len(jobs):,}", flush=True)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {args.out} ({len(rows):,} rows, {len(keys)} columns)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
