"""Precompute a compact per-token cache (input of analysis/length_truncation_analysis.py).

Reads the two per-token .npz score dumps (injected model, official seed-42
control) for every document in the attack table and folds each pair down to a
few dozen numbers.  Everything the content figures need about *within-document*
structure -- prefix-truncation curves, how concentrated the membership gain is
across tokens, whether it sits on hard or easy tokens -- comes out of this one
pass, so no later script ever has to touch the 45k raw files again.

Inputs (under the data root, see olmo_twin/paths.py):
analysis/per_document_attacks_prelim.csv.gz (document list) and the logprobs of
run/scores/{injected,official-seed42}-step23852/tokens/*.npz (token ids are not
read). Output: analysis/cache/token_cache.npz.
Pure CPU/IO, CPU only; WORKERS sets the number of processes (default 32).
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

TABLE = str(paths.ANALYSIS / "per_document_attacks_prelim.csv.gz")
SCORES = str(paths.RUN / "scores")
INJ_DIR = f"{SCORES}/injected-step23852/tokens"
CTL_DIR = f"{SCORES}/official-seed42-step23852/tokens"
OUT = str(paths.ANALYSIS / "cache/token_cache.npz")

# prefix lengths, in *valid scored tokens*
GRID = np.array([16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048,
                 3072, 4096, 6144, 8192, 12288, 16384], dtype=np.int64)
# quantiles of the per-token delta
QS = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
# top-fraction-of-tokens shares of the total gain
TOPF = np.array([0.01, 0.05, 0.10, 0.25, 0.50])
# Min-K%: fraction of tokens with the LOWEST control log-prob (hardest tokens)
MINK = np.array([0.05, 0.10, 0.20, 0.50])


def safe_id(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace(":", "_")


def one(doc_id: str):
    sid = safe_id(doc_id)
    pi = os.path.join(INJ_DIR, sid + ".npz")
    pc = os.path.join(CTL_DIR, sid + ".npz")
    nan = np.nan
    empty = dict(
        n_valid=0,
        cum_inj=np.full(GRID.size, nan), cum_ctl=np.full(GRID.size, nan),
        dq=np.full(QS.size, nan), topshare=np.full(TOPF.size, nan),
        lor_pos=np.full(TOPF.size, nan), lor_abs=np.full(TOPF.size, nan),
        mink_delta=np.full(MINK.size, nan), maxk_delta=np.full(MINK.size, nan),
        frac_pos=nan, mean_delta=nan, std_delta=nan, pos_sum=nan, neg_sum=nan,
        n_chunks=nan, mean_abs=nan,
    )
    if not (os.path.exists(pi) and os.path.exists(pc)):
        return doc_id, empty
    try:
        a = np.load(pi)["logprobs"].astype(np.float64)
        b = np.load(pc)["logprobs"].astype(np.float64)
    except Exception:
        return doc_id, empty
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    n_chunks = float((~np.isfinite(b)).sum())   # one NaN per 4096-chunk start
    a, b = a[ok], b[ok]
    m = a.size
    if m == 0:
        return doc_id, empty
    d = a - b

    ci = np.cumsum(a)
    cc = np.cumsum(b)
    idx = np.minimum(GRID, m) - 1
    cum_inj = np.where(GRID <= m, ci[idx], np.nan)
    cum_ctl = np.where(GRID <= m, cc[idx], np.nan)

    ds = np.sort(d)[::-1]           # descending
    tot = d.sum()
    pos_sum = float(ds[ds > 0].sum())
    neg_sum = float(ds[ds < 0].sum())
    das = np.sort(np.abs(d))[::-1]
    abs_sum = float(das.sum())
    topshare = np.full(TOPF.size, np.nan)
    lor_pos = np.full(TOPF.size, np.nan)
    lor_abs = np.full(TOPF.size, np.nan)
    for j, f in enumerate(TOPF):
        k = max(1, int(round(f * m)))
        if abs(tot) > 1e-12:
            topshare[j] = ds[:k].sum() / tot
        if pos_sum > 1e-12:
            lor_pos[j] = np.clip(ds[:k], 0, None).sum() / pos_sum
        if abs_sum > 1e-12:
            lor_abs[j] = das[:k].sum() / abs_sum

    order = np.argsort(b)           # ascending control log-prob = hardest first
    mink_delta = np.full(MINK.size, np.nan)
    maxk_delta = np.full(MINK.size, np.nan)
    for j, f in enumerate(MINK):
        k = max(1, int(round(f * m)))
        mink_delta[j] = d[order[:k]].mean()
        maxk_delta[j] = d[order[-k:]].mean()

    return doc_id, dict(
        n_valid=m,
        cum_inj=cum_inj, cum_ctl=cum_ctl,
        dq=np.quantile(d, QS), topshare=topshare,
        lor_pos=lor_pos, lor_abs=lor_abs,
        mink_delta=mink_delta, maxk_delta=maxk_delta,
        frac_pos=float((d > 0).mean()), mean_delta=float(d.mean()),
        std_delta=float(d.std()), pos_sum=pos_sum, neg_sum=neg_sum,
        n_chunks=n_chunks, mean_abs=float(np.abs(d).mean()),
    )


def main():
    csv.field_size_limit(sys.maxsize)
    with gzip.open(TABLE, "rt", newline="") as fh:
        doc_ids = [r["doc_id"] for r in csv.DictReader(fh)]
    print(f"{len(doc_ids)} documents", flush=True)

    nw = int(os.environ.get("WORKERS", "32"))
    with Pool(nw) as pool:
        res = pool.map(one, doc_ids, chunksize=64)

    N = len(res)
    out = dict(
        doc_id=np.array([r[0] for r in res], dtype=object),
        grid=GRID, qs=QS, topf=TOPF, mink=MINK,
        n_valid=np.array([r[1]["n_valid"] for r in res], dtype=np.int64),
        cum_inj=np.stack([r[1]["cum_inj"] for r in res]),
        cum_ctl=np.stack([r[1]["cum_ctl"] for r in res]),
        dq=np.stack([r[1]["dq"] for r in res]),
        topshare=np.stack([r[1]["topshare"] for r in res]),
        lor_pos=np.stack([r[1]["lor_pos"] for r in res]),
        lor_abs=np.stack([r[1]["lor_abs"] for r in res]),
        pos_sum=np.array([r[1]["pos_sum"] for r in res]),
        neg_sum=np.array([r[1]["neg_sum"] for r in res]),
        mink_delta=np.stack([r[1]["mink_delta"] for r in res]),
        maxk_delta=np.stack([r[1]["maxk_delta"] for r in res]),
        frac_pos=np.array([r[1]["frac_pos"] for r in res]),
        mean_delta=np.array([r[1]["mean_delta"] for r in res]),
        std_delta=np.array([r[1]["std_delta"] for r in res]),
        n_chunks=np.array([r[1]["n_chunks"] for r in res]),
        mean_abs=np.array([r[1]["mean_abs"] for r in res]),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    miss = int((out["n_valid"] == 0).sum())
    print(f"wrote {OUT}  N={N}  missing={miss}", flush=True)


if __name__ == "__main__":
    main()
