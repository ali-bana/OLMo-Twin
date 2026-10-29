#!/usr/bin/env python3
"""Paired change from DPO to each RL stage on the same pairs (stage minus DPO) for tau, SD, d_z, AUC,
TPR@1%FPR and %pos; pair bootstrap B=2000, default_rng(20261028), both stages evaluated on the same
resampled pairs; plus a paired t on the per-pair change of the contrast. Output rlvr/work/rl_minus_dpo.csv
under the data root (run after recompute.py, whose helpers and output directory it imports)."""
import math
from multiprocessing import Pool
import numpy as np, pandas as pd
from recompute import load_pairs, pops, arrays, point, B, SEED, OUT

STG = {"dpo": ("dpo-injected", "dpo-baseline"), "rlvr1": ("rlvr1-injected", "rlvr1-baseline"),
       "rlvr2i": ("rlvr2-step2000-injected", "rlvr2-step2000-baseline"), "rlvr2": ("rlvr2-injected", "rlvr2-baseline")}
KEYS = ("tau", "sd", "dz", "auc", "tpr1", "pos")

def job(args):
    pn, s, a, b, a0, b0 = args
    p, p0 = point(a, b), point(a0, b0)
    dd = (a - b) - (a0 - b0)
    rng = np.random.default_rng(SEED)
    bs = {k: np.empty(B) for k in KEYS}
    for i in range(B):
        j = rng.integers(0, a.size, a.size)
        q, q0 = point(a[j], b[j]), point(a0[j], b0[j])
        for k in KEYS: bs[k][i] = q[k] - q0[k]
    r = dict(population=pn, stage=s, n=a.size, paired_t_tau=dd.mean() / (dd.std(ddof=1) / math.sqrt(dd.size)),
             sd_per_pair_change=dd.std(ddof=1))
    for k in KEYS:
        r[f"d_{k}"] = p[k] - p0[k]; r[f"d_{k}_lo"] = np.percentile(bs[k], 2.5); r[f"d_{k}_hi"] = np.percentile(bs[k], 97.5)
    return r

if __name__ == "__main__":
    _, P = load_pairs(); PO = pops(P)
    jobs = []
    for pn in ("NAT1", "ALL7326", "ALLPAIRS7803"):
        a0, b0 = arrays(PO[pn], *STG["dpo"])
        for s in ("rlvr1", "rlvr2i", "rlvr2"):
            a, b = arrays(PO[pn], *STG[s]); jobs.append((pn, s, a, b, a0, b0))
    with Pool(9) as pool:
        rows = pool.map(job, jobs)
    pd.DataFrame(rows).to_csv(OUT / "rl_minus_dpo.csv", index=False)
    print(pd.DataFrame(rows)[["population", "stage", "d_tau", "d_tau_lo", "d_tau_hi", "d_dz", "d_dz_lo", "d_dz_hi", "d_auc", "d_auc_lo", "d_auc_hi", "d_tpr1", "d_tpr1_lo", "d_tpr1_hi", "paired_t_tau"]].round(4).to_string())
