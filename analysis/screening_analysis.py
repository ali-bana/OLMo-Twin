#!/usr/bin/env python3
"""Cross-fitted operating points and mixed-pool screening with FDR control.

Every true-positive rate at a
fixed false-positive rate in the paper sets its threshold on the same matched
non-members on which the false-positive rate is then measured (in-sample), and
every "nameable at FDR q" count ranks the members only, with the matched
non-members as the empirical null. This script measures what those numbers
become when the threshold or the null distribution comes from non-members that
are disjoint from the ones being scored.

  Part A  pair-level K-fold cross-fitting (K=5 primary, K=2 secondary), R=50
          repeated fold assignments; threshold on the calibration folds'
          non-members with the paper's rule, realized FPR and TPR on the
          evaluation fold; NAT1 pooled and per cohort; ALL pooled (secondary).
  Part B  the 1% (and 0.1%) threshold calibrated on the domain-matched null
          pool (never injected, never a partner) instead of on partners.
  Part C  mixed candidate pools at stated member prevalence built from the
          evaluation folds of the Part A cross-fit, conformal p-values from
          the calibration folds' non-members, Benjamini-Hochberg at q.

One command regenerates everything:
    python3 -u analysis/screening_analysis.py
Outputs (under the data root, see olmo_twin/paths.py):
    analysis/screening_analysis.{json,md}
If the generator's macro file (tables/numbers.tex in the default output
directory of figures/paper_figures.py) exists, the reproduced operating points
are also checked against its macros and the comparison is stored under
reproduction.paper_macros.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

paths.add_code_dirs("figures")
import paper_figures as pf  # noqa: E402

SEED = 20261013
OUT_JSON = pf.ANA / "screening_analysis.json"
OUT_MD = pf.ANA / "screening_analysis.md"

COHORTS = ["M", "L", "W-new", "W-gap", "S-se"]
SCORES_A = ["ratio_oracle", "ratio_sibling", "ratio_stage1", "injected__surprise", "z_oracle", "z_sibling"]
SCORES_B = ["ratio_oracle", "z_oracle"]
SCORES_C = ["ratio_oracle", "ratio_sibling", "z_oracle"]
FPRS = [("1%", 1, 100), ("0.1%", 1, 1000)]     # f = num/den, k = floor(n*num/den) in exact integers
KS = (5, 2)
K_PRIMARY = 5
R_REP = 50
B_BOOT = 400
PREVS = [0.002, 0.01, 0.05, 0.20, 0.50]
QS = [0.05, 0.20]
NULL_DOMAINS = {"arXiv": (["M", "L"], ["N-arxiv"]),
                "Wikipedia": (["W-new", "W-gap"], ["N-wikiA", "N-wikiB"])}

# random streams: rng = numpy.random.default_rng([SEED, stream, *indices])
S_FOLDS, S_BOOT_A, S_DRAW_C, S_SHUFFLE, S_BOOT_B = 1, 2, 3, 4, 5


def rng_for(*key):
    return np.random.default_rng([SEED, *[int(k) for k in key]])


def summ(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5))}


# ----------------------------------------------------------------- primitives
def threshold(neg_sorted, num, den):
    """The paper's rule (pf.tpr_at): the score admitting k=floor(f*n) negatives;
    a document is flagged when its score is strictly greater. Returns (thr, k, n)."""
    n = neg_sorted.size
    k = (n * num) // den
    return neg_sorted[n - k - 1], k, n


def make_folds(strata, K, rng):
    """Pair-level fold labels, balanced within each stratum (cohort)."""
    fold = np.empty(len(strata), int)
    for s in sorted(set(strata.tolist())):
        idx = np.nonzero(strata == s)[0]
        perm = rng.permutation(idx)
        fold[perm] = (np.arange(idx.size) + int(rng.integers(K))) % K
    return fold


def crossfit(a, b, fold, K):
    """One repeat of K-fold cross-fitting; evaluation-fold decisions pooled over folds."""
    res = {lab: dict(fp=0, tp=0, k=[], ncal=[], exp=0.0) for lab, _, _ in FPRS}
    for j in range(K):
        ev = fold == j
        cal = np.sort(b[~ev]); ae, be = a[ev], b[ev]
        for lab, num, den in FPRS:
            t, k, n = threshold(cal, num, den)
            r = res[lab]
            r["fp"] += int((be > t).sum()); r["tp"] += int((ae > t).sum())
            r["k"].append(k); r["ncal"].append(n)
            r["exp"] += be.size * (k + 1) / (n + 1)   # E[FPR] for exchangeable negatives
    for r in res.values():
        r["fpr"] = r["fp"] / b.size; r["tpr"] = r["tp"] / a.size; r["exp"] /= b.size
    return res


def insample(a, b):
    out = {}
    for lab, num, den in FPRS:
        t, k, n = threshold(np.sort(b), num, den)
        out[lab] = dict(tpr=float((a > t).mean()), fpr=float((b > t).mean()), k=k, n_neg=n,
                        exp_oos_fpr=(k + 1) / (n + 1))
    return out


def conformal_p(cal_sorted, s):
    """p = (1 + #{calibration negatives with score >= s}) / (1 + n_cal)."""
    n = cal_sorted.size
    return (1.0 + n - np.searchsorted(cal_sorted, s, side="left")) / (1.0 + n)


def bh_reject(p, q):
    """Benjamini-Hochberg step-up: reject p <= p_(k*), k* = max{k : p_(k) <= q k / n}."""
    n = p.size
    if n == 0:
        return np.zeros(0, bool)
    o = np.sort(p)
    ok = np.nonzero(o <= q * np.arange(1, n + 1) / n)[0]
    if ok.size == 0:
        return np.zeros(n, bool)
    return p <= o[ok[-1]]


# ----------------------------------------------------------------------- data
def load():
    cols, rows, arm, pairs = pf.load()
    selN = pf.population(cols, pairs, "NAT1")
    selA = pf.population(cols, pairs, "ALLPAIRS")   # every complete pair (7,803); the generator's "ALL" is ALL_noXF here
    coh = cols["cohort"]
    selAXF = [(i, h) for i, h in selA if coh[i] not in ("X", "F")]
    assert len(selN) == 4832 and len(selA) == 7803
    return cols, arm, {"NAT1": selN, "ALL": selA, "ALL_noXF": selAXF}


def pop_arrays(cols, sel, scores):
    coh = np.array([cols["cohort"][i] for i, _ in sel])
    assert all(cols["cohort"][i] == cols["cohort"][h] for i, h in sel)
    S = {}
    for key in scores:
        a, b = pf.arrays(cols, key, sel)
        assert np.isfinite(a).all() and np.isfinite(b).all(), key
        S[key] = (a, b)
    return coh, S


# --------------------------------------------------------------------- Part A
def part_a(cols, sels, folds_cache):
    out = {"population": {}}
    for pop in ("NAT1", "ALL", "ALL_noXF"):
        sel = sels[pop]
        coh, S = pop_arrays(cols, sel, SCORES_A)
        units = [("pooled", np.ones(len(sel), bool))]
        if pop == "NAT1":
            units += [(c, coh == c) for c in COHORTS]
        popres = {}
        for K in KS:
            folds = [make_folds(coh, K, rng_for(S_FOLDS, {"NAT1": 0, "ALL": 1, "ALL_noXF": 2}[pop], K, r))
                     for r in range(R_REP)]
            folds_cache[(pop, K)] = folds
            for uname, mask in units:
                for key in SCORES_A:
                    a, b = S[key][0][mask], S[key][1][mask]
                    reps = [crossfit(a, b, f[mask], K) for f in folds]
                    ins = insample(a, b)
                    d = popres.setdefault(uname, {}).setdefault(key, {})
                    d["n_pairs"] = int(mask.sum()); d["insample"] = ins
                    for lab, _, _ in FPRS:
                        fpr = [r[lab]["fpr"] for r in reps]; tpr = [r[lab]["tpr"] for r in reps]
                        exp = [r[lab]["exp"] for r in reps]
                        ks = sorted({k for r in reps for k in r[lab]["k"]})
                        nc = sorted({n for r in reps for n in r[lab]["ncal"]})
                        d.setdefault(f"K{K}", {})[lab] = {
                            "realized_fpr": summ(fpr), "tpr": summ(tpr), "expected_fpr_exchangeable": summ(exp),
                            "n_cal_range": [nc[0], nc[-1]], "k_values": ks,
                            "tpr_change_vs_insample": float(np.mean(tpr) - ins[lab]["tpr"]),
                            "fp_tp_counts_mean": [float(np.mean([r[lab]["fp"] for r in reps])),
                                                  float(np.mean([r[lab]["tp"] for r in reps]))]}
        # pair bootstrap (K = 5): sampling variability of the cross-fitted and in-sample rates
        K = K_PRIMARY
        for ui, (uname, mask) in enumerate(units):
            idx_u = np.nonzero(mask)[0]; strata_u = coh[idx_u]
            acc = {key: {lab: {"cf_fpr": [], "cf_tpr": [], "in_tpr": [], "d_tpr": []} for lab, _, _ in FPRS}
                   for key in SCORES_A}
            for bb in range(B_BOOT):
                rng = rng_for(S_BOOT_A, {"NAT1": 0, "ALL": 1, "ALL_noXF": 2}[pop], ui, bb)
                j = rng.integers(0, idx_u.size, idx_u.size)
                fold_b = make_folds(strata_u, K, rng)[j]
                for key in SCORES_A:
                    a, b = S[key][0][idx_u][j], S[key][1][idx_u][j]
                    cf = crossfit(a, b, fold_b, K); ins = insample(a, b)
                    for lab, _, _ in FPRS:
                        z = acc[key][lab]
                        z["cf_fpr"].append(cf[lab]["fpr"]); z["cf_tpr"].append(cf[lab]["tpr"])
                        z["in_tpr"].append(ins[lab]["tpr"]); z["d_tpr"].append(cf[lab]["tpr"] - ins[lab]["tpr"])
            for key in SCORES_A:
                for lab, _, _ in FPRS:
                    z = acc[key][lab]
                    popres[uname][key][f"K{K}"][lab]["pair_bootstrap"] = {
                        "draws": B_BOOT, "realized_fpr_ci": summ(z["cf_fpr"]), "tpr_ci": summ(z["cf_tpr"]),
                        "insample_tpr_ci": summ(z["in_tpr"]), "tpr_change_ci": summ(z["d_tpr"])}
        out["population"][pop] = popres
        print(f"  Part A {pop}: done", flush=True)
    return out


# --------------------------------------------------------------------- Part B
def part_b(cols, arm, sels):
    coh_all = cols["cohort"]
    out = {}
    selN = sels["NAT1"]
    coh, S = pop_arrays(cols, selN, SCORES_B)
    ntok = np.asarray(cols["injected__n_scored"], float)
    for dom, (cohs, ncohs) in NULL_DOMAINS.items():
        nidx = np.array([i for i in range(len(arm)) if arm[i] == "null" and coh_all[i] in ncohs])
        units = [(f"{dom} ({'+'.join(cohs)})", np.isin(coh, cohs))] + [(c, coh == c) for c in cohs]
        D = {"null_cohorts": ncohs, "n_null": int(nidx.size), "units": {}}
        for ui, (uname, mask) in enumerate(units):
            pidx = np.array([p for p, m in zip(selN, mask) if m])
            U = {"n_pairs": int(mask.sum()),
                 "median_tokens": {"members": float(np.median(ntok[pidx[:, 0]])),
                                   "partners": float(np.median(ntok[pidx[:, 1]])),
                                   "null": float(np.median(ntok[nidx]))}}
            for key in SCORES_B:
                a, b = S[key][0][mask], S[key][1][mask]
                z = np.asarray(cols[key], float)[nidx]
                assert np.isfinite(z).all()
                zs = np.sort(z); bs = np.sort(b)
                R = {"distribution": {nm: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                           "median": float(np.median(v)), "n": int(v.size)}
                                      for nm, v in (("members", a), ("partners", b), ("null", z))},
                     "auc_partners_vs_null": float(pf.ps.auc(b, z))}
                for lab, num, den in FPRS:
                    t, k, n = threshold(zs, num, den)
                    tp, kp, npn = threshold(bs, num, den)
                    R[lab] = {"null_threshold": float(t), "k": k, "n_cal_null": n,
                              "realized_fpr_partners": float((b > t).mean()), "fp_partners": int((b > t).sum()),
                              "tpr_members": float((a > t).mean()), "tp_members": int((a > t).sum()),
                              "expected_fpr_if_exchangeable": (k + 1) / (n + 1),
                              "partner_threshold_insample": float(tp), "partner_k": kp,
                              "insample_tpr": float((a > tp).mean()),
                              "null_share_above_partner_threshold": float((z > tp).mean())}
                # bootstrap: resample pairs of the unit and null documents independently
                boot = {lab: {"fpr": [], "tpr": []} for lab, _, _ in FPRS}; bauc = []
                for bb in range(B_BOOT):
                    rng = rng_for(S_BOOT_B, list(NULL_DOMAINS).index(dom), ui, SCORES_B.index(key), bb)
                    j = rng.integers(0, a.size, a.size); jz = rng.integers(0, z.size, z.size)
                    zz = np.sort(z[jz])
                    for lab, num, den in FPRS:
                        t, _, _ = threshold(zz, num, den)
                        boot[lab]["fpr"].append((b[j] > t).mean()); boot[lab]["tpr"].append((a[j] > t).mean())
                    if bb < 200:
                        bauc.append(pf.ps.auc(b[j], z[jz]))
                for lab, _, _ in FPRS:
                    R[lab]["bootstrap"] = {"draws": B_BOOT, "realized_fpr_ci": summ(boot[lab]["fpr"]),
                                           "tpr_ci": summ(boot[lab]["tpr"])}
                R["auc_partners_vs_null_ci"] = summ(bauc); R["auc_bootstrap_draws"] = len(bauc)
                U[key] = R
            D["units"][uname] = U
        out[dom] = D
    print("  Part B: done", flush=True)
    return out


# --------------------------------------------------------------------- Part C
def screen_unit(a_by, b_by, fold_list, K, unit_code, scores, prevs, tag, oracle=True):
    """All pools for one unit. Returns {score: {pi: {q: list of per-pool records}}}."""
    rec = {key: {pi: {q: [] for q in QS} for pi in prevs} for key in scores}
    for r, fold in enumerate(fold_list):
        for j in range(K):
            ev = np.nonzero(fold == j)[0]; ca = np.nonzero(fold != j)[0]
            n0 = ev.size
            draws = {}
            for pi_i, pi in enumerate(prevs):
                rng = rng_for(S_DRAW_C, tag, unit_code, r, j, pi_i)
                if pi >= 1.0:
                    m_exp = float("inf"); m = n0; chosen = ev.copy()
                else:
                    m_exp = pi / (1.0 - pi) * n0
                    m = int(math.floor(m_exp)); m += int(rng.random() < (m_exp - m)); m = min(m, n0)
                    chosen = rng.choice(ev, m, replace=False) if m else np.zeros(0, int)
                draws[pi] = (m_exp, m, chosen)
            for key in scores:
                a, b = a_by[key], b_by[key]
                cal = np.sort(b[ca])
                p_null = conformal_p(cal, b[ev])
                for pi in prevs:
                    m_exp, m, chosen = draws[pi]
                    pn = p_null if pi < 1.0 else np.zeros(0)
                    p = np.concatenate([pn, conformal_p(cal, a[chosen])])
                    N = p.size
                    for q in QS:
                        rej = bh_reject(p, q)
                        fd = int(rej[:pn.size].sum()); td = int(rej[pn.size:].sum()); R = fd + td
                        # smallest rejection set BH can form: p >= 1/(n_cal+1) so k >= N / (q (n_cal+1))
                        kmin = int(math.ceil(N / (q * (cal.size + 1)) - 1e-12))
                        # BH at level q/pi0 with the pool's true non-member share pi0 (prevalence known)
                        pi0 = pn.size / N
                        if pi0 > 0 and oracle:
                            rj2 = bh_reject(p, min(1.0, q / pi0))
                            fd2 = int(rj2[:pn.size].sum()); td2 = int(rj2[pn.size:].sum()); R2 = fd2 + td2
                        else:
                            fd2 = td2 = R2 = -1
                        rec[key][pi][q].append((r, j, m_exp, m, pn.size, cal.size, R, td, fd,
                                                fd / R if R else 0.0, kmin, R2, td2, fd2,
                                                (fd2 / R2 if R2 > 0 else 0.0) if R2 >= 0 else -1.0))
    return rec


def summarize_pools(lst, K):
    A = np.array(lst, float)   # r, j, m_exp, m, n_null, n_cal, R, TD, FD, FDP, kmin, R2, TD2, FD2, FDP2
    r = A[:, 0].astype(int)
    tot = np.array([A[r == rr][:, [3, 4, 6, 7, 8, 11, 12, 13]].sum(axis=0) for rr in sorted(set(r.tolist()))])
    finite_mexp = A[np.isfinite(A[:, 2]), 2]
    known = None
    if (A[:, 11] >= 0).all():
        known = {"discoveries": summ(A[:, 11]), "true_discoveries": summ(A[:, 12]),
                 "false_discoveries": summ(A[:, 13]), "realized_fdr": float(A[:, 14].mean()),
                 "realized_fdr_mc_se": float(A[:, 14].std(ddof=1) / math.sqrt(A.shape[0])),
                 "per_repeat_totals": {"true_discoveries": summ(tot[:, 6]), "false_discoveries": summ(tot[:, 7])}}
    return {
        "min_rejection_count": summ(A[:, 10]),
        "bh_known_prevalence": known,
        "pools": int(A.shape[0]),
        "n_null_per_pool": [int(A[:, 4].min()), int(A[:, 4].max())],
        "n_cal_per_pool": [int(A[:, 5].min()), int(A[:, 5].max())],
        "expected_members_per_pool": float(finite_mexp.mean()) if finite_mexp.size else None,
        "members_per_pool": summ(A[:, 3]),
        "share_pools_without_members": float((A[:, 3] == 0).mean()),
        "realized_prevalence": summ(A[:, 3] / (A[:, 3] + A[:, 4])),
        "discoveries": summ(A[:, 6]), "true_discoveries": summ(A[:, 7]), "false_discoveries": summ(A[:, 8]),
        "fdp": summ(A[:, 9]), "realized_fdr": float(A[:, 9].mean()),
        "realized_fdr_mc_se": float(A[:, 9].std(ddof=1) / math.sqrt(A.shape[0])),
        "share_pools_any_discovery": float((A[:, 6] > 0).mean()),
        "share_pools_any_false_discovery": float((A[:, 8] > 0).mean()),
        "ratio_false_to_discoveries_all_pools": float(A[:, 8].sum() / A[:, 6].sum()) if A[:, 6].sum() else None,
        "per_repeat_totals": {"members": summ(tot[:, 0]), "non_members": summ(tot[:, 1]),
                              "discoveries": summ(tot[:, 2]), "true_discoveries": summ(tot[:, 3]),
                              "false_discoveries": summ(tot[:, 4])},
    }


def part_c(cols, sels, folds_cache):
    K = K_PRIMARY
    selN = sels["NAT1"]
    coh, S = pop_arrays(cols, selN, SCORES_C)
    folds = folds_cache[("NAT1", K)]
    prevs = PREVS + [1.0]
    units = [("pooled", np.ones(len(selN), bool))] + [(c, coh == c) for c in COHORTS]
    out = {"units": {}, "stratified_sum": {}, "paper_rule": {}, "label_shuffle": {}}
    raw = {}
    for ui, (uname, mask) in enumerate(units):
        a_by = {k: S[k][0][mask] for k in SCORES_C}; b_by = {k: S[k][1][mask] for k in SCORES_C}
        rec = screen_unit(a_by, b_by, [f[mask] for f in folds], K, ui, SCORES_C, prevs, tag=0)
        raw[uname] = rec
        out["units"][uname] = {key: {str(pi): {str(q): summarize_pools(rec[key][pi][q], K) for q in QS}
                                     for pi in prevs} for key in SCORES_C}
        print(f"  Part C {uname}: done", flush=True)
    # domain-stratified screening: one screen per cohort, totals summed over cohorts per repeat
    for key in SCORES_C:
        out["stratified_sum"][key] = {}
        for pi in prevs:
            out["stratified_sum"][key][str(pi)] = {}
            for q in QS:
                td = np.zeros(R_REP); fd = np.zeros(R_REP); td2 = np.zeros(R_REP); fd2 = np.zeros(R_REP)
                for c in COHORTS:
                    for row in raw[c][key][pi][q]:
                        r = int(row[0]); td[r] += row[7]; fd[r] += row[8]; td2[r] += row[12]; fd2[r] += row[13]
                d = {"true_discoveries": summ(td), "false_discoveries": summ(fd),
                     "fdp_of_totals": summ(np.where(td + fd > 0, fd / np.maximum(td + fd, 1), 0.0))}
                if pi < 1.0:
                    d["bh_known_prevalence"] = {"true_discoveries": summ(td2), "false_discoveries": summ(fd2)}
                out["stratified_sum"][key][str(pi)][str(q)] = d
    # the paper's current rule, on the whole population, and its equivalence with BH on a
    # members-only pool with p = (1 + #{non-members > s}) / (1 + n_neg)
    for uname, mask in units:
        out["paper_rule"][uname] = {}
        for key in SCORES_C + ["z_sibling"]:
            a, b = pf.arrays(cols, key, [p for p, m in zip(selN, mask) if m])
            row = {}
            for q in QS:
                n_named = pf.fdr_named(a, b, q)
                bs = np.sort(b)
                p_strict = (1.0 + bs.size - np.searchsorted(bs, a, side="right")) / (1.0 + bs.size)
                n_bh = int(bh_reject(p_strict, q).sum())
                assert n_bh == n_named, (uname, key, q, n_bh, n_named)
                row[str(q)] = {"named": n_named, "bh_members_only_pool": n_bh, "n_members": int(a.size),
                               "n_nonmembers": int(b.size)}
                if n_named:
                    t = np.sort(a)[::-1][n_named - 1]          # score of the last member named
                    v = int((b > t).sum())
                    row[str(q)]["nonmembers_above_last_named"] = v
                    row[str(q)]["insample_fdp_if_partners_in_pool"] = v / (v + n_named)
            out["paper_rule"][uname][key] = row
    # sanity (iii): labels shuffled within pairs once; every candidate is then a null
    rng = rng_for(S_SHUFFLE)
    swap = rng.random(len(selN)) < 0.5
    out["label_shuffle"]["seed"] = [SEED, S_SHUFFLE]; out["label_shuffle"]["n_swapped"] = int(swap.sum())
    a_by = {k: np.where(swap, S[k][1], S[k][0]) for k in SCORES_C}
    b_by = {k: np.where(swap, S[k][0], S[k][1]) for k in SCORES_C}
    rec = screen_unit(a_by, b_by, folds, K, 0, SCORES_C, PREVS, tag=1, oracle=False)
    for key in SCORES_C:
        out["label_shuffle"][key] = {}
        for pi in PREVS:
            out["label_shuffle"][key][str(pi)] = {}
            for q in QS:
                A = np.array(rec[key][pi][q], float)
                out["label_shuffle"][key][str(pi)][str(q)] = {
                    "pools": int(A.shape[0]),
                    "fdr_under_global_null": float((A[:, 6] > 0).mean()),   # FDP = 1 whenever R > 0
                    "fdr_under_global_null_mc_se": float((A[:, 6] > 0).std(ddof=1) / math.sqrt(A.shape[0])),
                    "mean_discoveries": float(A[:, 6].mean()),
                    "mean_label_fdp": float(A[:, 9].mean())}
    print("  Part C label shuffle: done", flush=True)
    return out


# ------------------------------------------------------------- reproduction
def reproduction(cols, sels):
    selN = sels["NAT1"]
    rep = {}
    for key in ("ratio_oracle", "z_oracle"):
        a, b = pf.arrays(cols, key, selN)
        t1, k1, n1 = pf.tpr_at(a, b, 0.01); t01, k01, _ = pf.tpr_at(a, b, 0.001)
        ins = insample(a, b)
        assert abs(ins["1%"]["tpr"] - t1) < 1e-15 and ins["1%"]["k"] == k1
        assert abs(ins["0.1%"]["tpr"] - t01) < 1e-15 and ins["0.1%"]["k"] == k01
        rep[key] = {"tpr_1pct": t1, "k_1pct": k1, "tpr_0.1pct": t01, "k_0.1pct": k01, "n_neg": n1,
                    "fdr_named_0.05": pf.fdr_named(a, b, 0.05), "fdr_named_0.2": pf.fdr_named(a, b, 0.2)}
    a, b = pf.arrays(cols, "z_oracle", [p for p in selN if cols["cohort"][p[0]] == "M"])
    rep["z_oracle_M_fdr_named_0.05"] = pf.fdr_named(a, b, 0.05); rep["n_M"] = int(a.size)
    # every in-sample TPR used here agrees with pf.tpr_at
    for pop in ("NAT1", "ALL", "ALL_noXF"):
        coh, S = pop_arrays(cols, sels[pop], SCORES_A)
        for key in SCORES_A:
            for c in ([None] + COHORTS if pop == "NAT1" else [None]):
                m = np.ones(coh.size, bool) if c is None else coh == c
                aa, bb = S[key][0][m], S[key][1][m]
                for lab, num, den in FPRS:
                    assert insample(aa, bb)[lab]["tpr"] == pf.tpr_at(aa, bb, num / den)[0]
    # the generator's macros (figures/paper_figures.py output), when present
    mac = {}
    p = pf.TABS / "numbers.tex"
    if p.exists():
        for ln in p.read_text().splitlines():
            mm = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}", ln)
            if mm:
                mac[mm.group(1)] = mm.group(2)
        chk = {"tpronenat": f"{100*rep['ratio_oracle']['tpr_1pct']:.1f}",
               "tprpointonenat": f"{100*rep['ratio_oracle']['tpr_0.1pct']:.2f}",
               "tpronezorcnat": f"{100*rep['z_oracle']['tpr_1pct']:.1f}",
               "tprpointonezorcnat": f"{100*rep['z_oracle']['tpr_0.1pct']:.2f}",
               "fdrfive": str(rep["ratio_oracle"]["fdr_named_0.05"]),
               "fdrtwenty": str(rep["ratio_oracle"]["fdr_named_0.2"]),
               "fdrfivezorc": str(rep["z_oracle"]["fdr_named_0.05"]),
               "cohmnamedzorc": str(rep["z_oracle_M_fdr_named_0.05"])}
        rep["paper_macros"] = {k: {"paper": mac.get(k), "recomputed": v, "match": mac.get(k) == v}
                               for k, v in chk.items()}
        assert all(v["match"] for v in rep["paper_macros"].values()), rep["paper_macros"]
    return rep


# -------------------------------------------------------------------- report
def pct(x, d=2):
    return f"{100*x:.{d}f}"


def ci(s, d=2):
    return f"{pct(s['mean'], d)} [{pct(s['lo'], d)}, {pct(s['hi'], d)}]"


def write_md(J):
    A = J["part_a"]["population"]; B = J["part_b"]; C = J["part_c"]
    L = []
    w = L.append
    w("# Cross-fitted operating points and mixed-pool screening")
    w("")
    w("Generated by `analysis/screening_analysis.py` (one command, seed "
      f"{SEED}; every random stream is `numpy.random.default_rng([{SEED}, stream, ...])`, streams listed in the JSON). "
      "Analysis table loaded through `paper_figures.load()` (post-training exclusions applied, standardized-ratio "
      "columns merged). NAT1 = 4,832 pairs; ALL = 7,803 pairs; ALL_noXF = ALL without cohorts X and F "
      f"({J['meta']['n_pairs']['ALL_noXF']:,} pairs). All rates in percent.")
    w("")
    w("## Main results")
    w("")
    P = A["NAT1"]["pooled"]
    for key in ("ratio_oracle", "z_oracle"):
        x1 = P[key]["K5"]["1%"]; x01 = P[key]["K5"]["0.1%"]
        w(f"* {key}, NAT1, one threshold: cross-fitted TPR at 1% FPR {ci(x1['tpr'])} over repeats, pair-bootstrap "
          f"[{pct(x1['pair_bootstrap']['tpr_ci']['lo'])}, {pct(x1['pair_bootstrap']['tpr_ci']['hi'])}], realized FPR "
          f"{pct(x1['realized_fpr']['mean'], 3)}% (threshold = {x1['k_values'][0] + 1}th largest of "
          f"{x1['n_cal_range'][0]}-{x1['n_cal_range'][1]} calibration non-members); in-sample {pct(P[key]['insample']['1%']['tpr'])}. "
          f"At 0.1%: {ci(x01['tpr'])}, realized FPR {pct(x01['realized_fpr']['mean'], 3)}%; in-sample {pct(P[key]['insample']['0.1%']['tpr'])}.")
    pr = C["paper_rule"]["pooled"]; u = C["units"]
    for key in ("ratio_oracle", "z_oracle"):
        parts = []
        for pi in PREVS:
            t = u["pooled"][key][str(pi)]["0.05"]["per_repeat_totals"]
            parts.append(f"{pi:g}: {t['true_discoveries']['mean']:.1f} true / {t['false_discoveries']['mean']:.1f} false")
        kb = u["pooled"][key]["0.5"]["0.05"]["bh_known_prevalence"]["per_repeat_totals"]
        w(f"* {key}, NAT1 pooled, BH at q = 0.05 in mixed pools (totals over the five evaluation folds, mean over "
          f"repeats), by prevalence: " + "; ".join(parts) + f". With the prevalence known (BH at q/pi0), 50% pools: "
          f"{kb['true_discoveries']['mean']:.1f} true / {kb['false_discoveries']['mean']:.1f} false. "
          f"The paper's rule gives {pr[key]['0.05']['named']}.")
    for key in ("ratio_oracle", "z_oracle"):
        parts = []
        for pi in PREVS:
            t = u["M"][key][str(pi)]["0.05"]["per_repeat_totals"]
            parts.append(f"{pi:g}: {t['true_discoveries']['mean']:.1f} / {t['false_discoveries']['mean']:.1f}")
        w(f"* {key}, arXiv papers (M, {A['NAT1']['M'][key]['n_pairs']:,} pairs, within-cohort calibration), BH at q = 0.05, true / "
          f"false discoveries by prevalence: " + "; ".join(parts) + f". The paper's rule gives "
          f"{C['paper_rule']['M'][key]['0.05']['named']}.")
    w("")
    w("## What each procedure assumes")
    w("")
    w("* **In-sample operating point (paper).** Threshold = the score admitting k = floor(f n) of the n matched "
      "non-members (strict >), and the FPR is read on those same n non-members, so the in-sample FPR equals k/n by "
      "construction (no ties). For a new non-member exchangeable with the n, the probability of exceeding the "
      "threshold is (k+1)/(n+1), which is the out-of-sample FPR this threshold actually carries.")
    w("* **Cross-fitted operating point.** Pairs are split into K folds, balanced within cohort. The threshold is set "
      "with the same rule on the calibration folds' non-members only and applied to the evaluation fold's members "
      "and non-members; decisions are pooled over the K evaluation folds, so every pair is scored once by a threshold "
      "that did not see it. Expected realized FPR under exchangeability = average over folds of (k+1)/(n_cal+1).")
    w("* **Paper's 'nameable at FDR q' rule (`pf.fdr_named`).** Verified numerically here to be identical to "
      "Benjamini-Hochberg applied to a pool containing *only the members*, with p = (1 + #{non-members > s})/(1 + n_neg) "
      "computed against all matched non-members. The pool screened contains no non-member, so every discovery is "
      "true by construction and the count is the power of BH at 100% prevalence. Equivalently, with n_pos = n_neg the "
      "rule names the k top-scoring members whenever (v+1)/k <= q (n+1)/n, where v is the number of matched "
      "non-members above the k-th member: if those non-members were screened too, the same threshold would add v "
      "false discoveries, an in-sample FDP of v/(k+v), so the count approximates BH at level q/pi0 in a pool that is "
      "50% members with the prevalence known, except that the threshold is selected on the very non-members that "
      "define the null.")
    w("* **Mixed-pool screening (this note).** Each candidate pool = all evaluation-fold non-members plus members of "
      "the same fold drawn without replacement so that the member fraction is the stated prevalence (member count "
      "m = pi/(1-pi) n_0, rounded stochastically to an integer with the correct expectation; never padded). "
      "Conformal p-values p = (1 + #{calibration non-members >= s})/(1 + n_cal) from the calibration folds; BH at q. "
      "If calibration and pool non-members are exchangeable, these p-values are PRDS and BH controls FDR at "
      "q n_0/(n_0+m) <= q (Bates, Candes, Lei, Romano and Sesia, 2023, Ann. Statist. 51:149-178). The assumption is "
      "exchangeability of non-members across pairs within the population (cohort-balanced folds make the mixture "
      "identical); the guarantee is in expectation over calibration draws, not per pool.")
    w("")
    w("Intervals: *repeat* = 2.5/97.5 percentiles over the R = 50 fold assignments (fold-assignment variability "
      "only, not sampling variability); *bootstrap* = percentile interval over "
      f"{B_BOOT} pair-bootstrap draws, each with a fresh fold assignment (pairs resampled within the unit; "
      "both documents of a pair move together; duplicated pairs stay in the same fold).")
    w("")
    # ---- Part A tables
    w("## Part A. Cross-fitted true-positive rates (K = 5, primary)")
    w("")
    w("Caveat on the bootstrap columns: at 0.1% the threshold is the 1st-4th largest calibration non-member, and "
      "the nonparametric bootstrap is not consistent for such extreme order statistics (resampling duplicates "
      "documents, which lowers the extreme order statistics and inflates both FPR and TPR; compare the bootstrap "
      "means with the repeat means). Treat the 0.1% bootstrap intervals as indicative only. At 1% the bias is small "
      f"(ratio_oracle, pooled: bootstrap mean FPR {pct(A['NAT1']['pooled']['ratio_oracle']['K5']['1%']['pair_bootstrap']['realized_fpr_ci']['mean'], 3)}% "
      f"against {pct(A['NAT1']['pooled']['ratio_oracle']['K5']['1%']['realized_fpr']['mean'], 3)}% over repeats).")
    w("")
    w("K = 5 is primary: every cohort keeps at least 200 calibration non-members (L: 200; with K = 2 L has 125). "
      "K = 2 results are in the JSON and in the last table of this part.")
    w("")
    for key in ("ratio_oracle", "z_oracle"):
        for lab, _, _ in FPRS:
            w(f"### {key}, nominal FPR {lab}, NAT1")
            w("")
            w("| unit | pairs | n_cal | k | E[FPR] exch. | realized FPR (repeat) | realized FPR (bootstrap) | TPR cross-fit (repeat) | TPR cross-fit (bootstrap) | in-sample TPR [bootstrap] (k/n) | change in TPR [bootstrap] |")
            w("|---|---|---|---|---|---|---|---|---|---|---|")
            for u in ["pooled"] + COHORTS:
                d = A["NAT1"][u][key]; x = d["K5"][lab]; ins = d["insample"][lab]; pb = x["pair_bootstrap"]
                w(f"| {u} | {d['n_pairs']:,} | {x['n_cal_range'][0]}-{x['n_cal_range'][1]} | "
                  f"{','.join(map(str, x['k_values']))} | {pct(x['expected_fpr_exchangeable']['mean'], 3)} | "
                  f"{ci(x['realized_fpr'], 3)} | {ci(pb['realized_fpr_ci'], 3)} | {ci(x['tpr'])} | {ci(pb['tpr_ci'])} | "
                  f"{pct(ins['tpr'])} [{pct(pb['insample_tpr_ci']['lo'])}, {pct(pb['insample_tpr_ci']['hi'])}] "
                  f"({ins['k']}/{ins['n_neg']}) | {100*x['tpr_change_vs_insample']:+.2f} "
                  f"[{100*pb['tpr_change_ci']['lo']:+.2f}, {100*pb['tpr_change_ci']['hi']:+.2f}] |")
            w("")
    w("### All six scores, NAT1 pooled, K = 5")
    w("")
    w("| score | FPR | in-sample TPR (k/n) | cross-fit TPR (repeat) | cross-fit TPR (bootstrap) | realized FPR (repeat) | E[FPR] exch. |")
    w("|---|---|---|---|---|---|---|")
    for key in SCORES_A:
        d = A["NAT1"]["pooled"][key]
        for lab, _, _ in FPRS:
            x = d["K5"][lab]; ins = d["insample"][lab]
            w(f"| {key} | {lab} | {pct(ins['tpr'])} ({ins['k']}/{ins['n_neg']}) | {ci(x['tpr'])} | "
              f"{ci(x['pair_bootstrap']['tpr_ci'])} | {ci(x['realized_fpr'], 3)} | {pct(x['expected_fpr_exchangeable']['mean'], 3)} |")
    w("")
    w("### Per cohort, all six scores, K = 5, 1% FPR (cross-fit TPR mean over repeats; in-sample in parentheses)")
    w("")
    w("| score | " + " | ".join(COHORTS) + " |")
    w("|---|" + "---|" * len(COHORTS))
    for key in SCORES_A:
        cells = []
        for c in COHORTS:
            d = A["NAT1"][c][key]
            cells.append(f"{pct(d['K5']['1%']['tpr']['mean'], 1)} ({pct(d['insample']['1%']['tpr'], 1)})")
        w(f"| {key} | " + " | ".join(cells) + " |")
    w("")
    w("### Secondary: K = 2 (NAT1) and the ALL populations (K = 5), pooled")
    w("")
    w("| population | K | score | FPR | n_cal | k | realized FPR (repeat) | cross-fit TPR (repeat) | in-sample TPR (k/n) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for pop, K in (("NAT1", 2), ("ALL", 5), ("ALL_noXF", 5), ("ALL", 2), ("ALL_noXF", 2)):
        for key in ("ratio_oracle", "z_oracle", "ratio_sibling", "z_sibling"):
            d = A[pop]["pooled"][key]
            for lab, _, _ in FPRS:
                x = d[f"K{K}"][lab]; ins = d["insample"][lab]
                w(f"| {pop} | {K} | {key} | {lab} | {x['n_cal_range'][0]}-{x['n_cal_range'][1]} | "
                  f"{','.join(map(str, x['k_values']))} | {ci(x['realized_fpr'], 3)} | {ci(x['tpr'])} | "
                  f"{pct(ins['tpr'])} ({ins['k']}/{ins['n_neg']}) |")
    w("")
    w("K = 2, NAT1 per cohort, 1% FPR (cross-fit TPR mean; n_cal; k):")
    w("")
    w("| score | " + " | ".join(COHORTS) + " |")
    w("|---|" + "---|" * len(COHORTS))
    for key in ("ratio_oracle", "z_oracle"):
        cells = []
        for c in COHORTS:
            x = A["NAT1"][c][key]["K2"]["1%"]
            cells.append(f"{pct(x['tpr']['mean'], 1)} (n_cal {x['n_cal_range'][0]}-{x['n_cal_range'][1]}; k {','.join(map(str, x['k_values']))})")
        w(f"| {key} | " + " | ".join(cells) + " |")
    w("")
    # ---- Part B
    w("## Part B. Threshold calibrated on the domain-matched null pool")
    w("")
    for dom, D in B.items():
        w(f"### {dom}: null pool {'+'.join(D['null_cohorts'])} (n = {D['n_null']:,})")
        w("")
        w("| unit | score | FPR | k / n_null | realized FPR on partners [bootstrap] | TPR on members [bootstrap] | in-sample partner-calibrated TPR | null share above partner threshold |")
        w("|---|---|---|---|---|---|---|---|")
        for u, U in D["units"].items():
            for key in SCORES_B:
                for lab, _, _ in FPRS:
                    x = U[key][lab]; bt = x["bootstrap"]
                    w(f"| {u} | {key} | {lab} | {x['k']}/{x['n_cal_null']} | {pct(x['realized_fpr_partners'])} "
                      f"({x['fp_partners']}/{U['n_pairs']}) [{pct(bt['realized_fpr_ci']['lo'])}, {pct(bt['realized_fpr_ci']['hi'])}] | "
                      f"{pct(x['tpr_members'])} ({x['tp_members']}/{U['n_pairs']}) [{pct(bt['tpr_ci']['lo'])}, {pct(bt['tpr_ci']['hi'])}] | "
                      f"{pct(x['insample_tpr'])} | {pct(x['null_share_above_partner_threshold'])} |")
        w("")
        w("| unit | score | members mean (SD) | partners mean (SD) | null mean (SD) | AUC partners vs null [bootstrap] | median tokens member / partner / null |")
        w("|---|---|---|---|---|---|---|")
        for u, U in D["units"].items():
            for key in SCORES_B:
                ds = U[key]["distribution"]; f = "{:+.4f}" if key.startswith("ratio") else "{:+.2f}"
                g = "{:.4f}" if key.startswith("ratio") else "{:.2f}"
                au = U[key]["auc_partners_vs_null_ci"]; mt = U["median_tokens"]
                w(f"| {u} | {key} | {f.format(ds['members']['mean'])} ({g.format(ds['members']['sd'])}) | "
                  f"{f.format(ds['partners']['mean'])} ({g.format(ds['partners']['sd'])}) | "
                  f"{f.format(ds['null']['mean'])} ({g.format(ds['null']['sd'])}) | "
                  f"{U[key]['auc_partners_vs_null']:.3f} [{au['lo']:.3f}, {au['hi']:.3f}] | "
                  f"{mt['members']:.0f} / {mt['partners']:.0f} / {mt['null']:.0f} |")
        w("")
    # ---- Part C
    w("## Part C. Mixed-pool screening with Benjamini-Hochberg (K = 5, R = 50, 250 pools per unit)")
    w("")
    w("Per-pool numbers are means over the 250 pools with 2.5/97.5 percentiles; *totals* sum the 5 evaluation-fold "
      "pools of a repeat (so they cover the whole population once) and give mean [2.5, 97.5] over the 50 repeats. "
      "Realized FDR = mean FDP over pools (FDP = 0 when nothing is discovered). Prevalence 1.0 = members only, "
      "the pool the paper's rule screens.")
    w("")
    for key in SCORES_C:
        w(f"### {key}")
        w("")
        w("| unit | prevalence | q | members per pool (expected) | pools without members | min. rejections | TD per pool | FD per pool | realized FDR (MC s.e.) | TD total | FD total | known-prevalence BH: TD / FD total, FDR | paper rule (whole unit) |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for u in ["pooled"] + COHORTS:
            for pi in PREVS + [1.0]:
                for q in QS:
                    s = C["units"][u][key][str(pi)][str(q)]
                    em = "all" if s["expected_members_per_pool"] is None else f"{s['expected_members_per_pool']:.2f}"
                    pr = C["paper_rule"][u][key][str(q)]["named"] if pi in (0.5, 1.0) else ""
                    t = s["per_repeat_totals"]; kb = s["bh_known_prevalence"]
                    kcell = "" if kb is None else (f"{kb['per_repeat_totals']['true_discoveries']['mean']:.1f} / "
                                                   f"{kb['per_repeat_totals']['false_discoveries']['mean']:.1f}, {kb['realized_fdr']:.3f}")
                    w(f"| {u} | {pi:g} | {q:g} | {s['members_per_pool']['mean']:.2f} ({em}) | "
                      f"{100*s['share_pools_without_members']:.0f}% | {s['min_rejection_count']['mean']:.1f} | {s['true_discoveries']['mean']:.2f} "
                      f"[{s['true_discoveries']['lo']:.0f}, {s['true_discoveries']['hi']:.0f}] | "
                      f"{s['false_discoveries']['mean']:.2f} [{s['false_discoveries']['lo']:.0f}, {s['false_discoveries']['hi']:.0f}] | "
                      f"{s['realized_fdr']:.3f} ({s['realized_fdr_mc_se']:.3f}) | "
                      f"{t['true_discoveries']['mean']:.1f} [{t['true_discoveries']['lo']:.0f}, {t['true_discoveries']['hi']:.0f}] | "
                      f"{t['false_discoveries']['mean']:.1f} [{t['false_discoveries']['lo']:.0f}, {t['false_discoveries']['hi']:.0f}] | {kcell} | {pr} |")
        w("")
    w("### Domain-stratified screening (one screen per cohort, totals summed over the five cohorts)")
    w("")
    w("BH controls the FDR within each cohort's screen; the 'FDP of totals' column (mean over repeats of the summed "
      "false discoveries over the summed discoveries) is not a quantity BH controls, and at low prevalence it is "
      "dominated by repeats with a handful of discoveries.")
    w("")
    w("| score | prevalence | q | TD total | FD total | FDP of totals | known-prevalence BH: TD / FD total | paper rule, sum over cohorts |")
    w("|---|---|---|---|---|---|---|---|")
    for key in SCORES_C:
        for pi in PREVS + [1.0]:
            for q in QS:
                s = C["stratified_sum"][key][str(pi)][str(q)]
                pr = sum(C["paper_rule"][c][key][str(q)]["named"] for c in COHORTS) if pi in (0.5, 1.0) else ""
                kb = s.get("bh_known_prevalence")
                kcell = "" if kb is None else f"{kb['true_discoveries']['mean']:.1f} / {kb['false_discoveries']['mean']:.1f}"
                w(f"| {key} | {pi:g} | {q:g} | {s['true_discoveries']['mean']:.1f} [{s['true_discoveries']['lo']:.0f}, "
                  f"{s['true_discoveries']['hi']:.0f}] | {s['false_discoveries']['mean']:.1f} [{s['false_discoveries']['lo']:.0f}, "
                  f"{s['false_discoveries']['hi']:.0f}] | {s['fdp_of_totals']['mean']:.3f} | {kcell} | {pr} |")
    w("")
    w("### Paper's rule recomputed (pf.fdr_named on the whole unit; equals BH on a members-only pool)")
    w("")
    w("Cell = named count; in parentheses, the number v of matched non-members scoring above the last named member "
      "(in-sample), i.e. the false discoveries the same threshold would make if the partners were in the pool.")
    w("")
    w("| unit | " + " | ".join(f"{k} q={q:g}" for k in SCORES_C + ['z_sibling'] for q in QS) + " |")
    w("|---|" + "---|" * (2 * (len(SCORES_C) + 1)))
    for u in ["pooled"] + COHORTS:
        cells = []
        for k in SCORES_C + ["z_sibling"]:
            for q in QS:
                x = C["paper_rule"][u][k][str(q)]
                cells.append(str(x["named"]) + (f" ({x['nonmembers_above_last_named']})" if x["named"] else ""))
        w(f"| {u} | " + " | ".join(cells) + " |")
    w("")
    # ---- sanity
    S = J["sanity"]
    w("## Sanity checks")
    w("")
    w("**(i) Realized FPR at the nominal level (NAT1 pooled, K = 5).**")
    w("")
    w("| score | FPR | in-sample FPR k/n | in-sample threshold's E[out-of-sample FPR] (k+1)/(n+1) | cross-fit E[FPR] exch. | cross-fit realized FPR (repeat) |")
    w("|---|---|---|---|---|---|")
    for row in S["i_realized_fpr"]:
        w(f"| {row['score']} | {row['fpr']} | {pct(row['insample_fpr'], 3)} | {pct(row['insample_exp_oos'], 3)} | "
          f"{pct(row['crossfit_expected'], 3)} | {ci(row['crossfit_realized'], 3)} |")
    w("")
    w("**(ii) BH at prevalence 0.5 against the paper's rule (NAT1 pooled, totals over the 5 folds).** "
      + "; ".join(f"{r['score']} q={r['q']:g}: cross-fit mixed pool TD {r['td_total_mean']:.1f} "
                  f"[{r['td_total_lo']:.0f}, {r['td_total_hi']:.0f}], FD {r['fd_total_mean']:.1f}; same pool, BH at q/0.5 "
                  f"(prevalence known) TD {r['known_prevalence_td_total_mean']:.1f}, FD {r['known_prevalence_fd_total_mean']:.1f}; "
                  f"members-only cross-fit {r['td_members_only_mean']:.1f}; paper rule {r['paper_rule']}" for r in S["ii_bh_vs_paper"]))
    w("")
    w("Reading (ii): the paper's count is reproduced by the members-only screen (prevalence 1), which is what "
      "`pf.fdr_named` computes, and approximately by BH at level q/pi0 in a 50% pool, i.e. when the auditor knows "
      "that half the pool are members; standard BH, which does not know the prevalence, controls FDR at q pi0 and "
      "names fewer. Fold-level pools are not a rescaled whole-population screen: here the calibration set is four "
      "times the pool's non-member part, whereas in the paper's rule it equals the number of candidates, and the "
      "smallest rejection set BH can form is N/(q (n_cal+1)) (column 'min. rejections').")
    w("")
    w("**(iii) Labels shuffled within pairs once (every candidate is then a null; FDR = P(any discovery)).**")
    w("")
    w(f"{C['label_shuffle']['n_swapped']:,} of 4,832 pairs swapped (seed {C['label_shuffle']['seed']}). Same folds and "
      "pool construction as the real analysis. At q = 0.05 no pool can reject, because BH's smallest rejection set "
      "(about 6 to 10 candidates, see 'min. rejections') would all have to exceed every calibration non-member; the "
      "informative check is q = 0.20.")
    w("")
    w("| score | prevalence | q | P(any discovery) (MC s.e.) | mean discoveries |")
    w("|---|---|---|---|---|")
    for key in SCORES_C:
        for pi in PREVS:
            for q in QS:
                s = C["label_shuffle"][key][str(pi)][str(q)]
                w(f"| {key} | {pi:g} | {q:g} | {s['fdr_under_global_null']:.3f} ({s['fdr_under_global_null_mc_se']:.3f}) | {s['mean_discoveries']:.3f} |")
    w("")
    w(f"**(iv) FDR bound.** Pools in which the mean realized FDP exceeds q(1-prevalence) (standard BH) or q "
      f"(known-prevalence BH) by more than two Monte-Carlo standard errors: {len(S['iv_fdr_bound_violations'])}. "
      f"Largest ratio of realized FDR to q(1-prevalence): {S['iv_max_ratio_realized_fdr_to_q_pi0'][0]:.3f} "
      f"({', '.join(map(str, S['iv_max_ratio_realized_fdr_to_q_pi0'][1:]))}).")
    w("")
    OUT_MD.write_text("\n".join(L) + "\n")


def sanity(J):
    A = J["part_a"]["population"]["NAT1"]["pooled"]; C = J["part_c"]
    out = {"i_realized_fpr": [], "ii_bh_vs_paper": []}
    for key in SCORES_A:
        for lab, _, _ in FPRS:
            x = A[key]["K5"][lab]; ins = A[key]["insample"][lab]
            out["i_realized_fpr"].append({"score": key, "fpr": lab, "insample_fpr": ins["fpr"],
                                          "insample_exp_oos": ins["exp_oos_fpr"],
                                          "crossfit_expected": x["expected_fpr_exchangeable"]["mean"],
                                          "crossfit_realized": x["realized_fpr"]})
    for key in SCORES_C:
        for q in QS:
            s = C["units"]["pooled"][key]["0.5"][str(q)]["per_repeat_totals"]
            s1 = C["units"]["pooled"][key]["1.0"][str(q)]["per_repeat_totals"]
            out["ii_bh_vs_paper"].append({"score": key, "q": q, "td_total_mean": s["true_discoveries"]["mean"],
                                          "td_total_lo": s["true_discoveries"]["lo"], "td_total_hi": s["true_discoveries"]["hi"],
                                          "fd_total_mean": s["false_discoveries"]["mean"],
                                          "td_members_only_mean": s1["true_discoveries"]["mean"],
                                          "known_prevalence_td_total_mean": C["units"]["pooled"][key]["0.5"][str(q)]["bh_known_prevalence"]["per_repeat_totals"]["true_discoveries"]["mean"],
                                          "known_prevalence_fd_total_mean": C["units"]["pooled"][key]["0.5"][str(q)]["bh_known_prevalence"]["per_repeat_totals"]["false_discoveries"]["mean"],
                                          "paper_rule": C["paper_rule"]["pooled"][key][str(q)]["named"]})
    viol = []
    for u, U in C["units"].items():
        for key, V in U.items():
            for pi, W in V.items():
                for q, s in W.items():
                    if float(pi) >= 1.0:
                        continue
                    pi0 = 1.0 - s["realized_prevalence"]["mean"]
                    if s["realized_fdr"] > float(q) * pi0 + 2 * s["realized_fdr_mc_se"]:
                        viol.append([u, key, pi, q, "BH", s["realized_fdr"], float(q) * pi0])
                    kb = s["bh_known_prevalence"]
                    if kb and kb["realized_fdr"] > float(q) + 2 * kb["realized_fdr_mc_se"]:
                        viol.append([u, key, pi, q, "BH known prevalence", kb["realized_fdr"], float(q)])
    out["iv_fdr_bound_violations"] = viol
    out["iv_max_ratio_realized_fdr_to_q_pi0"] = max(
        (s["realized_fdr"] / (float(q) * (1.0 - s["realized_prevalence"]["mean"])), u, key, pi, q)
        for u, U in C["units"].items() for key, V in U.items() for pi, W in V.items() for q, s in W.items()
        if float(pi) < 1.0)
    return out


def main() -> int:
    t0 = time.time()
    print("loading ...", flush=True)
    cols, arm, sels = load()
    rep = reproduction(cols, sels)
    print(f"  reproduction: ratio_oracle TPR@1% {100*rep['ratio_oracle']['tpr_1pct']:.3f}%, "
          f"@0.1% {100*rep['ratio_oracle']['tpr_0.1pct']:.3f}%; z_oracle {100*rep['z_oracle']['tpr_1pct']:.3f}%", flush=True)
    folds_cache = {}
    J = {"meta": {
        "script": "analysis/screening_analysis.py", "table": str(pf.TABLE.relative_to(paths.DATA)),
        "seed": SEED,
        "random_streams": {"rng": f"numpy.random.default_rng([{SEED}, stream, *indices])",
                           "folds": [S_FOLDS, "population code (NAT1 0, ALL 1, ALL_noXF 2)", "K", "repeat"],
                           "part_a_bootstrap": [S_BOOT_A, "population code", "unit index", "draw"],
                           "part_c_member_draw": [S_DRAW_C, "tag (0 real, 1 shuffled)", "unit index", "repeat", "fold", "prevalence index"],
                           "label_shuffle": [S_SHUFFLE], "part_b_bootstrap": [S_BOOT_B, "domain", "unit", "score", "draw"]},
        "n_pairs": {k: len(v) for k, v in sels.items()},
        "K": list(KS), "K_primary": K_PRIMARY, "repeats": R_REP, "bootstrap_draws": B_BOOT,
        "prevalences": PREVS, "q": QS,
        "formulas": {
            "threshold": "k = floor(f * n_cal); thr = (k+1)-th largest calibration non-member score; flag if score > thr (pf.tpr_at)",
            "folds": "pair-level, balanced within cohort: within each cohort a random permutation is dealt to folds 0..K-1 from a random offset",
            "realized_fpr": "sum over folds of #{evaluation non-members > thr_fold} / n_pairs; TPR likewise on members",
            "expected_fpr_exchangeable": "sum over folds of n_eval_fold * (k_fold + 1) / (n_cal_fold + 1) / n_pairs",
            "insample": "threshold and FPR on the same n non-members: FPR = k/n; expected FPR on a new exchangeable non-member = (k+1)/(n+1)",
            "conformal_p": "p = (1 + #{calibration non-members with score >= s}) / (1 + n_cal)",
            "bh": "reject p <= p_(k*), k* = max{k: p_(k) <= q k / N}, N = pool size",
            "members_in_pool": "m = floor(x) + Bernoulli(x - floor(x)), x = pi/(1-pi) * n_0 (n_0 = evaluation-fold non-members); m <= fold members; never padded",
            "fdp": "false discoveries / discoveries, 0 if no discovery; realized FDR = mean FDP over pools",
            "paper_rule": "pf.fdr_named(members, non-members, q) == BH on members only with p = (1 + #{non-members > s})/(1 + n_neg) (asserted)",
            "part_b": "threshold from the null pool with the same rule; realized FPR on the matched partners, TPR on the members; bootstrap resamples pairs and null documents independently",
            "intervals": "repeat: 2.5/97.5 percentiles over fold assignments; bootstrap: 2.5/97.5 percentiles over pair-bootstrap draws (fresh folds each draw)"},
    }}
    J["reproduction"] = rep
    J["part_a"] = part_a(cols, sels, folds_cache)
    J["part_b"] = part_b(cols, arm, sels)
    J["part_c"] = part_c(cols, sels, folds_cache)
    J["sanity"] = sanity(J)
    J["meta"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(J, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    write_md(J)
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}\n({time.time()-t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
