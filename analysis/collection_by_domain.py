#!/usr/bin/env python3
"""Collection-level membership inference by domain, and its calibration against
the domain-matched null pool.

Run:   python3 -u analysis/collection_by_domain.py
Reads (under the data root, see olmo_twin/paths.py):
       analysis/per_document_attacks_prelim.csv.gz
       (through paper_figures.load(), so the post-training exclusions and the
       merged standardized-ratio columns are exactly the paper's)
       analysis/paper_numbers.json (key 'coll', recorded in the sanity block;
       written by figures/paper_figures.py, so run its --first-pass before this
       script on a fresh data root, see analysis/README.md)
Writes analysis/collection_by_domain.json
       analysis/collection_by_domain.md

Part A  For each score (ratio_oracle, ratio_sibling, ratio_stage1, z_oracle) and
        each NAT1 cohort (M, L, W-new, W-gap, S-se), the two arXiv / Wikipedia
        unions and the pooled NAT1 population:
          paired:   d_i = s(member_i) - s(partner_i);  d_z = mean(d)/SD(d)
                    n80 = ceil(((z_.95 + z_.80)/d_z)^2)        (paper_figures.n80)
                    n80_t = smallest n with noncentral-t power >= 0.80
                    empirical power at n80 and at n = 13: draw n pairs with
                    replacement from the observed d (R_POWER draws), one-sided
                    paired t-test at alpha = 0.05
                    size under the sharp null: same, with every drawn d_i
                    multiplied by an independent random sign
          unpaired: d_u = mean(d) / s_p,  s_p = sqrt((var(a) + var(b))/2)
                    n80_u = ceil(2((z_.95 + z_.80)/d_u)^2) per group
                    empirical power of the one-sided Welch test with m members
                    and m partners drawn independently (with replacement)
          collection AUC(m), m in MS: the paper's unpaired mean aggregator,
                    paper_figures.collection_auc replicated line for line but
                    with an explicit Generator (verified identical below):
                    reps = max(400, floor(40000/m)); draw reps x m members and,
                    independently, reps x m partners with replacement; AUC (rank,
                    ties 1/2) of the reps member means against the reps partner
                    means.  95% interval: percentile over B_CURVE bootstrap
                    replicates, each resampling PAIRS and recomputing the whole
                    curve (the interval therefore contains the Monte Carlo
                    error of the curve as well as the sampling error).
          paired collection AUC(m): P(mean of m drawn within-pair d > 0), ties
                    1/2, same reps -- the paired analogue, for comparison.
Part B  Calibration.  A "collection" X of m documents none of which is a member
        is compared with m reference non-members Y (the held-out partners):
          one-sided Welch two-sample t-test at alpha = 0.05 in the direction
          mean(X) > mean(Y) ("X is more member-like"), and also the opposite
          direction; R_CAL draws, each drawing m distinct documents from X and
          m distinct documents from Y (without replacement within a draw);
          collection AUC(m) of X against Y with the same aggregator, with a
          bootstrap interval resampling X and Y independently (B_CAL reps).
        Random-halves null: 2m distinct documents drawn from one group, the
          first m form X and the second m form Y (a fresh random split per
          draw); for the AUC, B_CAL fixed random splits into two halves.
        Comparisons: domain-matched (arXiv null pool vs M+L partners, vs M
          partners; Wikipedia null pools vs W-new+W-gap partners, and A vs W-new,
          B vs W-gap), cross-domain (arXiv null pool vs S-se partners, vs
          Wikipedia partners), and never-injected partners of each domain
          against a pooled multi-domain null.  Plus the operational test:
          members of a domain against that domain's null pool (unpaired).
Every random choice uses numpy.random.default_rng(SEED + crc32(label)), with the
label recorded next to each result.
"""

from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

paths.add_code_dirs("figures")
import paper_figures as pf  # noqa: E402

SEED = 20261013
ALPHA = 0.05
Z95, Z80 = pf.Z95, pf.Z80
MS = [1, 2, 3, 5, 10, 20, 50, 100]
R_POWER = 20000         # draws for empirical power / size
R_CAL = 20000           # draws for the calibration rejection rates
B_CURVE = 200           # bootstrap replicates of each whole collection-AUC curve
B_CAL = 200             # bootstrap replicates / random splits in Part B
B_DZ = 2000             # bootstrap replicates for d_z and mean-difference intervals
OUT_JSON = pf.ANA / "collection_by_domain.json"
OUT_MD = pf.ANA / "collection_by_domain.md"

SCORES_A = ["ratio_oracle", "ratio_sibling", "ratio_stage1", "z_oracle"]
SCORES_B = ["ratio_oracle", "ratio_sibling", "ratio_stage1", "z_oracle"]
REF = {"ratio_oracle": "exact counterfactual (AI2 seed-42 control), mean log-prob ratio",
       "ratio_sibling": "retrained reference (AI2 seed-42069 run), mean log-prob ratio",
       "ratio_stage1": "published pre-stage-2 checkpoint, mean log-prob ratio",
       "z_oracle": "exact counterfactual, standardized ratio (autocorrelation-corrected)"}
GROUPS = [("M", ["M"]), ("L", ["L"]), ("W-new", ["W-new"]), ("W-gap", ["W-gap"]), ("S-se", ["S-se"]),
          ("arXiv (M+L)", ["M", "L"]), ("Wikipedia (W-new+W-gap)", ["W-new", "W-gap"]),
          ("NAT1 pooled", ["M", "L", "W-new", "W-gap", "S-se"])]


def rng_for(label):
    return np.random.default_rng(SEED + zlib.crc32(label.encode()))


# --------------------------------------------------------------- primitives
def auc_fast(pos, neg):
    """Identical to injection_v2_plotstyle.auc (rank AUC, ties 1/2), vectorised."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def collection_auc(a, b, ms, rng, reps_floor=400, total=40000):
    """paper_figures.collection_auc with an explicit Generator (same draws, same order)."""
    out = []
    for m in ms:
        reps = max(reps_floor, int(total / m))
        pa = a[rng.integers(0, a.size, (reps, m))].mean(axis=1)
        pb = b[rng.integers(0, b.size, (reps, m))].mean(axis=1)
        out.append(auc_fast(pa, pb))
    return out


def paired_collection_auc(d, ms, rng, reps_floor=400, total=40000):
    out = []
    for m in ms:
        reps = max(reps_floor, int(total / m))
        md = d[rng.integers(0, d.size, (reps, m))].mean(axis=1)
        out.append(float((md > 0).mean() + 0.5 * (md == 0).mean()))
    return out


def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def rate(k, n):
    return dict(rate=float(k / n), k=int(k), n=int(n), wilson95=wilson(k, n))


def paired_t_reject(D):
    """D: (R, n) within-pair differences. One-sided paired t-test, H1 mean > 0."""
    n = D.shape[1]
    t = D.mean(1) / (D.std(1, ddof=1) / math.sqrt(n))
    return t > stats.t.ppf(1 - ALPHA, n - 1)


def welch(X, Y):
    """X: (R, m), Y: (R, k). Returns (t, df) of Welch's two-sample test, X - Y."""
    mx, my = X.mean(1), Y.mean(1)
    vx, vy = X.var(1, ddof=1) / X.shape[1], Y.var(1, ddof=1) / Y.shape[1]
    se2 = vx + vy
    t = (mx - my) / np.sqrt(se2)
    df = se2 ** 2 / (vx ** 2 / (X.shape[1] - 1) + vy ** 2 / (Y.shape[1] - 1))
    return t, df


def welch_rates(X, Y):
    t, df = welch(X, Y)
    up = int((stats.t.sf(t, df) < ALPHA).sum()); lo = int((stats.t.cdf(t, df) < ALPHA).sum())
    return dict(upper=rate(up, X.shape[0]), lower=rate(lo, X.shape[0]))


def draw_distinct(rng, N, k, R):
    """R draws of k distinct indices out of N, each in uniformly random order."""
    return np.stack([rng.choice(N, size=k, replace=False) for _ in range(R)])


def n80_normal(dz):
    return pf.n80(dz) if dz > 0 else None


def n80_t(dz, nmax=100000):
    """Smallest n with one-sided paired-t power >= 0.80 at standardized effect dz."""
    if not dz > 0:
        return None
    n = 2
    while n <= nmax:
        crit = stats.t.ppf(1 - ALPHA, n - 1)
        if stats.nct.sf(crit, n - 1, dz * math.sqrt(n)) >= 0.80:
            return n
        n += 1
    return None


def n80_unpaired(du):
    return int(math.ceil(2 * ((Z95 + Z80) / du) ** 2)) if du > 0 else None


def summ(x):
    x = np.asarray(x, float)
    return dict(n=int(x.size), mean=float(x.mean()), sd=float(x.std(ddof=1)))


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ------------------------------------------------------------------ Part A
def part_a(cols, pairs):
    coh = cols["cohort"]
    selN = pf.population(cols, pairs, "NAT1")
    res = {}
    for key in SCORES_A:
        res[key] = {}
        for gname, cl in GROUPS:
            sel = [(i, h) for i, h in selN if coh[i] in cl]
            a, b = pf.arrays(cols, key, sel)
            ok = np.isfinite(a) & np.isfinite(b)
            assert ok.all(), (key, gname, int((~ok).sum()))
            d = a - b; n = d.size
            lab = f"A|{key}|{gname}"
            rng = rng_for(lab)
            dz = float(d.mean() / d.std(ddof=1))
            J = rng.integers(0, n, (B_DZ, n))
            Db = d[J]; dzb = Db.mean(1) / Db.std(1, ddof=1)
            dz_ci = ci(dzb)
            n80 = n80_normal(dz)
            n80_ci = [n80_normal(dz_ci[1]), n80_normal(dz_ci[0])]
            # empirical power and sign-flip size
            pw = {}
            for nn in sorted({x for x in (13, n80) if x is not None and x >= 2 and x <= 5000}):
                D = d[rng.integers(0, n, (R_POWER, nn))]
                pw[str(nn)] = rate(int(paired_t_reject(D).sum()), R_POWER)
            # empirical n80: smallest n whose resampled paired-t power is >= 0.80
            emp = None; prev = None
            for nn in range(2, 2001):
                k = int(paired_t_reject(d[rng.integers(0, n, (R_POWER, nn))]).sum())
                if k / R_POWER >= 0.80:
                    emp = dict(n=nn, power=rate(k, R_POWER), power_at_n_minus_1=prev)
                    break
                prev = rate(k, R_POWER)
            size = {}
            for nn in sorted({x for x in (13, n80, 50) if x is not None and x >= 2 and x <= 5000}):
                D = d[rng.integers(0, n, (R_POWER, nn))] * rng.choice([-1.0, 1.0], size=(R_POWER, nn))
                size[str(nn)] = rate(int(paired_t_reject(D).sum()), R_POWER)
            # unpaired
            sp = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            du = float(d.mean() / sp)
            nu = n80_unpaired(du)
            upw = {}
            for mm in sorted({x for x in (13, nu) if x is not None and x >= 2 and x <= 5000}):
                X = a[rng.integers(0, n, (R_POWER, mm))]; Y = b[rng.integers(0, n, (R_POWER, mm))]
                upw[str(mm)] = welch_rates(X, Y)["upper"]
            # collection AUC (paper's unpaired aggregator) and bootstrap of the whole curve
            curve = collection_auc(a, b, MS, rng)
            pcurve = paired_collection_auc(d, MS, rng)
            boot = np.empty((B_CURVE, len(MS))); pboot = np.empty((B_CURVE, len(MS)))
            for r in range(B_CURVE):
                j = rng.integers(0, n, n)
                boot[r] = collection_auc(a[j], b[j], MS, rng)
                pboot[r] = paired_collection_auc(d[j], MS, rng)
            res[key][gname] = dict(
                rng_label=lab, cohorts=cl, n_pairs=n,
                member=summ(a), partner=summ(b),
                corr_member_partner=float(np.corrcoef(a, b)[0, 1]),
                mean_d=float(d.mean()), sd_d=float(d.std(ddof=1)),
                dz=dz, dz_ci95=dz_ci, n80=n80, n80_ci95_from_dz_ci=n80_ci, n80_noncentral_t=n80_t(dz),
                power_paired_t=pw, n80_empirical=emp, size_signflip_paired_t=size,
                d_unpaired=du, pooled_sd=sp, n80_unpaired_per_group=nu, power_welch_unpaired=upw,
                doc_auc=auc_fast(a, b),
                collection_auc={str(m): v for m, v in zip(MS, curve)},
                collection_auc_ci95={str(m): ci(boot[:, k]) for k, m in enumerate(MS)},
                paired_collection_auc={str(m): v for m, v in zip(MS, pcurve)},
                paired_collection_auc_ci95={str(m): ci(pboot[:, k]) for k, m in enumerate(MS)},
            )
            print(f"  A {key:13s} {gname:24s} n={n:5d} dz={dz:.3f} [{dz_ci[0]:.3f},{dz_ci[1]:.3f}] n80={n80} "
                  f"n80t={res[key][gname]['n80_noncentral_t']} n80emp={(emp or {}).get('n')} pw13={pw.get('13', {}).get('rate', float('nan')):.3f} "
                  f"AUC10={curve[4]:.3f} [{ci(boot[:, 4])[0]:.3f},{ci(boot[:, 4])[1]:.3f}] du={du:.3f} n80u={nu}",
                  flush=True)
    return res


# ------------------------------------------------------------------ Part B
def compare(x, y, label, ms_test=(13, 50)):
    """Collection X (tested as 'more member-like') against reference non-members Y."""
    rng = rng_for(label)
    out = dict(rng_label=label, X=summ(x), Y=summ(y))
    diff = x.mean() - y.mean()
    bx = x[rng.integers(0, x.size, (B_DZ, x.size))].mean(1)
    by = y[rng.integers(0, y.size, (B_DZ, y.size))].mean(1)
    out["mean_diff_X_minus_Y"] = float(diff)
    out["mean_diff_ci95"] = ci(bx - by)
    out["std_offset"] = float(diff / math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2))
    out["doc_auc"] = auc_fast(x, y)
    wt = {}
    for m in ms_test:
        IX = draw_distinct(rng, x.size, m, R_CAL); IY = draw_distinct(rng, y.size, m, R_CAL)
        wt[str(m)] = welch_rates(x[IX], y[IY])
    out["welch"] = wt
    curve = collection_auc(x, y, MS, rng)
    boot = np.empty((B_CAL, len(MS)))
    for r in range(B_CAL):
        boot[r] = collection_auc(x[rng.integers(0, x.size, x.size)], y[rng.integers(0, y.size, y.size)], MS, rng)
    out["collection_auc"] = {str(m): v for m, v in zip(MS, curve)}
    out["collection_auc_ci95"] = {str(m): ci(boot[:, k]) for k, m in enumerate(MS)}
    return out


def halves(y, label, ms_test=(13, 50)):
    """Pure resampling null inside one group."""
    rng = rng_for(label)
    out = dict(rng_label=label, group=summ(y))
    wt = {}
    for m in ms_test:
        I = draw_distinct(rng, y.size, 2 * m, R_CAL)
        wt[str(m)] = welch_rates(y[I[:, :m]], y[I[:, m:]])
    out["welch"] = wt
    curves = np.empty((B_CAL, len(MS))); offs = np.empty(B_CAL)
    h = y.size // 2
    for r in range(B_CAL):
        p = rng.permutation(y.size)
        u, v = y[p[:h]], y[p[h:2 * h]]
        offs[r] = u.mean() - v.mean()
        curves[r] = collection_auc(u, v, MS, rng)
    out["split_half_size"] = int(h)
    out["collection_auc_mean_over_splits"] = {str(m): float(curves[:, k].mean()) for k, m in enumerate(MS)}
    out["collection_auc_2.5_97.5_over_splits"] = {str(m): ci(curves[:, k]) for k, m in enumerate(MS)}
    out["half_mean_diff_2.5_97.5"] = ci(offs)
    return out


def part_b(cols, arm, pairs):
    coh = cols["cohort"]
    selN = pf.population(cols, pairs, "NAT1")
    res = {}
    for key in SCORES_B:
        s = np.asarray(cols[key], float)

        def partners(cl):
            v = np.array([s[h] for i, h in selN if coh[i] in cl]); assert np.isfinite(v).all(); return v

        def members(cl):
            v = np.array([s[i] for i, h in selN if coh[i] in cl]); assert np.isfinite(v).all(); return v

        def null(cl):
            v = s[(arm == "null") & np.isin(coh, cl)]; assert np.isfinite(v).all(); return v

        P = {"arXiv": partners(["M", "L"]), "M": partners(["M"]), "L": partners(["L"]),
             "Wikipedia": partners(["W-new", "W-gap"]), "W-new": partners(["W-new"]),
             "W-gap": partners(["W-gap"]), "S-se": partners(["S-se"])}
        N = {"N-arxiv": null(["N-arxiv"]), "N-wiki": null(["N-wikiA", "N-wikiB"]),
             "N-wikiA": null(["N-wikiA"]), "N-wikiB": null(["N-wikiB"]),
             "pooled N-arxiv+N-wikiA+N-wikiB": null(["N-arxiv", "N-wikiA", "N-wikiB"]),
             "pooled N-arxiv+N-wikiA+N-wikiB+N-edited": null(["N-arxiv", "N-wikiA", "N-wikiB", "N-edited"])}
        Mb = {"arXiv": members(["M", "L"]), "Wikipedia": members(["W-new", "W-gap"]), "S-se": members(["S-se"])}
        R = dict(matched={}, halves={}, cross_domain={}, pooled_null={}, members_vs_null={})
        jobs = [
            ("matched", "arXiv: N-arxiv vs M+L partners", N["N-arxiv"], P["arXiv"]),
            ("matched", "arXiv: N-arxiv vs M partners", N["N-arxiv"], P["M"]),
            ("matched", "arXiv: N-arxiv vs L partners", N["N-arxiv"], P["L"]),
            ("matched", "Wikipedia: N-wikiA+N-wikiB vs W-new+W-gap partners", N["N-wiki"], P["Wikipedia"]),
            ("matched", "Wikipedia: N-wikiA vs W-new partners", N["N-wikiA"], P["W-new"]),
            ("matched", "Wikipedia: N-wikiB vs W-gap partners", N["N-wikiB"], P["W-gap"]),
            ("cross_domain", "N-arxiv vs S-se partners", N["N-arxiv"], P["S-se"]),
            ("cross_domain", "N-arxiv vs W-new+W-gap partners", N["N-arxiv"], P["Wikipedia"]),
            ("pooled_null", "M+L partners vs pooled N-arxiv+N-wikiA+N-wikiB", P["arXiv"], N["pooled N-arxiv+N-wikiA+N-wikiB"]),
            ("pooled_null", "W-new+W-gap partners vs pooled N-arxiv+N-wikiA+N-wikiB", P["Wikipedia"], N["pooled N-arxiv+N-wikiA+N-wikiB"]),
            ("pooled_null", "S-se partners vs pooled N-arxiv+N-wikiA+N-wikiB", P["S-se"], N["pooled N-arxiv+N-wikiA+N-wikiB"]),
            ("pooled_null", "M+L partners vs pooled N-arxiv+N-wikiA+N-wikiB+N-edited", P["arXiv"], N["pooled N-arxiv+N-wikiA+N-wikiB+N-edited"]),
            ("pooled_null", "S-se partners vs pooled N-arxiv+N-wikiA+N-wikiB+N-edited", P["S-se"], N["pooled N-arxiv+N-wikiA+N-wikiB+N-edited"]),
            ("members_vs_null", "arXiv: M+L members vs N-arxiv", Mb["arXiv"], N["N-arxiv"]),
            ("members_vs_null", "Wikipedia: W-new+W-gap members vs N-wikiA+N-wikiB", Mb["Wikipedia"], N["N-wiki"]),
            ("members_vs_null", "unmatched: S-se members vs N-arxiv", Mb["S-se"], N["N-arxiv"]),
            ("members_vs_null", "unmatched: S-se members vs pooled N-arxiv+N-wikiA+N-wikiB", Mb["S-se"], N["pooled N-arxiv+N-wikiA+N-wikiB"]),
            ("members_vs_null", "unmatched: W-new+W-gap members vs N-arxiv", Mb["Wikipedia"], N["N-arxiv"]),
            ("members_vs_null", "arXiv: M+L members vs M+L partners (unpaired)", Mb["arXiv"], P["arXiv"]),
            ("members_vs_null", "Wikipedia: W-new+W-gap members vs W-new+W-gap partners (unpaired)", Mb["Wikipedia"], P["Wikipedia"]),
        ]
        for sec, lab, x, y in jobs:
            R[sec][lab] = compare(x, y, f"B|{key}|{lab}")
            o = R[sec][lab]
            print(f"  B {key:13s} {lab:64s} dX-Y={o['mean_diff_X_minus_Y']:+.4f} "
                  f"rej13={o['welch']['13']['upper']['rate']:.3f} rej50={o['welch']['50']['upper']['rate']:.3f} "
                  f"AUC10={o['collection_auc']['10']:.3f} AUC50={o['collection_auc']['50']:.3f}", flush=True)
        for lab, y in (("M+L partners", P["arXiv"]), ("W-new+W-gap partners", P["Wikipedia"]),
                       ("S-se partners", P["S-se"]), ("N-arxiv", N["N-arxiv"]), ("N-wikiA+N-wikiB", N["N-wiki"])):
            R["halves"][lab] = halves(y, f"H|{key}|{lab}")
            o = R["halves"][lab]
            print(f"  H {key:13s} {lab:24s} rej13={o['welch']['13']['upper']['rate']:.3f} "
                  f"rej50={o['welch']['50']['upper']['rate']:.3f} AUC50 mean={o['collection_auc_mean_over_splits']['50']:.3f} "
                  f"range={o['collection_auc_2.5_97.5_over_splits']['50']}", flush=True)
        res[key] = R
    return res


# ------------------------------------------------------------------ checks
def sanity(cols, pairs, A):
    selN = pf.population(cols, pairs, "NAT1")
    a, b = pf.arrays(cols, "ratio_oracle", selN)
    H = pf.headline(cols, selN)
    # replication of paper_figures.collection_auc: identical draws with the same Generator state
    pf.RNG = np.random.default_rng(12345)
    ref = [v for _, v in pf.collection_auc(a, b, MS)]
    mine = collection_auc(a, b, MS, np.random.default_rng(12345))
    gap = float(np.max(np.abs(np.array(ref) - np.array(mine))))
    assert gap < 1e-12, gap
    assert abs(auc_fast(a, b) - pf.ps.auc(a, b)) < 1e-12
    numbers = json.loads((pf.ANA / "paper_numbers.json").read_text())
    pooled = A["ratio_oracle"]["NAT1 pooled"]
    return dict(
        replication_of_pf_collection_auc_max_abs_diff=gap,
        auc_fast_equals_ps_auc=True,
        dz_ratio_oracle_NAT1=pooled["dz"], dz_from_pf_headline=H["dz"], paper_macro_dznat="0.71",
        n80_ratio_oracle_NAT1=pooled["n80"], paper_macro_npower="13",
        empirical_power_at_13=pooled["power_paired_t"]["13"]["rate"], paper_macro_powerempirical="81",
        collection_auc_m10_ratio_oracle_NAT1=pooled["collection_auc"]["10"],
        collection_auc_m10_ci95=pooled["collection_auc_ci95"]["10"],
        paper_macro_collorcmonezero="0.949",
        paper_numbers_json_coll_exact_counterfactual=numbers.get("coll", {}).get("exact counterfactual"),
        z_oracle_dz_NAT1=A["z_oracle"]["NAT1 pooled"]["dz"], paper_macro_dzzorcnat="1.04",
        z_oracle_n80_NAT1=A["z_oracle"]["NAT1 pooled"]["n80"],
        z_oracle_n80_empirical_NAT1=A["z_oracle"]["NAT1 pooled"]["n80_empirical"]["n"], paper_macro_npowerzorc="7",
    )


# ------------------------------------------------------------------ report
def fmt_ci(c, p=3):
    return f"[{c[0]:.{p}f}, {c[1]:.{p}f}]"


def fmt_rate(r):
    return f"{100*r['rate']:.1f}% ({r['k']}/{r['n']})"


def write_md(J):
    A, B, S = J["part_a"], J["part_b"], J["sanity"]
    L = ["# Collection-level membership inference by domain, and its calibration",
         "",
         "Generated by `analysis/collection_by_domain.py` (seed "
         f"{SEED}; every random choice uses `numpy.random.default_rng(SEED + crc32(label))`, labels in the JSON). "
         "Populations from `paper_figures.load()` / `population(..., 'NAT1')` (4,832 single-copy pairs, "
         "post-training exclusions applied). Scores oriented so that higher = more member-like.",
         "",
         "## Sanity checks",
         "",
         f"- pooled NAT1, exact counterfactual: d_z = {S['dz_ratio_oracle_NAT1']:.4f} (paper 0.71), "
         f"n80 = {S['n80_ratio_oracle_NAT1']} (paper 13), empirical power at 13 pairs = "
         f"{100*S['empirical_power_at_13']:.1f}% (paper 81%).",
         f"- collection AUC at m = 10 = {S['collection_auc_m10_ratio_oracle_NAT1']:.3f} "
         f"{fmt_ci(S['collection_auc_m10_ci95'])} (paper 0.949; the difference is Monte Carlo error of the "
         "40,000/m draws).",
         f"- standardized ratio: d_z = {S['z_oracle_dz_NAT1']:.3f} (paper 1.04), n80 = {S['z_oracle_n80_NAT1']} (paper 6).",
         f"- the collection-AUC routine reproduces `paper_figures.collection_auc` draw for draw "
         f"(max |diff| = {S['replication_of_pf_collection_auc_max_abs_diff']:.1e} at a common Generator state).",
         "",
         "## What each result assumes",
         "",
         "| result | reference model | non-member source | pairing | decision rule |",
         "|---|---|---|---|---|",
         "| paired d_z, n80, power | per column below | the matched held-out partner of each member (same cohort, same domain, coin-flipped) | yes: the statistic is the within-pair difference | one-sided paired t-test of mean(member - partner) > 0, alpha 0.05 |",
         "| collection AUC(m) (paper Fig. collection) | per column | matched partners of the same population, but drawn independently of the members | no: m members and m partners drawn independently with replacement | the set with the larger mean score is called the member set; AUC = P(correct), ties 1/2 |",
         "| paired collection AUC(m) | per column | the m partners of the m drawn members | yes | P(mean of m within-pair differences > 0) |",
         "| unpaired n80 / Welch power | per column | matched partners, drawn independently | no | one-sided Welch t-test of mean(members) > mean(non-members), alpha 0.05 |",
         "",
         "## Part A: per-domain collection results",
         ""]
    for key in SCORES_A:
        L += [f"### {key} — {REF[key]}", "",
              "| group | pairs | mean d | SD d | d_z [95% CI] | n80 (normal) [from d_z CI] | n80 (noncentral t) | power at n80 | power at 13 | sign-flip size at 13 | doc AUC | AUC m=10 [95% CI] | AUC m=20 | paired AUC m=10 | corr(member, partner) | d unpaired | n80 unpaired per group | Welch power at 13 (unpaired) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g, _ in GROUPS:
            r = A[key][g]
            n80 = r["n80"]
            pw80 = fmt_rate(r["power_paired_t"][str(n80)]) if n80 is not None and str(n80) in r["power_paired_t"] else "n/a"
            pw13 = fmt_rate(r["power_paired_t"]["13"])
            sz13 = fmt_rate(r["size_signflip_paired_t"]["13"])
            L.append(f"| {g} | {r['n_pairs']} | {r['mean_d']:+.{4 if key.startswith('ratio') else 3}f} | {r['sd_d']:.{4 if key.startswith('ratio') else 3}f} | {r['dz']:.3f} {fmt_ci(r['dz_ci95'])} | "
                     f"{n80} [{r['n80_ci95_from_dz_ci'][0]}, {r['n80_ci95_from_dz_ci'][1]}] | {r['n80_noncentral_t']} | {pw80} | {pw13} | {sz13} | "
                     f"{r['doc_auc']:.3f} | {r['collection_auc']['10']:.3f} {fmt_ci(r['collection_auc_ci95']['10'])} | "
                     f"{r['collection_auc']['20']:.3f} | {r['paired_collection_auc']['10']:.3f} | {r['corr_member_partner']:.3f} | "
                     f"{r['d_unpaired']:.3f} | {r['n80_unpaired_per_group']} | {fmt_rate(r['power_welch_unpaired']['13'])} |")
        L += ["", "Collection AUC curve (unpaired mean aggregator), point [95% bootstrap CI over pairs]:", "",
              "| group | " + " | ".join(f"m={m}" for m in MS) + " |", "|---|" + "---|" * len(MS)]
        for g, _ in GROUPS:
            r = A[key][g]
            L.append(f"| {g} | " + " | ".join(f"{r['collection_auc'][str(m)]:.3f} {fmt_ci(r['collection_auc_ci95'][str(m)])}" for m in MS) + " |")
        L.append("")
    L += ["## Part C: compact table (cohort x reference)", "",
          "Paired d_z [95% CI]; n80 = paper formula / exact noncentral t / empirical (resampled observed differences); "
          "collection AUC at m = 10 with the paper's unpaired mean aggregator [95% CI].", "",
          "| group | " + " | ".join(SCORES_A) + " |", "|---|" + "---|" * len(SCORES_A)]
    for g, _ in GROUPS:
        cells = []
        for key in SCORES_A:
            r = A[key][g]; e = r["n80_empirical"]
            cells.append(f"d_z {r['dz']:.2f} {fmt_ci(r['dz_ci95'], 2)}; n80 {r['n80']}/{r['n80_noncentral_t']}/{e['n'] if e else 'n/a'}; "
                         f"AUC10 {r['collection_auc']['10']:.3f} {fmt_ci(r['collection_auc_ci95']['10'])}")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L += ["", "Empirical n80 detail (power at n80_empirical and at one pair fewer):", "",
          "| score | group | n80 normal | n80 t | n80 empirical | power at it | power at n-1 |", "|---|---|---|---|---|---|---|"]
    for key in SCORES_A:
        for g, _ in GROUPS:
            r = A[key][g]; e = r["n80_empirical"]
            L.append(f"| {key} | {g} | {r['n80']} | {r['n80_noncentral_t']} | {e['n']} | {fmt_rate(e['power'])} | "
                     f"{fmt_rate(e['power_at_n_minus_1']) if e['power_at_n_minus_1'] else 'n/a'} |")
    L += [""]
    L += ["## Part B: calibration against non-member collections", "",
          "X is the collection tested as 'more member-like' (it contains no members except in the last block); "
          "Y is the reference non-member set. Rejection rate of the one-sided Welch test at alpha 0.05 over "
          f"{R_CAL} draws of m distinct documents from each; 'reverse' is the opposite direction. "
          "Collection AUC: paper's mean aggregator, X against Y; 95% interval from "
          f"{B_CAL} bootstrap replicates resampling X and Y independently.", ""]
    for key in SCORES_B:
        L += [f"### {key}", "",
              "| section | X vs Y | n_X | mean_X (SD) | n_Y | mean_Y (SD) | mean X−Y [95% CI] | std. offset | reject m=13 | reject m=50 | reverse m=13 | reverse m=50 | AUC m=1 | AUC m=10 [CI] | AUC m=50 [CI] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for sec in ("matched", "cross_domain", "pooled_null", "members_vs_null"):
            for lab, o in B[key][sec].items():
                p = 4 if key.startswith("ratio") else 2
                L.append(f"| {sec} | {lab} | {o['X']['n']} | {o['X']['mean']:+.{p}f} ({o['X']['sd']:.{p}f}) | {o['Y']['n']} | "
                         f"{o['Y']['mean']:+.{p}f} ({o['Y']['sd']:.{p}f}) | {o['mean_diff_X_minus_Y']:+.{p}f} {fmt_ci(o['mean_diff_ci95'], p)} | "
                         f"{o['std_offset']:+.3f} | {fmt_rate(o['welch']['13']['upper'])} | {fmt_rate(o['welch']['50']['upper'])} | "
                         f"{fmt_rate(o['welch']['13']['lower'])} | {fmt_rate(o['welch']['50']['lower'])} | {o['collection_auc']['1']:.3f} | "
                         f"{o['collection_auc']['10']:.3f} {fmt_ci(o['collection_auc_ci95']['10'])} | "
                         f"{o['collection_auc']['50']:.3f} {fmt_ci(o['collection_auc_ci95']['50'])} |")
        L += ["", "Random-halves null (pure resampling): a fresh random split per Welch draw; AUC over "
              f"{B_CAL} fixed random splits into halves (mean, and 2.5–97.5 percentile range across splits).", "",
              "| group | n | reject m=13 | reject m=50 | reverse m=13 | reverse m=50 | AUC m=10 mean [range] | AUC m=50 mean [range] | half-mean difference range |",
              "|---|---|---|---|---|---|---|---|---|"]
        for lab, o in B[key]["halves"].items():
            p = 4 if key.startswith("ratio") else 2
            L.append(f"| {lab} | {o['group']['n']} | {fmt_rate(o['welch']['13']['upper'])} | {fmt_rate(o['welch']['50']['upper'])} | "
                     f"{fmt_rate(o['welch']['13']['lower'])} | {fmt_rate(o['welch']['50']['lower'])} | "
                     f"{o['collection_auc_mean_over_splits']['10']:.3f} {fmt_ci(o['collection_auc_2.5_97.5_over_splits']['10'])} | "
                     f"{o['collection_auc_mean_over_splits']['50']:.3f} {fmt_ci(o['collection_auc_2.5_97.5_over_splits']['50'])} | "
                     f"{fmt_ci(o['half_mean_diff_2.5_97.5'], p)} |")
        L.append("")
    L += ["## Notes", "",
          "- There is no StackExchange null pool (one N-se document survives the exclusions), so the S-se collection "
          "test cannot be calibrated against an independent domain-matched null; only its random-halves null is reported.",
          "- n80 (normal) is the paper's formula ceil(((z_.95+z_.80)/d_z)^2); n80 (noncentral t) is the exact smallest n "
          "at which a one-sided paired t-test has power >= 0.80 for normally distributed differences with that d_z; "
          "'power at n80' and 'power at 13' are the empirical rejection rates when n pairs are resampled from the observed "
          "differences, which do not assume normality.",
          "- Bootstrap intervals for the collection AUC contain both the pair-level sampling error and the Monte Carlo "
          "error of the 40,000/m draws inside each curve.",
          ""]
    OUT_MD.write_text("\n".join(L) + "\n")


def main():
    t0 = time.time()
    cols, rows, arm, pairs = pf.load()
    print(f"loaded {len(rows):,} documents, {len(pairs):,} pairs", flush=True)
    A = part_a(cols, pairs)
    print(f"Part A done ({time.time()-t0:.0f}s)", flush=True)
    B = part_b(cols, arm, pairs)
    print(f"Part B done ({time.time()-t0:.0f}s)", flush=True)
    S = sanity(cols, pairs, A)
    print(json.dumps(S, indent=1), flush=True)
    J = dict(
        script="analysis/collection_by_domain.py", seed=SEED,
        rng="numpy.random.default_rng(SEED + zlib.crc32(rng_label.encode()))",
        table=str(pf.TABLE.relative_to(paths.DATA)),
        population="paper_figures.population(cols, pairs, 'NAT1'): r == 1 and cohort in {M, L, W-new, W-gap, S-se}; "
                   "post-training exclusions applied by paper_figures.load()",
        null_pools={"N-arxiv": "arXiv, never injected, never a partner (matched by source to M and L)",
                    "N-wikiA": "Wikipedia pages created after the cutoff (same source as W-new)",
                    "N-wikiB": "pre-cutoff Wikipedia pages absent from the snapshot (same source as W-gap)",
                    "N-edited": "edited Wikipedia (matched to the E family; used only in the pooled-null rows)",
                    "N-se": "1 document after exclusions: no StackExchange null pool"},
        constants=dict(alpha=ALPHA, z95=Z95, z80=Z80, ms=MS, r_power=R_POWER, r_cal=R_CAL, b_curve=B_CURVE,
                       b_cal=B_CAL, b_dz=B_DZ),
        references=REF,
        formulas={
            "d_i": "score(member_i) - score(partner_i)",
            "d_z": "mean(d) / SD(d), SD with ddof=1",
            "n80": "ceil(((z_0.95 + z_0.80) / d_z)^2), z_0.95=1.6449, z_0.80=0.8416 (paper_figures.n80)",
            "n80_ci95_from_dz_ci": "[n80(d_z upper), n80(d_z lower)], d_z interval = percentile bootstrap over pairs",
            "n80_noncentral_t": "min n: P(T > t_{0.95,n-1}) >= 0.8, T ~ noncentral t(df=n-1, nc=d_z*sqrt(n))",
            "power_paired_t": "fraction of R_POWER draws of n pairs (with replacement from observed d) where "
                              "mean(d)/(SD(d)/sqrt(n)) > t_{0.95,n-1}",
            "size_signflip_paired_t": "same with each drawn d_i multiplied by an independent +-1 sign",
            "n80_empirical": "smallest n >= 2 whose power_paired_t (R_POWER fresh draws per n, scanned upward) is >= 0.80; "
                             "the rate at n-1 is reported so that borderline cases are visible",
            "d_unpaired": "mean(d) / sqrt((var(members) + var(partners)) / 2)",
            "n80_unpaired_per_group": "ceil(2 ((z_0.95 + z_0.80) / d_unpaired)^2)",
            "power_welch_unpaired": "fraction of R_POWER draws (m members, m partners, independent, with replacement) "
                                    "where the one-sided Welch test of mean(members) > mean(partners) has p < 0.05",
            "collection_auc": "paper_figures.collection_auc replicated with an explicit Generator: reps = "
                              "max(400, int(40000/m)); member means and partner means of reps independent draws of m "
                              "with replacement; rank AUC (ties 1/2) of member means vs partner means",
            "collection_auc_ci95": "2.5/97.5 percentiles over B_CURVE replicates, each resampling pairs with "
                                   "replacement and recomputing the whole curve",
            "paired_collection_auc": "P(mean of m within-pair d drawn with replacement > 0) + 0.5 P(= 0)",
            "welch_calibration": "R_CAL draws of m distinct documents from X and m distinct from Y; Welch t = "
                                 "(mean X - mean Y)/sqrt(sX^2/m + sY^2/m), Welch-Satterthwaite df; 'upper' rejects "
                                 "when P(T > t) < 0.05 (X more member-like), 'lower' when P(T < t) < 0.05",
            "halves": "Welch: 2m distinct documents of one group per draw, first m = X, second m = Y; AUC: "
                      "B_CAL random splits of the group into two halves of floor(n/2), collection_auc of half 1 vs half 2",
            "std_offset": "(mean X - mean Y) / sqrt((var X + var Y)/2)",
            "mean_diff_ci95": "percentile bootstrap, X and Y resampled independently, B_DZ replicates",
        },
        part_a=A, part_b=B, sanity=S, runtime_s=round(time.time() - t0, 1))
    OUT_JSON.write_text(json.dumps(J, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    write_md(J)
    print(f"wrote {OUT_JSON} and {OUT_MD} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
