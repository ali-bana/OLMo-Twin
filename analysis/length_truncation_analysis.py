#!/usr/bin/env python3
"""Prefix-truncation analysis on fixed arXiv pairs.

Re-scores each cohort-M document on its first T tokens (T = 16 ... 4096) using
the per-token cache analysis/cache/token_cache.npz written by
analysis/precompute_token_cache.py, on the FIXED
set of pairs whose two documents both have at least 4,096 scored tokens, so no
point on the curve is a selection effect. Reports AUC / d_z / mean effect with
pair-bootstrap intervals, the log-log exponents on T <= 4096, the token-noise
share of the within-pair variance along T (Sokal-windowed n_eff from the raw
per-token files, logprobs only), cohort L by target length, and pooled L vs M.
Seed 20261013.

Output (under the data root, see olmo_twin/paths.py): analysis/length_truncation.json
"""

import sys, os, math, json
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402
paths.add_code_dirs("figures")
import numpy as np
import paper_figures as pf
from calibrated_attacks import sokal_tau

SEED = 20261013
B = 1000
TOK = str(paths.RUN / "scores")
INJ = TOK + "/injected-step23852/tokens"; CTL = TOK + "/official-seed42-step23852/tokens"
GRIDT = [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

def safe(d): return d.replace("/", "_").replace(":", "_")

def tokstats(doc):
    a = np.load(os.path.join(INJ, safe(doc) + ".npz"))["logprobs"].astype(np.float64)
    b = np.load(os.path.join(CTL, safe(doc) + ".npz"))["logprobs"].astype(np.float64)
    n = min(a.size, b.size); a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b); d = (a - b)[ok]
    out = []
    for T in GRIDT + [d.size]:
        if d.size < T:
            out.append((np.nan, np.nan, np.nan)); continue
        x = d[:T]; sd = x.std(ddof=1); tau, _ = sokal_tau(x)
        neff = float(T) if (not np.isfinite(tau) or tau <= 0) else min(max(T / tau, 1.0), T)
        out.append((x.mean(), sd ** 2 / neff, neff / T))
    return doc, np.array(out)

def main():
    rng = np.random.default_rng(SEED)
    cols, rows, arm, pairs = pf.load()
    selN = pf.population(cols, pairs, "NAT1")
    coh = cols["cohort"]; docid = cols["doc_id"]
    selM = [(i, h) for i, h in selN if coh[i] == "M"]
    C = np.load(pf.ANA / "cache/token_cache.npz", allow_pickle=True)
    cid = {d: k for k, d in enumerate(C["doc_id"].astype(str))}
    grid = C["grid"]
    I = np.array([cid[docid[i]] for i, _ in selM]); H = np.array([cid[docid[h]] for _, h in selM])
    nv = np.minimum(C["n_valid"][I], C["n_valid"][H])
    fixed = nv >= 4096
    res = {"seed": SEED, "bootstrap_draws": B, "n_M_pairs": len(selM), "n_fixed": int(fixed.sum())}
    pI = (C["cum_inj"][I] - C["cum_ctl"][I]) / grid[None, :]
    pH = (C["cum_inj"][H] - C["cum_ctl"][H]) / grid[None, :]
    fI, fH = pI[fixed], pH[fixed]
    nF = int(fixed.sum())
    gi = [int(np.nonzero(grid == T)[0][0]) for T in GRIDT]
    boots = [rng.integers(0, nF, nF) for _ in range(B)]
    # full length on the same fixed pairs
    aF = np.array([cols["ratio_oracle"][i] for (i, _), f in zip(selM, fixed) if f])
    bF = np.array([cols["ratio_oracle"][h] for (_, h), f in zip(selM, fixed) if f])
    def stats(a, b):
        d = a - b
        t, k, nn = pf.tpr_at(a, b, 0.01)
        return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), dz=float(d.mean() / d.std(ddof=1)),
                    auc=float(pf.ps.auc(a, b)), tpr1=t, k=k, n_neg=nn, n_pos=int(a.size))
    rowsT = []
    for T, g in zip(GRIDT + ["full"], gi + [None]):
        a, b = (fI[:, g], fH[:, g]) if g is not None else (aF, bF)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        s = stats(a, b)
        bs = np.array([[pf.ps.auc(a[j], b[j]), (a[j] - b[j]).mean() / (a[j] - b[j]).std(ddof=1), (a[j] - b[j]).mean()] for j in boots])
        s.update(auc_ci=list(np.percentile(bs[:, 0], [2.5, 97.5])), dz_ci=list(np.percentile(bs[:, 1], [2.5, 97.5])),
                 mean_ci=list(np.percentile(bs[:, 2], [2.5, 97.5])), T=T)
        rowsT.append(s)
    res["fixed_curve"] = rowsT
    def auc_at(g, j):
        a, b = (fI[:, g], fH[:, g]) if g is not None else (aF, bF)
        return pf.ps.auc(a[j], b[j])
    g16, g4096 = gi[GRIDT.index(16)], gi[GRIDT.index(4096)]
    d1 = np.array([auc_at(g4096, j) - auc_at(g16, j) for j in boots])
    d2 = np.array([auc_at(None, j) - auc_at(g4096, j) for j in boots])
    res["paired_auc_differences"] = {"4096_minus_16": {"diff": float(rowsT[GRIDT.index(4096)]["auc"] - rowsT[GRIDT.index(16)]["auc"]), "ci": list(np.percentile(d1, [2.5, 97.5]))},
                                     "full_minus_4096": {"diff": float(rowsT[-1]["auc"] - rowsT[GRIDT.index(4096)]["auc"]), "ci": list(np.percentile(d2, [2.5, 97.5]))}}
    # exponents on T<=4096 (13 grid points), unweighted OLS in log-log
    lx = np.log(np.array(GRIDT, float))
    def expo(a2, b2):
        d = a2 - b2; m = d.mean(0); s = d.std(0, ddof=1)
        return np.polyfit(lx, np.log(m / s), 1)[0], np.polyfit(lx, np.log(m), 1)[0], np.polyfit(lx, np.log(s), 1)[0]
    A2, B2 = fI[:, gi], fH[:, gi]
    e = expo(A2, B2)
    eb = np.array([expo(A2[j], B2[j]) for j in boots])
    res["exponents"] = {"dz": e[0], "mean": e[1], "sd": e[2],
                        "dz_ci": list(np.percentile(eb[:, 0], [2.5, 97.5])),
                        "mean_ci": list(np.percentile(eb[:, 1], [2.5, 97.5])),
                        "sd_ci": list(np.percentile(eb[:, 2], [2.5, 97.5]))}
    # token-noise share along T on the fixed pairs, from the raw per-token files
    docs = [docid[i] for (i, _), f in zip(selM, fixed) if f] + [docid[h] for (_, h), f in zip(selM, fixed) if f]
    with Pool(16) as pool:
        out = dict(pool.map(tokstats, docs, chunksize=16))
    TI = np.stack([out[docid[i]] for (i, _), f in zip(selM, fixed) if f])   # (nF, 14, 3)
    TH = np.stack([out[docid[h]] for (_, h), f in zip(selM, fixed) if f])
    # sanity: the raw prefix means equal the cache's
    assert np.nanmax(np.abs(TI[:, :13, 0] - A2)) < 1e-9 and np.nanmax(np.abs(TH[:, :13, 0] - B2)) < 1e-9
    sh = []
    for t in range(14):
        D = TI[:, t, 0] - TH[:, t, 0]; tv = TI[:, t, 1] + TH[:, t, 1]
        share = float(np.mean(tv) / D.var(ddof=1))
        bsh = [float(np.mean(tv[j]) / D[j].var(ddof=1)) for j in boots[:400]]
        sh.append(dict(T=(GRIDT + ["full"])[t], share=share, share_ci=list(np.percentile(bsh, [2.5, 97.5])),
                       var_D=float(D.var(ddof=1)), mean_tokvar=float(np.mean(tv)),
                       doc_var=float(D.var(ddof=1) - np.mean(tv)),
                       median_neff_over_T=float(np.median(np.concatenate([TI[:, t, 2], TH[:, t, 2]])))))
    res["token_noise_share_fixed"] = sh
    res["fixed_full_len_tokens_median"] = float(np.median(np.concatenate([C["n_valid"][I][fixed], C["n_valid"][H][fixed]])))
    # cohort L by target length, and pooled L vs M
    tl = cols["target_len"]
    selL = [(i, h) for i, h in selN if coh[i] == "L"]
    Lrows = []
    for T in sorted(set(int(tl[i]) for i, _ in selL)):
        s_ = [(i, h) for i, h in selL if int(tl[i]) == T]
        a, b = pf.arrays(cols, "ratio_oracle", s_)
        st = stats(a, b)
        bj = [rng.integers(0, a.size, a.size) for _ in range(B)]
        bs = np.array([[pf.ps.auc(a[j], b[j]), (a[j] - b[j]).mean() / (a[j] - b[j]).std(ddof=1)] for j in bj])
        st.update(T=T, auc_ci=list(np.percentile(bs[:, 0], [2.5, 97.5])), dz_ci=list(np.percentile(bs[:, 1], [2.5, 97.5])),
                  n_scored_median=float(np.median([cols["injected__n_scored"][i] for i, _ in s_])))
        Lrows.append(st)
    res["L_by_length"] = Lrows
    def auc_boot(sel):
        a, b = pf.arrays(cols, "ratio_oracle", sel)
        return pf.ps.auc(a, b), a, b
    aL, a1, b1 = auc_boot(selL); aM, a2, b2 = auc_boot(selM)
    dl = []
    for _ in range(B):
        j1 = rng.integers(0, a1.size, a1.size); j2 = rng.integers(0, a2.size, a2.size)
        dl.append((pf.ps.auc(a1[j1], b1[j1]), pf.ps.auc(a2[j2], b2[j2])))
    dl = np.array(dl)
    res["L_vs_M"] = dict(auc_L=aL, auc_L_ci=list(np.percentile(dl[:, 0], [2.5, 97.5])), auc_M=aM,
                         auc_M_ci=list(np.percentile(dl[:, 1], [2.5, 97.5])), diff=aL - aM,
                         diff_ci=list(np.percentile(dl[:, 0] - dl[:, 1], [2.5, 97.5])), n_L=a1.size, n_M=a2.size)
    out = pf.ANA / "length_truncation.json"
    json.dump(res, open(out, "w"), indent=1, default=float); print("wrote", out)

if __name__ == "__main__":
    main()
