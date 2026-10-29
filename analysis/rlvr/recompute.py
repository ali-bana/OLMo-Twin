#!/usr/bin/env python3
"""Independent recomputation of every post-training/RL number
(no code imported from figures/paper_figures.py; definitions follow the paper:
paper_figures.fig_posttrain / tpr_at / injection_v2_plotstyle.auc; the results
agreed with two other implementations).

score per document at a stage  s(d) = mean_logprob(treated arm, d) - mean_logprob(control arm, d)
within-pair contrast           D_p  = s(member_p) - s(partner_p)
tau = mean D, SD = sd(D, ddof=1), d_z = tau/SD, %pos = mean(D>0)
AUC = rank AUC of member scores vs partner scores (ties 1/2)
TPR@1%FPR = share of member scores > the (n-k)-th smallest partner score, k = floor(0.01 n)
Mid-training row: per_document_attacks_prelim.csv.gz column ratio_oracle (the paper's mid row).
CIs: pair bootstrap (member+partner together), B=2000, numpy default_rng(SEED) re-seeded per row,
percentile 95%; retentions/correlations use the mid-training replicate on the same resampled pairs.

Inputs (under the data root, see olmo_twin/paths.py): analysis/per_document_attacks_prelim.csv.gz,
analysis/posttrain_exclusions.json, run/scores_posttrain/<label>/scores.csv.gz and
rlvr/overlap/rlvr_overlap_per_doc.csv.gz (collection/rlvr_overlap_per_doc.py).
Outputs: rlvr/work/recompute_all.csv, recompute_drift.json, recompute_cohort.json.
Usage: python analysis/rlvr/recompute.py [WORKERS]   (default 32), then rl_minus_dpo.py and finalize.py.
"""
import json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from olmo_twin import paths  # noqa: E402

ANA = paths.ANALYSIS
TABLE = ANA / "per_document_attacks_prelim.csv.gz"
EXCL = ANA / "posttrain_exclusions.json"
SP = paths.RUN / "scores_posttrain"
HERE = Path(__file__).resolve().parent
OVERLAP = paths.RLVR / "overlap/rlvr_overlap_per_doc.csv.gz"
OUT = paths.RLVR / "work"
NAT = {"M", "L", "W-new", "W-gap", "S-se"}
COHORTS = ["M", "L", "W-new", "W-gap", "S-se", "E", "E0"]
B, SEED, FINAL = 2000, 20261028, 2604

STAGES = [("mid", None, None), ("sft", "sft-injected", "sft-baseline"), ("dpo", "dpo-injected", "dpo-baseline"),
          ("rlvr1", "rlvr1-injected", "rlvr1-baseline"),
          ("rlvr2i", "rlvr2-step2000-injected", "rlvr2-step2000-baseline"),
          ("rlvr2", "rlvr2-injected", "rlvr2-baseline")]
PREV = {"sft": "mid", "dpo": "sft", "rlvr1": "dpo", "rlvr2i": "rlvr1", "rlvr2": "rlvr1"}
TRAJ = [("dpo", 0, "dpo-injected", "dpo-baseline")]
for k in list(range(200, 2601, 200)) + [FINAL]:
    TRAJ.append(("rlvr1", k, *(("rlvr1-injected", "rlvr1-baseline") if k == FINAL else (f"rlvr1-step{k}-injected", f"rlvr1-step{k}-baseline"))))
for k in list(range(200, 2601, 200)) + [FINAL]:
    TRAJ.append(("rlvr2", k, *(("rlvr2-injected", "rlvr2-baseline") if k == FINAL else (f"rlvr2-step{k}-injected", f"rlvr2-step{k}-baseline"))))
AI2 = [("sft", "sft-injected", "ai2-sft", "sft-baseline"), ("dpo", "dpo-injected", "ai2-dpo", "dpo-baseline"),
       ("rlvr1", "rlvr1-injected", "ai2-rlvr1", "rlvr1-baseline"),
       ("rlvr2i", "rlvr2-step2000-injected", "ai2-instruct", "rlvr2-step2000-baseline"),
       ("rlvr2", "rlvr2-injected", "ai2-instruct", "rlvr2-baseline")]   # last row: extra (final vs Instruct)

_S = {}
def scores(name):
    if name not in _S:
        df = pd.read_csv(SP / name / "scores.csv.gz", keep_default_na=False, dtype=str)
        v = pd.to_numeric(df["mean_logprob"].where(df["status"] == "ok", ""), errors="coerce")
        s = pd.Series(v.values, index=df["doc_id"].values)
        assert not s.index.duplicated().any()
        _S[name] = s
    return _S[name]

def load_pairs():
    t = pd.read_csv(TABLE, keep_default_na=False, dtype=str, usecols=["doc_id", "cohort", "arm", "pair_id", "r", "ratio_oracle"])
    ex = json.loads(EXCL.read_text())
    t = t[~t.pair_id.isin(ex["exclude_pair_ids"]) & ~t.doc_id.isin(ex["exclude_doc_ids"])]
    t["r"] = pd.to_numeric(t["r"], errors="coerce"); t["ratio_oracle"] = pd.to_numeric(t["ratio_oracle"], errors="coerce")
    m = t[(t.pair_id != "") & (t.arm == "inject")].set_index("pair_id")
    h = t[(t.pair_id != "") & (t.arm == "holdout")].set_index("pair_id")
    ids = [p for p in m.index if p in set(h.index)]   # table order of the member rows (the order the generator uses;
                                                      # it fixes which pairs each bootstrap replicate draws)
    P = pd.DataFrame(dict(pair_id=ids, cohort=m.loc[ids, "cohort"].values, r=m.loc[ids, "r"].values,
                          doc_m=m.loc[ids, "doc_id"].values, doc_h=h.loc[ids, "doc_id"].values,
                          mid_m=m.loc[ids, "ratio_oracle"].values, mid_h=h.loc[ids, "ratio_oracle"].values))
    assert (h.loc[ids, "cohort"].values == P.cohort.values).all()
    return t, P

def pops(P):
    nat = P[(P.r == 1) & P.cohort.isin(NAT)]
    out = {"NAT1": nat, "ALL7326": P[~P.cohort.isin({"X", "F"})], "ALLPAIRS7803": P}
    # RL-overlap sensitivity (not the paper's rule; the rule flags nothing): drop M-01966, and drop any NAT1 pair
    # with a document sharing >=5 distinct 8-grams or a >=20-word run with an RLVR record
    ov = pd.read_csv(OVERLAP, keep_default_na=False)
    ov = ov[(pd.to_numeric(ov.max_shingles) >= 5) | (pd.to_numeric(ov.max_run) >= 20)]
    bad = set(ov.doc_id)
    out["NAT1_minus_M01966"] = nat[nat.pair_id != "M-01966"]
    out["NAT1_minus_RLoverlap5"] = nat[~nat.doc_m.isin(bad) & ~nat.doc_h.isin(bad)]
    return {k: v.reset_index(drop=True) for k, v in out.items()}

def arrays(pop, tr, co):
    if tr is None:
        return pop.mid_m.values.astype(float), pop.mid_h.values.astype(float)
    s = scores(tr) - scores(co).reindex(scores(tr).index)
    return s.reindex(pop.doc_m.values).values.astype(float), s.reindex(pop.doc_h.values).values.astype(float)

def auc(a, b):
    rk = rankdata(np.concatenate([a, b]))
    return (rk[:a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size)

def tpr1(a, b):
    bs = np.sort(b); k = int(math.floor(0.01 * bs.size))
    return float((a > bs[bs.size - k - 1]).mean())

def point(a, b):
    d = a - b; sd = d.std(ddof=1)
    return dict(n=d.size, tau=d.mean(), sd=sd, dz=d.mean() / sd, pos=(d > 0).mean(), auc=auc(a, b), tpr1=tpr1(a, b))

def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])

def boot(job):
    a, b, a0, b0 = job["a"], job["b"], job["a0"], job["b0"]
    n = a.size
    idx = np.random.default_rng(SEED).integers(0, n, (B, n))
    R = {k: np.empty(B) for k in ("tau", "sd", "dz", "auc", "tpr1", "ret_tau", "ret_sd", "ret_dz", "ret_auc", "tpr_change", "corr_mid")}
    for i in range(B):
        j = idx[i]; aa, bb = a[j], b[j]; d = aa - bb
        p = point(aa, bb)
        for k in ("tau", "sd", "dz", "auc", "tpr1"):
            R[k][i] = p[k]
        if a0 is not None:
            a0j, b0j = a0[j], b0[j]; d0 = a0j - b0j
            t0, s0 = d0.mean(), d0.std(ddof=1); u0 = auc(a0j, b0j); r0 = tpr1(a0j, b0j)
            R["ret_tau"][i] = p["tau"] / t0; R["ret_sd"][i] = p["sd"] / s0; R["ret_dz"][i] = p["dz"] / (t0 / s0)
            R["ret_auc"][i] = (p["auc"] - .5) / (u0 - .5); R["tpr_change"][i] = p["tpr1"] / r0 - 1
            R["corr_mid"][i] = corr(d0, d)
    keys = ("tau", "sd", "dz", "auc", "tpr1") + (("ret_tau", "ret_sd", "ret_dz", "ret_auc", "tpr_change", "corr_mid") if a0 is not None else ())
    return job["key"], {k: (float(np.percentile(R[k], 2.5)), float(np.percentile(R[k], 97.5))) for k in keys}

def row(pop, tr, co, mid, prev=None):
    a, b = arrays(pop, tr, co)
    assert np.isfinite(a).all() and np.isfinite(b).all(), (tr, co)
    r = point(a, b)
    a0, b0 = mid
    if tr is not None:
        r["corr_mid"] = corr(a0 - b0, a - b)
        if prev is not None:
            r["corr_prev"] = corr(prev[0] - prev[1], a - b)
    return r, (a, b)

def derived(r, m):
    return dict(ret_tau=100 * r["tau"] / m["tau"], ret_sd=100 * r["sd"] / m["sd"], ret_dz=100 * r["dz"] / m["dz"],
                ret_auc=100 * (r["auc"] - .5) / (m["auc"] - .5), tpr_change=100 * (r["tpr1"] / m["tpr1"] - 1),
                tpr_change_pp=100 * (r["tpr1"] - m["tpr1"]))

def main():
    t0 = time.time()
    table, P = load_pairs()
    PO = pops(P)
    print({k: len(v) for k, v in PO.items()}, flush=True)
    res, jobs = {}, []
    for pn, pop in PO.items():
        mid_ab = arrays(pop, None, None)
        AB = {}
        for s, tr, co in STAGES:
            r, ab = row(pop, tr, co, mid_ab, AB.get(PREV.get(s)))
            AB[s] = ab
            if s == "rlvr2":
                r["corr_rlvr2step2000"] = corr(AB["rlvr2i"][0] - AB["rlvr2i"][1], ab[0] - ab[1])
            res[("stage", pn, s)] = r
            jobs.append(dict(key=("stage", pn, s), a=ab[0], b=ab[1], a0=None if tr is None else mid_ab[0], b0=None if tr is None else mid_ab[1]))
        if not pn.startswith("NAT1_"):
            prevab = None
            for st, k, tr, co in TRAJ:
                r, ab = row(pop, tr, co, mid_ab, prevab)
                if prevab is None:
                    r["corr_prev"] = float("nan")
                prevab = ab
                res[("traj", pn, f"{st}:{k}")] = r
                jobs.append(dict(key=("traj", pn, f"{st}:{k}"), a=ab[0], b=ab[1], a0=mid_ab[0], b0=mid_ab[1]))
            for s, tr, co_ai2, co_ours in AI2:
                r, ab = row(pop, tr, co_ai2, mid_ab, AB[s])   # corr_prev here = corr with the main (matched-control) design
                r["corr_main_design"] = r.pop("corr_prev")
                res[("ai2", pn, s)] = r
                jobs.append(dict(key=("ai2", pn, s), a=ab[0], b=ab[1], a0=mid_ab[0], b0=mid_ab[1]))
                pl, pab = row(pop, co_ours, co_ai2, mid_ab)
                res[("placebo", pn, s)] = pl
                jobs.append(dict(key=("placebo", pn, s), a=pab[0], b=pab[1], a0=None, b0=None))
    # absolute drift of our control chain vs AI2's (per document, all docs / null pool / NAT1 docs)
    drift = {}
    for s, tr, co_ai2, co_ours in AI2[:4]:
        x = scores(co_ours) - scores(co_ai2).reindex(scores(co_ours).index)
        nulls = table[table.arm == "null"].doc_id.values
        natdocs = np.concatenate([PO["NAT1"].doc_m.values, PO["NAT1"].doc_h.values])
        d = {}
        for nm, ids in (("all_docs", None), ("null_pool", nulls), ("NAT1_docs", natdocs)):
            v = (x if ids is None else x.reindex(ids)).values; v = v[np.isfinite(v)]
            d[nm] = dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)))
        drift[s] = d
    print(f"points done {time.time()-t0:.0f}s; bootstrapping {len(jobs)} rows", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 32) as pool:
        cis = dict(pool.map(boot, jobs, chunksize=1))
    print(f"bootstrap done {time.time()-t0:.0f}s", flush=True)
    mids = {pn: res[("stage", pn, "mid")] for pn in PO}
    out = []
    for key, r in res.items():
        kind, pn, s = key
        rec = dict(kind=kind, population=pn, stage=s, **{k: float(v) for k, v in r.items()})
        if kind != "placebo" and s != "mid":
            rec.update(derived(r, mids[pn]))
        for k, (lo, hi) in cis[key].items():
            rec[f"{k}_lo"] = lo; rec[f"{k}_hi"] = hi
        out.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(OUT / "recompute_all.csv", index=False)
    (OUT / "recompute_drift.json").write_text(json.dumps(drift, indent=1))
    coh = {}
    for ch in COHORTS:
        pop = P[P.cohort == ch].reset_index(drop=True)   # pairs of ALL within the cohort, all doses (as the generator)
        coh[ch] = {"n": len(pop)}
        for s, tr, co in STAGES:
            a, b = arrays(pop, tr, co); coh[ch][s] = float(auc(a, b))
        for s, tr, co_ai2, _ in AI2:
            a, b = arrays(pop, tr, co_ai2); coh[ch][f"ai2_{s}"] = float(auc(a, b))
    (OUT / "recompute_cohort.json").write_text(json.dumps(coh, indent=1))
    print(f"done {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
