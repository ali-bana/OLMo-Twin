#!/usr/bin/env python3
"""Reference-calibrated membership-inference attacks on the injection-v2 per-token scores.

Two attacks are computed for every evaluation document and evaluated on the
HOLDOUT split of the matched pairs (fixed in the design), with every hyperparameter
selected on the DEV split only:

  1. RMIA, offline (Zarifzadeh, Liu, Shokri, "Low-Cost High-Power Membership
     Inference Attacks", ICML 2024, arXiv:2312.03262).
  2. A per-document standardized reference ratio (a per-document t-statistic
     of the per-token log-probability difference).

Models (all score the same 22,680 documents with identical token ids):
  TREATED   scores_ext/ext-injected                 target model theta
  CONTROL   scores_ext/ext-seed42                   AI2 seed-42 run, exact counterfactual (OUT for all docs)
  SIBLING   scores/sibling-seed42069-step23852      AI2 seed-42069 run, same recipe/mixture, other order (OUT)
  STAGE1    scores_ext/ext-stage1-pretreatment      published pre-stage-2 checkpoint (OUT, never saw stage 2)
No IN reference model exists, so RMIA is run in its OFFLINE form.

Notation. For document x and model m, lp_m(t) is the log-probability of token t;
"scored tokens" are the positions where lp is finite (the first position of
every 4096-token window is NaN). The NaN mask is identical across the four
models (checked). L_m(x) = mean_t lp_m(t) is exactly the `<model>__loss` column
of per_document_attacks_prelim.csv.gz; S_m(x) = sum_t lp_m(t).

---------------------------------------------------------------- RMIA (offline)
Paper, eq. 3 and 5:
    LR_theta(x, z) = [Pr(x|theta) / Pr(x)] / [Pr(z|theta) / Pr(z)]
    Score(x)       = Pr_{z ~ Z}[ LR_theta(x, z) > gamma ]
The paper writes >= in eq. 5 and > in Algorithm 1 line 19 and in the official
code (privacytrustlab/ml_privacy_meter, `ratios > 1.0`); we use the strict
form. With continuous ratios the two differ only on a null set.

Pr(x|theta), two variants (the paper defines it for classifiers only):
    gm   per-token geometric-mean probability  exp(L_theta(x))      [primary]
    seq  full sequence probability             exp(S_theta(x))
`seq` scales with length: exp(S) is ~exp(-2000) for a 1,000-token document.

Pr(x), offline (Algorithm 1 lines 12-13, eq. 10, Appendix B.2.2):
    Pr_OUT(x) = (1/k) sum_{theta' in refset} Pr(x|theta')          (arithmetic mean of probabilities)
    Pr(x)     = 0.5 * [ (1 + a) * Pr_OUT(x) + (1 - a) ]
derived from the linear model Pr_IN(x) = a * (Pr_OUT(x) - 1) + 1 and
Pr(x) = (Pr_IN + Pr_OUT) / 2. The SAME computation is applied to Pr(z): the
paper's Section 3 says "The same computation process applies to Pr(z)" and the
official code applies the a-scaling to the population samples too (Algorithm 1
line 17 omits it; we follow the text and the code). All quantities are formed
in log space (logsumexp / logaddexp), so the `seq` variant does not underflow.

Reference sets: {SIBLING} (realistic), {CONTROL} (oracle), {CONTROL, SIBLING}
(two OUT models averaged as above), {STAGE1}.

Population samples Z. Documents known to be non-members and disjoint from the
evaluated documents: the held-out partners (arm == holdout) of the DEV-split
pairs of the population being evaluated (NAT1 or ALL). Holdout-split documents
are scored against this full Z. Dev-split documents (used only for selecting
hyperparameters) are scored against Z with their own pair's held-out member
removed (leave-one-pair-out), so no document is ever compared with itself or
with its matched partner. The paper's default is "all non-members of the target
model ... excluding the query x itself" (Section 5.6).

Score as computed:  Score(x) = #{z in Z : log R(z) < log R(x) - log gamma} / |Z|,
with log R(.) = log Pr(.|theta) - log Pr(.). For a FIXED Z and gamma this is the
empirical CDF of {log R(z)} evaluated at log R(x) - log gamma, a non-decreasing
step function of log R(x). RMIA therefore ranks documents exactly as log R(x)
does except that documents falling between the same two population points tie
(resolution 1/|Z|); it cannot change the AUC or the TPR at a fixed FPR except
through those ties. gamma only shifts where the CDF is read and thus only moves
ties/saturation. Whatever RMIA changes relative to the plain reference ratio
comes from a < 1:
  * gm, single reference, a = 1:  log R(x) = L_theta(x) - L_ref(x), the plain ratio exactly;
  * gm, a < 1: log R(x) = L_theta - log(0.5((1+a) e^{L_ref} + 1 - a)), a mixture of the
    plain ratio and the target-only LOSS statistic L_theta (a = 0 is ~LOSS);
  * seq, a < 1: e^{S_ref} is ~0 for every document, so log R(x) = S_theta - log((1-a)/2),
    i.e. the total log-likelihood attack; seq, a = 1: S_theta - S_ref = T * mean_t d_t.

Hyperparameters, chosen per (population, reference set, variant) on the DEV
pairs by maximising the dev rank AUC (the paper's criterion for a; Appendix
B.2.2), ties broken by dev pooled TPR@1%FPR, then by grid order:
    a     in {0, 0.1, ..., 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1}
          (the official code tunes a on 0..1 step 0.1; the finer points near 1
           matter because document-level gm probabilities are ~0.02-0.3, so
           the (1-a) term dominates unless a is close to 1)
    gamma = exp(g), g in {0, .001, .002, .005, .01, .02, .05, .1, .2, .5, ln 2, 1, 2, 5}
          (gamma >= 1 as the paper requires; the log scale is needed because
           gm log-ratios are O(0.01-0.1) nats while seq log-ratios are O(10))
The paper's own procedure for a (attack one reference model with another) is
impossible here: every reference is OUT for every evaluation document, so a
reference used as a temporary target has no members. Dev-split selection
replaces it; it gives the attacker labelled members for 20% of the pairs, which
is more than the paper's threat model grants.
Defaults reported alongside: a = 0.3 (official-code default; the paper's CIFAR-10
/ CINIC-10 value) with gamma = 2 (the paper: "Throughout our experiments, we set
gamma to 2"), and a = 1, gamma = 1 (the pure-reference-ratio point; the official
code's hard-coded gamma).

------------------------------------------------- standardized reference ratio
For reference ref in {SIBLING, CONTROL, STAGE1}, over the T scored positions
(NaN positions dropped and the remaining series treated as contiguous):
    d_t       = lp_TREATED(t) - lp_ref(t)
    ratio(x)  = mean_t d_t                                 (= the existing ratio_* column)
    SE(x)     = sd_t(d_t) / sqrt(n_eff(x))                 (sd with ddof = 1)
    z(x)      = mean_t d_t / SE(x)
Variants:
    T    n_eff = T (tokens treated as independent)
    ac   n_eff = T / tau, clamped to [1, T] (tau <= 0 -> n_eff = T), where tau is the
         integrated autocorrelation time of d_t with Sokal's automatic window:
           rho_k     = c_k / c_0, c_k = (1/T) sum_{t < T-k} (d_t - dbar)(d_{t+k} - dbar)
                       (FFT with zero padding to >= 2T, i.e. no circular wrap)
           tau(W)    = 1 + 2 sum_{k=1..W} rho_k
           W*        = smallest W >= 1 with W >= c * tau(W), c = 5 (W* = T-1 if none)
           tau       = tau(W*)
Documents with T < 2 get z = NaN. No hyperparameters. z is used as a ranking
statistic only: among non-members it is NOT N(0,1) (the treated model's
log-probabilities are shifted upward on unseen documents of the injected
domains, and document-level drift between runs is not token noise), which the
summary records under null_behaviour.

---------------------------------------------------------------- evaluation
Populations (pairs with both members scored, after the post-training exclusions
of posttrain_exclusions.json -- 10 pairs and 16 documents -- and the one
unscored document):
    NAT1  r == 1 and cohort in {M, L, W-new, W-gap, S-se}        (4,832 pairs)
    ALL   every analysed pair                                     (7,803 pairs)
Positives = injected members, negatives = their held-out partners. All reported
numbers use HOLDOUT-split pairs; DEV-split pairs are used for selection and for
the dev-vs-holdout optimism check only.
    AUC            rank (Mann-Whitney) AUC, ties counted 1/2 (identical to
                   injection_v2_plotstyle.auc; vectorised with scipy rankdata)
    TPR@1%FPR      pooled: threshold = the score that admits floor(0.01 * n_neg)
                   negatives, strict '>' (identical to paper_figures.tpr_at);
                   NOTE injection_v2_plotstyle.tpr_at_fpr uses an interpolated
                   quantile instead and is not used here
    stratified     one such threshold per cohort, cohort TPRs averaged with
                   weights = number of holdout pairs in the cohort
                   (paper_figures' domain-stratified operating point)
    CIs            percentile 95%, 400 bootstrap draws resampling holdout PAIRS;
                   the same draws are used for every attack, so the paired
                   differences vs the plain ratio also get CIs

Outputs (under the data root, see olmo_twin/paths.py)
  analysis/per_document_attacks_calibrated.csv.gz
     one row per scored doc_id; columns:
       doc_id cohort stratum arm pair_id split r n_scored
       loss_<m>, sumlp_<m>               m in treated, control, sibling, stage1
       ratio_<ref>, sd_d_<ref>, tau_<ref>, n_eff_<ref>     ref in sibling, control, stage1
       zratio_<ref>_T, zratio_<ref>_ac
       rmia_<refset>_<gm|seq>_<selected|default>_<NAT1|ALL>
            refset in sibling, control, control_sibling, stage1; NaN outside the
            population; holdout docs scored against the full dev Z, dev docs
            against the leave-one-pair-out Z
  analysis/calibrated_attacks_summary.json

Inputs: bundle/evaluation_manifest.csv.gz, analysis/per_document_attacks_prelim.csv.gz,
analysis/posttrain_exclusions.json, analysis/token_neff.npz (optional cross-check) and
the per-token score files run/<score dir>/tokens/*.npz of the four models above.
The per-document statistics extracted from the per-token files are cached in
--cache (default analysis/cache/doc_stats.npz) and reused when the document list
matches. With --recompute (or without a cache) the per-token files are read,
including their token_ids (alignment check), so this mode needs per-token files
that still contain the token ids.

Usage (CPU only):
  python analysis/calibrated_attacks.py --workers 16
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

ANA = paths.ANALYSIS
MANIFEST = paths.BUNDLE / "evaluation_manifest.csv.gz"
PRELIM = ANA / "per_document_attacks_prelim.csv.gz"
EXCL = ANA / "posttrain_exclusions.json"
NEFF_PREV = ANA / "token_neff.npz"
OUT_CSV = ANA / "per_document_attacks_calibrated.csv.gz"
OUT_JSON = ANA / "calibrated_attacks_summary.json"
RUN = paths.RUN


def _rel(path: Path) -> str:
    """Path relative to the data root for the metadata (absolute if outside it)."""
    try:
        return str(Path(path).relative_to(paths.DATA))
    except ValueError:
        return str(path)

# label -> (score dir under RUN, column prefix in the prelim table)
MODELS = {
    "treated": ("scores_ext/ext-injected", "injected"),
    "control": ("scores_ext/ext-seed42", "seed42"),
    "sibling": ("scores/sibling-seed42069-step23852", "sibling"),
    "stage1": ("scores_ext/ext-stage1-pretreatment", "stage1"),
}
REFS = ("sibling", "control", "stage1")
PRELIM_RATIO = {"sibling": "ratio_sibling", "control": "ratio_oracle", "stage1": "ratio_stage1"}
REFSETS = {
    "sibling": ("sibling",),
    "control": ("control",),
    "control_sibling": ("control", "sibling"),
    "stage1": ("stage1",),
}
NAT = {"M", "L", "W-new", "W-gap", "S-se"}
SOKAL_C = 5.0
A_GRID = [round(0.1 * i, 1) for i in range(10)] + [0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
LOG_GAMMA_GRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, math.log(2.0), 1.0, 2.0, 5.0]
HP_DEFAULT = {"a": 0.3, "gamma": 2.0}
HP_RATIO_POINT = {"a": 1.0, "gamma": 1.0}
FPR = 0.01


def safe(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace(":", "_")


# ===================================================================== per document
def sokal_tau(x: np.ndarray, c: float = SOKAL_C):
    """Integrated autocorrelation time with Sokal's automatic window.

    tau(W) = 1 + 2 sum_{k=1..W} rho_k; window W* = smallest W >= 1 with
    W >= c * tau(W) (T-1 if never reached). Returns (tau, W*); (nan, 0) if the
    series is constant or shorter than 2.
    """
    n = x.size
    if n < 2:
        return float("nan"), 0
    y = x - x.mean()
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(y, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n] / n
    if not acov[0] > 0:
        return float("nan"), 0
    rho = acov / acov[0]
    taus = 2.0 * np.cumsum(rho) - 1.0          # taus[W] = 1 + 2 sum_{k=1..W} rho_k
    lags = np.arange(n)
    ok = (lags >= 1) & (lags >= c * taus)
    w = int(np.argmax(ok)) if ok.any() else n - 1
    return float(taus[w]), w


def per_doc(doc_id: str) -> dict:
    out = {"doc_id": doc_id, "missing": ""}
    data = {}
    for m, (d, _) in MODELS.items():
        f = RUN / d / "tokens" / f"{safe(doc_id)}.npz"
        if not f.exists():
            out["missing"] = m
            return out
        with np.load(f) as z:
            data[m] = (z["token_ids"], z["logprobs"].astype(np.float64))
    ids_t, lp_t = data["treated"]
    ok_t = np.isfinite(lp_t)
    for m, (ids, lp) in data.items():
        v = np.isfinite(lp)
        out[f"loss_{m}"] = float(lp[v].mean()) if v.any() else float("nan")
        out[f"sumlp_{m}"] = float(lp[v].sum()) if v.any() else float("nan")
        out[f"n_{m}"] = int(v.sum())
        out[f"ids_eq_{m}"] = bool(ids.shape == ids_t.shape and np.array_equal(ids, ids_t))
        out[f"mask_eq_{m}"] = bool(lp.shape == lp_t.shape and np.array_equal(v, ok_t))
    for r in REFS:
        lp_r = data[r][1]
        n = min(lp_r.size, lp_t.size)
        ok = np.isfinite(lp_t[:n]) & np.isfinite(lp_r[:n])
        d = lp_t[:n][ok] - lp_r[:n][ok]
        T = int(d.size)
        mean = float(d.mean()) if T else float("nan")
        sd = float(d.std(ddof=1)) if T > 1 else float("nan")
        tau, w = sokal_tau(d)
        if T < 2:
            neff = float(T) if T else float("nan")
        elif not np.isfinite(tau) or tau <= 0:
            neff = float(T)
        else:
            neff = float(min(max(T / tau, 1.0), T))
        out[f"T_{r}"] = T
        out[f"ratio_{r}"] = mean
        out[f"sd_d_{r}"] = sd
        out[f"tau_{r}"] = tau
        out[f"win_{r}"] = w
        out[f"n_eff_{r}"] = neff
        if T > 1 and sd > 0:
            out[f"zratio_{r}_T"] = mean / (sd / math.sqrt(T))
            out[f"zratio_{r}_ac"] = mean / (sd / math.sqrt(neff))
        else:
            out[f"zratio_{r}_T"] = float("nan")
            out[f"zratio_{r}_ac"] = float("nan")
    return out


def extract(doc_ids, workers, cache: Path, recompute: bool):
    if cache.exists() and not recompute:
        z = np.load(cache, allow_pickle=True)
        if list(z["doc_ids"]) == list(doc_ids):
            print(f"[extract] using cache {cache}", flush=True)
            return {k: z[k] for k in z.files}
    t0 = time.time()
    with Pool(workers) as pool:
        res = []
        for i, r in enumerate(pool.imap(per_doc, doc_ids, chunksize=32)):
            res.append(r)
            if (i + 1) % 4000 == 0:
                print(f"[extract] {i + 1}/{len(doc_ids)} docs, {time.time() - t0:.0f}s", flush=True)
    print(f"[extract] {len(res)} docs in {time.time() - t0:.0f}s with {workers} workers", flush=True)
    keys = sorted({k for r in res for k in r if k not in ("doc_id", "missing")})
    cols = {"doc_ids": np.array(doc_ids, dtype=object),
            "missing": np.array([r["missing"] for r in res], dtype=object)}
    for k in keys:
        cols[k] = np.array([r.get(k, np.nan) for r in res], dtype=float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **cols)
    return cols


# ===================================================================== metrics
def auc(pos, neg) -> float:
    """Rank AUC, ties 1/2; same definition as injection_v2_plotstyle.auc."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    u = r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def tpr_at(pos, neg, fpr=FPR):
    """TPR at the threshold admitting floor(fpr*n_neg) negatives, strict '>'
    (paper_figures.tpr_at). Returns (tpr, k, n_neg)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = np.sort(neg[np.isfinite(neg)])
    if pos.size == 0 or neg.size == 0:
        return float("nan"), 0, int(neg.size)
    k = int(math.floor(fpr * neg.size))
    thr = neg[neg.size - k - 1]
    return float((pos > thr).mean()), k, int(neg.size)


def strat_tpr(score, pairs, cohort, fpr=FPR):
    """Domain-stratified TPR: one threshold per cohort, weighted by cohort pair count."""
    by = defaultdict(list)
    for i, h in pairs:
        by[cohort[i]].append((i, h))
    tp = tot = 0.0
    per = {}
    for ch in sorted(by):
        sel = by[ch]
        t, k, nn = tpr_at(score[[i for i, _ in sel]], score[[h for _, h in sel]], fpr)
        per[ch] = {"tpr": t, "k": k, "n_pairs": len(sel)}
        tp += t * len(sel); tot += len(sel)
    return float(tp / tot), per


def null_behaviour(score, pairs, cohort, z_crit=2.326):
    """Distribution of a score among the non-members (and members) of `pairs`.

    For the z-ratio, z_crit = 2.326 is the one-sided 1% point of N(0,1): if z were
    a calibrated per-document test, ~1% of non-members would exceed it.
    """
    P = np.array([i for i, _ in pairs]); N = np.array([h for _, h in pairs])
    sn, sp = score[N], score[P]
    out = {"mean_nonmembers": float(np.nanmean(sn)), "sd_nonmembers": float(np.nanstd(sn)),
           "mean_members": float(np.nanmean(sp)), "sd_members": float(np.nanstd(sp)),
           f"share_nonmembers_above_{z_crit}": float(np.nanmean(sn > z_crit)),
           f"share_members_above_{z_crit}": float(np.nanmean(sp > z_crit)), "by_cohort": {}}
    cn = cohort[N]
    for ch in sorted(set(cn)):
        m = cn == ch
        out["by_cohort"][ch] = {"mean_nonmembers": float(np.nanmean(sn[m])), "sd_nonmembers": float(np.nanstd(sn[m])),
                                f"share_nonmembers_above_{z_crit}": float(np.nanmean(sn[m] > z_crit))}
    return out


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def evaluate(score, pairs, cohort, boot_idx, base=None, dev_pairs=None):
    """Holdout metrics of `score` on `pairs` (list of (inject_idx, holdout_idx)).

    boot_idx: (B, n_pairs) resampling indices into `pairs`, shared across attacks.
    base: the plain-ratio score on the same documents, for paired differences.
    """
    P = np.array([i for i, _ in pairs]); N = np.array([h for _, h in pairs])
    s_p, s_n = score[P], score[N]
    t1, k1, nn = tpr_at(s_p, s_n)
    st, per = strat_tpr(score, pairs, cohort)
    res = {"n_pairs": len(pairs), "auc": auc(s_p, s_n), "tpr1": t1, "tpr1_k_negatives": k1,
           "n_neg": nn, "tpr1_strat": st, "tpr1_strat_by_cohort": per,
           "n_nonfinite": int((~np.isfinite(s_p)).sum() + (~np.isfinite(s_n)).sum())}
    ba = np.array([auc(s_p[j], s_n[j]) for j in boot_idx])
    bt = np.array([tpr_at(s_p[j], s_n[j])[0] for j in boot_idx])
    res["auc_ci"] = ci(ba); res["tpr1_ci"] = ci(bt)
    if base is not None:
        b_p, b_n = base[P], base[N]
        bba = np.array([auc(b_p[j], b_n[j]) for j in boot_idx])
        bbt = np.array([tpr_at(b_p[j], b_n[j])[0] for j in boot_idx])
        res["delta_auc_vs_plain"] = res["auc"] - auc(b_p, b_n)
        res["delta_auc_vs_plain_ci"] = ci(ba - bba)
        res["delta_tpr1_vs_plain"] = t1 - tpr_at(b_p, b_n)[0]
        res["delta_tpr1_vs_plain_ci"] = ci(bt - bbt)
        res["delta_tpr1_strat_vs_plain"] = st - strat_tpr(base, pairs, cohort)[0]
    if dev_pairs is not None:
        res["auc_dev"] = auc(score[[i for i, _ in dev_pairs]], score[[h for _, h in dev_pairs]])
        res["tpr1_dev"] = tpr_at(score[[i for i, _ in dev_pairs]], score[[h for _, h in dev_pairs]])[0]
    return res


# ===================================================================== RMIA
def log_ratio(D, refset, kind, a):
    """log Pr(x|theta) - log Pr(x), offline approximation, for every document."""
    stat = "loss" if kind == "gm" else "sumlp"
    tgt = D[f"{stat}_treated"]
    refs = np.stack([D[f"{stat}_{r}"] for r in REFSETS[refset]])
    lpo = logsumexp(refs, axis=0) - math.log(refs.shape[0])     # log Pr_OUT(x)
    if a >= 1.0:
        lpx = lpo
    else:
        with np.errstate(invalid="ignore"):     # NaN only for the one unscored document
            lpx = np.logaddexp(math.log1p(a) + lpo, math.log(1.0 - a)) - math.log(2.0)
    return tgt - lpx


def rmia_scores(lr, X, Z, log_gamma, loo_self=None):
    """Score(x) = #{z in Z: lr_z < lr_x - log gamma} / |Z| (strict '>' on LR).

    loo_self: for each x, the index of a Z member to leave out (its pair's
    held-out document); then the denominator is |Z| - 1.
    """
    zs = np.sort(lr[Z])
    thr = lr[X] - log_gamma
    cnt = np.searchsorted(zs, thr, side="left").astype(float)
    if loo_self is None:
        return cnt / zs.size
    cnt -= (lr[loo_self] < thr).astype(float)
    return cnt / (zs.size - 1)


def rmia_population(D, refset, kind, a, gamma, dev_pairs, hold_pairs, n_docs):
    """Score array over all documents (NaN outside the population)."""
    lr = log_ratio(D, refset, kind, a)
    Z = np.array([h for _, h in dev_pairs])
    out = np.full(n_docs, np.nan)
    lg = math.log(gamma)
    Xh = np.array([d for p in hold_pairs for d in p])
    out[Xh] = rmia_scores(lr, Xh, Z, lg)
    Xd = np.array([d for p in dev_pairs for d in p])
    self_ = np.array([h for _, h in dev_pairs for _ in (0, 1)])
    out[Xd] = rmia_scores(lr, Xd, Z, lg, loo_self=self_)
    return out, lr


# ===================================================================== main
def read_csv_gz(path):
    csv.field_size_limit(sys.maxsize)
    with gzip.open(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", type=Path, default=ANA / "cache/doc_stats.npz",
                    help="per-document statistics cache")
    ap.add_argument("--recompute", action="store_true", help="ignore the cache")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    t_start = time.time()

    # ---------------------------------------------------------- documents
    man = read_csv_gz(MANIFEST)
    doc_ids = [r["doc_id"] for r in man]
    D = extract(doc_ids, args.workers, args.cache, args.recompute)
    missing = [(doc_ids[i], m) for i, m in enumerate(D["missing"]) if m]
    print(f"[docs] {len(doc_ids)} manifest docs, {len(missing)} without scores: {missing}", flush=True)
    scored = np.array([not m for m in D["missing"]])
    n_docs = len(doc_ids)
    cohort = np.array([r["cohort"] for r in man], dtype=object)
    arm = [r["arm"] for r in man]
    split = [r["split"] for r in man]
    rcop = np.array([fnum(r["r"]) for r in man])

    # ---------------------------------------------------------- pairs and populations
    excl = json.loads(EXCL.read_text())
    bad_pairs, bad_docs = set(excl["exclude_pair_ids"]), set(excl["exclude_doc_ids"])
    byp = defaultdict(dict)
    for i, r in enumerate(man):
        if not r["pair_id"] or not scored[i] or r["pair_id"] in bad_pairs or r["doc_id"] in bad_docs:
            continue
        byp[r["pair_id"]][r["arm"]] = i
    pairs_all = sorted((v["inject"], v["holdout"]) for v in byp.values() if "inject" in v and "holdout" in v)
    assert all(split[i] == split[h] for i, h in pairs_all), "pair members disagree on split"
    pops = {
        "NAT1": [(i, h) for i, h in pairs_all if rcop[i] == 1 and cohort[i] in NAT],
        "ALL": pairs_all,
    }
    splits = {p: {s: [(i, h) for i, h in v if split[i] == s] for s in ("dev", "holdout")} for p, v in pops.items()}
    for p in pops:
        print(f"[pop] {p}: {len(pops[p])} pairs (dev {len(splits[p]['dev'])}, holdout {len(splits[p]['holdout'])})", flush=True)

    # ---------------------------------------------------------- sanity checks
    rng = np.random.default_rng(args.seed)
    sanity = {}
    prelim = read_csv_gz(PRELIM)
    pre = {r["doc_id"]: r for r in prelim}
    idx_scored = np.nonzero(scored)[0]
    samp = rng.choice(idx_scored, 100, replace=False)
    loss_diff = {}
    for m, (_, pref) in MODELS.items():
        d_samp = [abs(D[f"loss_{m}"][i] - fnum(pre[doc_ids[i]][f"{pref}__loss"])) for i in samp]
        d_all = [abs(D[f"loss_{m}"][i] - fnum(pre[doc_ids[i]][f"{pref}__loss"])) for i in idx_scored]
        loss_diff[m] = {"prelim_column": f"{pref}__loss", "max_abs_diff_100_random": float(np.nanmax(d_samp)),
                        "max_abs_diff_all_docs": float(np.nanmax(d_all)),
                        "n_over_1e-6_all_docs": int(np.sum(np.array(d_all) > 1e-6))}
    nsc = [abs(D["n_treated"][i] - fnum(pre[doc_ids[i]]["injected__n_scored"])) for i in idx_scored]
    sanity["a_mean_logprob_reproduces_prelim"] = {**loss_diff, "n_scored_max_abs_diff_all": float(np.nanmax(nsc)),
                                                  "pass": all(v["max_abs_diff_100_random"] <= 1e-6 for v in loss_diff.values())}
    tok = {m: {"identical_100_random": bool(all(D[f"ids_eq_{m}"][samp] == 1)),
               "identical_all_docs": int(np.sum(D[f"ids_eq_{m}"][idx_scored] == 1)),
               "nan_mask_identical_all_docs": int(np.sum(D[f"mask_eq_{m}"][idx_scored] == 1))}
           for m in MODELS if m != "treated"}
    sanity["b_token_ids_identical"] = {"vs_treated": tok, "n_docs": int(idx_scored.size),
                                       "pass": all(v["identical_100_random"] for v in tok.values())}
    sanity["sample_doc_ids_100"] = [doc_ids[i] for i in samp]
    # ratio column cross-check over every scored doc
    rr = {}
    for r in REFS:
        dd = np.array([abs(D[f"ratio_{r}"][i] - fnum(pre[doc_ids[i]][PRELIM_RATIO[r]])) for i in idx_scored])
        rr[r] = {"prelim_column": PRELIM_RATIO[r], "max_abs_diff_all_docs": float(np.nanmax(dd)),
                 "T_equals_n_scored_all_docs": bool(np.all(D[f"T_{r}"][idx_scored] == D["n_treated"][idx_scored]))}
    sanity["ratio_columns_reproduce_prelim"] = rr
    print(f"[sanity a] {json.dumps(loss_diff)}", flush=True)
    print(f"[sanity b] {json.dumps(tok)}", flush=True)

    # ---------------------------------------------------------- bootstrap draws (shared)
    boots = {p: rng.integers(0, len(splits[p]["holdout"]), size=(args.n_boot, len(splits[p]["holdout"])))
             for p in pops}

    plain = {r: D[f"ratio_{r}"] for r in REFS}
    plain["control_sibling"] = 0.5 * (D["ratio_control"] + D["ratio_sibling"])

    results = {}
    csv_cols = {}
    for p in pops:
        dev, hold = splits[p]["dev"], splits[p]["holdout"]
        R = {"plain_ratio": {}, "rmia": {}, "zratio": {}}
        # sanity (c): plain-ratio holdout AUC, our pipeline vs prelim table
        c_chk = {}
        for r in REFS:
            col = np.array([fnum(pre[d][PRELIM_RATIO[r]]) if d in pre else np.nan for d in doc_ids])
            a_ours = auc(plain[r][[i for i, _ in hold]], plain[r][[h for _, h in hold]])
            a_tab = auc(col[[i for i, _ in hold]], col[[h for _, h in hold]])
            c_chk[r] = {"auc_ours": a_ours, "auc_prelim_table": a_tab, "abs_diff": abs(a_ours - a_tab)}
        sanity.setdefault("c_plain_ratio_holdout_auc_vs_prelim", {})[p] = c_chk

        for r, s in plain.items():
            R["plain_ratio"][r] = evaluate(s, hold, cohort, boots[p], dev_pairs=dev)
            nb = null_behaviour(s, hold, cohort)
            R["plain_ratio"][r]["null_behaviour"] = {k: v for k, v in nb.items() if not k.startswith("share_")}
            print(f"[{p}] plain {r:16s} AUC {R['plain_ratio'][r]['auc']:.4f} "
                  f"TPR1 {100 * R['plain_ratio'][r]['tpr1']:.2f}% strat {100 * R['plain_ratio'][r]['tpr1_strat']:.2f}%", flush=True)

        # ---- RMIA
        Zn = len(dev)
        for refset in REFSETS:
            R["rmia"][refset] = {}
            for kind in ("gm", "seq"):
                t0 = time.time()
                grid_auc = np.full((len(A_GRID), len(LOG_GAMMA_GRID)), np.nan)
                grid_tpr = np.full_like(grid_auc, np.nan)
                Pd = [i for i, _ in dev]; Nd = [h for _, h in dev]
                for ia, a in enumerate(A_GRID):
                    for ig, lg in enumerate(LOG_GAMMA_GRID):
                        sc, _ = rmia_population(D, refset, kind, a, math.exp(lg), dev, hold, n_docs)
                        grid_auc[ia, ig] = auc(sc[Pd], sc[Nd])
                        grid_tpr[ia, ig] = tpr_at(sc[Pd], sc[Nd])[0]
                order = sorted(((-grid_auc[ia, ig], -grid_tpr[ia, ig], ia, ig)
                                for ia in range(len(A_GRID)) for ig in range(len(LOG_GAMMA_GRID))))
                _, _, ia, ig = order[0]
                sel = {"a": A_GRID[ia], "gamma": math.exp(LOG_GAMMA_GRID[ig]), "log_gamma": LOG_GAMMA_GRID[ig]}
                block = {"selected_hyperparameters": sel,
                         "dev_objective": {"criterion": "max dev AUC, tie-break dev pooled TPR@1%FPR",
                                           "dev_auc": float(grid_auc[ia, ig]), "dev_tpr1": float(grid_tpr[ia, ig])},
                         "grid": {"a": A_GRID, "log_gamma": LOG_GAMMA_GRID,
                                  "dev_auc": np.round(grid_auc, 6).tolist(),
                                  "dev_tpr1": np.round(grid_tpr, 6).tolist()},
                         "population_Z": {"definition": f"held-out partners of the {p} DEV-split pairs; "
                                          "dev docs scored leave-one-pair-out", "size": Zn}}
                for tag, hp in (("selected", sel), ("default", HP_DEFAULT), ("a1_gamma1", HP_RATIO_POINT)):
                    sc, lr = rmia_population(D, refset, kind, hp["a"], hp["gamma"], dev, hold, n_docs)
                    ev = evaluate(sc, hold, cohort, boots[p], base=plain[refset], dev_pairs=dev)
                    # monotonicity of the holdout score in log R(x) (fixed Z, fixed gamma)
                    Xh = np.array([d for q in hold for d in q])
                    o = np.argsort(lr[Xh], kind="mergesort")
                    ev["monotone_in_logratio_holdout"] = bool(np.all(np.diff(sc[Xh][o]) >= 0))
                    ev["n_distinct_scores_holdout"] = int(np.unique(sc[Xh]).size)
                    ev["auc_of_logratio_holdout"] = auc(lr[[i for i, _ in hold]], lr[[h for _, h in hold]])
                    ev["tpr1_of_logratio_holdout"] = tpr_at(lr[[i for i, _ in hold]], lr[[h for _, h in hold]])[0]
                    ev["spearman_logratio_vs_plain_holdout"] = float(spearmanr(lr[Xh], plain[refset][Xh])[0])
                    ev["hyperparameters"] = {"a": hp["a"], "gamma": hp["gamma"]}
                    block[tag] = ev
                    if tag in ("selected", "default"):
                        csv_cols[f"rmia_{refset}_{kind}_{tag}_{p}"] = sc
                R["rmia"][refset][kind] = block
                s = block["selected"]
                print(f"[{p}] rmia {refset:16s} {kind:3s} a={sel['a']:<5} g={sel['gamma']:.4g} "
                      f"devAUC {block['dev_objective']['dev_auc']:.4f} | hold AUC {s['auc']:.4f} "
                      f"(plain {s['auc'] - s['delta_auc_vs_plain']:.4f}) TPR1 {100 * s['tpr1']:.2f}% "
                      f"strat {100 * s['tpr1_strat']:.2f}% | default AUC {block['default']['auc']:.4f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)

        # ---- standardized ratio
        for r in REFS:
            R["zratio"][r] = {}
            for v in ("T", "ac"):
                s = D[f"zratio_{r}_{v}"]
                R["zratio"][r][v] = evaluate(s, hold, cohort, boots[p], base=plain[r], dev_pairs=dev)
                R["zratio"][r][v]["null_behaviour"] = null_behaviour(s, hold, cohort)
                e = R["zratio"][r][v]
                print(f"[{p}] zratio {r:8s} {v:2s} AUC {e['auc']:.4f} [{e['auc_ci'][0]:.4f},{e['auc_ci'][1]:.4f}] "
                      f"dAUC {e['delta_auc_vs_plain']:+.4f} TPR1 {100 * e['tpr1']:.2f}% "
                      f"dTPR {100 * e['delta_tpr1_vs_plain']:+.2f} strat {100 * e['tpr1_strat']:.2f}%", flush=True)
        results[p] = R

    # sanity (d): dev vs holdout AUC for every attack
    dvh = {}
    for p, R in results.items():
        dvh[p] = {}
        for r, e in R["plain_ratio"].items():
            dvh[p][f"plain_{r}"] = {"dev": e["auc_dev"], "holdout": e["auc"]}
        for rs, kinds in R["rmia"].items():
            for k, b in kinds.items():
                for tag in ("selected", "default"):
                    dvh[p][f"rmia_{rs}_{k}_{tag}"] = {"dev": b[tag]["auc_dev"], "holdout": b[tag]["auc"]}
        for r, vs in R["zratio"].items():
            for v, e in vs.items():
                dvh[p][f"zratio_{r}_{v}"] = {"dev": e["auc_dev"], "holdout": e["auc"]}
        for k in dvh[p]:
            dvh[p][k]["dev_minus_holdout"] = dvh[p][k]["dev"] - dvh[p][k]["holdout"]
    sanity["d_dev_vs_holdout_auc"] = dvh

    # auxiliary: the autocorrelation estimate vs the earlier token_neff.npz (c=6, maxlag=256, other score dirs)
    if NEFF_PREV.exists():
        Zp = np.load(NEFF_PREV, allow_pickle=True)
        pos = {d: i for i, d in enumerate(Zp["doc_ids"])}
        a_, b_ = [], []
        for i in idx_scored:
            j = pos.get(doc_ids[i])
            if j is not None and np.isfinite(Zp["n_eff"][j]) and np.isfinite(D["n_eff_control"][i]):
                a_.append(D["n_eff_control"][i]); b_.append(Zp["n_eff"][j])
        a_, b_ = np.array(a_), np.array(b_)
        sanity["aux_neff_control_vs_token_neff_npz"] = {
            "note": "token_neff.npz used c=6, maxlag=256, tau clamped >=1, tau=1 for n<32; ours c=5, no lag cap",
            "n": int(a_.size), "spearman": float(spearmanr(a_, b_)[0]),
            "median_ratio_ours_over_prev": float(np.median(a_ / b_))}

    # n_eff / tau summaries
    neff_sum = {}
    for r in REFS:
        T = D[f"T_{r}"][idx_scored]; ne = D[f"n_eff_{r}"][idx_scored]; ta = D[f"tau_{r}"][idx_scored]
        neff_sum[r] = {"median_T": float(np.median(T)), "median_tau": float(np.nanmedian(ta)),
                       "p10_p90_tau": [float(np.nanpercentile(ta, 10)), float(np.nanpercentile(ta, 90))],
                       "median_neff_over_T": float(np.nanmedian(ne / T)),
                       "share_tau_below_1": float(np.nanmean(ta < 1)),
                       "median_window": float(np.median(D[f"win_{r}"][idx_scored]))}

    # ---------------------------------------------------------- outputs
    summary = {
        "meta": {
            "script": "analysis/calibrated_attacks.py",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": {m: _rel(RUN / d) for m, (d, _) in MODELS.items()},
            "paper": "Zarifzadeh, Liu, Shokri, Low-Cost High-Power Membership Inference Attacks, ICML 2024, arXiv:2312.03262",
            "rmia_definition": {
                "score": "Score(x) = #{z in Z: LR(x,z) > gamma}/|Z|, LR(x,z) = [Pr(x|theta)/Pr(x)]/[Pr(z|theta)/Pr(z)] (strict >, Alg. 1 line 19 / official code)",
                "pr_x_given_theta": {"gm": "exp(mean_t logprob_t) over scored tokens (= <model>__loss)",
                                     "seq": "exp(sum_t logprob_t); scales with length"},
                "pr_x_offline": "0.5*((1+a)*Pr_OUT(x) + (1-a)), Pr_OUT = arithmetic mean of Pr(x|theta') over the reference set; same formula for z (paper Sec. 3, official code)",
                "population_Z": "held-out partners of the DEV-split pairs of the evaluated population; holdout docs scored against all of Z, dev docs leave-one-pair-out",
                "selection": "per (population, refset, variant): argmax dev AUC over the (a, gamma) grid, tie-break dev pooled TPR@1%FPR, then grid order",
                "a_grid": A_GRID, "log_gamma_grid": LOG_GAMMA_GRID,
                "defaults": {"default": HP_DEFAULT, "a1_gamma1": HP_RATIO_POINT,
                             "why": "a=0.3 is the official-code default (paper: CIFAR-10/CINIC-10); gamma=2 is the paper's setting; a=1,gamma=1 is the pure reference-ratio point"},
                "monotonicity": "for fixed Z and gamma, Score(x) is a non-decreasing step function of log Pr(x|theta)/Pr(x); RMIA changes AUC/TPR only via ties at resolution 1/|Z|, and differs from the plain ratio only through a<1",
            },
            "zratio_definition": {
                "d_t": "logprob_TREATED(t) - logprob_ref(t) over scored positions (NaN window starts dropped, series treated as contiguous)",
                "z": "mean_t d_t / (sd_t(d_t, ddof=1) / sqrt(n_eff))",
                "T": "n_eff = T",
                "ac": f"n_eff = T/tau clamped to [1,T]; tau = 1 + 2 sum_(k=1..W) rho_k, W = smallest W>=1 with W >= {SOKAL_C}*tau(W)",
            },
            "evaluation": {
                "populations": {"NAT1": "r==1 and cohort in {M,L,W-new,W-gap,S-se}", "ALL": "every analysed pair"},
                "n_pairs": {p: {"total": len(pops[p]), "dev": len(splits[p]["dev"]), "holdout": len(splits[p]["holdout"])} for p in pops},
                "exclusions": {"posttrain_pairs": len(bad_pairs), "posttrain_docs": len(bad_docs),
                               "unscored_docs": [d for d, _ in missing]},
                "reported_split": "holdout", "selection_split": "dev",
                "auc": "rank AUC, ties 1/2", "tpr1": "threshold admitting floor(0.01*n_neg) negatives, strict >",
                "tpr1_strat": "one threshold per cohort, weighted by holdout pairs per cohort",
                "bootstrap": {"draws": args.n_boot, "unit": "holdout pairs", "seed": args.seed,
                              "interval": "percentile 95%", "shared_across_attacks": True},
                "plain_baseline_control_sibling": "0.5*(ratio_control + ratio_sibling)",
            },
            "neff_summary": neff_sum,
        },
        "sanity": sanity,
        "results": results,
    }
    args.out_json.write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"[out] {args.out_json}", flush=True)

    # per-document table
    base_cols = ["doc_id", "cohort", "stratum", "arm", "pair_id", "split", "r"]
    num = {"n_scored": D["n_treated"]}
    for m in MODELS:
        num[f"loss_{m}"] = D[f"loss_{m}"]
    for m in MODELS:
        num[f"sumlp_{m}"] = D[f"sumlp_{m}"]
    for r in REFS:
        num[f"ratio_{r}"] = D[f"ratio_{r}"]
        num[f"sd_d_{r}"] = D[f"sd_d_{r}"]
        num[f"tau_{r}"] = D[f"tau_{r}"]
        num[f"n_eff_{r}"] = D[f"n_eff_{r}"]
        num[f"zratio_{r}_T"] = D[f"zratio_{r}_T"]
        num[f"zratio_{r}_ac"] = D[f"zratio_{r}_ac"]
    num.update(csv_cols)
    names = list(num)
    with gzip.open(args.out_csv, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(base_cols + names)
        for i, r in enumerate(man):
            if not scored[i]:
                continue
            vals = []
            for k in names:
                x = num[k][i]
                if k == "n_scored":
                    vals.append(str(int(x)))
                else:
                    vals.append("" if not np.isfinite(x) else repr(float(x)))
            w.writerow([r[c] for c in base_cols] + vals)
    print(f"[out] {args.out_csv} ({int(scored.sum())} rows, {len(base_cols) + len(names)} columns)", flush=True)
    print(f"[done] {time.time() - t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
