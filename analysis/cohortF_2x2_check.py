#!/usr/bin/env python3
"""Independent re-derivation and audit of the cohort-F form-matched (2x2) analysis.

Re-derives every number in analysis/cohortF_2x2_stats.json (written by an
exploratory figure script, fig_cohortF_2x2.py, that is not part of this
repository; the file is released with the analysis outputs) from the
per-document and per-token scores, and adds the checks that the stored analysis
does not make. If cohortF_2x2_stats.json is absent, step 10 (stored vs
re-derived) is skipped and recorded as skipped; nothing else depends on it.

Design. Cohort F has 278 matched pairs of post-cutoff arXiv papers. In each pair
a coin chose paper A (injected, in its LaTeX-derived rendering) and paper B
(held out). Every paper exists in two renderings produced by the bundle's
converters: LaTeX-derived (redpajama_latex) and PDF-derived (pes2o_style_pdf).

    cell        document id in the score files          source of scores
    LaTeX(A)    <A>        (the injected token array)   analysis table (pf.load)
    PDF(B)      <B>        (the original holdout arm)   analysis table (pf.load)
    PDF(A)      Fpdf:<A>   (never injected)             scores_cohortF_2x2/F2x2-*
    LaTeX(B)    Flatex:<B> (never injected)             scores_cohortF_2x2/F2x2-*

delta(x) = mean_t log p_treated(x_t | x_<t) - mean_t log p_seed42(x_t | x_<t)
(nats/token; identical to the analysis table's ratio_oracle for the two
original cells).

    tau_native_i   = delta(LaTeX A_i) - delta(LaTeX B_i)
    tau_transfer_i = delta(PDF A_i)   - delta(PDF B_i)
    tau_orig_i     = delta(LaTeX A_i) - delta(PDF B_i)     (the confounded original)
    transfer coefficient rho = mean(tau_transfer) / mean(tau_native)

Inputs (under the data root, see olmo_twin/paths.py): analysis/ (attack table
through paper_figures.load(), cohortF_data_exclusions.json from
analysis/audit_cohortF_data.py, optional cohortF_2x2_stats.json); bundle/
(manifest.csv.gz, manifest_v1a_prepairfix.csv.gz, scoring_manifest_extra.csv.gz,
texts_index.csv.gz, stream_plan/placement.jsonl.gz, tokens/F/ and
logprobs_extra_stage1/scores.csv.gz); run/scores/, run/scores_ext/,
run/scores_cohortF_2x2/ (per-document and per-token scores, with token ids) and
run/cohortF_2x2/tokens/ (written by scoring/cohortF_2x2_tokenise.py).
Output: analysis/cohortF_2x2_check.json.

Run:  python3 -u analysis/cohortF_2x2_check.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
paths.add_code_dirs("figures")
import paper_figures as pf  # noqa: E402
import injection_v2_plotstyle as ps  # noqa: E402

ANA = paths.ANALYSIS
OUT_JSON = ANA / "cohortF_2x2_check.json"
STORED = ANA / "cohortF_2x2_stats.json"
EXCL = ANA / "cohortF_data_exclusions.json"
BUNDLE = paths.BUNDLE
REPARSE = paths.COHORTF_REPARSE
RUN = paths.RUN
NEWS = {"injected": RUN / "scores_cohortF_2x2/F2x2-injected",
        "seed42": RUN / "scores_cohortF_2x2/F2x2-seed42",
        "stage1": RUN / "scores_cohortF_2x2/F2x2-stage1"}
EXT = {"injected": RUN / "scores_ext/ext-injected",           # analysis-table source
       "seed42": RUN / "scores_ext/ext-seed42",
       "stage1": RUN / "scores_ext/ext-stage1-pretreatment"}
ORIG = {"injected": RUN / "scores/injected-step23852",         # fig_cohortF_2x2.py source
        "seed42": RUN / "scores/official-seed42-step23852"}
EXTRA_STAGE1 = BUNDLE / "logprobs_extra_stage1"               # pre-treatment LaTeX(B) scores
NEW_TOKENS = RUN / "cohortF_2x2/tokens"

SEED = 20261013          # every bootstrap draw in this script comes from this seed
NBOOT = 4000             # pair-bootstrap replicates
NGRAMS = (4, 8, 16, 32)  # token n-gram orders for the shared-content analysis
NPRIMARY = 8


# ----------------------------------------------------------------- helpers
def read_scores(path: Path) -> dict:
    csv.field_size_limit(sys.maxsize)
    with gzip.open(path, "rt", newline="") as fh:
        return {r["doc_id"]: r for r in csv.DictReader(fh)}


def mlp(d: dict, key: str) -> float:
    r = d.get(key)
    return float(r["mean_logprob"]) if r and r.get("status") == "ok" else np.nan


def npz(dir_: Path, doc_id: str):
    p = dir_ / "tokens" / f"{doc_id.replace(':', '_').replace('/', '_')}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["token_ids"].astype(np.int64), z["logprobs"].astype(np.float64)


def fast_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC with ties at 1/2; identical definition to ps.auc."""
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def pct(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def paired_summary(diff: np.ndarray, a: np.ndarray, b: np.ndarray, IDX: np.ndarray) -> dict:
    """diff = a - b per pair (member minus matched non-member)."""
    n = diff.size
    t = stats.ttest_1samp(diff, 0.0, alternative="greater")
    bm = diff[IDX].mean(axis=1)
    bauc = np.array([fast_auc(a[k], b[k]) for k in IDX])
    auc = ps.auc(a, b)
    assert abs(auc - fast_auc(a, b)) < 1e-12
    return {"n_pairs": int(n), "mean": float(diff.mean()), "ci_pair_boot": pct(bm),
            "se": float(diff.std(ddof=1) / math.sqrt(n)), "sd": float(diff.std(ddof=1)),
            "d_z": float(diff.mean() / diff.std(ddof=1)),
            "t": float(t.statistic), "p_one_sided": float(t.pvalue),
            "frac_pos": float((diff > 0).mean()), "n_pos": int((diff > 0).sum()),
            "auc": float(auc), "auc_ci_pair_boot": pct(bauc)}


def ngram_hash(x: np.ndarray, n: int) -> np.ndarray:
    """64-bit polynomial hash of every length-n window of x (index = window start)."""
    L = x.size - n + 1
    if L <= 0:
        return np.zeros(0, np.uint64)
    P = np.uint64(1000003)
    h = np.zeros(L, np.uint64)
    xu = (x.astype(np.uint64) + np.uint64(1))
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * P + xu[j:j + L]
    return h


def coverage(probe: np.ndarray, ref: np.ndarray, n: int) -> np.ndarray:
    """cov[t] = True if the n-gram probe[t-n+1..t] occurs anywhere in ref.
    Positions t < n-1 are returned as False and must be masked by the caller."""
    cov = np.zeros(probe.size, bool)
    hp, hr = ngram_hash(probe, n), ngram_hash(ref, n)
    if hp.size and hr.size:
        cov[n - 1:] = np.isin(hp, hr)
    return cov


# ------------------------------------------------------------------- main
def main() -> int:
    out: dict = {"generated_by": "analysis/cohortF_2x2_check.py",
                 "seed": SEED, "n_boot": NBOOT,
                 "rng": "numpy.random.default_rng(SEED); one (NBOOT x n_pairs) matrix of "
                        "pair indices drawn once and reused for every pair-bootstrap "
                        "statistic so that ratios and decompositions are resampled jointly"}

    # ================= 1. accounting: which pairs, which documents
    csv.field_size_limit(sys.maxsize)
    man = [r for r in csv.DictReader(gzip.open(BUNDLE / "manifest.csv.gz", "rt"))
           if r["cohort"] == "F"]
    byp = defaultdict(dict)
    for r in man:
        byp[r["pair_id"]][r["arm"]] = r
    design_pairs = sorted(p for p, v in byp.items() if "inject" in v and "holdout" in v)
    forms = Counter((r["arm"], r["inject_text_form"]) for r in man)

    cols, rows, arm, pairs = pf.load()
    coh, docid, pid = cols["cohort"], cols["doc_id"], cols["pair_id"]
    tab = {pid[i]: (i, h) for i, h in pairs if coh[i] == "F"}
    f_rows = [k for k in range(len(coh)) if coh[k] == "F"]
    tab_docs = {docid[k] for k in f_rows}

    excl = json.loads(EXCL.read_text())
    excl_pairs = set(excl["exclude_pair_ids"])

    new = {k: read_scores(v / "scores.csv.gz") for k, v in NEWS.items()}
    orig = {k: read_scores(v / "scores.csv.gz") for k, v in ORIG.items()}

    acct = []
    for p in design_pairs:
        A, B = byp[p]["inject"]["doc_id"], byp[p]["holdout"]["doc_id"]
        st = {"LaTeX(A)": orig["injected"].get(A, {}).get("status", "absent"),
              "PDF(B)": orig["injected"].get(B, {}).get("status", "absent"),
              "PDF(A)": new["injected"].get(f"Fpdf:{A}", {}).get("status", "absent"),
              "LaTeX(B)": new["injected"].get(f"Flatex:{B}", {}).get("status", "absent")}
        for m in ("seed42",):
            for cell, key, src in (("LaTeX(A)", A, orig), ("PDF(B)", B, orig),
                                   ("PDF(A)", f"Fpdf:{A}", new), ("LaTeX(B)", f"Flatex:{B}", new)):
                s2 = src[m].get(key, {}).get("status", "absent")
                if s2 != "ok":
                    st[cell] = st[cell] if st[cell] != "ok" else f"{m}:{s2}"
        complete = all(v == "ok" for v in st.values())
        acct.append({"pair_id": p, "A": A, "B": B, "in_table": p in tab,
                     "audit_excluded": p in excl_pairs, "four_cells_scored": complete,
                     "status": st})
    n_design = len(acct)
    not_scored = [a for a in acct if not a["four_cells_scored"]]
    excluded = [a for a in acct if a["audit_excluded"]]
    analysed = [a for a in acct if a["four_cells_scored"] and not a["audit_excluded"]]
    out["accounting"] = {
        "design_pairs_in_manifest": n_design,
        "manifest_forms": {f"{k[0]}:{k[1]}": v for k, v in forms.items()},
        "F_pairs_in_analysis_table_after_pf_load": len(tab),
        "F_rows_in_analysis_table": len(f_rows),
        "pairs_missing_a_scored_cell": [{"pair_id": a["pair_id"], "A": a["A"], "B": a["B"],
                                         "status": {k: v for k, v in a["status"].items() if v != "ok"},
                                         "also_audit_excluded": a["audit_excluded"]}
                                        for a in not_scored],
        "pairs_excluded_by_data_audit": [{"pair_id": a["pair_id"], "A": a["A"], "B": a["B"],
                                          "four_cells_scored": a["four_cells_scored"],
                                          "reasons": [excl["detail"][d]["reasons"] for d in (a["A"], a["B"])
                                                      if d in excl["detail"]]}
                                         for a in excluded],
        "pairs_analysed": len(analysed),
        "identity": f"{n_design} design pairs - {len(not_scored)} with a cell over the scorer's "
                    f"65,536-token cap - {len(excluded)} excluded by the data audit "
                    f"({len([a for a in excluded if not a['four_cells_scored']])} overlap) = {len(analysed)}",
    }
    assert len(analysed) == n_design - len(not_scored) - len([a for a in excluded if a["four_cells_scored"]])

    # pairing covariate actually used by build_injection_manifest.make_pairs for F:
    # attach_pair_score tries "F:<id>:inj" first; in the pre-pair-fix build that key
    # held the LaTeX text for that build's injected arm and the PDF text for its
    # holdout arm, so the covariate's rendering was set by a superseded coin flip.
    v1 = {r["doc_id"]: r["arm"] for r in csv.DictReader(gzip.open(BUNDLE / "manifest_v1a_prepairfix.csv.gz", "rt"))
          if r["cohort"] == "F"}
    exman0 = {r["doc_id"]: r["raw_text_path"] for r in csv.DictReader(gzip.open(BUNDLE / "scoring_manifest_extra.csv.gz", "rt"))}
    cov_form = Counter()
    tab22 = Counter()
    for r in man:
        k = f"F:{r['doc_id']}:inj"
        cov_form[(r["arm"], "latex" if "/raw/latex/" in exman0.get(k, "") else "pdf" if k in exman0 else "none")] += 1
        tab22[(v1.get(r["doc_id"]), r["arm"])] += 1
    ct = np.array([[tab22[("inject", "inject")], tab22[("inject", "holdout")]],
                   [tab22[("holdout", "inject")], tab22[("holdout", "holdout")]]])
    chi = stats.chi2_contingency(ct, correction=False)
    out["randomization_audit"] = {
        "pairing_covariate_rendering_by_final_arm": {f"{a}:{f}": v for (a, f), v in cov_form.items()},
        "prepairfix_arm_vs_final_arm": {"rows_prepairfix_inject_holdout_cols_final_inject_holdout": ct.tolist(),
                                        "chi2": float(chi[0]), "p": float(chi[1])},
        "reading": "pairs were formed on a pre-treatment score computed on the LaTeX rendering "
                   "for half of the papers and on the PDF rendering for the other half; the "
                   "coin was flipped after pairing and is independent of the superseded arm, "
                   "so arm assignment remains randomized; the pairs are matched less tightly "
                   "than intended, which costs precision, not unbiasedness",
    }

    # ================= 2. injection integrity: what exactly was trained on
    plc = [json.loads(l) for l in gzip.open(BUNDLE / "stream_plan/placement.jsonl.gz", "rt")]
    all_sha = {d["tokens_sha256"] for d in plc}
    all_placed_ids = Counter(d["doc_id"] for d in plc)
    fpl = [d for d in plc if d["cohort"] == "F"]
    A_ids = {byp[p]["inject"]["doc_id"] for p in design_pairs}
    B_ids = {byp[p]["holdout"]["doc_id"] for p in design_pairs}
    sha_ok = 0
    for d in fpl:
        arr = np.load(BUNDLE / d["tokens_relpath"])
        sha_ok += hashlib.sha256(arr.tobytes()).hexdigest() == d["tokens_sha256"]
    pdfA_sha_hits, latexB_sha_hits, pdfA_vs_inj_identical = 0, 0, 0
    for p in design_pairs:
        A, B = byp[p]["inject"]["doc_id"], byp[p]["holdout"]["doc_id"]
        ta = np.load(NEW_TOKENS / "pdf_of_injected" / f"{A}.npy")
        tb = np.load(NEW_TOKENS / "latex_of_holdout" / f"{B}.npy")
        pdfA_sha_hits += hashlib.sha256(ta.tobytes()).hexdigest() in all_sha
        latexB_sha_hits += hashlib.sha256(tb.tobytes()).hexdigest() in all_sha
        inj = np.load(BUNDLE / "tokens/F" / f"{A}.npy")
        pdfA_vs_inj_identical += (inj.shape == ta.shape and bool((inj == ta).all()))
    out["injection_integrity"] = {
        "F_placement_entries": len(fpl),
        "F_placed_doc_ids_equal_inject_arm": {d["doc_id"] for d in fpl} == A_ids,
        "F_placed_copies_per_doc": dict(Counter(Counter(d["doc_id"] for d in fpl).values())),
        "F_placed_token_arrays_sha256_match_bundle_LaTeX_arrays": f"{sha_ok}/{len(fpl)}",
        "F_placed_tokens_relpath_all_tokens_F_A": all(d["tokens_relpath"] == f"tokens/F/{d['doc_id']}.npy" for d in fpl),
        "inject_text_form_of_A": dict(Counter(byp[p]["inject"]["inject_text_form"] for p in design_pairs)),
        "PDF(A)_token_arrays_whose_sha256_is_in_the_placement": pdfA_sha_hits,
        "LaTeX(B)_token_arrays_whose_sha256_is_in_the_placement": latexB_sha_hits,
        "PDF(A)_token_array_identical_to_injected_LaTeX(A)_array": pdfA_vs_inj_identical,
        "B_doc_ids_placed_under_any_cohort": len(B_ids & set(all_placed_ids)),
        "A_doc_ids_placed_under_another_cohort": len({d["doc_id"] for d in plc if d["cohort"] != "F"} & A_ids),
        "bundle_texts_all_form_of_A": dict(Counter(r["form"] for r in csv.DictReader(gzip.open(BUNDLE / "texts_index.csv.gz", "rt")) if r["doc_id"] in A_ids and r["cohort"] == "F")),
        "note": "the placement plan is the trainer's input; it lists, for every injected copy, "
                "the token file and its sha256. The PDF rendering of an injected paper appears "
                "nowhere in it.",
    }

    # ================= 3. per-pair cells
    sel = [(a["pair_id"], a["A"], a["B"]) for a in analysed]
    n = len(sel)
    g_tab = lambda key, which: np.array([cols[key][tab[p][0 if which == "A" else 1]] for p, _, _ in sel], float)  # noqa: E731
    # original cells via the paper's loader
    d_LA = g_tab("ratio_oracle", "A")
    d_PB = g_tab("ratio_oracle", "B")
    c_LA = g_tab("seed42__loss", "A")
    c_PB = g_tab("seed42__loss", "B")
    s_LA = g_tab("stage1__loss", "A")
    s_PB = g_tab("stage1__loss", "B")
    nt_LA = g_tab("seed42__n_scored", "A")
    nt_PB = g_tab("seed42__n_scored", "B")
    # new cells
    gi = lambda m, pre, ids: np.array([mlp(new[m], f"{pre}:{x}") for x in ids])  # noqa: E731
    As = [a for _, a, _ in sel]
    Bs = [b for _, _, b in sel]
    d_PA = gi("injected", "Fpdf", As) - gi("seed42", "Fpdf", As)
    d_LB = gi("injected", "Flatex", Bs) - gi("seed42", "Flatex", Bs)
    c_PA, c_LB = gi("seed42", "Fpdf", As), gi("seed42", "Flatex", Bs)
    s_PA, s_LB = gi("stage1", "Fpdf", As), gi("stage1", "Flatex", Bs)
    nt_PA = np.array([float(new["seed42"][f"Fpdf:{x}"]["n_tokens"]) for x in As])
    nt_LB = np.array([float(new["seed42"][f"Flatex:{x}"]["n_tokens"]) for x in Bs])
    for arr in (d_LA, d_PB, d_PA, d_LB, c_LA, c_PB, c_PA, c_LB, s_LA, s_PB, s_PA, s_LB):
        assert np.isfinite(arr).all()

    # the stored script read the two original cells from scores/ rather than the table
    d_LA_o = np.array([mlp(orig["injected"], a) - mlp(orig["seed42"], a) for a in As])
    d_PB_o = np.array([mlp(orig["injected"], b) - mlp(orig["seed42"], b) for b in Bs])
    out["source_agreement"] = {
        "max_abs_diff_delta_LaTeX(A)_table_vs_scores_dir": float(np.max(np.abs(d_LA - d_LA_o))),
        "max_abs_diff_delta_PDF(B)_table_vs_scores_dir": float(np.max(np.abs(d_PB - d_PB_o))),
        "note": "the analysis table is built from scores_ext/ (per-token npz, float64 means); "
                "fig_cohortF_2x2.py read scores/ (6-decimal CSV means). Same checkpoints.",
    }

    rng = np.random.default_rng(SEED)
    IDX = rng.integers(0, n, size=(NBOOT, n))

    tau_nat = d_LA - d_LB
    tau_tr = d_PA - d_PB
    tau_orig = d_LA - d_PB
    est = {"native (LaTeX A vs LaTeX B)": paired_summary(tau_nat, d_LA, d_LB, IDX),
           "transfer (PDF A vs PDF B)": paired_summary(tau_tr, d_PA, d_PB, IDX),
           "original, confounded (LaTeX A vs PDF B)": paired_summary(tau_orig, d_LA, d_PB, IDX)}
    # TPR at 1% FPR with counts (resolution floor: n_neg = n)
    for k, (a, b) in zip(est, [(d_LA, d_LB), (d_PA, d_PB), (d_LA, d_PB)]):
        tpr, kk, nn = pf.tpr_at(a, b, 0.01)
        est[k]["tpr_at_1pct_fpr"] = {"tpr": tpr, "k_neg_admitted": kk, "n_neg": nn, "n_pos": int(a.size),
                                     "n_pos_above": int(round(tpr * a.size))}
    # replicate the stored script's own interval methods exactly
    def boot_stored(x, nb=4000, seed=0):
        r = np.random.default_rng(seed)
        m = np.array([r.choice(x, x.size, replace=True).mean() for _ in range(nb)])
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    for k, tau, a, b in [("native (LaTeX A vs LaTeX B)", tau_nat, d_LA, d_LB),
                         ("transfer (PDF A vs PDF B)", tau_tr, d_PA, d_PB)]:
        est[k]["stored_method_ci"] = boot_stored(tau)
        est[k]["stored_method_auc_ci_unpaired"] = list(ps.auc_ci(a, b, n=600))
    out["estimators"] = est

    # ================= 4. transfer coefficient
    rho = tau_tr.mean() / tau_nat.mean()
    rb = tau_tr[IDX].mean(1) / tau_nat[IDX].mean(1)
    # Fieller interval for a ratio of two correlated means
    tq = stats.t.ppf(0.975, n - 1)
    S = np.cov(np.vstack([tau_tr, tau_nat]), ddof=1) / n
    mx, my = tau_tr.mean(), tau_nat.mean()
    qa = my ** 2 - tq ** 2 * S[1, 1]
    qb = -2 * (mx * my - tq ** 2 * S[0, 1])
    qc = mx ** 2 - tq ** 2 * S[0, 0]
    disc = qb ** 2 - 4 * qa * qc
    fieller = sorted([(-qb - math.sqrt(disc)) / (2 * qa), (-qb + math.sqrt(disc)) / (2 * qa)]) if qa > 0 and disc > 0 else None
    r7 = np.random.default_rng(7)
    rr7 = []
    for _ in range(4000):
        k = r7.integers(0, n, n)
        rr7.append(tau_tr[k].mean() / tau_nat[k].mean())
    gap = tau_nat - tau_tr
    out["transfer_coefficient"] = {
        "estimator": "ratio of means, mean_i(tau_transfer_i) / mean_i(tau_native_i), over the same pairs",
        "value": float(rho), "ci_pair_boot": pct(rb), "ci_fieller_t": fieller,
        "stored_method_ci_seed7": pct(rr7),
        "P_boot_rho_le_0": float((rb <= 0).mean()), "P_boot_rho_ge_1": float((rb >= 1).mean()),
        "native_minus_transfer": {"mean": float(gap.mean()), "ci_pair_boot": pct(gap[IDX].mean(1)),
                                  "t": float(stats.ttest_1samp(gap, 0).statistic),
                                  "p_two_sided": float(stats.ttest_1samp(gap, 0).pvalue)},
    }

    # ================= 5. the stored decomposition, re-derived and re-read
    def decomp(ix=None):
        sl = slice(None) if ix is None else ix
        m_loss = (d_PA[sl] - d_LA[sl]).mean()
        f_rise = (d_PB[sl] - d_LB[sl]).mean()
        drop = tau_tr[sl].mean() - tau_nat[sl].mean()
        return m_loss, f_rise, drop, m_loss / drop, -f_rise / drop
    dm, dfl, dd, sm, sf = decomp()
    bd = np.array([decomp(k) for k in IDX])
    out["decomposition"] = {
        "identity": "mean(tau_transfer) - mean(tau_native) = mean(delta PDF A - delta LaTeX A) "
                    "- mean(delta PDF B - delta LaTeX B)",
        "member_raw_change_PDF_minus_LaTeX": {"value": float(dm), "ci": pct(bd[:, 0])},
        "nonmember_floor_change_PDF_minus_LaTeX": {"value": float(dfl), "ci": pct(bd[:, 1])},
        "tau_change_transfer_minus_native": {"value": float(dd), "ci": pct(bd[:, 2])},
        "share_from_member": {"value": float(sm), "ci": pct(bd[:, 3])},
        "share_from_floor": {"value": float(sf), "ci": pct(bd[:, 4])},
        "arm_means": {"delta_LaTeX_A": float(d_LA.mean()), "delta_LaTeX_B": float(d_LB.mean()),
                      "delta_PDF_A": float(d_PA.mean()), "delta_PDF_B": float(d_PB.mean())},
        "arm_mean_cis": {k: pct(v[IDX].mean(1)) for k, v in
                         [("delta_LaTeX_A", d_LA), ("delta_LaTeX_B", d_LB),
                          ("delta_PDF_A", d_PA), ("delta_PDF_B", d_PB)]},
        "reading": "This is an accounting identity of four arm means, not a causal split. "
                   "Under the additive model that the paired estimator itself assumes "
                   "(delta = form-specific non-member floor + membership component), the "
                   "membership component is tau: it falls from mean(tau_native) to "
                   "mean(tau_transfer). The member's raw delta is nearly unchanged only "
                   "because the PDF-form floor is higher by about the same amount. The "
                   "'shares' therefore do not show that the trace survives re-rendering. "
                   "They also decompose a difference of mean effects, not a loss of AUC.",
    }

    # ================= 6. placebos and balance checks
    def placebo(a, b, label, what):
        diff = a - b
        return {"compares": what, "auc": float(ps.auc(a, b)),
                "auc_ci_pair_boot": pct([fast_auc(a[k], b[k]) for k in IDX]),
                "auc_ci_stored_method_unpaired": list(ps.auc_ci(a, b, n=600)),
                "mean_diff": float(diff.mean()), "mean_diff_ci": pct(diff[IDX].mean(1)),
                "p_two_sided": float(stats.ttest_1samp(diff, 0).pvalue), "n_pairs": int(diff.size)}
    pl = {
        "P1_native_control_level": placebo(c_LA, c_LB, "native",
            "seed-42 control mean log-prob, LaTeX(A) vs LaTeX(B); stored 'placebo'"),
        "P1_transfer_control_level": placebo(c_PA, c_PB, "transfer",
            "seed-42 control mean log-prob, PDF(A) vs PDF(B); stored 'placebo'"),
        "P1_original_control_level": placebo(c_LA, c_PB, "orig",
            "seed-42 control mean log-prob, LaTeX(A) vs PDF(B); the confounded original"),
        "P2_native_drift": placebo(c_LA - s_LA, c_LB - s_LB, "native",
            "(seed-42 minus stage-1) mean log-prob, LaTeX(A) vs LaTeX(B): a difference score "
            "built like delta from two models neither of which saw A"),
        "P2_transfer_drift": placebo(c_PA - s_PA, c_PB - s_PB, "transfer",
            "(seed-42 minus stage-1) mean log-prob, PDF(A) vs PDF(B)"),
        "P2_original_drift": placebo(c_LA - s_LA, c_PB - s_PB, "orig",
            "(seed-42 minus stage-1) mean log-prob, LaTeX(A) vs PDF(B)"),
        "P3_native_length": placebo(np.log(nt_LA), np.log(nt_LB), "native",
            "log scored tokens, LaTeX(A) vs LaTeX(B)"),
        "P3_transfer_length": placebo(np.log(nt_PA), np.log(nt_PB), "transfer",
            "log scored tokens, PDF(A) vs PDF(B)"),
    }
    # does document length explain the per-pair effects? (descriptive)
    for k, tau, la, lb in [("native", tau_nat, nt_LA, nt_LB), ("transfer", tau_tr, nt_PA, nt_PB)]:
        X = np.column_stack([np.ones(n), np.log(la) - np.log(lb)])
        beta, *_ = np.linalg.lstsq(X, tau, rcond=None)
        bb = np.array([np.linalg.lstsq(X[k2], tau[k2], rcond=None)[0] for k2 in IDX[:1000]])
        pl[f"length_adjusted_{k}"] = {"model": "tau_i = b0 + b1*(log len A_i - log len B_i)",
                                      "b0_adjusted_effect": float(beta[0]), "b0_ci": pct(bb[:, 0]),
                                      "b1": float(beta[1]), "b1_ci": pct(bb[:, 1]),
                                      "boot_draws": 1000}
    out["placebos"] = pl

    # ================= 7. cross-job scoring reproducibility
    ex1 = read_scores(EXTRA_STAGE1 / "scores.csv.gz")
    # the bundle's pre-treatment pass scored each F paper's LaTeX rendering under
    # "F:<id>:latex" or, for papers that were the injected arm of the pre-pair-fix
    # build, under "F:<id>:inj" with a LaTeX source path
    exman = {r["doc_id"]: r["raw_text_path"] for r in csv.DictReader(gzip.open(BUNDLE / "scoring_manifest_extra.csv.gz", "rt"))}
    def latex_key(b):
        for k in (f"F:{b}:latex", f"F:{b}:inj"):
            if k in exman and "/raw/latex/" in exman[k] and ex1.get(k, {}).get("status") == "ok":
                return k
        return None
    dif_lb = np.array([mlp(new["stage1"], f"Flatex:{b}") - mlp(ex1, latex_key(b)) if latex_key(b) else np.nan for b in Bs])
    man_bpb = {r["doc_id"]: float(r["bpb_pre"]) for r in man}
    dif_pa_bpb = np.array([float(new["stage1"][f"Fpdf:{a}"]["bpb"]) - man_bpb[a] for a in As])
    ext_i, ext_c = read_scores(EXT["injected"] / "scores.csv.gz"), read_scores(EXT["seed42"] / "scores.csv.gz")
    dif_job = np.array([(mlp(ext_i, a) - mlp(ext_c, a)) - (mlp(orig["injected"], a) - mlp(orig["seed42"], a)) for a in As])
    out["scoring_reproducibility"] = {
        "LaTeX(B)_stage1_mean_logprob_F2x2_job_minus_bundle_pretreatment_job": {
            "n": int(np.isfinite(dif_lb).sum()), "mean": float(np.nanmean(dif_lb)),
            "max_abs": float(np.nanmax(np.abs(dif_lb)))},
        "PDF(A)_stage1_bpb_F2x2_job_minus_manifest_bpb_pre": {
            "n": int(np.isfinite(dif_pa_bpb).sum()), "mean": float(np.nanmean(dif_pa_bpb)),
            "max_abs": float(np.nanmax(np.abs(dif_pa_bpb)))},
        "LaTeX(A)_delta_scores_ext_job_minus_scores_job": {
            "n": int(np.isfinite(dif_job).sum()), "mean": float(np.nanmean(dif_job)),
            "max_abs": float(np.nanmax(np.abs(dif_job)))},
        "note": "the two original cells and the two new cells were scored in different jobs; "
                "these differences bound any job-level offset, to be compared with tau",
    }

    # ================= 8. shared content between the two renderings, and where the signal sits
    per = defaultdict(list)
    for (p, A, B) in sel:
        LA_i, LA_c = npz(EXT["injected"], A), npz(EXT["seed42"], A)
        PB_i, PB_c = npz(EXT["injected"], B), npz(EXT["seed42"], B)
        PA_i, PA_c = npz(NEWS["injected"], f"Fpdf:{A}"), npz(NEWS["seed42"], f"Fpdf:{A}")
        LB_i, LB_c = npz(NEWS["injected"], f"Flatex:{B}"), npz(NEWS["seed42"], f"Flatex:{B}")
        cells = {"LA": (LA_i, LA_c), "PB": (PB_i, PB_c), "PA": (PA_i, PA_c), "LB": (LB_i, LB_c)}
        tok, dl = {}, {}
        for c, (zi, zc) in cells.items():
            assert zi is not None and zc is not None, (p, c)
            assert zi[0].shape == zc[0].shape and (zi[0] == zc[0]).all()
            tok[c] = zi[0]
            dl[c] = zi[1] - zc[1]
        inj = np.load(BUNDLE / "tokens/F" / f"{A}.npy").astype(np.int64)
        assert inj.shape == tok["LA"].shape and (inj == tok["LA"]).all()
        per["npz_mean_delta_LA"].append(np.nanmean(dl["LA"]))
        per["npz_mean_delta_PA"].append(np.nanmean(dl["PA"]))
        # probe -> reference: PDF A against the injected LaTeX A; PDF B against LaTeX B;
        # LaTeX A against PDF A; LaTeX B against PDF B; cross-paper baseline PDF A vs LaTeX B
        for nn in NGRAMS:
            for probe, ref in (("PA", "LA"), ("PB", "LB"), ("LA", "PA"), ("LB", "PB")):
                cov = coverage(tok[probe], tok[ref], nn)
                ok = np.isfinite(dl[probe]); ok[:nn - 1] = False
                per[f"cont_{probe}_in_{ref}_{nn}"].append(cov[ok].mean())
                per[f"dcov_{probe}_{nn}"].append(dl[probe][ok & cov].mean() if (ok & cov).any() else np.nan)
                per[f"dunc_{probe}_{nn}"].append(dl[probe][ok & ~cov].mean() if (ok & ~cov).any() else np.nan)
            cov = coverage(tok["PA"], tok["LB"], nn)
            ok = np.isfinite(dl["PA"]); ok[:nn - 1] = False
            per[f"cont_PA_in_LB_{nn}"].append(cov[ok].mean())
    per = {k: np.array(v, float) for k, v in per.items()}
    out["source_agreement"]["max_abs_diff_npz_mean_delta_vs_table_LaTeX(A)"] = float(np.max(np.abs(per["npz_mean_delta_LA"] - d_LA)))
    out["source_agreement"]["max_abs_diff_npz_mean_delta_vs_score_csv_PDF(A)"] = float(np.max(np.abs(per["npz_mean_delta_PA"] - d_PA)))

    def q(x):
        x = x[np.isfinite(x)]
        return {"median": float(np.median(x)), "p25": float(np.percentile(x, 25)),
                "p75": float(np.percentile(x, 75)), "mean": float(x.mean()), "n": int(x.size)}
    shared = {"definition": "containment(probe in ref) = share of the probe's scored token "
                            "positions t (t >= n-1, finite log-prob) whose token n-gram ending at t "
                            "occurs anywhere in ref's token-id sequence; 64-bit polynomial hash",
              "by_n": {}}
    for nn in NGRAMS:
        blk = {"containment": {
            "PDF(A) in injected LaTeX(A)": q(per[f"cont_PA_in_LA_{nn}"]),
            "PDF(B) in LaTeX(B)": q(per[f"cont_PB_in_LB_{nn}"]),
            "LaTeX(A) in PDF(A)": q(per[f"cont_LA_in_PA_{nn}"]),
            "LaTeX(B) in PDF(B)": q(per[f"cont_LB_in_PB_{nn}"]),
            "PDF(A) in LaTeX(B) (different papers)": q(per[f"cont_PA_in_LB_{nn}"])}}
        for lab, a_cov, a_unc, b_cov, b_unc in [
                ("transfer", "dcov_PA", "dunc_PA", "dcov_PB", "dunc_PB"),
                ("native", "dcov_LA", "dunc_LA", "dcov_LB", "dunc_LB")]:
            tc = per[f"{a_cov}_{nn}"] - per[f"{b_cov}_{nn}"]
            tu = per[f"{a_unc}_{nn}"] - per[f"{b_unc}_{nn}"]
            okk = np.isfinite(tc) & np.isfinite(tu)
            bi = np.array([np.nanmean(tc[k]) for k in IDX]), np.array([np.nanmean(tu[k]) for k in IDX])
            blk[f"{lab}_effect_on_shared_positions"] = {
                "mean": float(np.nanmean(tc)), "ci": pct(bi[0]), "n_pairs": int(np.isfinite(tc).sum()),
                "member_mean_delta": float(np.nanmean(per[f"{a_cov}_{nn}"])),
                "nonmember_mean_delta": float(np.nanmean(per[f"{b_cov}_{nn}"]))}
            blk[f"{lab}_effect_on_unshared_positions"] = {
                "mean": float(np.nanmean(tu)), "ci": pct(bi[1]), "n_pairs": int(np.isfinite(tu).sum()),
                "p_one_sided": float(stats.ttest_1samp(tu[np.isfinite(tu)], 0, alternative="greater").pvalue),
                "member_mean_delta": float(np.nanmean(per[f"{a_unc}_{nn}"])),
                "nonmember_mean_delta": float(np.nanmean(per[f"{b_unc}_{nn}"]))}
            dd2 = tc - tu
            blk[f"{lab}_shared_minus_unshared"] = {"mean": float(np.nanmean(dd2[okk])),
                                                   "ci": pct([np.nanmean(dd2[k]) for k in IDX])}
        shared["by_n"][str(nn)] = blk
    # does the per-pair transfer effect track how much of PDF(A) is shared with what was trained?
    cA = per[f"cont_PA_in_LA_{NPRIMARY}"]
    X = np.column_stack([np.ones(n), cA - cA.mean()])
    beta = np.linalg.lstsq(X, tau_tr, rcond=None)[0]
    bb = np.array([np.linalg.lstsq(X[k], tau_tr[k], rcond=None)[0] for k in IDX[:1000]])
    sp = stats.spearmanr(cA, tau_tr)
    shared["transfer_vs_containment"] = {
        "model": f"tau_transfer_i = b0 + b1*(containment_{NPRIMARY}(PDF A_i in LaTeX A_i) - mean)",
        "b1": float(beta[1]), "b1_ci": pct(bb[:, 1]), "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue), "boot_draws": 1000}
    out["shared_content"] = shared

    # ================= 9. sensitivity: the four audit-excluded pairs put back
    back = [a for a in acct if a["four_cells_scored"]]
    sA = [a["A"] for a in back]; sB = [a["B"] for a in back]
    d_LA2 = np.array([mlp(orig["injected"], a) - mlp(orig["seed42"], a) for a in sA])
    d_PB2 = np.array([mlp(orig["injected"], b) - mlp(orig["seed42"], b) for b in sB])
    d_PA2 = gi("injected", "Fpdf", sA) - gi("seed42", "Fpdf", sA)
    d_LB2 = gi("injected", "Flatex", sB) - gi("seed42", "Flatex", sB)
    tn2, tt2 = d_LA2 - d_LB2, d_PA2 - d_PB2
    I2 = np.random.default_rng(SEED + 1).integers(0, len(back), size=(NBOOT, len(back)))
    out["sensitivity_without_audit_exclusions"] = {
        "n_pairs": len(back),
        "native_mean": float(tn2.mean()), "native_ci": pct(tn2[I2].mean(1)),
        "native_auc": float(ps.auc(d_LA2, d_LB2)),
        "transfer_mean": float(tt2.mean()), "transfer_ci": pct(tt2[I2].mean(1)),
        "transfer_auc": float(ps.auc(d_PA2, d_PB2)),
        "rho": float(tt2.mean() / tn2.mean()),
        "rho_ci": pct(tt2[I2].mean(1) / tn2[I2].mean(1)),
        "seed": SEED + 1,
        "note": "original cells from scores/ here (the four excluded pairs are in the table, "
                "but this keeps one source for all 275 pairs)",
    }

    # ================= 10. stored vs re-derived
    # cohortF_2x2_stats.json is a released input written by a script that is not part
    # of this repository; without it the comparison is skipped (nothing below uses it)
    if STORED.exists():
        st = json.loads(STORED.read_text())
        sn, stt = st["estimators"]["native (LaTeX vs LaTeX)"], st["estimators"]["transfer (PDF vs PDF)"]
        en, et = est["native (LaTeX A vs LaTeX B)"], est["transfer (PDF A vs PDF B)"]
        cmp_rows = []
        def add(name, stored, mine, tol):
            cmp_rows.append({"quantity": name, "stored": stored, "rederived": mine,
                             "abs_diff": None if stored is None or mine is None else abs(stored - mine),
                             "beyond_rounding": None if stored is None or mine is None else abs(stored - mine) > tol})
        add("n_pairs", st["n_pairs"], n, 0)
        add("dropped_incomplete", st["dropped_incomplete"], len(not_scored) - len([a for a in not_scored if a["audit_excluded"]]), 0)
        for lab, s_, e_, P1 in (("native", sn, en, "P1_native_control_level"),
                                ("transfer", stt, et, "P1_transfer_control_level")):
            add(f"{lab} mean", s_["mean"], e_["mean"], 1e-5)
            add(f"{lab} ci lo (stored method)", s_["ci"][0], est[f"{lab} ({'LaTeX' if lab=='native' else 'PDF'} A vs {'LaTeX' if lab=='native' else 'PDF'} B)"]["stored_method_ci"][0], 1e-4)
            add(f"{lab} ci hi (stored method)", s_["ci"][1], est[f"{lab} ({'LaTeX' if lab=='native' else 'PDF'} A vs {'LaTeX' if lab=='native' else 'PDF'} B)"]["stored_method_ci"][1], 1e-4)
            add(f"{lab} ci lo (pair bootstrap, seed {SEED})", s_["ci"][0], e_["ci_pair_boot"][0], 5e-4)
            add(f"{lab} ci hi (pair bootstrap, seed {SEED})", s_["ci"][1], e_["ci_pair_boot"][1], 5e-4)
            add(f"{lab} se", s_["se"], e_["se"], 1e-5)
            add(f"{lab} d_z", s_["d_z"], e_["d_z"], 2e-3)
            add(f"{lab} frac_pos", s_["frac_pos"], e_["frac_pos"], 1e-9)
            add(f"{lab} auc", s_["auc"], e_["auc"], 1e-3)
            add(f"{lab} auc ci lo (stored method, unpaired)", s_["auc_ci"][0], e_["stored_method_auc_ci_unpaired"][0], 2e-3)
            add(f"{lab} auc ci hi (stored method, unpaired)", s_["auc_ci"][1], e_["stored_method_auc_ci_unpaired"][1], 2e-3)
            add(f"{lab} placebo auc", s_["placebo"], pl[P1]["auc"], 1e-3)
            add(f"{lab} placebo ci lo (stored method)", s_["placebo_ci"][0], pl[P1]["auc_ci_stored_method_unpaired"][0], 2e-3)
            add(f"{lab} placebo ci hi (stored method)", s_["placebo_ci"][1], pl[P1]["auc_ci_stored_method_unpaired"][1], 2e-3)
        add("transfer coefficient", st["transfer_coefficient"], float(rho), 1e-3)
        add("transfer coefficient ci lo (stored method, seed 7)", st["transfer_coefficient_ci"][0], out["transfer_coefficient"]["stored_method_ci_seed7"][0], 5e-3)
        add("transfer coefficient ci hi (stored method, seed 7)", st["transfer_coefficient_ci"][1], out["transfer_coefficient"]["stored_method_ci_seed7"][1], 5e-3)
        for k_s, k_m in (("member_loss", dm), ("floor_rise", dfl), ("tau_drop", dd),
                         ("share_from_member", sm), ("share_from_floor", sf)):
            add(f"decomposition {k_s}", st["decomposition"][k_s], float(k_m), 1e-3 if "share" in k_s else 1e-5)
        for k_s, v in (("delta_latexA", d_LA), ("delta_latexB", d_LB), ("delta_pdfA", d_PA), ("delta_pdfB", d_PB)):
            add(f"arm mean {k_s}", st["arm_means"][k_s], float(v.mean()), 1e-5)
        out["stored_vs_rederived"] = cmp_rows
        out["discrepancies_beyond_rounding"] = [r for r in cmp_rows if r["beyond_rounding"]]
    else:
        out["stored_vs_rederived"] = []
        out["discrepancies_beyond_rounding"] = []
        out["stored_comparison"] = "skipped: cohortF_2x2_stats.json absent"

    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("accounting", "randomization_audit", "injection_integrity", "source_agreement",
                                          "scoring_reproducibility")}, indent=1))
    print(json.dumps(out["estimators"], indent=1))
    print(json.dumps(out["transfer_coefficient"], indent=1))
    print(json.dumps(out["decomposition"], indent=1))
    print(json.dumps(out["placebos"], indent=1))
    print(json.dumps(out["shared_content"], indent=1))
    print(json.dumps(out["sensitivity_without_audit_exclusions"], indent=1))
    print("DISCREPANCIES:", json.dumps(out["discrepancies_beyond_rounding"], indent=1))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
