#!/usr/bin/env python3
"""Table 4 (tables/calibrated.tex) and the holdout-dependent calib* macros
recomputed on ALL NAT1 pairs instead of the 80% holdout split.

Reuses, unmodified, the functions of
  analysis/calibrated_attacks.py  (evaluate, auc, tpr_at, strat_tpr, ...)
  figures/paper_figures.py        (load, population, headline, tpr_at, ...)
The driver below replicates calibrated_attacks.main() lines 494-570 and 638-645
(population construction, RNG stream, shared pair bootstrap, evaluate() calls)
with RMIA dropped and ONE change: the evaluated pair list is pops["NAT1"]
(all 4,832 pairs) instead of splits["NAT1"]["holdout"] (3,886 pairs). Line
numbers in the comments refer to analysis/calibrated_attacks.py. Neither
module's main() is called, so neither writes anything.

Inputs (under the data root, see olmo_twin/paths.py): bundle/evaluation_manifest.csv.gz,
analysis/per_document_attacks_calibrated.csv.gz and calibrated_attacks_summary.json
(written by analysis/calibrated_attacks.py; values written with repr(), exact
round trip, so the per-token npz files are not re-read),
analysis/posttrain_exclusions.json and, through paper_figures.load(), the attack
table. The scores are checked against the extraction cache (--cache, default
analysis/cache/doc_stats.npz) when it exists.

Step 0 reproduces the published holdout JSON exactly (validation of the driver).
Step 1 is the recomputation on all NAT1 pairs.
Step 2 cross-checks against paper_figures' whole-NAT1 numbers (sanity anchors);
with --paper-tables DIR also against the macros of DIR/numbers.tex (the paper's
reformatted file or the generator's own, both are parsed), and the commented
RMIA rows of DIR/calibrated.tex (if present) are carried into the table.
Outputs go to --out (default analysis/table4_all_pairs/): values.json,
results_full.json and calibrated_allNAT1.tex; then run analysis/format_table4.py
--in <the same directory>.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

ANA = paths.ANALYSIS
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--out", type=Path, default=ANA / "table4_all_pairs",
                 help="output directory (default: analysis/table4_all_pairs under the data root)")
_ap.add_argument("--cache", type=Path, default=ANA / "cache/doc_stats.npz",
                 help="calibrated_attacks.py extraction cache; compared with the CSV only if it exists")
_ap.add_argument("--paper-tables", type=Path, default=None,
                 help="optional directory with numbers.tex (and calibrated.tex) of the paper or of "
                      "figures/paper_figures.py: anchors of step 2 and the commented RMIA rows")
ARGS = _ap.parse_args()
HERE = ARGS.out                       # output directory
HERE.mkdir(parents=True, exist_ok=True)

paths.add_code_dirs("analysis", "figures")
import calibrated_attacks as ca  # noqa: E402  (paths default to the data root; main() is not called)
import paper_figures as pf  # noqa: E402  (main() is not called, so nothing is written under its output directory)

N_BOOT, SEED = 400, 20261012          # calibrated_attacks.main() defaults (--n-boot, --seed)
CALIB_CSV = ANA / "per_document_attacks_calibrated.csv.gz"
CALIB_JSON = ANA / "calibrated_attacks_summary.json"
CACHE = ARGS.cache
REFS = ca.REFS                        # ("sibling", "control", "stage1")
REFK = {"control": "orc", "sibling": "sib", "stage1": "pub"}

# ------------------------------------------------------------------ documents (main() 494-506)
man = ca.read_csv_gz(ca.MANIFEST)
doc_ids = [r["doc_id"] for r in man]
n_docs = len(doc_ids)
calrows = {r["doc_id"]: r for r in ca.read_csv_gz(CALIB_CSV)}
scored = np.array([d in calrows for d in doc_ids])
missing = [d for d, s in zip(doc_ids, scored) if not s]
print(f"[docs] manifest {n_docs}, scored (rows in calibrated CSV) {int(scored.sum())}, unscored {missing}")
NUMCOLS = ["n_scored"] + [f"loss_{m}" for m in ca.MODELS] + [f"ratio_{r}" for r in REFS] + \
          [f"tau_{r}" for r in REFS] + [f"n_eff_{r}" for r in REFS] + \
          [f"zratio_{r}_{v}" for r in REFS for v in ("T", "ac")]
D = {k: np.array([ca.fnum(calrows[d][k]) if d in calrows else np.nan for d in doc_ids]) for k in NUMCOLS}
for k in ("cohort", "arm", "pair_id", "split", "r"):    # the CSV copies these manifest columns
    assert all(calrows[d][k] == man[i][k] for i, d in enumerate(doc_ids) if d in calrows), k
cohort = np.array([r["cohort"] for r in man], dtype=object)
split = [r["split"] for r in man]
rcop = np.array([ca.fnum(r["r"]) for r in man])

# CSV vs extraction cache (the arrays calibrated_attacks.main() actually used)
cache_chk = {}
if CACHE.exists():
    Z = np.load(CACHE, allow_pickle=True)
    assert list(Z["doc_ids"]) == doc_ids
    for k in NUMCOLS:
        ck = "n_treated" if k == "n_scored" else k
        a, b = D[k][scored], Z[ck][scored]
        same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        ok = np.isfinite(a) & np.isfinite(b)
        cache_chk[k] = {"max_abs_diff": float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0, "nan_mask_equal": same_nan}
    print(f"[cache] max |CSV - cache| over {len(cache_chk)} columns: "
          f"{max(v['max_abs_diff'] for v in cache_chk.values()):.3g}; NaN masks equal: {all(v['nan_mask_equal'] for v in cache_chk.values())}")

# ------------------------------------------------------------------ pairs and populations (main() 509-525)
excl = json.loads(ca.EXCL.read_text())
bad_pairs, bad_docs = set(excl["exclude_pair_ids"]), set(excl["exclude_doc_ids"])
byp = defaultdict(dict)
for i, r in enumerate(man):
    if not r["pair_id"] or not scored[i] or r["pair_id"] in bad_pairs or r["doc_id"] in bad_docs:
        continue
    byp[r["pair_id"]][r["arm"]] = i
pairs_all = sorted((v["inject"], v["holdout"]) for v in byp.values() if "inject" in v and "holdout" in v)
assert all(split[i] == split[h] for i, h in pairs_all)
NAT1 = [(i, h) for i, h in pairs_all if rcop[i] == 1 and cohort[i] in ca.NAT]
splits = {s: [(i, h) for i, h in NAT1 if split[i] == s] for s in ("dev", "holdout")}
print(f"[pop] NAT1 {len(NAT1)} pairs (dev {len(splits['dev'])}, holdout {len(splits['holdout'])}); all pairs {len(pairs_all)}")

# ------------------------------------------------------------------ RNG stream (main() 528-563)
idx_scored = np.nonzero(scored)[0]


def boots_for(n):
    """Exactly main()'s stream: default_rng(seed); 100-doc sanity sample; then NAT1's draws."""
    rng = np.random.default_rng(SEED)
    rng.choice(idx_scored, 100, replace=False)
    return rng.integers(0, n, size=(N_BOOT, n))


plain = {r: D[f"ratio_{r}"] for r in REFS}


def run(pairs, dev_pairs):
    """main() 581-645 without RMIA: plain ratio (no base) and both z variants (base = plain ratio)."""
    B = boots_for(len(pairs))
    out = {"plain_ratio": {}, "zratio": {}}
    for r in REFS:
        out["plain_ratio"][r] = ca.evaluate(plain[r], pairs, cohort, B, dev_pairs=dev_pairs)
        out["zratio"][r] = {v: ca.evaluate(D[f"zratio_{r}_{v}"], pairs, cohort, B, base=plain[r], dev_pairs=dev_pairs)
                            for v in ("T", "ac")}
    return out


# ------------------------------------------------------------------ step 0: reproduce the published holdout JSON
S = json.loads(CALIB_JSON.read_text())
hold = run(splits["holdout"], splits["dev"])
FIELDS = ("auc", "auc_ci", "tpr1", "tpr1_ci", "tpr1_strat", "tpr1_k_negatives", "n_pairs",
          "delta_auc_vs_plain", "delta_auc_vs_plain_ci", "delta_tpr1_vs_plain", "delta_tpr1_vs_plain_ci",
          "delta_tpr1_strat_vs_plain", "auc_dev", "tpr1_dev")
maxdiff = 0.0; nchk = 0
for r in REFS:
    blocks = [(hold["plain_ratio"][r], S["results"]["NAT1"]["plain_ratio"][r])] + \
             [(hold["zratio"][r][v], S["results"]["NAT1"]["zratio"][r][v]) for v in ("T", "ac")]
    for mine, pub in blocks:
        for f in FIELDS:
            if f in pub:
                a, b = np.ravel(mine[f]).astype(float), np.ravel(pub[f]).astype(float)
                maxdiff = max(maxdiff, float(np.max(np.abs(a - b)))); nchk += a.size
print(f"[step 0] holdout reproduction: {nchk} published JSON values, max |diff| = {maxdiff:.3g}")
assert maxdiff < 1e-12, "driver does not reproduce the published holdout numbers"

# ------------------------------------------------------------------ step 1: all NAT1 pairs
full = run(NAT1, None)
n = len(NAT1); kneg = full["plain_ratio"]["control"]["tpr1_k_negatives"]
V = {}
for r in REFS:
    k = REFK[r]
    for row, e in (("ratio", full["plain_ratio"][r]), ("z", full["zratio"][r]["ac"]), ("zT", full["zratio"][r]["T"])):
        tk = "table" if row != "zT" else "appx"          # zT (n_eff = T) is not a Table 4 row; Appendix B only
        V[f"{tk}.{k}.{row}.auc"] = e["auc"]
        V[f"{tk}.{k}.{row}.auc_lo"], V[f"{tk}.{k}.{row}.auc_hi"] = e["auc_ci"]
        V[f"{tk}.{k}.{row}.tpr"] = 100 * e["tpr1"]
        V[f"{tk}.{k}.{row}.tpr_lo"], V[f"{tk}.{k}.{row}.tpr_hi"] = (100 * x for x in e["tpr1_ci"])
        V[f"{tk}.{k}.{row}.strat"] = 100 * e["tpr1_strat"]
        # macro names as paper_figures.calibrated() builds them: calib{att}{k}{field}
        mrow = row
        if row == "zT":                                    # paper_figures emits only auc/tpr/strat for zT
            V[f"macro.calibzT{k}auc"] = e["auc"]; V[f"macro.calibzT{k}tpr"] = 100 * e["tpr1"]
            V[f"macro.calibzT{k}strat"] = 100 * e["tpr1_strat"]
            continue
        V[f"macro.calib{mrow}{k}auc"] = e["auc"]
        V[f"macro.calib{mrow}{k}auclo"], V[f"macro.calib{mrow}{k}auchi"] = e["auc_ci"]
        V[f"macro.calib{mrow}{k}tpr"] = 100 * e["tpr1"]
        V[f"macro.calib{mrow}{k}tprlo"], V[f"macro.calib{mrow}{k}tprhi"] = (100 * x for x in e["tpr1_ci"])
        V[f"macro.calib{mrow}{k}strat"] = 100 * e["tpr1_strat"]
        if "delta_auc_vs_plain" in e:
            V[f"macro.calib{mrow}{k}dauc"] = e["delta_auc_vs_plain"]
            V[f"macro.calib{mrow}{k}dauclo"], V[f"macro.calib{mrow}{k}dauchi"] = e["delta_auc_vs_plain_ci"]
            V[f"macro.calib{mrow}{k}dtpr"] = 100 * e["delta_tpr1_vs_plain"]
            V[f"macro.calib{mrow}{k}dtprlo"], V[f"macro.calib{mrow}{k}dtprhi"] = (100 * x for x in e["delta_tpr1_vs_plain_ci"])
            V[f"macro.calib{mrow}{k}dstrat"] = 100 * e["delta_tpr1_strat_vs_plain"]
V["macro.calibn"] = n
V["macro.calibnholdout"] = n
V["macro.calibkneg"] = kneg
V["macro.calibdraws"] = N_BOOT
# \calibtaumedian / \calibneffshare: neff_summary of main() 681-688 is over ALL scored documents (idx_scored),
# never over a split, so the recomputation is the identity; reported, plus the NAT1-pair-document variant
T_all = D["n_scored"][idx_scored]
V["macro.calibtaumedian"] = float(np.nanmedian(D["tau_control"][idx_scored]))
V["macro.calibneffshare"] = 100 * float(np.nanmedian(D["n_eff_control"][idx_scored] / T_all))
nat_docs = np.array(sorted({d for p in NAT1 for d in p}))
V["alt.calibtaumedian_NAT1docs"] = float(np.nanmedian(D["tau_control"][nat_docs]))
V["alt.calibneffshare_NAT1docs"] = 100 * float(np.nanmedian(D["n_eff_control"][nat_docs] / D["n_scored"][nat_docs]))
V["check.neff_summary_json.median_tau"] = S["meta"]["neff_summary"]["control"]["median_tau"]
V["check.neff_summary_json.median_neff_over_T_pct"] = 100 * S["meta"]["neff_summary"]["control"]["median_neff_over_T"]
V["check.step0_holdout_reproduction_max_abs_diff"] = maxdiff
V["check.step0_holdout_values_compared"] = nchk
V["check.n_scored_docs"] = int(scored.sum())

# ------------------------------------------------------------------ step 2: paper_figures cross-check on all NAT1
cols, rows, arm, pairs_pf = pf.load()
selN = pf.population(cols, pairs_pf, "NAT1")
did = cols["doc_id"]
setA = {(doc_ids[i], doc_ids[h]) for i, h in NAT1}
setB = {(did[i], did[h]) for i, h in selN}
V["check.pf_NAT1_pairs"] = len(selN)
V["check.pf_vs_ca_pairsets_identical"] = int(setA == setB)
coh = cols["cohort"]


def pf_strat(key, sel):                     # paper_figures.calibrated() inner strat_tpr, verbatim
    tp = tot = 0
    for ch in sorted(set(coh[i] for i, _ in sel)):
        s_ = [(i, h) for i, h in sel if coh[i] == ch]
        aa, bb = pf.arrays(cols, key, s_)
        tp += pf.tpr_at(aa, bb, 0.01)[0] * len(s_); tot += len(s_)
    return tp / tot


ANCH = {  # paper macro -> (paper value, recomputed value from paper_figures functions, recomputed from ca on all NAT1)
}
for key, k, pm in (("ratio_oracle", "orc", ("aucnat", "tpronenat", "tprstratorc")),
                   ("z_oracle", "zorc", ("auczorcnat", "tpronezorcnat", "tprstratzorc")),
                   ("ratio_sibling", "sib", ("aucrsibling", "tprrsibling", "tprstratsib")),
                   ("z_sibling", "zsib", ("auczsibnat", "tpronezsibnat", "tprstratzsib")),
                   ("ratio_stage1", "pub", ("aucrstage", "tprrstage", "tprstratpub")),
                   ("z_stage1", "zpub", ("auczpubnat", "tpronezpubnat", "tprstratzpub"))):
    a, b = pf.arrays(cols, key, selN)
    V[f"pf.{pm[0]}"] = pf.ps.auc(a, b)
    V[f"pf.{pm[1]}"] = 100 * pf.tpr_at(a, b, 0.01)[0]
    V[f"pf.{pm[2]}"] = 100 * pf_strat(key, selN)

# paper macros (--paper-tables DIR/numbers.tex) for the anchors and the existing whole-NAT1 CIs
NUM = ARGS.paper_tables / "numbers.tex" if ARGS.paper_tables else None
pm = {}
if NUM is not None and NUM.exists():
    # the paper's copy holds bare values; the generator's own numbers.tex adds \xspace and {,}
    _num = NUM.read_text().replace("{,}", "").replace("\\xspace", "")
    pm = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", _num)}
else:
    print("[step 2] no --paper-tables numbers.tex: paper.* anchors skipped")
PAPER_ANCHORS = ("aucnat", "tpronenat", "tprstratorc", "auczorcnat", "tpronezorcnat", "tprstratzorc",
             "aucrsibling", "tprrsibling", "tprstratsib", "auczsibnat", "tpronezsibnat", "tprstratzsib",
             "aucrstage", "tprrstage", "tprstratpub", "auczpubnat", "tpronezpubnat", "tprstratzpub",
             "aucnatlo", "aucnathi", "tpronenatlo", "tpronenathi", "auczorcnatlo", "auczorcnathi",
             "tpronezorcnatlo", "tpronezorcnathi", "auczsibnatlo", "auczsibnathi", "tpronezsibnatlo",
             "tpronezsibnathi", "auczpubnatlo", "auczpubnathi", "tpronezpubnatlo", "tpronezpubnathi",
             "npairsnat", "negnat")
for name in (PAPER_ANCHORS if pm else ()):
    V[f"paper.{name}"] = float(pm[name])
json.dump(V, open(HERE / "values.json", "w"), indent=1, sort_keys=True)
json.dump({"all_NAT1": full, "holdout_reproduced": hold, "cache_check": cache_chk}, open(HERE / "results_full.json", "w"),
          indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ------------------------------------------------------------------ proposed table (same layout as the paper's hand-formatted calibrated.tex)
d3 = lambda x: f"{x:.3f}"[1:]
def row(lab, e):
    return (f"\\quad {lab} & {e['auc']:.3f}& {{\\scriptsize[{d3(e['auc_ci'][0])},{d3(e['auc_ci'][1])}]}} & "
            f"{100*e['tpr1']:.1f}& {{\\scriptsize[{100*e['tpr1_ci'][0]:.1f},{100*e['tpr1_ci'][1]:.1f}]}} & "
            f"{100*e['tpr1_strat']:.1f} \\\\")
TAB = ARGS.paper_tables / "calibrated.tex" if ARGS.paper_tables else None
paper_tab = TAB.read_text().splitlines() if TAB is not None and TAB.exists() else []   # source of the commented RMIA rows
rmia_old = defaultdict(list); cur = None
for ln in paper_tab:
    if "emph{Exact" in ln: cur = "control"
    elif "emph{Retrained" in ln: cur = "sibling"
    elif "emph{Published" in ln: cur = "stage1"
    if ln.startswith("%RMIA-old%") and cur:
        rmia_old[cur].append(ln)
L = [r"% generated by analysis/recompute_table4_all_pairs.py -- do not edit by hand",
     r"% 2026-10-28: recomputed on all NAT1 pairs (was the 80% holdout split); RMIA rows unchanged and commented out",
     r"\begin{tabular}{@{}l S[table-format=1.3]@{\,}l S[table-format=1.1]@{\,}l S[table-format=2.1]@{}}",
     r"\toprule",
     r"Attack & \multicolumn{2}{c}{\auc\ [95\% CI]} & \multicolumn{2}{c}{\tprat{1} [95\% CI]} & {per domain} \\",
     r"\midrule"]
for r, lab in (("control", "Exact counterfactual"), ("sibling", "Retrained reference"), ("stage1", "Published checkpoint")):
    L.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{lab}}}}} \\\\")
    L.append(row("Ratio", full["plain_ratio"][r]))
    L += rmia_old[r]
    L.append(row("Standardized ratio", full["zratio"][r]["ac"]))
    if r != "stage1":
        L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
(HERE / "calibrated_allNAT1.tex").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ report
print("\n[step 1] all NAT1 pairs (n = %d, k_neg = %d)" % (n, kneg))
for r in REFS:
    for lab, e in (("ratio", full["plain_ratio"][r]), ("z(ac)", full["zratio"][r]["ac"]), ("z(T)", full["zratio"][r]["T"])):
        s = (f"  {r:8s} {lab:6s} AUC {e['auc']:.4f} [{e['auc_ci'][0]:.4f},{e['auc_ci'][1]:.4f}]  "
             f"TPR {100*e['tpr1']:.2f} [{100*e['tpr1_ci'][0]:.2f},{100*e['tpr1_ci'][1]:.2f}]  strat {100*e['tpr1_strat']:.2f}")
        if "delta_auc_vs_plain" in e:
            s += (f"  dAUC {e['delta_auc_vs_plain']:+.4f} [{e['delta_auc_vs_plain_ci'][0]:+.4f},{e['delta_auc_vs_plain_ci'][1]:+.4f}]"
                  f"  dTPR {100*e['delta_tpr1_vs_plain']:+.2f} [{100*e['delta_tpr1_vs_plain_ci'][0]:+.2f},{100*e['delta_tpr1_vs_plain_ci'][1]:+.2f}]")
        print(s)
print("\n[step 2] anchors: paper macro | paper_figures fn | calibrated_attacks fn (all NAT1)")
A = [("aucnat", "table.orc.ratio.auc", 1), ("tpronenat", "table.orc.ratio.tpr", 1), ("tprstratorc", "table.orc.ratio.strat", 1),
     ("auczorcnat", "table.orc.z.auc", 1), ("tpronezorcnat", "table.orc.z.tpr", 1), ("tprstratzorc", "table.orc.z.strat", 1),
     ("aucrsibling", "table.sib.ratio.auc", 1), ("tprrsibling", "table.sib.ratio.tpr", 1), ("tprstratsib", "table.sib.ratio.strat", 1),
     ("auczsibnat", "table.sib.z.auc", 1), ("tpronezsibnat", "table.sib.z.tpr", 1), ("tprstratzsib", "table.sib.z.strat", 1),
     ("aucrstage", "table.pub.ratio.auc", 1), ("tprrstage", "table.pub.ratio.tpr", 1), ("tprstratpub", "table.pub.ratio.strat", 1),
     ("auczpubnat", "table.pub.z.auc", 1), ("tpronezpubnat", "table.pub.z.tpr", 1), ("tprstratzpub", "table.pub.z.strat", 1)]
for name, key, _ in A:
    print(f"  {name:15s} paper {pm.get(name, '-'):>6s} | pf {V['pf.' + name]:.5f} | ca {V[key]:.5f}")
print(f"  pair sets identical (pf vs ca): {setA == setB} ({len(selN)} vs {len(NAT1)})")
print(f"  calibtaumedian {V['macro.calibtaumedian']:.4f} (JSON {V['check.neff_summary_json.median_tau']:.4f}); "
      f"calibneffshare {V['macro.calibneffshare']:.3f} (JSON {V['check.neff_summary_json.median_neff_over_T_pct']:.3f}); "
      f"NAT1-docs variant: tau {V['alt.calibtaumedian_NAT1docs']:.4f}, share {V['alt.calibneffshare_NAT1docs']:.3f}")
print(f"[out] {HERE / 'values.json'}, {HERE / 'results_full.json'}, {HERE / 'calibrated_allNAT1.tex'}")
