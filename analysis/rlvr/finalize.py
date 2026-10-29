#!/usr/bin/env python3
"""Write the FINAL post-training + RL results from rlvr/work/recompute_*.{csv,json} and
rlvr/work/rl_minus_dpo.csv (written by recompute.py and rl_minus_dpo.py; the recomputation was
number-by-number identical to two other implementations, which are not included).

Writes into FINAL_DIR (default: rlvr/ under the data root): stages_extended.csv, derived.csv,
sensitivity_rl_overlap.csv, rl_minus_dpo.csv, cohort_auc.csv, rl_trajectory.csv,
ai2_control_robustness.csv, ai2_control_drift.csv and proposed_macros.tex (the RLVR macros).
With a paper directory (--paper-dir DIR or the second positional argument; it must hold
tables/numbers.tex and tables/stages.tex) it also writes reproduction_check.csv and marks the
macros that numbers.tex already defines ('%%exists', '% paper has ...'). Without one, every macro
is emitted as \\newcommand and reproduction_check.csv is not written; the macro values do not
depend on the paper.

usage: python3 analysis/rlvr/finalize.py [FINAL_DIR] [PAPER_DIR] [--paper-dir PAPER_DIR]"""
import argparse, json, re, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from recompute import STAGES, TRAJ, AI2, FINAL, B, SEED, COHORTS, OUT   # OUT = rlvr/work under the data root
from recompute import paths   # olmo_twin.paths, put on sys.path by recompute.py

H = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("final_dir", nargs="?", type=Path, default=paths.RLVR,
                 help="output directory of the final tables (default: rlvr/ under the data root)")
_ap.add_argument("paper_dir", nargs="?", type=Path, default=None, help="same as --paper-dir")
_ap.add_argument("--paper-dir", dest="paper_dir_opt", metavar="PAPER_DIR", type=Path, default=None,
                 help="paper sources with tables/numbers.tex and tables/stages.tex (optional)")
_args = _ap.parse_args()
FD = _args.final_dir
FD.mkdir(parents=True, exist_ok=True)
PAPER = _args.paper_dir_opt or _args.paper_dir   # None: no reproduction check, no '%%exists' annotations

POPDESC = {
    "NAT1": "verified-unseen single-exposure pairs (cohorts M, L, W-new, W-gap, S-se; r=1); the paper's primary population",
    "ALLPAIRS7803": "every complete pair incl. cohorts X and F (7,803); the population of the paper's existing \\pt*all macros (the generator's ALL in an earlier version)",
    "ALL7326": "every analysed pair except cohorts X and F (7,326); the current generator's ALL and the population named in the Q7 prose ('all cohorts except F and X')",
    "NAT1_minus_M01966": "sensitivity: NAT1 without pair M-01966 (holdout doc quotes a MATH problem present in both RLVR sets; 19 shared 8-grams, below the audit threshold)",
    "NAT1_minus_RLoverlap5": "sensitivity: NAT1 without every pair with a document sharing >=5 distinct 8-grams or a >=20-word run with any RLVR record (all below the audit threshold)",
}
LABEL = {"mid": "Mid-training", "sft": "+ SFT", "dpo": "+ SFT + DPO", "rlvr1": "+ SFT + DPO + RLVR1 (final, step 2604)",
         "rlvr2i": "+ RLVR2 step 2000 (released-Instruct analogue)", "rlvr2": "+ RLVR2 final (step 2604)"}
PREV = {"sft": "mid", "dpo": "sft", "rlvr1": "dpo", "rlvr2i": "rlvr1", "rlvr2": "rlvr1"}
DIRS = {s: (t, c) for s, t, c in STAGES}
MAINPOPS = ["NAT1", "ALLPAIRS7803", "ALL7326"]

R = pd.read_csv(OUT / "recompute_all.csv")
COH = json.loads((OUT / "recompute_cohort.json").read_text())
DRIFT = json.loads((OUT / "recompute_drift.json").read_text())
RLD = pd.read_csv(OUT / "rl_minus_dpo.csv")
get = lambda kind, pop, st: R[(R.kind == kind) & (R.population == pop) & (R.stage == st)].iloc[0]

def ci_cols(r, k, scale=1.0, name=None):
    name = name or k
    return {name: r[k] * scale, f"{name}_lo": r[f"{k}_lo"] * scale, f"{name}_hi": r[f"{k}_hi"] * scale}

# ------------------------------------------------------------ stages_extended.csv + derived.csv
st_rows, dv_rows = [], []
for pop in MAINPOPS + ["NAT1_minus_M01966", "NAT1_minus_RLoverlap5"]:
    for s, tr, co in STAGES:
        r = get("stage", pop, s)
        row = dict(population=pop, stage=s, stage_label=LABEL[s],
                   treated=tr or "per_document_attacks_prelim.csv.gz:ratio_oracle (injected endpoint)",
                   control=co or "(seed-42 endpoint; exact counterfactual)", n=int(r.n))
        row.update(ci_cols(r, "tau")); row.update(ci_cols(r, "sd")); row.update(ci_cols(r, "dz")); row.update(ci_cols(r, "auc"))
        row.update(ci_cols(r, "tpr1", 100, "tpr1_pct")); row["pos_pct"] = 100 * r.pos
        row["table_row"] = f"{LABEL[s]} & {r.tau:+.4f} & {r.sd:.4f} & {r.dz:.2f} & {r.auc:.3f} & {100*r.tpr1:.1f} \\\\"
        st_rows.append(row)
        if s == "mid": continue
        d = dict(population=pop, stage=s, stage_label=LABEL[s], prev_stage=PREV[s], n=int(r.n))
        d.update({"ret_tau_pct": r.ret_tau, "ret_tau_lo": 100 * r.ret_tau_lo, "ret_tau_hi": 100 * r.ret_tau_hi,
                  "ret_sd_pct": r.ret_sd, "ret_sd_lo": 100 * r.ret_sd_lo, "ret_sd_hi": 100 * r.ret_sd_hi,
                  "ret_dz_pct": r.ret_dz, "ret_dz_lo": 100 * r.ret_dz_lo, "ret_dz_hi": 100 * r.ret_dz_hi,
                  "ret_aucadv_pct": r.ret_auc, "ret_aucadv_lo": 100 * r.ret_auc_lo, "ret_aucadv_hi": 100 * r.ret_auc_hi,
                  "tpr_change_rel_pct": r.tpr_change, "tpr_change_rel_lo": 100 * r.tpr_change_lo, "tpr_change_rel_hi": 100 * r.tpr_change_hi,
                  "tpr_change_pp": r.tpr_change_pp,
                  "corr_with_mid": r.corr_mid, "corr_with_mid_lo": r.corr_mid_lo, "corr_with_mid_hi": r.corr_mid_hi,
                  "corr_with_prev_stage": r.corr_prev,
                  "corr_with_rlvr2_step2000": r.get("corr_rlvr2step2000", np.nan) if s == "rlvr2" else np.nan})
        dv_rows.append(d)
ST = pd.DataFrame(st_rows); DV = pd.DataFrame(dv_rows)
ST[ST.population.isin(MAINPOPS)].to_csv(FD / "stages_extended.csv", index=False)
DV[DV.population.isin(MAINPOPS)].to_csv(FD / "derived.csv", index=False)
sens = ST[ST.population.str.startswith("NAT1")].merge(DV[["population", "stage", "ret_tau_pct", "ret_dz_pct", "ret_aucadv_pct", "corr_with_mid"]],
                                                       on=["population", "stage"], how="left")
sens.to_csv(FD / "sensitivity_rl_overlap.csv", index=False)
rld = RLD.copy()
for k in ("tpr1", "pos"):
    for suf in ("", "_lo", "_hi"):
        rld[f"d_{k}{suf}"] = 100 * rld[f"d_{k}{suf}"]
rld = rld.rename(columns=lambda c: c.replace("d_tpr1", "d_tpr1_pp").replace("d_pos", "d_pos_pp"))
rld.to_csv(FD / "rl_minus_dpo.csv", index=False)

# ------------------------------------------------------------ cohort_auc.csv
crow = []
for ch in COHORTS:
    c = COH[ch]; row = dict(cohort=ch, n_pairs=c["n"])
    for s, _, _ in STAGES: row[f"auc_{s}"] = c[s]
    row["change_dpo_to_rlvr2i"] = c["rlvr2i"] - c["dpo"]; row["change_dpo_to_rlvr2"] = c["rlvr2"] - c["dpo"]
    for s, *_ in AI2: row[f"robustness_ai2ctl_auc_{s}"] = c[f"ai2_{s}"]
    crow.append(row)
pd.DataFrame(crow).to_csv(FD / "cohort_auc.csv", index=False)

# ------------------------------------------------------------ rl_trajectory.csv
tr_rows = []
for pop in MAINPOPS:
    for st, k, tdir, cdir in TRAJ:
        r = get("traj", pop, f"{st}:{k}")
        row = dict(population=pop, rl_stage="dpo_end" if st == "dpo" else st, stage_step=k,
                   rl_update=k if st in ("dpo", "rlvr1") else FINAL + k,
                   checkpoint="DPO end (RL step 0)" if st == "dpo" else f"{st.upper()} step {k}" + (" (final)" if k == FINAL else ""),
                   treated=tdir, control=cdir, n=int(r.n))
        for m, sc, nm in (("tau", 1, "tau"), ("sd", 1, "sd"), ("dz", 1, "dz"), ("auc", 1, "auc"), ("tpr1", 100, "tpr1_pct")):
            row.update(ci_cols(r, m, sc, nm))
        row["pos_pct"] = 100 * r.pos
        for m, nm in (("ret_tau", "ret_tau_pct"), ("ret_sd", "ret_sd_pct"), ("ret_dz", "ret_dz_pct"), ("ret_auc", "ret_aucadv_pct"), ("tpr_change", "tpr_change_rel_pct")):
            row[nm] = r[m]; row[nm.replace("_pct", "") + "_lo"] = 100 * r[f"{m}_lo"]; row[nm.replace("_pct", "") + "_hi"] = 100 * r[f"{m}_hi"]
        row.update(ci_cols(r, "corr_mid", 1, "corr_with_mid"))
        row["corr_with_prev_checkpoint"] = r.corr_prev if st != "dpo" else get("stage", pop, "dpo").corr_prev
        row["prev_checkpoint"] = "SFT end" if st == "dpo" else None
        tr_rows.append(row)
TR = pd.DataFrame(tr_rows)
TR.to_csv(FD / "rl_trajectory.csv", index=False)

# ------------------------------------------------------------ ai2_control_robustness.csv
ar = []
for pop in MAINPOPS:
    for s, tdir, ai2dir, ourctl in AI2:
        r = get("ai2", pop, s); p = get("placebo", pop, s); m = get("stage", pop, s)
        row = dict(population=pop, stage=s, stage_label=LABEL[s] + (" vs ai2-instruct (extra row)" if s == "rlvr2" else ""),
                   treated=tdir, control=ai2dir, n=int(r.n), design="ROBUSTNESS: L(ours injected, stage) - L(AI2 released, stage)")
        for k, sc, nm in (("tau", 1, "tau"), ("sd", 1, "sd"), ("dz", 1, "dz"), ("auc", 1, "auc"), ("tpr1", 100, "tpr1_pct")):
            row.update(ci_cols(r, k, sc, nm))
        row["pos_pct"] = 100 * r.pos
        row.update(ret_tau_pct=r.ret_tau, ret_sd_pct=r.ret_sd, ret_dz_pct=r.ret_dz, ret_aucadv_pct=r.ret_auc, tpr_change_rel_pct=r.tpr_change,
                   corr_with_mid=r.corr_mid, corr_with_matched_control_design=r.corr_main_design,
                   matched_tau=m.tau, matched_sd=m.sd, matched_dz=m.dz, matched_auc=m.auc, matched_tpr1_pct=100 * m.tpr1)
        row.update({f"placebo_{k}": p[k] for k in ("tau", "tau_lo", "tau_hi", "sd", "dz", "auc", "auc_lo", "auc_hi")})
        row["placebo_tpr1_pct"] = 100 * p.tpr1; row["placebo_pos_pct"] = 100 * p.pos
        row["placebo_definition"] = f"L({ourctl}) - L({ai2dir}): our control chain minus AI2's, same within-pair contrast (neither arm saw the documents)"
        ar.append(row)
pd.DataFrame(ar).to_csv(FD / "ai2_control_robustness.csv", index=False)
dr = []
for s, d in DRIFT.items():
    for g, v in d.items():
        dr.append(dict(stage=s, documents=g, n=v["n"], mean_ours_ctl_minus_ai2=v["mean"], sd=v["sd"]))
pd.DataFrame(dr).to_csv(FD / "ai2_control_drift.csv", index=False)

# ------------------------------------------------------------ reproduction_check.csv
paper = {}
CK = {"m": "M", "l": "L", "wnew": "W-new", "wgap": "W-gap", "sse": "S-se", "e": "E", "ezero": "E0"}   # also used below
if PAPER is None:
    print("no paper directory: reproduction_check.csv not written; all macros emitted as \\newcommand")
else:
    for line in (PAPER / "tables/numbers.tex").read_text().splitlines():
        g = re.match(r"\\newcommand\{\\(pt[a-z]+)\}\{([^}]*)\}", line)
        if g: paper[g.group(1)] = g.group(2)
    def mac(pop, suf):
        mm = {}
        S = {s: get("stage", pop, s) for s, *_ in STAGES}
        for s in ("mid", "sft", "dpo"):
            v = S[s]
            mm[f"pt{s}tau{suf}"] = f"{v.tau:+.4f}"; mm[f"pt{s}dz{suf}"] = f"{v.dz:.2f}"; mm[f"pt{s}auc{suf}"] = f"{v.auc:.3f}"
            mm[f"pt{s}tpr{suf}"] = f"{100*v.tpr1:.1f}"; mm[f"pt{s}sd{suf}"] = f"{v.sd:.4f}"; mm[f"pt{s}pos{suf}"] = f"{100*v.pos:.1f}"
            if s != "mid": mm[f"pt{s}corr{suf}"] = f"{v.corr_mid:.2f}"
        mm[f"ptsftretain{suf}"] = f"{S['sft'].ret_tau:.0f}"; mm[f"ptsftsdretain{suf}"] = f"{S['sft'].ret_sd:.0f}"
        mm[f"ptdporetain{suf}"] = f"{S['dpo'].ret_tau:.0f}"; mm[f"ptdpotprgain{suf}"] = f"{S['dpo'].tpr_change:+.0f}"
        if suf == "":
            mm["ptdpodzretain"] = f"{S['dpo'].ret_dz:.0f}"; mm["ptdpoaucretain"] = f"{S['dpo'].ret_auc:.0f}"
        return mm
    ours = mac("NAT1", ""); ours.update(mac("ALLPAIRS7803", "all"))
    alt = mac("ALL7326", "all")
    for k, ch in CK.items():
        for s in ("mid", "sft", "dpo"): ours[f"ptcoh{k}{s}"] = f"{COH[ch][s]:.3f}"
    rc = []
    for k in paper:
        pop = "ALLPAIRS7803" if k.endswith("all") else ("cohort pairs" if k.startswith("ptcoh") else "NAT1")
        rc.append(dict(item="numbers.tex macro", name="\\" + k, paper=paper[k], ours=ours.get(k), match=paper[k] == ours.get(k), population=pop,
                       value_if_pooled_is_ALL7326=alt.get(k, "") if k.endswith("all") else "",
                       match_if_ALL7326=(alt.get(k) == paper[k]) if k.endswith("all") else ""))
    rows_tex = {}
    for line in (PAPER / "tables/stages.tex").read_text().splitlines():
        if "&" in line and not line.strip().startswith("Stage"):
            c = [x.strip().rstrip("\\").strip() for x in line.split("&")]; rows_tex[c[0]] = c[1:]
    for s, lab in (("mid", "Mid-training"), ("sft", "+ SFT"), ("dpo", "+ SFT + DPO")):
        v = get("stage", "NAT1", s)
        for col, pv, ov in zip(("tau", "SD", "d_z", "AUC", "TPR@1%FPR (%)"), rows_tex[lab],
                               (f"{v.tau:+.4f}", f"{v.sd:.4f}", f"{v.dz:.2f}", f"{v.auc:.3f}", f"{100*v.tpr1:.1f}")):
            rc.append(dict(item="stages.tex cell", name=f"{lab} | {col}", paper=pv, ours=ov, match=pv == ov, population="NAT1"))
    cnt = {}
    for line in (PAPER / "tables/numbers.tex").read_text().splitlines():
        g = re.match(r"\\newcommand\{\\(npairs[a-zA-Z]*)\}\{([^}]*)\}", line)
        if g: cnt[g.group(1)] = g.group(2)
    for k, pop in (("npairsnat", "NAT1"), ("npairs", "ALLPAIRS7803"), ("npairspooled", "ALL7326")):
        if k in cnt:
            n = str(int(get("stage", pop, "mid").n))
            rc.append(dict(item="pair count", name="\\" + k, paper=cnt[k], ours=n, match=cnt[k] == n, population=pop))
    RC = pd.DataFrame(rc); RC.to_csv(FD / "reproduction_check.csv", index=False)
    print(f"reproduction: {RC.match.sum()}/{len(RC)} match; *all under ALL7326 would match {sum(1 for r in rc if r.get('match_if_ALL7326') is True)}/24")

# ------------------------------------------------------------ proposed_macros.tex
TOK = {"mid": "mid", "sft": "sft", "dpo": "dpo", "rlvr1": "rlone", "rlvr2i": "instruct", "rlvr2": "rltwo"}
SUF = {"NAT1": "", "ALLPAIRS7803": "all", "ALL7326": "pooled"}
L = ["% Proposed macros for the RL stages of Q7 (sec:results-posttrain). Generated by",
     "% analysis/rlvr/finalize.py; NOT inserted into the paper.",
     "% Stage tokens: rlone = RLVR1 final (step 2604); instruct = RLVR2 step 2000 (analogue of the released",
     "% OLMo-2-0425-1B-Instruct); rltwo = RLVR2 final (step 2604). Population suffix: none = NAT1 (4,832 pairs);",
     "% 'all' = every complete pair (7,803; the population of the existing \\pt*all macros); 'pooled' = all but X and F",
     "% (7,326; the population the Q7 prose names and the current generator's ALL). Choose ONE pooled population",
     "% for the paper; the 'pooled' versions of the existing mid/sft/dpo macros are given for that switch.",
     f"% CIs: pair bootstrap, B={B}, numpy default_rng({SEED}), percentile 95%. Retentions are % of the mid-training value",
     "% (aucretain: of the AUC advantage over 0.5); tprgain = relative TPR change vs mid-training (%); corr = Pearson r of the",
     "% per-pair contrast with mid-training; corrprev = with the previous stage (rlone: DPO; instruct and rltwo: RLVR1).",
     "% Lines starting with '%%exists' are already in numbers.tex with exactly this value."]
def emit(name, val):
    if name in paper:
        assert paper[name] == val or name.endswith("all"), (name, paper[name], val)
        L.append(f"%%exists \\newcommand{{\\{name}}}{{{val}}}" + ("" if paper[name] == val else f"   % paper has {paper[name]}"))
    else:
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")
for pop, suf in SUF.items():
    L.append(f"\n% ---- {pop}: {POPDESC[pop]}")
    for s, *_ in STAGES:
        r = get("stage", pop, s); t = TOK[s]
        emit(f"pt{t}tau{suf}", f"{r.tau:+.4f}"); emit(f"pt{t}sd{suf}", f"{r.sd:.4f}"); emit(f"pt{t}dz{suf}", f"{r.dz:.2f}")
        emit(f"pt{t}auc{suf}", f"{r.auc:.3f}"); emit(f"pt{t}tpr{suf}", f"{100*r.tpr1:.1f}"); emit(f"pt{t}pos{suf}", f"{100*r.pos:.1f}")
        if s in ("rlvr1", "rlvr2i", "rlvr2"):
            emit(f"pt{t}taulo{suf}", f"{r.tau_lo:+.4f}"); emit(f"pt{t}tauhi{suf}", f"{r.tau_hi:+.4f}")
            emit(f"pt{t}dzlo{suf}", f"{r.dz_lo:.2f}"); emit(f"pt{t}dzhi{suf}", f"{r.dz_hi:.2f}")
            emit(f"pt{t}auclo{suf}", f"{r.auc_lo:.3f}"); emit(f"pt{t}auchi{suf}", f"{r.auc_hi:.3f}")
            emit(f"pt{t}tprlo{suf}", f"{100*r.tpr1_lo:.1f}"); emit(f"pt{t}tprhi{suf}", f"{100*r.tpr1_hi:.1f}")
        if s != "mid":
            emit(f"pt{t}corr{suf}", f"{r.corr_mid:.2f}")
            emit(f"pt{t}retain{suf}", f"{r.ret_tau:.0f}"); emit(f"pt{t}sdretain{suf}", f"{r.ret_sd:.0f}")
            emit(f"pt{t}dzretain{suf}", f"{r.ret_dz:.0f}"); emit(f"pt{t}aucretain{suf}", f"{r.ret_auc:.0f}")
            emit(f"pt{t}tprgain{suf}", f"{r.tpr_change:+.0f}")
        if s in ("rlvr1", "rlvr2i", "rlvr2"):
            emit(f"pt{t}corrprev{suf}", f"{r.corr_prev:.2f}")
            emit(f"pt{t}retainlo{suf}", f"{100*r.ret_tau_lo:.0f}"); emit(f"pt{t}retainhi{suf}", f"{100*r.ret_tau_hi:.0f}")
            emit(f"pt{t}aucretainlo{suf}", f"{100*r.ret_auc_lo:.0f}"); emit(f"pt{t}aucretainhi{suf}", f"{100*r.ret_auc_hi:.0f}")
    r = get("stage", pop, "rlvr2"); emit(f"ptrltwocorrinstruct{suf}", f"{r.corr_rlvr2step2000:.2f}")
    # paired change DPO -> RL stage
    for s in ("rlvr1", "rlvr2i", "rlvr2"):
        q = RLD[(RLD.population == pop) & (RLD.stage == s)].iloc[0]; t = TOK[s]
        emit(f"pt{t}dauc{suf}", f"{q.d_auc:+.3f}"); emit(f"pt{t}dauclo{suf}", f"{q.d_auc_lo:+.3f}"); emit(f"pt{t}dauchi{suf}", f"{q.d_auc_hi:+.3f}")
        emit(f"pt{t}ddz{suf}", f"{q.d_dz:+.2f}"); emit(f"pt{t}ddzlo{suf}", f"{q.d_dz_lo:+.2f}"); emit(f"pt{t}ddzhi{suf}", f"{q.d_dz_hi:+.2f}")
        emit(f"pt{t}dtpr{suf}", f"{100*q.d_tpr1:+.1f}"); emit(f"pt{t}dtprlo{suf}", f"{100*q.d_tpr1_lo:+.1f}"); emit(f"pt{t}dtprhi{suf}", f"{100*q.d_tpr1_hi:+.1f}")
    # trajectory ranges over the 28 RL checkpoints (DPO end excluded)
    T = TR[(TR.population == pop) & (TR.rl_stage != "dpo_end")]
    for nm, col, f in (("retain", "ret_tau_pct", ".0f"), ("tau", "tau", "+.4f"), ("dz", "dz", ".2f"), ("auc", "auc", ".3f"),
                       ("tpr", "tpr1_pct", ".1f"), ("corr", "corr_with_mid", ".2f"), ("dzretain", "ret_dz_pct", ".0f"),
                       ("aucretain", "ret_aucadv_pct", ".0f")):
        emit(f"ptrltraj{nm}min{suf}", format(T[col].min(), f)); emit(f"ptrltraj{nm}max{suf}", format(T[col].max(), f))
    emit(f"ptrltrajcorrprevmin{suf}", format(T.corr_with_prev_checkpoint.min(), ".2f"))
L.append("\n% ---- per-cohort AUC (pairs of the cohort, all doses; as the existing \\ptcoh* macros)")
for k, ch in CK.items():
    for s, *_ in STAGES: emit(f"ptcoh{k}{TOK[s]}", f"{COH[ch][s]:.3f}")
L.append("\n% ---- ROBUSTNESS: AI2's released post-trained models as the control, NAT1 (L(ours injected) - L(AI2) at the same stage)")
for s, *_ in AI2:
    r = get("ai2", "NAT1", s); p = get("placebo", "NAT1", s); t = TOK[s]
    for nm, v in (("tau", f"{r.tau:+.4f}"), ("taulo", f"{r.tau_lo:+.4f}"), ("tauhi", f"{r.tau_hi:+.4f}"), ("sd", f"{r.sd:.4f}"),
                  ("dz", f"{r.dz:.2f}"), ("auc", f"{r.auc:.3f}"), ("tpr", f"{100*r.tpr1:.1f}"), ("retain", f"{r.ret_tau:.0f}"),
                  ("corr", f"{r.corr_mid:.2f}"), ("corrmatched", f"{r.corr_main_design:.2f}")):
        emit(f"ptaictl{t}{nm}", v)
    if s != "rlvr2":
        for nm, v in (("tau", f"{p.tau:+.4f}"), ("taulo", f"{p.tau_lo:+.4f}"), ("tauhi", f"{p.tau_hi:+.4f}"), ("sd", f"{p.sd:.4f}"), ("auc", f"{p.auc:.3f}")):
            emit(f"ptaiplacebo{t}{nm}", v)
for s, d in DRIFT.items():
    t = TOK[s]; emit(f"ptaidrift{t}mean", f"{d['NAT1_docs']['mean']:+.3f}"); emit(f"ptaidrift{t}sd", f"{d['NAT1_docs']['sd']:.3f}")
names = [re.search(r"\\newcommand\{\\([a-z]+)\}", l).group(1) for l in L if l.startswith("\\newcommand")]
assert len(names) == len(set(names)), "duplicate macro"
assert all(re.fullmatch(r"[a-z]+", n) for n in names)
(FD / "proposed_macros.tex").write_text("\n".join(L) + "\n")
print(f"proposed macros: {len(names)} new, {sum(1 for l in L if l.startswith('%%exists'))} existing (commented)")
