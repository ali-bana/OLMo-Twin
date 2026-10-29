#!/usr/bin/env python3
"""Regenerate the paper's figures, tables and every number the prose cites.

Writes vector PDFs at IEEE two-column geometry into <out>/figures/, the
tables/*.tex fragments into <out>/tables/, and <out>/tables/numbers.tex, a
macro file so that no number in the prose is typed by hand. <out> is the
--out directory (default: paper/ under the data root, see olmo_twin.paths).
Everything is recomputed from the analysis tables under the data root
(analysis/). The headline numbers are also written to
analysis/paper_numbers.json, which later analysis steps read.

This module is also the library of the analysis scripts (load, population,
headline, tpr_at, ...); importing it does not parse the command line, so
importers always see the default output directory.

Usage:
    python figures/paper_figures.py [--out DIR] [--first-pass]

--first-pass is for a fresh data root: collection_by_domain.py and
exclusion_sensitivity.py read analysis/paper_numbers.json, while the full run
needs their JSON outputs (and those of screening_analysis.py,
cohortF_2x2_check.py and length_truncation_analysis.py) for checklist(). The
first pass skips checklist() and write_tables() and writes paper_numbers.json
with the same NAT1, ALL, E and coll entries as the full run; run the five
analyses, then this script again without --first-pass (analysis/README.md).
figures/published_figures.py afterwards redraws the data figures exactly as
published (figures/README.md).

Populations (fixed on 2026-10-12):

  NAT1   the primary member-vs-non-member population: single-copy pairs in the
         cohorts whose documents are real text verified absent from both
         training stages -- M (post-cutoff arXiv), L (arXiv prefixes), W-new,
         W-gap (Wikipedia), S-se (StackExchange). 4,832 pairs. Cohort X
         (synthetic anchors, unstable between runs) and cohort F (failed
         placebo) are excluded; the E family (near-duplicates of trained
         text, by design) is reported separately as re-exposure.
  ALL    the pooled "every exposure" view: every analysed pair except cohorts
         X (synthetic anchors, unstable between runs) and F (failed placebo),
         7,326 pairs; ALLPAIRS is every complete pair (7,803), used only for
         counts. Methods promises that X and F enter no pooled negative set,
         and since 2026-10-13 the generator keeps that promise for ALL too.

All AUC / TPR numbers use the matched held-out partners of the same population
as negatives. The post-training exclusions (10 pairs, 16 documents) are applied
by ps.load_table everywhere.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root (olmo_twin)
from olmo_twin import paths  # noqa: E402
paths.add_code_dirs("analysis")   # injection_v2_plotstyle, trajectory_analysis
import injection_v2_plotstyle as ps  # noqa: E402
import trajectory_analysis as traj  # noqa: E402

ANA = paths.ANALYSIS
TABLE = ANA / "per_document_attacks_prelim.csv.gz"
LADDER = ANA / "ladder_trajectory.csv.gz"
LADDER_META = ANA / "ladder_meta.json"
NEFF = ANA / "token_neff.npz"
CALIB_CSV = ANA / "per_document_attacks_calibrated.csv.gz"   # analysis/calibrated_attacks.py
CALIB_JSON = ANA / "calibrated_attacks_summary.json"
BUNDLE = paths.BUNDLE
RUN = paths.RUN
PT = {"sft": (RUN / "scores_posttrain/sft-injected/scores.csv.gz",
              RUN / "scores_posttrain/sft-baseline/scores.csv.gz"),
      "dpo": (RUN / "scores_posttrain/dpo-injected/scores.csv.gz",
              RUN / "scores_posttrain/dpo-baseline/scores.csv.gz")}
PAPER = paths.PAPER_OUT      # output directory; --out rebinds it (set_output)
FIGS = PAPER / "figures"
TABS = PAPER / "tables"

COL = 3.40          # inches, one IEEE column
FULL = 7.16         # inches, both columns
C = ps.C
NAT = {"M", "L", "W-new", "W-gap", "S-se"}
EFAM = {"E", "E0", "E0-D"}
RNG = np.random.default_rng(20261012)
Z95, Z80 = 1.6449, 0.8416     # one-sided 0.05, power 0.80

TIERS = [  # label, column, colour, linestyle, marker
    ("no reference (surprise)", "injected__surprise", C["bad"], ":", "x"),
    ("published checkpoint", "ratio_stage1", C["pretrained"], "-.", "^"),
    ("retrained reference", "ratio_sibling", C["accent"], "--", "s"),
    ("exact counterfactual", "ratio_oracle", C["injected"], "-", "o"),
]


def setup():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.8, "legend.frameon": False,
        "legend.handlelength": 1.8, "legend.columnspacing": 1.0,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.1, "lines.markersize": 3.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#dddddd", "grid.alpha": 0.9,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    })


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / f"{name}.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.relative_to(PAPER)}")


def _texesc(s):
    return re.sub(r"(?<!\\)%", r"\\%", str(s))


def fmt_int(n):
    return f"{int(n):,}".replace(",", "{,}")


# ------------------------------------------------------------------ stats
def tpr_at(pos, neg, fpr):
    """TPR at the threshold that admits floor(fpr*n_neg) negatives. Returns
    (tpr, k, n_neg); k=0 means the threshold is the top negative score, i.e.
    the operating point is below the resolution floor 1/n_neg."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = np.sort(neg[np.isfinite(neg)])
    k = int(math.floor(fpr * neg.size))
    thr = neg[neg.size - k - 1]
    return float((pos > thr).mean()), k, int(neg.size)


def signflip_p(diff, nperm=100000):
    """Exact-randomisation (sign-flip) one-sided p for mean(diff) > 0, Monte
    Carlo with nperm draws; returns (p_upper_bound, z_analytic)."""
    d = diff[np.isfinite(diff)]
    obs = d.mean()
    sd_null = math.sqrt((d ** 2).sum()) / d.size
    z = obs / sd_null
    hits = 0
    B = 2000
    for _ in range(nperm // B):
        signs = RNG.choice([-1.0, 1.0], size=(B, d.size))
        hits += int(((signs * d).mean(axis=1) >= obs).sum())
    return (hits + 1) / (nperm + 1), z


def fdr_named(pos, neg, q):
    """Number of positives nameable at FDR<=q with the (v+1)/(n+1) envelope."""
    pos = np.sort(pos[np.isfinite(pos)])[::-1]; neg = np.sort(neg[np.isfinite(neg)])[::-1]
    k = np.arange(1, pos.size + 1)
    v = np.searchsorted(-neg, -pos, side="left").astype(float)
    fdr = np.minimum(1.0, (v + 1.0) * (pos.size / (neg.size + 1.0)) / k)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    ok = np.nonzero(fdr <= q)[0]
    return int(k[ok].max()) if ok.size else 0


def auc_ci_pairs(a, b, n=400):
    """Percentile 95% interval for the rank AUC, resampling PAIRS (member and
    partner move together), which is the paper's bootstrap convention."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    bs = [ps.auc(a[j], b[j]) for j in (RNG.integers(0, a.size, a.size) for _ in range(n))]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def collection_auc(a, b, ms, reps_floor=400, total=40000):
    out = []
    for m in ms:
        reps = max(reps_floor, int(total / m))
        pa = a[RNG.integers(0, a.size, (reps, m))].mean(axis=1)
        pb = b[RNG.integers(0, b.size, (reps, m))].mean(axis=1)
        out.append((m, ps.auc(pa, pb)))
    return out


def n80(dz):
    return int(math.ceil(((Z95 + Z80) / dz) ** 2))


def n80_resampled(d, draws=20000, seed=20261012):
    """Smallest n at which a one-sided paired t-test at alpha 0.05 rejects in at
    least 80% of `draws` resamples of n pairs (with replacement) from the
    observed contrasts d. Used for the standardized ratio, whose d_z is large
    enough that n is small and the normal approximation n80() is too optimistic
    (it ignores the heavier tails of the t distribution). Uses its own
    generator, so the shared stream RNG is not consumed."""
    from scipy import stats as _st
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    n = 2
    while True:
        x = d[rng.integers(0, d.size, (draws, n))]
        with np.errstate(divide="ignore", invalid="ignore"):  # a draw of n identical pairs has SD 0
            t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / math.sqrt(n))
        if (t > _st.t.ppf(0.95, n - 1)).mean() >= 0.80:
            return n
        n += 1


# ------------------------------------------------------------------- load
CALIB_COLS = {  # calibrated_attacks.py column -> name used here (control == exact counterfactual)
    "zratio_control_ac": "z_oracle", "zratio_sibling_ac": "z_sibling", "zratio_stage1_ac": "z_stage1",
    "zratio_control_T": "zT_oracle", "zratio_sibling_T": "zT_sibling", "zratio_stage1_T": "zT_stage1",
    "rmia_control_gm_selected_NAT1": "rmia_oracle", "rmia_sibling_gm_selected_NAT1": "rmia_sibling",
    "rmia_stage1_gm_selected_NAT1": "rmia_stage1", "split": "split",
}


def merge_calibrated(cols):
    """Attach the per-document standardized-ratio and RMIA scores written by
    analysis/calibrated_attacks.py, keyed by doc_id, and check that its
    reference ratio agrees with the analysis table."""
    if not CALIB_CSV.exists():
        print(f"  [calibrated] {CALIB_CSV.name} missing; standardized-ratio rows skipped")
        return
    rec = {}
    with gzip.open(CALIB_CSV, "rt") as f:
        for r in csv.DictReader(f):
            rec[r["doc_id"]] = r
    docid = cols["doc_id"]
    for src, dst in CALIB_COLS.items():
        vals = [rec.get(d, {}).get(src, "nan") for d in docid]
        cols[dst] = np.array(vals) if dst == "split" else np.array([float(v) if v not in ("", None) else np.nan for v in vals])
    dev = cols["split"] == "dev"
    for dst in CALIB_COLS.values():
        if dst.startswith("rmia_"):
            cols[dst][dev] = np.nan   # dev rows are leave-one-out scores used for selection
    chk = np.array([float(rec.get(d, {}).get("ratio_control") or "nan") for d in docid])
    ok = np.isfinite(chk) & np.isfinite(np.asarray(cols["ratio_oracle"], float))
    gap = float(np.max(np.abs(chk[ok] - np.asarray(cols["ratio_oracle"], float)[ok])))
    assert gap < 1e-9, f"calibrated CSV disagrees with the analysis table (max |diff| {gap})"
    print(f"  [calibrated] merged {int(ok.sum()):,} documents from {CALIB_CSV.name}")


def load():
    cols, rows = ps.load_table(TABLE)
    merge_calibrated(cols)
    arm = cols["arm"]; pid = cols["pair_id"]
    byp = defaultdict(dict)
    for i, p in enumerate(pid):
        if isinstance(p, str) and p:
            byp[p][arm[i]] = i
    pairs = [(v["inject"], v["holdout"]) for v in byp.values()
             if "inject" in v and "holdout" in v]
    return cols, rows, arm, pairs


def population(cols, pairs, which):
    coh, r = cols["cohort"], cols["r"]
    if which == "NAT1":
        return [(i, h) for i, h in pairs if r[i] == 1 and coh[i] in NAT]
    if which == "ALL":
        return [(i, h) for i, h in pairs if coh[i] not in ("X", "F")]
    if which == "ALLPAIRS":
        return pairs
    if which == "ALLnoX":
        return [(i, h) for i, h in pairs if coh[i] != "X"]
    if which == "E":
        return [(i, h) for i, h in pairs if coh[i] in EFAM]
    raise ValueError(which)


def arrays(cols, key, sel):
    a = np.array([cols[key][i] for i, _ in sel], float)
    b = np.array([cols[key][h] for _, h in sel], float)
    return a, b


def headline(cols, sel, key="ratio_oracle"):
    a, b = arrays(cols, key, sel)
    d = a - b; d = d[np.isfinite(d)]
    lo, hi = ps.bootstrap_ci(d)
    alo, ahi = auc_ci_pairs(a, b, n=400)
    t1, k1, nn = tpr_at(a, b, 0.01)
    t01, k01, _ = tpr_at(a, b, 0.001)
    return dict(n=int(d.size), tau=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(d.size)),
                lo=float(lo), hi=float(hi), dz=float(d.mean() / d.std(ddof=1)),
                sd=float(d.std(ddof=1)), pos=float((d > 0).mean()),
                auc=float(ps.auc(a, b)), auc_lo=float(alo), auc_hi=float(ahi),
                tpr1=t1, k1=k1, tpr01=t01, k01=k01, nneg=nn)


def calibrated(cols, selN):
    """Reference-calibrated attacks from analysis/calibrated_attacks.py:
    the per-document standardized ratio and offline RMIA. The standardized
    ratio has no tuned hyperparameter, so it is also reported on the whole
    population like every other statistic; the RMIA comparison selects (a,
    gamma) on the dev split and is read from the summary JSON on the holdout
    split. Writes tables/calibrated.tex and returns (name, value) macros."""
    out = []
    if "z_oracle" not in cols or not CALIB_JSON.exists():
        return out
    S = json.loads(CALIB_JSON.read_text())
    REF = (("orc", "control", "Exact counterfactual"), ("sib", "sibling", "Retrained reference"),
           ("pub", "stage1", "Published checkpoint"))
    ATT = (("ratio", "Ratio"), ("rmia", "RMIA, dev-selected"), ("rmiadef", "RMIA, defaults"),
           ("z", "Standardized ratio"))

    def pick(pop, ref, att):
        R = S["results"][pop]
        return {"ratio": lambda: R["plain_ratio"][ref],
                "rmia": lambda: R["rmia"][ref]["gm"]["selected"],
                "rmiadef": lambda: R["rmia"][ref]["gm"]["default"],
                "rmiaseq": lambda: R["rmia"][ref]["seq"]["selected"],
                "z": lambda: R["zratio"][ref]["ac"],
                "zT": lambda: R["zratio"][ref]["T"]}[att]()

    ev = S["meta"]["evaluation"]
    out += [("calibn", fmt_int(ev["n_pairs"]["NAT1"]["holdout"])), ("calibdev", fmt_int(ev["n_pairs"]["NAT1"]["dev"])),
            ("calibnall", fmt_int(ev["n_pairs"]["ALL"]["holdout"])),
            ("calibzsize", fmt_int(S["results"]["NAT1"]["rmia"]["control"]["gm"]["population_Z"]["size"])),
            ("calibkneg", str(pick("NAT1", "control", "ratio")["tpr1_k_negatives"])),
            ("calibdraws", str(ev["bootstrap"]["draws"]))]
    L = [r"% generated by figures/paper_figures.py -- do not edit by hand",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"Attack & \auc\ [95\% CI] & \tprat{1} [95\% CI] & per domain \\", r"\midrule"]
    for k, ref, glab in REF:
        L.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{glab}}}}} \\\\")
        for att, alab in ATT:
            v = pick("NAT1", ref, att)
            d3 = lambda x: f"{x:.3f}"[1:]   # .761 rather than 0.761 inside the interval
            L.append(f"\\quad {alab} & {v['auc']:.3f} {{\\scriptsize[{d3(v['auc_ci'][0])},{d3(v['auc_ci'][1])}]}} & "
                     f"{100*v['tpr1']:.1f} {{\\scriptsize[{100*v['tpr1_ci'][0]:.1f},{100*v['tpr1_ci'][1]:.1f}]}} & "
                     f"{100*v['tpr1_strat']:.1f} \\\\")
            for pop, suf in (("NAT1", ""), ("ALL", "all")):
                v = pick(pop, ref, att)
                out += [(f"calib{att}{k}auc{suf}", f"{v['auc']:.3f}"),
                        (f"calib{att}{k}auclo{suf}", f"{v['auc_ci'][0]:.3f}"), (f"calib{att}{k}auchi{suf}", f"{v['auc_ci'][1]:.3f}"),
                        (f"calib{att}{k}tpr{suf}", f"{100*v['tpr1']:.1f}"),
                        (f"calib{att}{k}tprlo{suf}", f"{100*v['tpr1_ci'][0]:.1f}"), (f"calib{att}{k}tprhi{suf}", f"{100*v['tpr1_ci'][1]:.1f}"),
                        (f"calib{att}{k}strat{suf}", f"{100*v['tpr1_strat']:.1f}")]
                if "delta_auc_vs_plain" in v:
                    out += [(f"calib{att}{k}dauc{suf}", f"{v['delta_auc_vs_plain']:+.3f}"),
                            (f"calib{att}{k}dauclo{suf}", f"{v['delta_auc_vs_plain_ci'][0]:+.3f}"),
                            (f"calib{att}{k}dauchi{suf}", f"{v['delta_auc_vs_plain_ci'][1]:+.3f}"),
                            (f"calib{att}{k}dtpr{suf}", f"{100*v['delta_tpr1_vs_plain']:+.1f}"),
                            (f"calib{att}{k}dtprlo{suf}", f"{100*v['delta_tpr1_vs_plain_ci'][0]:+.1f}"),
                            (f"calib{att}{k}dtprhi{suf}", f"{100*v['delta_tpr1_vs_plain_ci'][1]:+.1f}")]
        if k != "pub":
            L.append(r"\addlinespace")
        hp = S["results"]["NAT1"]["rmia"][ref]["gm"]["selected_hyperparameters"]
        out += [(f"calibrmiaa{k}", f"{hp['a']:g}"), (f"calibrmiagamma{k}", f"{hp['gamma']:.2f}")]
        for att in ("zT", "rmiaseq"):
            v = pick("NAT1", ref, att)
            out += [(f"calib{att}{k}auc", f"{v['auc']:.3f}"), (f"calib{att}{k}tpr", f"{100*v['tpr1']:.1f}"),
                    (f"calib{att}{k}strat", f"{100*v['tpr1_strat']:.1f}")]
    L += [r"\bottomrule", r"\end{tabular}"]
    TABS.mkdir(parents=True, exist_ok=True)
    (TABS / "calibrated.tex").write_text("\n".join(L) + "\n")
    # null behaviour of the standardized ratio on the whole population (the
    # holdout-split values live in the JSON; the prose quotes whole-NAT1 numbers)
    za, zb = arrays(cols, "z_oracle", selN); zb = zb[np.isfinite(zb)]; za = za[np.isfinite(za)]
    out += [("znullmean", f"{zb.mean():+.1f}"), ("znullsd", f"{zb.std(ddof=1):.1f}"),
            ("znullshare", f"{100*(zb > 2.326).mean():.0f}"), ("zmemmean", f"{za.mean():+.1f}")]
    # geometric-mean token probability of the treated model on the population's documents
    gm = np.exp(np.concatenate(arrays(cols, "injected__loss", selN)))
    out += [("gmproblo", f"{np.percentile(gm, 1):.2f}"), ("gmprobhi", f"{np.percentile(gm, 99):.2f}")]
    dg = S["sanity"]["d_dev_vs_holdout_auc"]["NAT1"]
    out += [("calibdevgapratio", f"{dg['plain_control']['dev_minus_holdout']:+.3f}"),
            ("calibdevgaprmia", f"{dg['rmia_control_gm_selected']['dev_minus_holdout']:+.3f}")]
    ne = S["meta"]["neff_summary"]["control"]
    out += [("calibtaumedian", f"{ne['median_tau']:.2f}"), ("calibneffshare", f"{100*ne['median_neff_over_T']:.0f}")]
    g = S["results"]["NAT1"]["rmia"]["control"]["gm"]["grid"]
    assert len(g["dev_auc"]) == len(g["a"]) and len(g["dev_auc"][0]) == len(g["log_gamma"])
    ia, i1, ig = g["a"].index(0.3), g["a"].index(1.0), g["log_gamma"].index(0.0)
    out += [("rmiadevaucathree", f"{g['dev_auc'][ia][ig]:.3f}"), ("rmiadevaucaone", f"{g['dev_auc'][i1][ig]:.3f}")]
    # whole-population numbers for the standardized ratio, like every other statistic
    for k, key in (("zorc", "z_oracle"), ("zsib", "z_sibling"), ("zpub", "z_stage1")):
        h = headline(cols, selN, key)
        out += [(f"auc{k}nat", f"{h['auc']:.3f}"), (f"auc{k}natlo", f"{h['auc_lo']:.3f}"), (f"auc{k}nathi", f"{h['auc_hi']:.3f}"),
                (f"tprone{k}nat", f"{100*h['tpr1']:.1f}"), (f"tprpointone{k}nat", f"{100*h['tpr01']:.2f}"),
                (f"dz{k}nat", f"{h['dz']:.2f}"), (f"pos{k}nat", f"{100*h['pos']:.1f}")]
        a, b = arrays(cols, key, selN)
        out.append((f"npower{k}", str(n80_resampled(a - b))))
        bs = [tpr_at(a[j], b[j], 0.01)[0] for j in (RNG.integers(0, a.size, a.size) for _ in range(400))]
        out += [(f"tprone{k}natlo", f"{100*np.percentile(bs, 2.5):.1f}"), (f"tprone{k}nathi", f"{100*np.percentile(bs, 97.5):.1f}")]
        for q, nmq in ((0.05, "five"), (0.20, "twenty")):
            out.append((f"fdr{nmq}{k}", str(fdr_named(a, b, q))))
    coh = cols["cohort"]

    def strat_tpr(key, sel):
        tp = tot = 0
        for ch in sorted(set(coh[i] for i, _ in sel)):
            s_ = [(i, h) for i, h in sel if coh[i] == ch]
            aa, bb = arrays(cols, key, s_)
            tp += tpr_at(aa, bb, 0.01)[0] * len(s_); tot += len(s_)
        return tp / tot

    for k, key in (("orc", "ratio_oracle"), ("zorc", "z_oracle"), ("sib", "ratio_sibling"), ("zsib", "z_sibling")):
        bs = []
        for _ in range(400):
            j = RNG.integers(0, len(selN), len(selN))
            bs.append(strat_tpr(key, [selN[t] for t in j]))
        out += [(f"tprstrat{k}lo", f"{100*np.percentile(bs, 2.5):.1f}"), (f"tprstrat{k}hi", f"{100*np.percentile(bs, 97.5):.1f}")]
    # RMIA's score is a quantile against Z, so a threshold set on Z transfers: FPR among the
    # holdout partners at the 0.99 quantile, holdout rows only (dev rows are blanked above)
    selH = [(i, h) for i, h in selN if cols["split"][i] == "holdout"]
    a, b = arrays(cols, "rmia_oracle", selH)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    out += [("rmiathrfpr", f"{100*(b >= 0.99).mean():.1f}"), ("rmiathrtpr", f"{100*(a >= 0.99).mean():.1f}"),
            ("calibnholdout", fmt_int(len(selH)))]
    # whom the pooled 1% threshold catches: count and arXiv share, ratio vs standardized
    for k, key in (("ratio", "ratio_oracle"), ("z", "z_oracle")):
        a, b = arrays(cols, key, selN)
        kneg = int(math.floor(0.01 * b.size)); thr = np.sort(b)[b.size - kneg - 1]
        caught = a > thr
        isM = np.array([coh[i] == "M" for i, _ in selN])
        out += [(f"{k}pooledcaught", str(int(caught.sum()))), (f"{k}pooledcaughtm", str(int((caught & isM).sum())))]
    print(f"  tables/calibrated.tex ({len(out)} calibrated macros)")
    return out


def validation(cols, selN):
    """Design-validation macros: the randomization check. The control model
    never saw either arm, so its own mean log-probability must not separate
    injected members from their partners (AUC 0.5 up to sampling noise)."""
    out = []
    coh = cols["cohort"]
    a, b = arrays(cols, "seed42__loss", selN)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    lo, hi = auc_ci_pairs(a, b, n=400)
    out += [("placeboauc", f"{ps.auc(a, b):.3f}"), ("placeboauclo", f"{lo:.3f}"), ("placeboauchi", f"{hi:.3f}")]
    per = {}
    for ch in sorted(set(coh[i] for i, _ in selN)):
        s_ = [(i, h) for i, h in selN if coh[i] == ch]
        aa, bb = arrays(cols, "seed42__loss", s_); ok = np.isfinite(aa) & np.isfinite(bb)
        per[ch] = float(ps.auc(aa[ok], bb[ok]))
        out.append((f"placebo{re.sub(r'[^a-z0-9]', '', ch.lower())}", f"{per[ch]:.3f}"))
    out += [("placebocohlo", f"{min(per.values()):.3f}"), ("placebocohhi", f"{max(per.values()):.3f}")]
    return out


PREV = {"0.002": "pointtwo", "0.01": "one", "0.05": "five", "0.2": "twenty", "0.5": "fifty", "1.0": "all"}
COHK = {"M": "m", "L": "l", "W-new": "wnew", "W-gap": "wgap", "S-se": "sse", "arXiv (M+L)": "arxiv",
        "Wikipedia (W-new+W-gap)": "wiki", "NAT1 pooled": "pooled", "pooled": "pooled"}


def checklist(cols, selN):
    """Macros and tables from the 2026-10-13 checklist analyses, each a
    separate script with its own JSON: screening_analysis (cross-fitted
    operating points, mixed-pool screening), collection_by_domain, exclusion
    sensitivity, cohortF_2x2_check and length_truncation."""
    out = []
    f3 = lambda x: f"{x:.3f}"
    pct1 = lambda x: f"{100*x:.1f}"
    # ----------------------------------------------------- screening
    S = json.loads((ANA / "screening_analysis.json").read_text())
    A = S["part_a"]["population"]["NAT1"]
    for sk, key in (("ratio", "ratio_oracle"), ("z", "z_oracle"), ("sib", "ratio_sibling"), ("zsib", "z_sibling")):
        P = A["pooled"][key]["K5"]
        for fk, fname in (("1%", ""), ("0.1%", "pointone")):
            v = P[fk]
            pc = pct1 if fk == "1%" else (lambda x: f"{100*x:.2f}")
            out += [(f"xfittpr{fname}{sk}", pc(v["tpr"]["mean"])), (f"xfittpr{fname}{sk}replo", pc(v["tpr"]["lo"])),
                    (f"xfittpr{fname}{sk}rephi", pc(v["tpr"]["hi"])), (f"xfitfpr{fname}{sk}", f"{100*v['realized_fpr']['mean']:.2f}"),
                    (f"xfitk{fname}{sk}", str(v["k_values"][0])), (f"xfitncal{fname}{sk}", fmt_int(v["n_cal_range"][0]))]
            if "pair_bootstrap" in v:
                b = v["pair_bootstrap"]
                out += [(f"xfittpr{fname}{sk}lo", pct1(b["tpr_ci"]["lo"])), (f"xfittpr{fname}{sk}hi", pct1(b["tpr_ci"]["hi"])),
                        (f"xfitdtpr{fname}{sk}", f"{100*b['tpr_change_ci']['mean']:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00")),
                        (f"xfitdtpr{fname}{sk}lo", f"{100*b['tpr_change_ci']['lo']:+.2f}"), (f"xfitdtpr{fname}{sk}hi", f"{100*b['tpr_change_ci']['hi']:+.2f}")]
        tp = tot = 0.0
        for ch in ("M", "L", "W-new", "W-gap", "S-se"):
            c = A[ch][key]; n = c["n_pairs"]; v = c["K5"]["1%"]
            out += [(f"xfit{COHK[ch]}tpr{sk}", pct1(v["tpr"]["mean"])), (f"xfit{COHK[ch]}fpr{sk}", f"{100*v['realized_fpr']['mean']:.2f}")]
            tp += v["tpr"]["mean"] * n; tot += n
        out.append((f"xfittprstrat{sk}", pct1(tp / tot)))
    Bp = S["part_b"]
    for lab, unit, dom in (("m", "M", "arXiv"), ("wiki", "Wikipedia (W-new+W-gap)", "Wikipedia")):
        for sk, key in (("ratio", "ratio_oracle"), ("z", "z_oracle")):
            v = Bp[dom]["units"][unit][key]["1%"]
            out += [(f"nullcal{lab}fpr{sk}", f"{100*v['realized_fpr_partners']:.2f}"), (f"nullcal{lab}tpr{sk}", pct1(v["tpr_members"]))]
    Cc = S["part_c"]
    for sk, key in (("ratio", "ratio_oracle"), ("z", "z_oracle"), ("sib", "ratio_sibling")):
        for pv, pname in PREV.items():
            if pv == "1.0":
                continue
            u = Cc["units"]["pooled"][key][pv]["0.05"]; t = u["per_repeat_totals"]
            out += [(f"screen{sk}{pname}true", f"{t['true_discoveries']['mean']:.1f}"), (f"screen{sk}{pname}false", f"{t['false_discoveries']['mean']:.1f}"),
                    (f"screen{sk}{pname}truelo", f"{t['true_discoveries']['lo']:.0f}"), (f"screen{sk}{pname}truehi", f"{t['true_discoveries']['hi']:.0f}"),
                    (f"screen{sk}{pname}share", f"{100*u['share_pools_any_discovery']:.0f}"), (f"screen{sk}{pname}fdp", f"{100*u['realized_fdr']:.1f}"),
                    (f"screen{sk}{pname}minrej", f"{u['min_rejection_count']['mean']:.0f}")]
            if key in ("ratio_oracle", "z_oracle"):
                um = Cc["units"]["M"][key][pv]["0.05"]["per_repeat_totals"]
                out += [(f"screenm{sk}{pname}true", f"{um['true_discoveries']['mean']:.0f}"), (f"screenm{sk}{pname}false", f"{um['false_discoveries']['mean']:.0f}")]
            st = Cc["stratified_sum"][key][pv]["0.05"]
            out += [(f"screenstrat{sk}{pname}true", f"{st['true_discoveries']['mean']:.0f}"), (f"screenstrat{sk}{pname}false", f"{st['false_discoveries']['mean']:.0f}")]
        for u, uk in (("S-se", "sse"), ("M", "m"), ("W-new", "wnew"), ("W-gap", "wgap"), ("L", "l")):
            tu = Cc["units"][u][key]["0.5"]["0.05"]["per_repeat_totals"]
            out.append((f"screendom{uk}{sk}fiftytrue", f"{tu['true_discoveries']['mean']:.0f}"))
        kp = Cc["units"]["pooled"][key]["0.5"]["0.05"]["bh_known_prevalence"]["per_repeat_totals"]
        out += [(f"screenknown{sk}fiftytrue", f"{kp['true_discoveries']['mean']:.0f}"), (f"screenknown{sk}fiftyfalse", f"{kp['false_discoveries']['mean']:.1f}")]
    u = Cc["units"]["pooled"]["z_oracle"]["0.5"]["0.05"]
    out += [("screennpool", fmt_int(u["n_null_per_pool"][0])), ("screenncal", fmt_int(u["n_cal_per_pool"][0])),
            ("screenpmin", fmt_int(u["n_cal_per_pool"][0] + 1)), ("screenzfiftyratio", f"{100*u['ratio_false_to_discoveries_all_pools']:.1f}")]
    # non-members above the last member the members-only rule names (the in-sample view)
    for sk, key in (("z", "z_oracle"), ("ratio", "ratio_oracle")):
        a, b = arrays(cols, key, selN); a = np.sort(a[np.isfinite(a)])[::-1]; b = b[np.isfinite(b)]
        kq = fdr_named(a, b, 0.05)
        out += [(f"screen{sk}named", str(kq)), (f"screen{sk}namedabove", str(int((b > a[kq - 1]).sum()) if kq else "0"))]
    # ----------------------------------------------------- collection by domain
    C = json.loads((ANA / "collection_by_domain.json").read_text())
    for sk, key in (("orc", "ratio_oracle"), ("sib", "ratio_sibling"), ("pub", "ratio_stage1"), ("zorc", "z_oracle")):
        for grp, v in C["part_a"][key].items():
            k = COHK[grp]
            pw = v["power_paired_t"].get(str(v["n80"]))
            out += [(f"coll{k}{sk}dz", f"{v['dz']:.2f}"), (f"coll{k}{sk}dzlo", f"{v['dz_ci95'][0]:.2f}"), (f"coll{k}{sk}dzhi", f"{v['dz_ci95'][1]:.2f}"),
                    (f"coll{k}{sk}npower", str(v["n80"])), (f"coll{k}{sk}npowert", str(v["n80_noncentral_t"])),
                    (f"coll{k}{sk}npoweremp", str(v["n80_empirical"]["n"])), (f"coll{k}{sk}aucten", f"{v['collection_auc']['10']:.2f}"),
                    (f"coll{k}{sk}auctenlo", f"{v['collection_auc_ci95']['10'][0]:.2f}"), (f"coll{k}{sk}auctenhi", f"{v['collection_auc_ci95']['10'][1]:.2f}")]
            if pw is not None:
                out.append((f"coll{k}{sk}power", f"{100*pw['rate']:.0f}"))
            ne = v["n80_empirical"]
            out += [(f"coll{k}{sk}poweremp", f"{100*ne['power']['rate']:.0f}"),
                    (f"coll{k}{sk}powerempminus", f"{100*ne['power_at_n_minus_1']['rate']:.0f}")]
            if "13" in v["power_paired_t"]:
                out.append((f"coll{k}{sk}powerthirteen", f"{100*v['power_paired_t']['13']['rate']:.0f}"))
            if "13" in v["size_signflip_paired_t"]:
                out.append((f"coll{k}{sk}sizethirteen", f"{100*v['size_signflip_paired_t']['13']['rate']:.1f}"))
            out += [(f"coll{k}{sk}dunpaired", f"{v['d_unpaired']:.2f}"), (f"coll{k}{sk}nunpaired", str(v["n80_unpaired_per_group"]))]
            if "13" in v["power_welch_unpaired"]:
                out.append((f"coll{k}{sk}welchthirteen", f"{100*v['power_welch_unpaired']['13']['rate']:.0f}"))
            out.append((f"coll{k}{sk}corr", f"{v['corr_member_partner']:.2f}"))
    zp = C["part_a"]["z_oracle"]["NAT1 pooled"]["power_paired_t"]
    for n_, nm_ in (("6", "six"), ("7", "seven"), ("8", "eight")):
        if n_ in zp:
            out.append((f"collzorcpower{nm_}", f"{100*zp[n_]['rate']:.0f}"))
    B = C["part_b"]["ratio_oracle"]
    def rej(e, m="13", d="upper"): return f"{100*e['welch'][m][d]['rate']:.1f}"
    mt = B["matched"]
    e = mt["arXiv: N-arxiv vs M+L partners"]
    out += [("calarxivrej", rej(e)), ("calarxivrejfifty", rej(e, "50")), ("calarxivoffset", f"{e['mean_diff_X_minus_Y']:+.4f}"),
            ("calarxivoffsetlo", f"{e['mean_diff_ci95'][0]:+.4f}"), ("calarxivoffsethi", f"{e['mean_diff_ci95'][1]:+.4f}"), ("calarxivoffsetsd", f"{e['std_offset']:.2f}"),
            ("calarxivaucten", f"{e['collection_auc']['10']:.2f}")]
    e = mt["Wikipedia: N-wikiA+N-wikiB vs W-new+W-gap partners"]
    out += [("calwikirej", rej(e)), ("calwikirejfifty", rej(e, "50")), ("calwikiaucten", f"{e['collection_auc']['10']:.2f}")]
    out += [("calwikiarej", rej(mt["Wikipedia: N-wikiA vs W-new partners"])), ("calwikiarejfifty", rej(mt["Wikipedia: N-wikiA vs W-new partners"], "50")),
            ("calwikibrej", rej(mt["Wikipedia: N-wikiB vs W-gap partners"])), ("calwikibrejfifty", rej(mt["Wikipedia: N-wikiB vs W-gap partners"], "50"))]
    hv = [B["halves"][g]["welch"][m][d]["rate"] for g in B["halves"] for m in ("13", "50") for d in ("upper", "lower")]
    out += [("calhalveslo", f"{100*min(hv):.1f}"), ("calhalveshi", f"{100*max(hv):.1f}")]
    cd = B["cross_domain"]
    out += [("crosssserej", rej(cd["N-arxiv vs S-se partners"])), ("crosswikirej", rej(cd["N-arxiv vs W-new+W-gap partners"])),
            ("crosswikirejfifty", rej(cd["N-arxiv vs W-new+W-gap partners"], "50"))]
    pn = B["pooled_null"]
    out += [("pooledarxivrej", rej(pn["M+L partners vs pooled N-arxiv+N-wikiA+N-wikiB"])), ("pooledarxivrejfifty", rej(pn["M+L partners vs pooled N-arxiv+N-wikiA+N-wikiB"], "50")),
            ("pooledsserej", rej(pn["S-se partners vs pooled N-arxiv+N-wikiA+N-wikiB"]))]
    mv = B["members_vs_null"]
    out += [("mvnarxivrej", rej(mv["arXiv: M+L members vs N-arxiv"])), ("mvnarxivrejfifty", rej(mv["arXiv: M+L members vs N-arxiv"], "50")),
            ("mvnwikirej", rej(mv["Wikipedia: W-new+W-gap members vs N-wikiA+N-wikiB"])), ("mvnssepooledrej", rej(mv["unmatched: S-se members vs pooled N-arxiv+N-wikiA+N-wikiB"]))]
    out += [("collrpower", fmt_int(C["constants"]["r_power"])), ("collrcal", fmt_int(C["constants"]["r_cal"]))]
    # collection table: cohort x reference, d_z / n80 (normal) / AUC at m=10
    L = [r"% generated by figures/paper_figures.py from collection_by_domain.json -- do not edit by hand",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Cohort & Counterfactual & Retrained & Published & Standardized \\", r"\midrule"]
    for grp, lab in (("M", "arXiv (M)"), ("L", "arXiv prefixes (L)"), ("W-new", "W-new"), ("W-gap", "W-gap"), ("S-se", "S-se"), ("NAT1 pooled", "pooled")):
        cells = []
        for key in ("ratio_oracle", "ratio_sibling", "ratio_stage1", "z_oracle"):
            v = C["part_a"][key][grp]
            cells.append(f"{v['dz']:.2f}/{v['n80']}/{v['collection_auc']['10']:.2f}")
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TABS / "collection.tex").write_text("\n".join(L) + "\n")
    # ----------------------------------------------------- exclusion sensitivity
    E = json.loads((ANA / "exclusion_sensitivity.json").read_text())
    rows = {pop: {r["label"]: r for r in E["results"][pop]} for pop in ("NAT1", "ALL")}
    N = rows["NAT1"]; prim = N["primary analysis (paper)"]
    var = [v for k, v in N.items() if k != "primary analysis (paper)"]
    for sk in ("ratio_oracle", "z_oracle"):
        tag = "" if sk == "ratio_oracle" else "z"
        out += [(f"sensnat{tag}dtaumax", f"{max(abs(v['delta_vs_primary'][sk]['tau']) for v in var):.1e}".replace("e-0", "\\times10^{-") + "}"),
                (f"sensnat{tag}ddzmax", f"{max(abs(v['delta_vs_primary'][sk]['dz']) for v in var):.4f}"),
                (f"sensnat{tag}daucmax", f"{max(abs(v['delta_vs_primary'][sk]['auc']) for v in var):.4f}"),
                (f"sensnat{tag}tprlo", f"{100*min(v['scores'][sk]['tpr1'] for v in N.values()):.2f}"), (f"sensnat{tag}tprhi", f"{100*max(v['scores'][sk]['tpr1'] for v in N.values()):.2f}"),
                (f"sensnat{tag}fdrlo", str(min(v['scores'][sk]['fdr05'] for v in N.values()))), (f"sensnat{tag}fdrhi", str(max(v['scores'][sk]['fdr05'] for v in N.values())))]
    Al = rows["ALL"]
    for lab, k in (("primary analysis (paper)", "withxf"), ("drop cohorts F and X", "noxf"), ("drop cohort X", "nox"), ("drop cohort F", "nof")):
        v = Al[lab]
        for sk, tag in (("ratio_oracle", ""), ("z_oracle", "z")):
            sc = v["scores"][sk]
            out += [(f"sensall{k}{tag}auc", f3(sc["auc"])), (f"sensall{k}{tag}tpr", f"{100*sc['tpr1']:.2f}"), (f"sensall{k}{tag}fdr", str(sc["fdr05"])),
                    (f"sensall{k}{tag}dz", f"{sc['dz']:.2f}"), (f"sensall{k}{tag}n", fmt_int(v["n_pairs"]))]
    rm = E["repetition_masking"]["summary"] if isinstance(E["repetition_masking"].get("summary"), dict) else {}
    masked_row = N["drop pairs of all 18 masking-affected documents"]
    out += [("sensnatmaskedpairs", str(prim["n_pairs"] - masked_row["n_pairs"]))]
    LT = [r"% generated by figures/paper_figures.py from exclusion_sensitivity.json -- do not edit by hand",
          r"\begin{tabular}{lrrrrrr}", r"\toprule",
          r"Variant & pairs & $\hat\tau$ & $d_z$ & \auc & \tprat{1} & named \\", r"\midrule"]
    def row(lab, v, sk="ratio_oracle"):
        sc = v["scores"][sk]
        return f"{_texesc(lab)} & {fmt_int(v['n_pairs'])} & ${sc['tau']:+.5f}$ & {sc['dz']:.3f} & {sc['auc']:.4f} & {100*sc['tpr1']:.2f} & {sc['fdr05']} \\\\"
    LT.append(r"\multicolumn{7}{l}{\emph{Verified-unseen single-exposure pairs, reference ratio}} \\")
    for lab, v in N.items():
        LT.append(row(lab.replace("primary analysis (paper)", "primary analysis"), v))
    LT += [r"\addlinespace", r"\multicolumn{7}{l}{\emph{Verified-unseen single-exposure pairs, standardized ratio}} \\"]
    for lab, v in N.items():
        LT.append(row(lab.replace("primary analysis (paper)", "primary analysis"), v, "z_oracle"))
    LT += [r"\addlinespace", r"\multicolumn{7}{l}{\emph{Pooled pairs, reference ratio}} \\"]
    for lab, v in Al.items():
        LT.append(row(lab.replace("primary analysis (paper)", "every analysed pair (cohorts F and X included)").replace("drop cohorts F and X", "primary pooled analysis (without F and X)"), v))
    LT += [r"\bottomrule", r"\end{tabular}"]
    (TABS / "sensitivity.tex").write_text("\n".join(LT) + "\n")
    # ----------------------------------------------------- cohort F 2x2
    F = json.loads((ANA / "cohortF_2x2_check.json").read_text())
    nat = F["estimators"]["native (LaTeX A vs LaTeX B)"]; tr = F["estimators"]["transfer (PDF A vs PDF B)"]
    out += [("cohfnpairs", str(nat["n_pairs"])), ("cohfnative", f"{nat['mean']:+.4f}"), ("cohfnativelo", f"{nat['ci_pair_boot'][0]:.4f}"), ("cohfnativehi", f"{nat['ci_pair_boot'][1]:.4f}"),
            ("cohfnativedz", f"{nat['d_z']:.2f}"), ("cohfnativeauc", f3(nat["auc"])), ("cohfnativeauclo", f3(nat["auc_ci_pair_boot"][0])), ("cohfnativeauchi", f3(nat["auc_ci_pair_boot"][1])),
            ("cohfnativetpr", pct1(nat["tpr_at_1pct_fpr"]["tpr"])),
            ("cohftransfer", f"{tr['mean']:+.4f}"), ("cohftransferlo", f"{tr['ci_pair_boot'][0]:.4f}"), ("cohftransferhi", f"{tr['ci_pair_boot'][1]:.4f}"),
            ("cohftransferdz", f"{tr['d_z']:.2f}"), ("cohftransferauc", f3(tr["auc"])), ("cohftransferauclo", f3(tr["auc_ci_pair_boot"][0])), ("cohftransferauchi", f3(tr["auc_ci_pair_boot"][1])),
            ("cohftransferp", f"{tr['p_one_sided']:.3f}"), ("cohftransfertpr", pct1(tr["tpr_at_1pct_fpr"]["tpr"]))]
    rho = F["transfer_coefficient"]; sv = F["sensitivity_without_audit_exclusions"]
    out += [("cohfrho", f"{rho['value']:.2f}"), ("cohfrholo", f"{rho['ci_fieller_t'][0]:.2f}"), ("cohfrhohi", f"{rho['ci_fieller_t'][1]:.2f}"),
            ("cohfrhoall", f"{sv['rho']:.2f}"), ("cohfrhoalllo", f"{sv['rho_ci'][0]:.2f}"), ("cohfrhoallhi", f"{sv['rho_ci'][1]:.2f}"), ("cohfnall", str(sv["n_pairs"]))]
    pl = F["placebos"]
    out += [("cohfplacebonative", f3(pl["P1_native_control_level"]["auc"])), ("cohfplacebotransfer", f3(pl["P1_transfer_control_level"]["auc"])),
            ("cohfplaceboorig", f"{pl['P1_original_control_level']['auc']:.2f}"), ("cohfplaceboorighi", f"{pl['P1_original_control_level']['auc_ci_pair_boot'][1]:.2f}"),
            ("cohfplaceboorigdlo", f"{pl['P1_original_control_level']['auc_ci_pair_boot'][0]:.2f}"),
            ("cohfdriftnative", f3(pl["P2_native_drift"]["auc"])), ("cohfdrifttransfer", f3(pl["P2_transfer_drift"]["auc"])), ("cohfdriftorig", f"{pl['P2_original_drift']['auc']:.2f}")]
    dc = F["decomposition"]; am = dc["arm_means"]
    out += [("cohffloorgap", f"{dc['nonmember_floor_change_PDF_minus_LaTeX']['value']:.4f}"), ("cohforigeffect", f"{am['delta_LaTeX_A'] - am['delta_PDF_B']:+.4f}")]
    sh = F["shared_content"]["by_n"]["8"]
    out += [("cohfsharedcont", f"{100*sh['containment']['PDF(A) in injected LaTeX(A)']['median']:.0f}"),
            ("cohfsharedeffect", f"{sh['transfer_effect_on_shared_positions']['mean']:+.4f}"), ("cohfsharedeffectlo", f"{sh['transfer_effect_on_shared_positions']['ci'][0]:.4f}"), ("cohfsharedeffecthi", f"{sh['transfer_effect_on_shared_positions']['ci'][1]:.4f}"),
            ("cohfunsharedeffect", f"{sh['transfer_effect_on_unshared_positions']['mean']:+.4f}"), ("cohfunsharedeffectlo", f"{sh['transfer_effect_on_unshared_positions']['ci'][0]:+.4f}"), ("cohfunsharedeffecthi", f"{sh['transfer_effect_on_unshared_positions']['ci'][1]:.4f}")]
    acc = F["accounting"]
    out += [("cohfdesign", str(acc["design_pairs_in_manifest"])), ("cohftoolong", str(len(acc["pairs_missing_a_scored_cell"]))), ("cohfaudit", str(len(acc["pairs_excluded_by_data_audit"])))]
    # ----------------------------------------------------- length truncation
    Lj = json.loads((ANA / "length_truncation.json").read_text())
    TN = {16: "sixteen", 128: "onetwoeight", 1024: "onezerotwofour", 4096: "fourzeroninesix", "full": "full"}
    for r in Lj["fixed_curve"]:
        if r["T"] in TN:
            k = TN[r["T"]]
            out += [(f"lenauc{k}", f3(r["auc"])), (f"lenauc{k}lo", f3(r["auc_ci"][0])), (f"lenauc{k}hi", f3(r["auc_ci"][1])),
                    (f"lenmu{k}", f"{r['mean']:+.4f}"), (f"lendz{k}", f"{r['dz']:.2f}")]
    for r in Lj["token_noise_share_fixed"]:
        if r["T"] in TN:
            out.append((f"lenshare{TN[r['T']]}", f"{100*r['share']:.0f}"))
    ex = Lj["exponents"]
    out += [("lenexpdz", f"{ex['dz']:.2f}"), ("lenexpdzlo", f"{ex['dz_ci'][0]:.2f}"), ("lenexpdzhi", f"{ex['dz_ci'][1]:.2f}"),
            ("lenexpmean", f"{ex['mean']:.2f}"), ("lenexpsd", f"{ex['sd']:.2f}"), ("lennfixed", fmt_int(Lj["n_fixed"])),
            ("lenfullmedian", fmt_int(round(Lj["fixed_full_len_tokens_median"])))]
    pd_ = Lj["paired_auc_differences"]
    out += [("lendiffsixteen", f"{pd_['4096_minus_16']['diff']:+.3f}"), ("lendiffsixteenlo", f"{pd_['4096_minus_16']['ci'][0]:+.3f}"), ("lendiffsixteenhi", f"{pd_['4096_minus_16']['ci'][1]:+.3f}"),
            ("lendifffull", f"{pd_['full_minus_4096']['diff']:+.3f}"), ("lendifffulllo", f"{pd_['full_minus_4096']['ci'][0]:+.3f}"), ("lendifffullhi", f"{pd_['full_minus_4096']['ci'][1]:+.3f}")]
    LN = {512: "fiveonetwo", 1024: "onezerotwofour", 2048: "twozerofoureight", 4096: "fourzeroninesix", 8192: "eightoneninetwo"}
    for r in Lj["L_by_length"]:
        k = LN[int(r["T"])]
        out += [(f"cohlauc{k}", f3(r["auc"])), (f"cohlauc{k}lo", f3(r["auc_ci"][0])), (f"cohlauc{k}hi", f3(r["auc_ci"][1])), (f"cohln{k}", str(r["n_pos"]))]
    lm = Lj["L_vs_M"]
    out += [("lvsmdiff", f"{lm['diff']:+.3f}"), ("lvsmdifflo", f"{lm['diff_ci'][0]:+.3f}"), ("lvsmdiffhi", f"{lm['diff_ci'][1]:+.3f}")]
    print(f"  tables/collection.tex tables/sensitivity.tex ({len(out)} checklist macros)")
    return out


# ---------------------------------------------------------------- figures
def fig_intro(cols, selN, M):
    """The introductory figure (intro.pdf): four reference tiers at one copy, and what pooling buys."""
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.3), gridspec_kw={"height_ratios": [1.0, 1.05]})
    ax = axes[0]
    labels, tprs, aucs, colr = [], [], [], []
    for lab, key, c, ls, mk in TIERS:
        a, b = arrays(cols, key, selN)
        t, k, n = tpr_at(a, b, 0.01)
        labels.append(lab.replace(" (surprise)", "")); tprs.append(100 * t); aucs.append(ps.auc(a, b)); colr.append(c)
    x = np.arange(len(labels))
    ax.bar(x, tprs, color=colr, width=0.62, zorder=3)
    for i, (t, a) in enumerate(zip(tprs, aucs)):
        ax.text(i, t + 0.12, f"{t:.1f}%", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
        ax.text(i, 0.25, f"AUC\n{a:.2f}", ha="center", va="bottom", fontsize=5.8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["no\nreference", "published\ncheckpoint", "retrained\nreference", "exact\ncounterfactual"], fontsize=6.3)
    ax.set_ylabel("TPR at 1% FPR (%)")
    ax.set_ylim(0, max(tprs) * 1.35)
    ax.set_title("(a) One document, one exposure: what the adversary's reference buys", loc="left", fontsize=7.4)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ms = [1, 2, 3, 5, 10, 20, 50, 100]
    for lab, key, c, ls, mk in TIERS[:0:-1]:
        a, b = arrays(cols, key, selN)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        cur = collection_auc(a, b, ms)
        ax.plot([m for m, _ in cur], [v for _, v in cur], color=c, ls=ls, marker=mk, label=lab)
    ax.axhline(0.5, color="#222", ls="--", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("documents pooled, $m$")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 1.03)
    ax.set_title("(b) A collection of $m$ documents, one exposure each", loc="left", fontsize=7.4)
    # lower right, above the chance line: the one region of panel (b) no curve enters
    ax.legend(fontsize=6.2, loc="lower right", bbox_to_anchor=(1.0, 0.17), borderaxespad=0.0, labelspacing=0.35, borderpad=0.3,
              frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    save(fig, "intro")
    M["intro_tpr"] = dict(zip(labels, tprs)); M["intro_auc"] = dict(zip(labels, aucs))


def fig_signal(cols, arm, selN, H):
    d = cols["ratio_oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.85))
    ax = axes[0]
    lo, hi = np.quantile(d[np.isfinite(d)], [0.002, 0.995])
    for m, lab, c, hs in [(arm == "null", "never injected (null pool)", C["null"], "//"),
                          (arm == "holdout", "matched held-out", C["holdout"], None),
                          (arm == "inject", "injected", C["injected"], None)]:
        ax.hist(d[m], bins=160, range=(lo, hi), histtype="stepfilled",
                alpha=0.45, color=c, density=True, label=lab, lw=0.5)
        ax.axvline(np.nanmean(d[m]), color=c, ls="--", lw=0.8)
    ax.axvline(0, color="#222", lw=0.6)
    ax.set_xlabel(r"per-document effect $\tau(d)$ (nats/token)")
    ax.set_ylabel("density")
    ax.set_title("(a) Three arms, all documents", loc="left")
    ax.legend(fontsize=6.2, loc="upper right")

    ax = axes[1]
    a, b = arrays(cols, "ratio_oracle", selN)
    diff = a - b; diff = diff[np.isfinite(diff)]
    lo2, hi2 = np.quantile(diff, [0.004, 0.996])
    ax.hist(diff, bins=160, range=(lo2, hi2), histtype="stepfilled",
            color=C["injected"], alpha=0.55, density=True, lw=0.5)
    ax.axvline(0, color="#222", lw=0.6)
    ax.axvline(diff.mean(), color=C["bad"], ls="--", lw=0.9)
    ax.annotate(f"mean {diff.mean():+.4f}\n{100*(diff>0).mean():.1f}% positive",
                xy=(diff.mean(), ax.get_ylim()[1] * 0.72),
                xytext=(6, 0), textcoords="offset points", fontsize=6.6)
    ax.set_xlabel(r"within-pair contrast $\tau(d_{\mathrm{in}})-\tau(d_{\mathrm{out}})$ (nats/token)")
    ax.set_ylabel("density")
    ax.set_title(f"(b) Within-pair, verified-unseen single copy, $n={len(selN):,}$", loc="left")
    save(fig, "signal_distributions")


def fig_roc(cols, selN):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.0))
    npos = len(selN)
    for ax, logscale in zip(axes, (False, True)):
        for lab, key, c, ls, mk in TIERS:
            a, b = arrays(cols, key, selN)
            f, t = ps.roc(a, b)
            au = ps.auc(a, b)
            if logscale:
                k = (f > 0) & (t > 0)
                ax.plot(f[k], t[k], color=c, ls=ls, label=lab)
            else:
                ax.plot(f, t, color=c, ls=ls, label=f"{lab} ({au:.3f})")
        if "z_oracle" in cols:
            a, b = arrays(cols, "z_oracle", selN)
            f, t = ps.roc(a, b); au = ps.auc(a, b)
            lab = "exact counterfactual, standardized"
            if logscale:
                k = (f > 0) & (t > 0)
                ax.plot(f[k], t[k], color=C["injected"], ls=(0, (1, 1.2)), lw=1.0, label=lab)
            else:
                ax.plot(f, t, color=C["injected"], ls=(0, (1, 1.2)), lw=1.0, label=f"{lab} ({au:.3f})")
        if logscale:
            lim = 1.0 / npos
            ax.plot([lim, 1], [lim, 1], color="#222", ls="--", lw=0.6)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lim, 1); ax.set_ylim(lim, 1)
            ax.axvline(0.01, color="#444", lw=0.7, ls=(0, (2, 2)))
            ax.text(0.0115, 1.5 * lim, "1% FPR", fontsize=6)
            ax.set_title("(b) Low-FPR regime (log-log)", loc="left")
            ax.legend(fontsize=6.2, loc="upper left")
        else:
            ax.plot([0, 1], [0, 1], color="#222", ls="--", lw=0.6)
            ax.set_title("(a) Full ROC (AUC in legend)", loc="left")
            ax.legend(fontsize=6.2, loc="lower right")
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    save(fig, "roc")


def fig_attacks(cols, selN):
    """Every attack statistic against difficulty-matched non-members."""
    groups = [
        ("Reference-free", C["holdout"], [
            ("LOSS", "injected__loss", -1), ("Perplexity", "injected__ppl", -1),
            ("Bits-per-byte", "bpb", -1), ("zlib ratio", "zlib", 1),
            (r"Min-K% (k=20)", "injected__mink_20", 1),
            (r"Min-K%++ (k=20)", "injected__minkpp_20", 1),
            ("Surprise", "injected__surprise", 1),
        ]),
        ("Published checkpoint", C["pretrained"], [
            ("Ratio", "ratio_stage1", 1),
            (r"Ratio + Min-K%", "mink_20_ratio_stage1", 1),
            (r"Ratio + Min-K%++", "minkpp_20_ratio_stage1", 1),
            ("Standardized ratio", "z_stage1", 1),
        ]),
        ("Retrained reference", C["accent"], [
            ("Ratio", "ratio_sibling", 1),
            (r"Ratio + Min-K%", "mink_20_ratio_sibling", 1),
            (r"Ratio + Min-K%++", "minkpp_20_ratio_sibling", 1),
            ("Standardized ratio", "z_sibling", 1),
        ]),
        ("Exact counterfactual", C["injected"], [
            ("Ratio", "ratio_oracle", 1),
            (r"Ratio + Min-K%", "mink_20_ratio_oracle", 1),
            (r"Ratio + Min-K%++", "minkpp_20_ratio_oracle", 1),
            ("Standardized ratio", "z_oracle", 1),
        ]),
    ]
    items = []
    for gname, gcol, entries in groups:
        for lab, key, sgn in entries:
            if key not in cols:
                continue
            a, b = arrays(cols, key, selN)
            # every column in the attack table is stored so that larger =
            # more member-like ("loss" is the mean log-probability, "ppl" is
            # -exp(-mean log p), "bpb" and "zlib" are -NLL per byte); the
            # sgn field documents that and is not applied
            au = ps.auc(a, b)
            lo, hi = auc_ci_pairs(a, b, n=300)
            t1, k, n = tpr_at(a, b, 0.01)
            items.append(dict(lab=lab, key=key, auc=au, lo=lo, hi=hi, tpr1=t1, group=gname, col=gcol))
    gbest = defaultdict(float)
    for it in items:
        gbest[it["group"]] = max(gbest[it["group"]], it["auc"])
    items.sort(key=lambda t: (gbest[t["group"]], t["auc"]))

    fig, ax = plt.subplots(figsize=(COL, 0.158 * len(items) + 0.8))
    y = np.arange(len(items))
    ax.barh(y, [t["auc"] for t in items], color=[t["col"] for t in items], height=0.66, zorder=3)
    ax.errorbar([t["auc"] for t in items], y,
                xerr=[[t["auc"] - t["lo"] for t in items], [t["hi"] - t["auc"] for t in items]],
                fmt="none", ecolor="#444", elinewidth=0.6, capsize=1.4, zorder=4)
    ax.axvline(0.5, color="#222", ls="--", lw=0.7, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([t["lab"] for t in items], fontsize=6.4)
    ax.set_xlim(0.45, 0.88)
    ax.set_xlabel("AUC, injected vs. matched held-out")
    HATCH = {"Reference-free": "", "Published checkpoint": "//", "Retrained reference": "..", "Exact counterfactual": "xx"}
    for bar, it in zip(ax.patches, items):
        bar.set_hatch(HATCH[it["group"]]); bar.set_edgecolor("white"); bar.set_linewidth(0.3)
    handles = [mpatches.Patch(facecolor=g[1], hatch=HATCH[g[0]], edgecolor="white", label=g[0]) for g in groups[::-1]]
    ax.legend(handles=handles, loc="lower right", fontsize=6.0)
    ax.grid(axis="y", visible=False)
    save(fig, "attacks_auc")
    return items


def fig_tiers(cols, arm, selN, M):
    """What the reference buys, and the noise it has to clear."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.9), gridspec_kw={"width_ratios": [0.9, 1.3]})
    ax = axes[0]
    rec = []
    for lab, key, c, ls, mk in TIERS[1:]:
        h = headline(cols, selN, key)
        rec.append((lab, h, c, mk))
    y = np.arange(len(rec))[::-1]
    for yy, (lab, h, c, mk) in zip(y, rec):
        ax.errorbar(h["tau"], yy, xerr=[[h["tau"] - h["lo"]], [h["hi"] - h["tau"]]],
                    fmt=mk, color=c, ms=5, capsize=2.5, lw=1.0)
        ax.text(h["tau"], yy + 0.28, f"{h['tau']:+.4f}  $d_z$={h['dz']:.2f}", ha="center", fontsize=6.0, color=c)
    ax.axvline(0, color="#222", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rec], fontsize=6.4)
    ax.set_ylim(-0.6, len(rec) - 0.2)
    ax.set_xlabel(r"within-pair effect $\hat\tau$ (nats/token), 95% CI")
    ax.set_title("(a) The estimate barely depends on the reference", loc="left", fontsize=7.0)
    ax.grid(axis="y", visible=False)
    M["tier"] = {lab: h for lab, h, _, _ in rec}

    ax = axes[1]
    coh = cols["cohort"]
    nonmem = (arm != "inject") & np.isin(coh, list(NAT) + ["N-arxiv", "N-wikiA", "N-wikiB"])
    drift = (cols["seed42__loss"] - cols["sibling__loss"])[nonmem]
    drift = drift[np.isfinite(drift)]
    holdN = np.array([h for _, h in selN])
    spill = cols["ratio_oracle"][holdN]; spill = spill[np.isfinite(spill)]
    a, b = arrays(cols, "ratio_oracle", selN); eff = a - b; eff = eff[np.isfinite(eff)]
    lim = 0.12
    bins = np.linspace(-lim, lim, 121)
    for v, lab, c, al in [(drift, f"seed noise: two seeds, same recipe,\nnever-injected text (SD {drift.std(ddof=1):.4f})", C["holdout"], 0.85),
                          (spill, f"spillover onto held-out partners\n(mean {spill.mean():+.4f}, SD {spill.std(ddof=1):.4f})", C["null"], 0.6),
                          (eff, f"within-pair membership effect\n(mean {eff.mean():+.4f}, SD {eff.std(ddof=1):.4f})", C["injected"], 0.45)]:
        ax.hist(np.clip(v, -lim, lim), bins=bins, density=True, color=c, alpha=al, label=lab, lw=0)
    ax.axvline(0, color="#222", lw=0.6)
    ax.set_xlim(-0.16, lim)
    ax.set_xlabel("difference in mean log-probability (nats/token)")
    ax.set_ylabel("density")
    ax.set_title("(b) Seed noise is small; spillover and heterogeneity are not", loc="left", fontsize=7.4)
    ax.legend(fontsize=5.6, loc="upper left", borderaxespad=0.3)
    save(fig, "reference_tiers")
    M["drift"] = dict(mean=float(drift.mean()), sd=float(drift.std(ddof=1)), n=int(drift.size))
    M["spill"] = dict(mean=float(spill.mean()), sd=float(spill.std(ddof=1)), n=int(spill.size))
    M["effsd"] = float(eff.std(ddof=1))


def fig_dose(cols, pairs, M):
    """The matched arXiv ladder: M at r=1, D at 2-16, D32 at 32."""
    d, r, coh = cols["ratio_oracle"], cols["r"], cols["cohort"]
    levels = []
    for rv, chs in [(1, {"M"}), (2, {"D"}), (4, {"D"}), (8, {"D"}), (16, {"D"}), (32, {"D32"})]:
        sel = [(i, h) for i, h in pairs if int(r[i]) == rv and coh[i] in chs]
        a, b = arrays(cols, "ratio_oracle", sel); dd = a - b; dd = dd[np.isfinite(dd)]
        lo, hi = ps.bootstrap_ci(dd)
        t1, k, n = tpr_at(a, b, 0.01)
        levels.append(dict(r=rv, n=int(dd.size), mu=float(dd.mean()), lo=float(lo), hi=float(hi),
                           sd=float(dd.std(ddof=1)), dz=float(dd.mean() / dd.std(ddof=1)),
                           auc=float(ps.auc(a, b)), tpr1=t1, k=k, nneg=n))
    xs = np.array([l["r"] for l in levels], float); mu = np.array([l["mu"] for l in levels])
    sl = np.polyfit(np.log(xs), np.log(mu), 1)
    # bootstrap the exponent by resampling pairs within each level
    bs = []
    per = [arrays(cols, "ratio_oracle", [(i, h) for i, h in pairs if int(r[i]) == l["r"] and coh[i] in ({"M"} if l["r"] == 1 else {"D32"} if l["r"] == 32 else {"D"})]) for l in levels]
    per = [(a - b)[np.isfinite(a - b)] for a, b in per]
    for _ in range(1000):
        m2 = [p[RNG.integers(0, p.size, p.size)].mean() for p in per]
        bs.append(np.polyfit(np.log(xs), np.log(m2), 1)[0])
    elo, ehi = np.percentile(bs, [2.5, 97.5])
    pred = np.exp(sl[1]) * xs ** sl[0]
    r2 = 1 - ((mu - pred) ** 2).sum() / ((mu - mu.mean()) ** 2).sum()

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.95))
    ax = axes[0]
    ax.errorbar(xs, mu, yerr=[mu - np.array([l["lo"] for l in levels]), np.array([l["hi"] for l in levels]) - mu],
                color=C["injected"], marker="o", capsize=1.6, elinewidth=0.6, ls="none", label="within-pair effect, 95% CI")
    ax.plot(xs, pred, color=C["neutral"], ls="--", lw=0.8, label=f"fit $\\propto r^{{{sl[0]:.2f}}}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel("copies injected $r$ (arXiv ladder)")
    ax.set_ylabel("within-pair effect\n(nats/token)")
    ax.set_title("(a) Effect is linear in exposures", loc="left")
    ax.legend(fontsize=6.4, loc="upper left")
    ax = axes[1]
    ax.plot(xs, [100 * l["tpr1"] for l in levels], color=C["bad"], marker="s", label="TPR at 1% FPR")
    ax2 = ax.twinx()
    ax2.plot(xs, [l["dz"] for l in levels], color=C["accent"], marker="^", ls="--", label="$d_z$")
    ax2.set_ylabel("$d_z$", color=C["accent"]); ax2.grid(False)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs); ax.set_xticklabels([f"{int(l['r'])}\n$n$={l['n']}" for l in levels], fontsize=6.2)
    ax.set_xlabel("copies injected $r$ (pairs per level)")
    ax.set_ylabel("TPR at 1% FPR (%)", color=C["bad"])
    ax.set_ylim(0, 112)
    ax.set_title("(b) Detection lags the effect", loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.2, loc="upper left")
    save(fig, "dose")
    M["dose"] = dict(levels=levels, exponent=float(sl[0]), exp_lo=float(elo), exp_hi=float(ehi), r2=float(r2))


def fig_drift(cols, arm, M):
    """Never-injected documents still gain, by domain."""
    coh = cols["cohort"]
    d = cols["ratio_oracle"]
    summ = json.loads((BUNDLE / "summary.json").read_text())["cohorts"]
    dom = {"M": "arXiv", "D": "arXiv", "D32": "arXiv", "DB": "arXiv", "L": "arXiv", "F": "arXiv",
           "W-new": "Wikipedia", "W-gap": "Wikipedia", "S-se": "StackExchange",
           "E0": "edited Wikipedia", "E0-D": "edited Wikipedia", "E": "edited Wikipedia", "X": "synthetic"}
    toks = defaultdict(float)
    for ch, v in summ.items():
        toks[dom[ch]] += v["tokens_injected"] / 1e6
    groups = [
        ("arXiv", ["M", "L"], ["N-arxiv"]),
        ("Wikipedia", ["W-new", "W-gap"], ["N-wikiA", "N-wikiB"]),
        ("StackExchange", ["S-se"], []),
        ("edited Wikipedia", ["E", "E0", "E0-D"], ["N-edited"]),
    ]
    rec = []
    for name, hold, nulls in groups:
        mh = (arm == "holdout") & np.isin(coh, hold)
        mn = (arm == "null") & np.isin(coh, nulls)
        vh = d[mh]; vh = vh[np.isfinite(vh)]
        vn = d[mn]; vn = vn[np.isfinite(vn)]
        rec.append(dict(name=name, tokens=toks[name],
                        hold=float(vh.mean()), hold_se=float(vh.std(ddof=1) / math.sqrt(vh.size)), nh=int(vh.size),
                        null=float(vn.mean()) if vn.size > 20 else float("nan"),
                        null_se=float(vn.std(ddof=1) / math.sqrt(vn.size)) if vn.size > 20 else float("nan"), nn=int(vn.size)))
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    x = np.arange(len(rec)); w = 0.36
    ax.bar(x - w / 2, [r["hold"] for r in rec], w, yerr=[1.96 * r["hold_se"] for r in rec],
           color=C["holdout"], label="held-out partners", zorder=3, capsize=1.5, error_kw=dict(lw=0.6))
    ax.bar(x + w / 2, [r["null"] if np.isfinite(r["null"]) else 0 for r in rec], w,
           yerr=[1.96 * r["null_se"] if np.isfinite(r["null_se"]) else 0 for r in rec],
           color=C["null"], hatch="//", label="null pool", zorder=3, capsize=1.5, error_kw=dict(lw=0.6))
    for i, r in enumerate(rec):
        ax.text(i, max(r["hold"], r["null"] if np.isfinite(r["null"]) else 0) + 0.003,
                f"{r['tokens']:.1f}M tokens\ninjected", ha="center", fontsize=5.6, color="#444")
    ax.axhline(0, color="#222", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([r["name"].replace(" ", "\n").replace("StackExchange", "Stack-\nExchange") for r in rec], fontsize=6.4)
    ax.set_ylabel(r"mean $\tau(d)$ (nats/token)")
    ax.set_ylim(0, max(r["hold"] for r in rec) * 1.45)
    ax.set_title("Never-injected documents, by domain", loc="left")
    ax.legend(fontsize=6.0, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "domain_transfer")
    M["drift_dom"] = rec


def fig_cohort(cols, pairs, M):
    d, coh, r = cols["ratio_oracle"], cols["cohort"], cols["r"]
    rows_ = []
    for ch in sorted(set(coh[np.array([i for i, _ in pairs])])):
        sel = [(i, h) for i, h in pairs if coh[i] == ch]
        if len(sel) < 30:
            continue
        a, b = arrays(cols, "ratio_oracle", sel); dd = a - b; dd = dd[np.isfinite(dd)]
        lo, hi = ps.bootstrap_ci(dd)
        t1, k, n = tpr_at(a, b, 0.01)
        rows_.append(dict(ch=ch, n=len(sel), mu=float(dd.mean()), lo=float(lo), hi=float(hi),
                          auc=float(ps.auc(a, b)), r=int(np.median([r[i] for i, _ in sel])),
                          tpr1=t1, k=k, dz=float(dd.mean() / dd.std(ddof=1)),
                          flag="E" if ch in EFAM else "F" if ch == "F" else "X" if ch == "X" else ""))
    single = [t for t in rows_ if t["r"] == 1]
    single.sort(key=lambda t: t["auc"])
    # one column: effect with CI per cohort, the AUC as a right-hand column (rows ordered by it)
    fig, ax = plt.subplots(figsize=(COL, 2.25))
    y = np.arange(len(single))
    colr = [C["pretrained"] if t["flag"] == "E" else C["bad"] if t["flag"] in ("F", "X") else C["injected"] for t in single]
    ax.errorbar([t["mu"] for t in single], y,
                xerr=[[t["mu"] - t["lo"] for t in single], [t["hi"] - t["mu"] for t in single]],
                fmt="none", ecolor="#444", capsize=1.5, elinewidth=0.6)
    ax.scatter([t["mu"] for t in single], y, c=colr, s=14, zorder=4)
    ax.axvline(0, color="#222", lw=0.6)
    ax.set_yticks(y)
    lab = {"E": " (re-exposure)", "F": " (failed placebo)", "X": " (synthetic)", "": ""}
    ax.set_yticklabels([f"{t['ch']}{lab[t['flag']]}  $n$={t['n']}" for t in single], fontsize=6.2)
    ax.set_ylim(-0.6, len(single) - 0.4)
    ax.set_xlabel("within-pair effect (nats/token), 95% CI", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.grid(axis="y", visible=False)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{t['auc']:.3f}" for t in single], fontsize=6.2)
    ax2.tick_params(axis="y", length=0, pad=2)
    ax2.grid(False)
    for s_ in ("top", "right", "left", "bottom"):
        ax2.spines[s_].set_visible(False)
    ax2.text(1.0, 1.0, "AUC", transform=ax2.transAxes, ha="left", va="bottom", fontsize=6.4, fontweight="bold")
    handles = [mpatches.Patch(color=C["injected"], label="verified unseen"),
               mpatches.Patch(color=C["pretrained"], label="re-exposure"),
               mpatches.Patch(color=C["bad"], label="F and X")]
    ax.legend(handles=handles, fontsize=5.8, loc="upper right", handlelength=1.0, borderaxespad=0.2)
    save(fig, "cohort")
    M["cohort"] = rows_


def fig_trajectory(M):
    steps, TAU, FU, lr, meta = traj.load_ladder(LADDER, pair_filter=lambda x: x.get("cohort") in NAT)
    win = traj.by_window(steps, TAU, FU)
    curves = traj.window_curves(steps, TAU, FU, windows=traj.WINDOWS)
    decay = traj.decay_curve(steps, TAU, FU)
    ev = traj.event_align(steps, TAU, FU)
    seam = json.loads(LADDER_META.read_text())["seam_test"]

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.05), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    ax = axes[0]
    shades = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]
    for cv, c in zip(curves, shades):
        ax.plot(steps, cv["mu"], color=c, lw=1.1, label=f"placed {cv['a']//1000}k-{cv['b']//1000}k")
        ax.fill_between(steps, cv["mu"] - 1.96 * cv["se"], cv["mu"] + 1.96 * cv["se"], color=c, alpha=0.15, lw=0)
    ax.axhline(0, color="#222", lw=0.6)
    ax.set_xlabel("checkpoint read (update)")
    ax.set_ylabel(r"$\hat	au$ (nats/token)")
    ax.set_title("(a) Each copy spikes, then decays", loc="left")
    ax.legend(fontsize=5.4, loc="upper right")
    ax.set_xlim(0, 24000)

    ax = axes[1]
    mid = [(w["a"] + w["b"]) / 2 for w in win]
    ax.errorbar(mid, [w["peak"] for w in win], yerr=[1.96 * w["peak_se"] for w in win], color=C["injected"],
                marker="o", capsize=1.5, elinewidth=0.6, label="peak (first checkpoint after the copy)")
    ax.errorbar(mid, [w["end"] for w in win], yerr=[1.96 * w["end_se"] for w in win], color=C["holdout"],
                marker="s", ls="--", capsize=1.5, elinewidth=0.6, label="endpoint (update 23,852)")
    ax.plot(mid, [w["pre"] for w in win], color=C["neutral"], marker="x", ls=":", lw=0.8, label="last checkpoint before the copy")
    ax.axhline(0, color="#222", lw=0.6)
    ax.set_xlabel("update at which the copy was placed")
    ax.set_ylabel(r"$\hat\tau$ (nats/token)")
    ax.set_ylim(-0.01, 0.115)
    ax.set_title("(b) Peak falls, endpoint flat", loc="left")
    ax.legend(fontsize=5.2, loc="upper right")

    ax = axes[2]
    xs = [(w["a"] + w["b"]) / 2 for w in win]
    ax.errorbar([23852 - x for x in xs], [100 * w["survive"] for w in win],
                yerr=[[100 * (w["survive"] - w["survive_lo"]) for w in win], [100 * (w["survive_hi"] - w["survive"]) for w in win]],
                color=C["accent"], marker="o", capsize=1.5, elinewidth=0.6, label="by placement window")
    ax.plot([d["a"] + (d["b"] - d["a"]) / 2 for d in decay], [100 * d["ratio"] for d in decay], color=C["neutral"],
            ls="--", marker="^", lw=0.9, label="pooled, vs. time since copy")
    ax.axhline(100, color="#222", ls=":", lw=0.7)
    ax.set_xlabel("updates after the copy")
    ax.set_ylabel("surviving fraction of peak (%)")
    ax.set_ylim(0, 135)
    ax.set_title("(c) Forgetting is large", loc="left")
    ax.legend(fontsize=5.4, loc="upper right")
    save(fig, "trajectory")
    M["traj"] = dict(windows=win, decay=decay, n=meta["n_pairs"], lag=float(np.nanmean(ev["lag"][ev["valid"]])),
                     seam=seam, n_valid=int(ev["valid"].sum()))
    # the cross-sectional slope the old analysis reported
    end = ev["end"]; ok = np.isfinite(end) & np.isfinite(FU)
    pre90 = ok & (FU <= 0.9 * 23852)
    sl_all = np.polyfit(FU[ok], end[ok], 1)[0]; sl_pre = np.polyfit(FU[pre90], end[pre90], 1)[0]
    bs = []
    ii = np.nonzero(pre90)[0]
    for _ in range(500):
        j = RNG.choice(ii, ii.size, replace=True); bs.append(np.polyfit(FU[j], end[j], 1)[0])
    M["xsec"] = dict(slope_all=float(sl_all * 1000), slope_pre=float(sl_pre * 1000),
                     lo=float(np.percentile(bs, 2.5) * 1000), hi=float(np.percentile(bs, 97.5) * 1000),
                     n=int(pre90.sum()), n_all=int(ok.sum()))


def fig_posttrain(cols, selN, selA, M):
    """Both twins through the identical SFT and DPO: per-document scores."""
    def load_scores(p):
        out = {}
        with gzip.open(p, "rt", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    out[row["doc_id"]] = float(row["mean_logprob"])
                except (KeyError, ValueError):
                    pass
        return out
    docid = cols["doc_id"]
    stages = {"mid": None}
    for st, (pt, pc) in PT.items():
        if pt.exists() and pc.exists():
            stages[st] = (load_scores(pt), load_scores(pc))
    res = {}
    for pop, sel in (("NAT1", selN), ("ALL", selA)):
        res[pop] = {}
        for st in ("mid", "sft", "dpo"):
            if st not in stages:
                continue
            if st == "mid":
                a, b = arrays(cols, "ratio_oracle", sel)
            else:
                T, Cc = stages[st]
                a = np.array([T.get(docid[i], np.nan) - Cc.get(docid[i], np.nan) for i, _ in sel])
                b = np.array([T.get(docid[h], np.nan) - Cc.get(docid[h], np.nan) for _, h in sel])
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            d = a - b
            lo, hi = ps.bootstrap_ci(d)
            t1, k, n = tpr_at(a, b, 0.01)
            res[pop][st] = dict(n=int(d.size), tau=float(d.mean()), lo=float(lo), hi=float(hi),
                                sd=float(d.std(ddof=1)), dz=float(d.mean() / d.std(ddof=1)),
                                pos=float((d > 0).mean()), auc=float(ps.auc(a, b)), tpr1=t1)
            if st != "mid":
                a0, b0 = arrays(cols, "ratio_oracle", sel); d0 = (a0 - b0)[ok]
                res[pop][st]["corr_with_mid"] = float(np.corrcoef(d0, d)[0, 1])
    coh = cols["cohort"]
    res["cohort_auc"] = {}
    for ch in ("M", "L", "W-new", "W-gap", "S-se", "E", "E0"):
        sel = [(i, h) for i, h in selA if coh[i] == ch]
        row = {}
        for st in ("mid", "sft", "dpo"):
            if st not in stages:
                continue
            if st == "mid":
                a, b = arrays(cols, "ratio_oracle", sel)
            else:
                T, Cc = stages[st]
                a = np.array([T.get(docid[i], np.nan) - Cc.get(docid[i], np.nan) for i, _ in sel])
                b = np.array([T.get(docid[h], np.nan) - Cc.get(docid[h], np.nan) for _, h in sel])
            row[st] = float(ps.auc(a, b))
        res["cohort_auc"][ch] = row
    order = [s for s in ("mid", "sft", "dpo") if s in res["NAT1"]]
    LBL = {"mid": "mid-training", "sft": "+ SFT", "dpo": "+ SFT + DPO"}
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.85))
    x = np.arange(len(order))
    cols_ = [C["injected"], C["accent"], C["pretrained"]][:len(order)]
    for ax, (key, lab, pct) in zip(axes, [("tau", "effect (nats/token)", False),
                                          ("dz", r"Cohen's $d_z$", False),
                                          ("tpr1", r"TPR at 1% FPR (%)", True)]):
        v = [res["NAT1"][s][key] * (100 if pct else 1) for s in order]
        ax.bar(x, v, color=cols_, width=0.6, zorder=3)
        if key == "tau":
            ax.errorbar(x, v, yerr=[[v[i] - res["NAT1"][s]["lo"] for i, s in enumerate(order)],
                                    [res["NAT1"][s]["hi"] - v[i] for i, s in enumerate(order)]],
                        fmt="none", ecolor="#444", elinewidth=0.6, capsize=1.6, zorder=4)
        for i, val in enumerate(v):
            ax.text(i, val * 1.07 if key == "tau" else val, f"{val:.3f}" if not pct else f"{val:.1f}", ha="center", va="bottom", fontsize=6.4)
        ax.set_xticks(x); ax.set_xticklabels([LBL[s] for s in order], fontsize=6.6)
        ax.set_ylabel(lab); ax.set_ylim(0, max(v) * 1.28); ax.grid(axis="x", visible=False)
    axes[0].set_title("(a) Raw effect shrinks", loc="left")
    axes[1].set_title("(b) Standardized effect falls", loc="left")
    axes[2].set_title("(c) Detection at 1% FPR", loc="left")
    save(fig, "posttrain")
    M["pt"] = res


def fig_collection(cols, selN, H, M):
    ms = [1, 2, 3, 5, 10, 20, 31, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.9))
    ax = axes[0]
    coll = {}
    for lab, key, c, ls, mk in TIERS[:0:-1]:
        a, b = arrays(cols, key, selN); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        cur = collection_auc(a, b, ms)
        coll[lab] = cur
        ax.plot([m for m, _ in cur], [v for _, v in cur], color=c, ls=ls, marker=mk, label=lab)
    ax.axhline(0.5, color="#222", ls="--", lw=0.6)
    ax.axhline(0.95, color=C["neutral"], ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("documents in the collection $m$")
    ax.set_ylabel("AUC (mean score of $m$ vs. $m$)")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("(a) Collection-level inference", loc="left")
    ax.legend(fontsize=6.0, loc="lower right")
    ax = axes[1]
    base = np.logspace(-3, -0.3, 60)
    tpr = H["tpr1"]
    ppv = (tpr * base) / (tpr * base + 0.01 * (1 - base))
    ax.plot(base * 100, ppv * 100, color=C["bad"])
    ax.axvline(0.197, color=C["neutral"], ls=":", lw=0.8)
    ax.annotate("this experiment's\nprevalence, 0.197%", xy=(0.197, 3), fontsize=6.0,
                xytext=(4, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("prevalence of members in the candidate pool (%)")
    ax.set_ylabel("precision at 1% FPR (%)")
    ax.set_title(f"(b) Precision at the measured TPR ({100*tpr:.1f}%)", loc="left")
    save(fig, "collection")
    M["coll"] = coll
    M["ppv_at_prev"] = float(100 * (tpr * 0.00197) / (tpr * 0.00197 + 0.01 * (1 - 0.00197)))


# ------------------------------------------------------------------ tables
def write_tables(items, M, H, HA, HE, cols, selN):
    TABS.mkdir(parents=True, exist_ok=True)
    # attacks
    L = [r"% generated by figures/paper_figures.py -- do not edit by hand",
         r"\begin{tabular}{lrr}", r"\toprule",
         r"Attack & \auc & \tprat{1} \\", r"\midrule"]
    gbest = defaultdict(float)
    for it in items:
        gbest[it["group"]] = max(gbest[it["group"]], it["auc"])
    last = None
    for it in sorted(items, key=lambda t: (-gbest[t["group"]], -t["auc"])):
        if it["group"] != last:
            if last is not None:
                L.append(r"\addlinespace")
            L.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{_texesc(it['group'])}}}}} \\\\")
            last = it["group"]
        L.append(f"\\quad {_texesc(it['lab'])} & {it['auc']:.3f} & {100*it['tpr1']:.1f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TABS / "attacks.tex").write_text("\n".join(L) + "\n")

    # cohorts
    L = [r"% generated by figures/paper_figures.py",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Cohort & Note & $n$ & $r$ & $\hat\tau$ & \auc & \tprat{1} \\", r"\midrule"]
    note = {"E": "re-exposure", "F": "failed placebo", "X": "synthetic", "": ""}
    RR = {"D": "2--16"}
    for t in sorted(M["cohort"], key=lambda t: (t["flag"] != "", -t["auc"])):
        L.append(f"{_texesc(t['ch'])} & {note[t['flag']]} & {fmt_int(t['n'])} & {RR.get(t['ch'], t['r'])} & ${t['mu']:+.4f}$ & {t['auc']:.3f} & {100*t['tpr1']:.1f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TABS / "cohorts.tex").write_text("\n".join(L) + "\n")

    # post-training stages
    pt = M["pt"]["NAT1"]
    order = [s for s in ("mid", "sft", "dpo") if s in pt]
    LBL = {"mid": "Mid-training", "sft": "+ SFT", "dpo": "+ SFT + DPO"}
    L = [r"% generated by figures/paper_figures.py",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Stage & $\hat\tau$ & SD & $d_z$ & \auc & \tprat{1} \\", r"\midrule"]
    for s in order:
        v = pt[s]
        L.append(f"{LBL[s]} & ${v['tau']:+.4f}$ & {v['sd']:.4f} & {v['dz']:.2f} & {v['auc']:.3f} & {100*v['tpr1']:.1f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TABS / "stages.tex").write_text("\n".join(L) + "\n")

    # numbers
    m = [r"% generated by figures/paper_figures.py -- every number the prose cites",
         r"% macro names cannot contain digits, so digits are spelled out: 1->one, 10->onezero, 32->threetwo"]
    DIG = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
           "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    def nm(n): return "".join(DIG.get(ch, ch) for ch in n).replace("-", "")
    def mac(n, v): m.append(f"\\newcommand{{\\{nm(n)}}}{{{v}\\xspace}}")
    mac("npairsdesign", "7{,}814"); mac("npairs", fmt_int(HA["n"])); mac("npairsnat", fmt_int(H["n"]))
    mac("npairsE", fmt_int(HE["n"]))
    for pre, h in (("nat", H), ("all", HA), ("efam", HE)):
        mac(f"tau{pre}", f"{h['tau']:+.4f}"); mac(f"tau{pre}lo", f"{h['lo']:+.4f}"); mac(f"tau{pre}hi", f"{h['hi']:+.4f}")
        mac(f"se{pre}", f"{h['se']:.4f}"); mac(f"dz{pre}", f"{h['dz']:.2f}"); mac(f"sd{pre}", f"{h['sd']:.4f}")
        mac(f"pos{pre}", f"{100*h['pos']:.1f}"); mac(f"auc{pre}", f"{h['auc']:.3f}")
        mac(f"auc{pre}lo", f"{h['auc_lo']:.3f}"); mac(f"auc{pre}hi", f"{h['auc_hi']:.3f}")
        mac(f"tprone{pre}", f"{100*h['tpr1']:.1f}"); mac(f"tprpointone{pre}", f"{100*h['tpr01']:.2f}")
        mac(f"neg{pre}", fmt_int(h["nneg"]))
    mac("permz", f"{M['perm_z']:.0f}"); mac("permp", M["perm_p"])
    mac("npower", str(M["n80"])); mac("npowerall", str(M["n80_all"]))
    mac("fdrfive", str(M["fdr"][0.05])); mac("fdrtwenty", str(M["fdr"][0.20]))
    mac("ppvprev", f"{M['ppv_at_prev']:.1f}")
    for lab, cur in M["coll"].items():
        key = {"published checkpoint": "pub", "retrained reference": "sib", "exact counterfactual": "orc"}[lab]
        for mm, v in cur:
            mac(f"coll{key}m{mm}", f"{v:.3f}")
    for lab, h in M["tier"].items():
        key = {"published checkpoint": "pub", "retrained reference": "sib", "exact counterfactual": "orc"}[lab]
        mac(f"tier{key}tau", f"{h['tau']:+.4f}"); mac(f"tier{key}dz", f"{h['dz']:.2f}")
        mac(f"tier{key}auc", f"{h['auc']:.3f}"); mac(f"tier{key}tpr", f"{100*h['tpr1']:.1f}")
        mac(f"tier{key}npower", str(n80(h["dz"])))
    mac("sibshare", f"{100*(M['tier']['retrained reference']['auc']-0.5)/(M['tier']['exact counterfactual']['auc']-0.5):.0f}")
    mac("driftmean", f"{M['drift']['mean']:+.4f}"); mac("driftsd", f"{M['drift']['sd']:.4f}")
    mac("spillmean", f"{M['spill']['mean']:+.4f}"); mac("spillsd", f"{M['spill']['sd']:.4f}")
    mac("effsd", f"{M['effsd']:.4f}")
    mac("driftvarshare", f"{100*M['drift']['sd']**2/M['effsd']**2:.0f}")
    for it in items:
        k = re.sub(r"[^a-z]", "", it["key"].replace("injected__", "").replace("ratio_", "r").lower())
        mac(f"auc{k}", f"{it['auc']:.3f}"); mac(f"tpr{k}", f"{100*it['tpr1']:.1f}")
    best_free = max((it for it in items if it["group"] == "Reference-free"), key=lambda t: t["auc"])
    mac("bestfreeauc", f"{best_free['auc']:.3f}"); mac("bestfreename", _texesc(best_free["lab"]))
    D = M["dose"]
    mac("doseexponent", f"{D['exponent']:.2f}"); mac("doseexplo", f"{D['exp_lo']:.2f}"); mac("doseexphi", f"{D['exp_hi']:.2f}")
    mac("doser2", f"{D['r2']:.3f}")
    for l in D["levels"]:
        mac(f"dosemu{l['r']}", f"{l['mu']:+.4f}"); mac(f"dosetpr{l['r']}", f"{100*l['tpr1']:.1f}")
        mac(f"dosedz{l['r']}", f"{l['dz']:.2f}"); mac(f"dosen{l['r']}", str(l["n"])); mac(f"doseauc{l['r']}", f"{l['auc']:.3f}")
    for r in M["drift_dom"]:
        k = re.sub(r"[^a-z]", "", r["name"].lower())
        mac(f"drift{k}hold", f"{r['hold']:+.4f}"); mac(f"drift{k}tok", f"{r['tokens']:.1f}")
        if np.isfinite(r["null"]):
            mac(f"drift{k}null", f"{r['null']:+.4f}")
    T = M["traj"]
    mac("trajn", fmt_int(T["n"])); mac("trajlag", f"{T['lag']:.0f}")
    for w in T["windows"]:
        k = f"{w['a']//1000}{w['b']//1000}"
        mac(f"trajpeak{k}", f"{w['peak']:.4f}"); mac(f"trajend{k}", f"{w['end']:.4f}")
        mac(f"trajsurv{k}", f"{100*w['survive']:.0f}"); mac(f"trajsurvlo{k}", f"{100*w['survive_lo']:.0f}")
        mac(f"trajsurvhi{k}", f"{100*w['survive_hi']:.0f}"); mac(f"trajpre{k}", f"{w['pre']:+.4f}")
    for d in T["decay"]:
        mac(f"decay{d['a']//1000}{d['b']//1000}", f"{100*d['ratio']:.0f}")
    S = T["seam"]
    mac("seamjump", f"{S['seam_jump_vs_control']:+.5f}"); mac("seammedian", f"{S['median_other_step_jump_vs_control']:.5f}")
    mac("seamratio", f"{S['seam_ratio_vs_control']:.2f}"); mac("seamratiosib", f"{S['seam_ratio_vs_sibling']:.2f}")
    X = M["xsec"]
    mac("xsecslopeall", f"{X['slope_all']:+.5f}"); mac("xsecslopepre", f"{X['slope_pre']:+.5f}")
    mac("xsecci", f"[{X['lo']:+.5f},\\,{X['hi']:+.5f}]"); mac("xsecn", fmt_int(X["n"]))
    for pop in ("NAT1", "ALL"):
        suf = "" if pop == "NAT1" else "all"
        for s, v in M["pt"][pop].items():
            mac(f"pt{s}tau{suf}", f"{v['tau']:+.4f}"); mac(f"pt{s}dz{suf}", f"{v['dz']:.2f}")
            mac(f"pt{s}auc{suf}", f"{v['auc']:.3f}"); mac(f"pt{s}tpr{suf}", f"{100*v['tpr1']:.1f}")
            mac(f"pt{s}sd{suf}", f"{v['sd']:.4f}"); mac(f"pt{s}pos{suf}", f"{100*v['pos']:.1f}")
            if "corr_with_mid" in v:
                mac(f"pt{s}corr{suf}", f"{v['corr_with_mid']:.2f}")
        pt = M["pt"][pop]
        if "sft" in pt:
            mac(f"ptsftretain{suf}", f"{100*pt['sft']['tau']/pt['mid']['tau']:.0f}")
            mac(f"ptsftsdretain{suf}", f"{100*pt['sft']['sd']/pt['mid']['sd']:.0f}")
        if "dpo" in pt:
            mac(f"ptdporetain{suf}", f"{100*pt['dpo']['tau']/pt['mid']['tau']:.0f}")
            mac(f"ptdpotprgain{suf}", f"{100*(pt['dpo']['tpr1']/pt['mid']['tpr1']-1):+.0f}")
    for ch, row in M["pt"].get("cohort_auc", {}).items():
        k = re.sub(r"[^a-z0-9]", "", ch.lower())
        for st, v in row.items():
            mac(f"ptcoh{k}{st}", f"{v:.3f}")
    tpr = H["tpr1"]
    for prev, nmn in ((0.05, "five"), (0.5, "fifty")):
        mac(f"ppvat{nmn}", f"{100*(tpr*prev)/(tpr*prev+0.01*(1-prev)):.0f}")
    tpr8 = [l for l in M["dose"]["levels"] if l["r"] == 8][0]["tpr1"]
    mac("ppveightprev", f"{100*(tpr8*0.00197)/(tpr8*0.00197+0.01*(1-0.00197)):.0f}")
    mac("ppveightfive", f"{100*(tpr8*0.05)/(tpr8*0.05+0.01*0.95):.0f}")
    pre = [w["pre"] for w in M["traj"]["windows"]]
    mac("trajpreabsmax", f"{max(abs(v) for v in pre):.4f}")

    # domain-stratified operating points: one threshold per cohort, which is
    # what an adversary who knows the domain and holds domain-matched
    # non-members can do; the pooled numbers are the single-threshold case
    coh = cols["cohort"]
    strat_keys = [({"no reference (surprise)": "free", "published checkpoint": "pub",
                    "retrained reference": "sib", "exact counterfactual": "orc"}[lab], key)
                  for lab, key, _, _, _ in TIERS]
    strat_keys += [(k, key) for k, key in (("zorc", "z_oracle"), ("zsib", "z_sibling"), ("zpub", "z_stage1")) if key in cols]
    for k, key in strat_keys:
        a, b = arrays(cols, key, selN)
        tp = tot = named = 0; caught = {}
        for ch in sorted(set(coh[i] for i, _ in selN)):
            sel = [(i, h) for i, h in selN if coh[i] == ch]
            aa, bb = arrays(cols, key, sel)
            t, _, _ = tpr_at(aa, bb, 0.01)
            tp += t * len(sel); tot += len(sel); caught[ch] = t * len(sel)
            cnt = fdr_named(aa, bb, 0.05); named += cnt
            chk = re.sub(r'[^a-z0-9]', '', ch.lower())
            if k == "orc":
                mac(f"coh{chk}named", str(cnt))
            mac(f"coh{chk}named{k}", str(cnt)); mac(f"coh{chk}tpr{k}", f"{100*t:.1f}")
        mac(f"tprstrat{k}", f"{100*tp/tot:.1f}"); mac(f"namedstrat{k}", fmt_int(named))
        mac(f"stratsseshare{k}", f"{100*caught.get('S-se', 0)/tp:.0f}" if tp else "0")
        mac(f"namedpooled{k}", str(fdr_named(a, b, 0.05)))
    a, b = arrays(cols, "ratio_oracle", selN)
    bs = []
    for _ in range(400):
        j = RNG.integers(0, a.size, a.size); bs.append(tpr_at(a[j], b[j], 0.01)[0])
    mac("tpronenatlo", f"{100*np.percentile(bs, 2.5):.1f}"); mac("tpronenathi", f"{100*np.percentile(bs, 97.5):.1f}")
    from scipy import stats as _st
    d = a - b; d = d[np.isfinite(d)]; rej = 0; n13 = M["n80"]
    for _ in range(4000):
        x = d[RNG.integers(0, d.size, n13)]; rej += x.mean() / (x.std(ddof=1) / math.sqrt(n13)) > _st.t.ppf(0.95, n13 - 1)
    mac("powerempirical", f"{100*rej/4000:.0f}")
    mac("seedsharetwo", f"{100*2*M['drift']['sd']**2/M['effsd']**2:.0f}")
    pt = M["pt"]["NAT1"]
    if "dpo" in pt:
        mac("ptdpodzretain", f"{100*pt['dpo']['dz']/pt['mid']['dz']:.0f}")
        mac("ptdpoaucretain", f"{100*(pt['dpo']['auc']-0.5)/(pt['mid']['auc']-0.5):.0f}")

    for n_, v_ in M.get("calib_macros", []):
        mac(n_, v_)
    N = M["noise"]
    mac("tokshare", f"{100*N['share']:.0f}"); mac("toksharearxiv", f"{100*N['share_M']:.0f}")
    mac("tokshareshort", f"{100*N['share_short']:.0f}")
    for t in M["cohort"]:
        k = re.sub(r"[^a-z0-9]", "", t["ch"].lower())
        mac(f"coh{k}auc", f"{t['auc']:.3f}"); mac(f"coh{k}tau", f"{t['mu']:+.4f}"); mac(f"coh{k}n", fmt_int(t["n"]))
        mac(f"coh{k}tpr", f"{100*t['tpr1']:.1f}")
    (TABS / "numbers.tex").write_text("\n".join(m) + "\n")
    print(f"  tables/: attacks.tex cohorts.tex stages.tex numbers.tex ({len(m)-1} macros)")


# -------------------------------------------------------------------- main
def set_output(out) -> None:
    """Write figures to <out>/figures and tables to <out>/tables (--out)."""
    global PAPER, FIGS, TABS
    PAPER = Path(out)
    FIGS = PAPER / "figures"
    TABS = PAPER / "tables"


def main(first_pass: bool = False) -> int:
    setup()
    print("loading analysis table ...")
    cols, rows, arm, pairs = load()
    selN = population(cols, pairs, "NAT1"); selA = population(cols, pairs, "ALL"); selE = population(cols, pairs, "E")
    print(f"  {len(rows):,} documents; pairs: all {len(pairs):,}, verified-unseen single-copy {len(selN):,}, E family {len(selE):,}")
    M = {}
    H = headline(cols, selN); HA = headline(cols, selA); HE = headline(cols, selE)
    for lab, h in (("NAT1", H), ("ALL", HA), ("E", HE)):
        print(f"  {lab:5s} n={h['n']} tau={h['tau']:+.5f} se={h['se']:.5f} dz={h['dz']:.3f} pos={100*h['pos']:.1f}% "
              f"AUC={h['auc']:.4f} TPR@1%={100*h['tpr1']:.2f}% (k={h['k1']}) TPR@0.1%={100*h['tpr01']:.2f}% (k={h['k01']})")
    a, b = arrays(cols, "ratio_oracle", selN)
    p, z = signflip_p(a - b)
    M["perm_p"] = f"<{p:.0e}".replace("e-0", "\\times10^{-") + "}" if p < 1e-4 else f"={p:.2g}"
    M["perm_p"] = r"<10^{-5}" if p <= 1.1e-5 else f"={p:.1e}"
    M["perm_z"] = z
    M["n80"] = n80(H["dz"]); M["n80_all"] = n80(HA["dz"])
    M["fdr"] = {q: fdr_named(a, b, q) for q in (0.05, 0.20)}
    print(f"  sign-flip z={z:.1f}, p{M['perm_p']}; n80={M['n80']} (all-pairs dz gives {M['n80_all']}); "
          f"FDR-nameable {M['fdr']}")
    # token-noise share (mean of per-pair token variance / Var(D)), single copy
    Z = np.load(NEFF, allow_pickle=True)
    pos = {d: i for i, d in enumerate(Z["doc_ids"])}
    docid = cols["doc_id"]; coh = cols["cohort"]
    def share(sel):
        tv, dd = [], []
        for i, h in sel:
            pi, ph = pos.get(docid[i]), pos.get(docid[h])
            if pi is None or ph is None:
                continue
            v = Z["delta_sd"][pi] ** 2 / Z["n_eff"][pi] + Z["delta_sd"][ph] ** 2 / Z["n_eff"][ph]
            if np.isfinite(v):
                tv.append(v); dd.append(cols["ratio_oracle"][i] - cols["ratio_oracle"][h])
        dd = np.array(dd); return float(np.mean(tv) / dd.var(ddof=1))
    M["noise"] = dict(share=share(selN), share_M=share([(i, h) for i, h in selN if coh[i] == "M"]),
                      share_short=share([(i, h) for i, h in selN if coh[i] in ("W-new", "W-gap", "S-se")]))
    print(f"  token-noise share of within-pair variance: {100*M['noise']['share']:.0f}% (arXiv {100*M['noise']['share_M']:.0f}%, short {100*M['noise']['share_short']:.0f}%)")
    print("figures:")
    fig_intro(cols, selN, M)
    fig_signal(cols, arm, selN, H)
    fig_roc(cols, selN)
    items = fig_attacks(cols, selN)
    fig_tiers(cols, arm, selN, M)
    fig_dose(cols, pairs, M)
    fig_drift(cols, arm, M)
    fig_cohort(cols, pairs, M)
    fig_trajectory(M)
    fig_posttrain(cols, selN, selA, M)
    fig_collection(cols, selN, H, M)
    print("tables:")
    if first_pass:
        # checklist() reads the five analysis JSONs and draws no random numbers;
        # nothing above it depends on them, so NAT1, ALL, E and coll below equal
        # the full run's values
        M["calib_macros"] = calibrated(cols, selN) + validation(cols, selN)
    else:
        M["calib_macros"] = calibrated(cols, selN) + validation(cols, selN) + checklist(cols, selN)
    M["calib_macros"].append(("npairsanalysed", fmt_int(len(pairs))))
    if first_pass:
        print("  first pass: checklist() and write_tables() skipped; run the analyses, then this script again")
    else:
        write_tables(items, M, H, HA, HE, cols, selN)
    (ANA / "paper_numbers.json").write_text(json.dumps(
        {"NAT1": H, "ALL": HA, "E": HE, **{k: v for k, v in M.items() if k != "cohort"}, "cohort": M["cohort"]},
        indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Regenerate the paper's tables, macro file (numbers.tex), "
                                             "data figures and analysis/paper_numbers.json.")
    ap.add_argument("--out", type=Path, default=paths.PAPER_OUT,
                    help="output directory; figures/ and tables/ are written below it "
                         "(default: %(default)s, i.e. paper/ under the data root)")
    ap.add_argument("--first-pass", action="store_true",
                    help="fresh data root: skip checklist() and write_tables(); the figures, "
                         "tables/calibrated.tex and analysis/paper_numbers.json are still written")
    args = ap.parse_args()
    set_output(args.out)
    raise SystemExit(main(first_pass=args.first_pass))
