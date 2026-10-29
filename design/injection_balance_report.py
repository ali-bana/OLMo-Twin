#!/usr/bin/env python3
"""Arm-balance report for the injection set: inject vs holdout, per cohort,
under the pre-treatment checkpoint and the released (control) model.

Nothing has been injected yet, so the two arms of every cohort must have the
same log-prob distribution under BOTH checkpoints and the same
pre-treatment -> released shift. Any difference now is sampling imbalance
(or a bug in pairing), not membership.

Per document, from the scorer's per-token arrays (NaN chunk starts dropped):
  mean_lp, bpb, median_lp, p1/p5/p10/p25/p90/p99 of token log-probs,
  mink20 (mean of the lowest 20% of token log-probs, the Min-K% statistic),
  frac_lt_m10 (share of tokens below -10 nats), n_tokens
under each checkpoint, plus the per-document shift d_* = main - stage1.

Per cohort (and per stratum), inject vs holdout: n, mean, median, sd,
standardized mean difference (Cohen's d), Mann-Whitney U and KS p-values,
and the within-pair difference (inject - holdout) with a 95% CI.

Outputs: <out>/per_document.csv.gz, arm_balance.csv, arm_balance_by_stratum.csv,
figures/*.png, REPORT.md.

Defaults: --injection-dir is the bundle/ directory of the data root;
--audit-dir and --edited-dir are audit_v2/ and wiki_edited/ under its
collection/ directory (see olmo_twin/paths.py).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# repository root on sys.path, so that `olmo_twin` imports from a checkout
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

W = paths.COLLECTION   # collection working directory (audit_v2/, wiki_edited/)
COHORT_ORDER = ["M", "D", "D32", "DB", "L", "F", "X", "W-new", "W-gap", "S-se", "E0", "E0-D", "E"]
STATS = ["mean_lp", "bpb", "median_lp", "p1", "p5", "p10", "p25", "p90", "p99", "mink20", "frac_lt_m10"]
PRIMARY = ["mean_lp", "bpb", "mink20", "p5"]


def safe(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace(":", "_")


def rd(p: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with gzip.open(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def token_stats(lp: np.ndarray, n_bytes: int | None) -> dict:
    v = lp[~np.isnan(lp)]
    if v.size == 0:
        return {}
    k = max(1, int(round(0.2 * v.size)))
    q = np.percentile(v, [1, 5, 10, 25, 50, 90, 99])
    d = {"mean_lp": float(v.mean()), "median_lp": float(q[4]), "p1": float(q[0]), "p5": float(q[1]),
         "p10": float(q[2]), "p25": float(q[3]), "p90": float(q[5]), "p99": float(q[6]),
         "mink20": float(np.sort(v)[:k].mean()), "frac_lt_m10": float((v < -10).mean()),
         "n_tokens_scored": int(v.size), "nll_sum": float(-v.sum())}
    if n_bytes:
        d["bpb"] = float(-v.sum() / np.log(2) / n_bytes)
    return d


def locate(row: dict, arm_form: str, dirs: dict) -> tuple[Path, Path, str] | None:
    """(stage1 npz, main npz, doc key) for the text the arm is scored on."""
    c, did = row["cohort"], row["doc_id"]
    if c in ("L", "X"):
        key = safe(f"{c}:{did}:inj")
        return dirs["extra_s1"] / f"{key}.npz", dirs["extra_main"] / f"{key}.npz", key
    if c == "F" and (row["arm"] == "inject" or arm_form == "latex"):
        # the LaTeX form was scored under either suffix depending on the arm at scoring time
        for suf in ("inj", "latex"):
            key = safe(f"F:{did}:{suf}")
            if (dirs["extra_s1"] / f"{key}.npz").exists():
                return dirs["extra_s1"] / f"{key}.npz", dirs["extra_main"] / f"{key}.npz", key
        key = safe(f"F:{did}:inj")
        return dirs["extra_s1"] / f"{key}.npz", dirs["extra_main"] / f"{key}.npz", key
    if c.startswith("E"):
        key = safe(did)
        return dirs["ed_s1"] / f"{key}.npz", dirs["ed_main"] / f"{key}.npz", key
    key = safe(did)
    return dirs["a_s1"] / f"{key}.npz", dirs["a_main"] / f"{key}.npz", key


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / max(a.size + b.size - 2, 1))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def compare(df: pd.DataFrame, col: str) -> dict:
    a = df.loc[df.arm == "inject", col].dropna().to_numpy()
    b = df.loc[df.arm == "holdout", col].dropna().to_numpy()
    if a.size < 3 or b.size < 3:
        return {}
    # within-pair difference
    piv = df.pivot_table(index="pair_id", columns="arm", values=col)
    piv = piv.dropna()
    diff = (piv["inject"] - piv["holdout"]).to_numpy()
    se = diff.std(ddof=1) / np.sqrt(diff.size) if diff.size > 1 else np.nan
    return {
        f"{col}__inj_mean": a.mean(), f"{col}__hold_mean": b.mean(),
        f"{col}__inj_median": np.median(a), f"{col}__hold_median": np.median(b),
        f"{col}__inj_sd": a.std(ddof=1), f"{col}__hold_sd": b.std(ddof=1),
        f"{col}__cohen_d": cohens_d(a, b),
        f"{col}__mwu_p": stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
        f"{col}__ks_p": stats.ks_2samp(a, b).pvalue,
        f"{col}__pair_diff_mean": diff.mean(), f"{col}__pair_diff_ci95": 1.96 * se,
        f"{col}__pair_diff_wilcoxon_p": stats.wilcoxon(diff).pvalue if diff.size > 10 and np.any(diff != 0) else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--injection-dir", type=Path, default=paths.BUNDLE)
    ap.add_argument("--audit-dir", type=Path, default=W / "audit_v2")
    ap.add_argument("--edited-dir", type=Path, default=W / "wiki_edited")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    I = args.injection_dir
    dirs = {"a_s1": args.audit_dir / "logprobs_stage1" / "tokens", "a_main": args.audit_dir / "logprobs_main" / "tokens",
            "ed_s1": args.edited_dir / "logprobs_stage1" / "tokens", "ed_main": args.edited_dir / "logprobs_main" / "tokens",
            "extra_s1": I / "logprobs_extra_stage1" / "tokens", "extra_main": I / "logprobs_extra_main" / "tokens"}
    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    man = rd(I / "manifest.csv.gz")

    # ---- per document ------------------------------------------------------
    recs, missing = [], defaultdict(int)
    for r in man:
        forms = [("as_injected", None)]
        if r["cohort"] == "F" and r["arm"] == "holdout":
            forms.append(("latex", "latex"))
        if r["cohort"] == "F" and r["arm"] == "inject":
            forms.append(("pdf", "pdf"))
        for form_name, form in forms:
            if form == "pdf":
                key = safe(r["doc_id"])
                p1, p2 = dirs["a_s1"] / f"{key}.npz", dirs["a_main"] / f"{key}.npz"
            else:
                p1, p2, key = locate(r, form, dirs)
            if not (p1.exists() and p2.exists()):
                missing[r["cohort"]] += 1
                continue
            a, b = np.load(p1), np.load(p2)
            assert np.array_equal(a["token_ids"], b["token_ids"]), f"token ids differ across checkpoints: {key}"
            text_path = {"latex": r["alt_text_path"], "pdf": r["text_path"]}.get(form, r["inject_text_path"])
            n_bytes = len(Path(text_path).read_bytes()) if text_path and Path(text_path).exists() else None
            s1, mn = token_stats(a["logprobs"], n_bytes), token_stats(b["logprobs"], n_bytes)
            rec = {"doc_id": r["doc_id"], "cohort": r["cohort"], "stratum": r["stratum"], "arm": r["arm"],
                   "pair_id": r["pair_id"], "split": r["split"], "r": r["r"], "form": form_name,
                   "n_tokens": s1.get("n_tokens_scored")}
            for k in STATS:
                rec[f"s1_{k}"] = s1.get(k, np.nan)
                rec[f"main_{k}"] = mn.get(k, np.nan)
                rec[f"d_{k}"] = mn.get(k, np.nan) - s1.get(k, np.nan)
            recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(out / "per_document.csv.gz", index=False)
    print(f"per-document rows: {len(df):,}; missing arrays by cohort: {dict(missing)}")

    # ---- per cohort comparisons -------------------------------------------
    cols = [f"{ck}_{k}" for ck in ("s1", "main", "d") for k in STATS]
    rows_c, rows_s = [], []
    for c in COHORT_ORDER:
        sub = df[(df.cohort == c) & (df.form == "as_injected")]
        if sub.empty:
            continue
        label = c if c != "F" else "F (as injected: LaTeX inject vs PDF holdout — expected to differ)"
        row = {"cohort": label, "n_inject": int((sub.arm == "inject").sum()), "n_holdout": int((sub.arm == "holdout").sum())}
        for col in cols:
            row.update(compare(sub, col))
        rows_c.append(row)
        for s, ss in sub.groupby("stratum"):
            row = {"cohort": c, "stratum": s, "n_inject": int((ss.arm == "inject").sum()),
                   "n_holdout": int((ss.arm == "holdout").sum())}
            for col in cols:
                row.update(compare(ss, col))
            rows_s.append(row)
    # F: both arms on the LaTeX form (inject as injected = latex; holdout latex form)
    f_latex = pd.concat([df[(df.cohort == "F") & (df.arm == "inject") & (df.form == "as_injected")],
                         df[(df.cohort == "F") & (df.arm == "holdout") & (df.form == "latex")]])
    if not f_latex.empty:
        row = {"cohort": "F (LaTeX form, both arms)", "n_inject": int((f_latex.arm == "inject").sum()),
               "n_holdout": int((f_latex.arm == "holdout").sum())}
        for col in cols:
            row.update(compare(f_latex, col))
        rows_c.append(row)
    f_pdf = pd.concat([df[(df.cohort == "F") & (df.arm == "inject") & (df.form == "pdf")],
                       df[(df.cohort == "F") & (df.arm == "holdout") & (df.form == "as_injected")]])
    if not f_pdf.empty:
        row = {"cohort": "F (PDF form, both arms)", "n_inject": int((f_pdf.arm == "inject").sum()),
               "n_holdout": int((f_pdf.arm == "holdout").sum())}
        for col in cols:
            row.update(compare(f_pdf, col))
        rows_c.append(row)
    bal = pd.DataFrame(rows_c)
    bal["noise_sd_d"] = np.sqrt(1 / bal.n_inject + 1 / bal.n_holdout)   # SD of Cohen's d under pure randomization
    bal.to_csv(out / "arm_balance.csv", index=False)
    pd.DataFrame(rows_s).to_csv(out / "arm_balance_by_stratum.csv", index=False)

    # ---- figures ----------------------------------------------------------
    plot_df = pd.concat([df[(df.form == "as_injected") & (df.cohort != "F")], f_latex])   # F shown on its LaTeX form
    cohorts = [c for c in COHORT_ORDER if c in set(plot_df.cohort)]
    colors = {"inject": "#d62728", "holdout": "#1f77b4"}

    def box_panel(ax, ck, stat, title, cohort_list=None):
        data, pos, labels = [], [], []
        for i, c in enumerate(cohort_list or [c for c in cohorts if c != "X"]):   # X (anchors) has its own figure
            for j, arm in enumerate(("inject", "holdout")):
                v = plot_df[(plot_df.cohort == c) & (plot_df.arm == arm)][f"{ck}_{stat}"].dropna()
                data.append(v.to_numpy()); pos.append(i * 3 + j)
            labels.append(c)
        bp = ax.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True)
        for k, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors["inject"] if k % 2 == 0 else colors["holdout"]); patch.set_alpha(0.6)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(labels))]); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)

    for stat, ylabel in (("mean_lp", "mean token log-prob (nats)"), ("bpb", "bits per byte"),
                         ("mink20", "Min-K% (k=20) mean log-prob"), ("p5", "5th-percentile token log-prob")):
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=False)
        box_panel(axes[0], "s1", stat, f"pre-treatment checkpoint — {ylabel}")
        box_panel(axes[1], "main", stat, f"released model (control) — {ylabel}")
        box_panel(axes[2], "d", stat, f"shift main − pre-treatment — {ylabel}")
        axes[0].set_ylabel(ylabel)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a], alpha=0.6) for a in ("inject", "holdout")]
        axes[0].legend(handles, ["inject", "holdout"], loc="lower left")
        fig.suptitle(f"Inject vs holdout per cohort, before any injection: {stat}", y=1.02)
        fig.tight_layout(); fig.savefig(out / "figures" / f"box_{stat}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # anchors (X) by stratum, three checkpoints views
    if "X" in cohorts:
        xs = plot_df[plot_df.cohort == "X"]
        strata = sorted(xs.stratum.unique())
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (ck, title) in zip(axes, (("s1", "pre-treatment"), ("main", "released"), ("d", "shift main − pre"))):
            data, pos = [], []
            for i, st in enumerate(strata):
                for j, arm in enumerate(("inject", "holdout")):
                    data.append(xs[(xs.stratum == st) & (xs.arm == arm)][f"{ck}_mean_lp"].dropna().to_numpy()); pos.append(i * 3 + j)
            bp = ax.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True)
            for k, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors["inject"] if k % 2 == 0 else colors["holdout"]); patch.set_alpha(0.6)
            ax.set_xticks([i * 3 + 0.5 for i in range(len(strata))]); ax.set_xticklabels(strata, rotation=30, ha="right")
            ax.set_title(f"X anchors — {title} mean token log-prob"); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "figures" / "box_anchors_X.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ECDF overlays of mean_lp per cohort, both checkpoints
    fig, axes = plt.subplots(3, 5, figsize=(22, 11))
    for ax, c in zip(axes.flat, cohorts):
        for ck, ls in (("s1", "-"), ("main", "--")):
            for arm in ("inject", "holdout"):
                v = np.sort(plot_df[(plot_df.cohort == c) & (plot_df.arm == arm)][f"{ck}_mean_lp"].dropna().to_numpy())
                if v.size:
                    ax.step(v, np.arange(1, v.size + 1) / v.size, ls, color=colors[arm], lw=1.4,
                            label=f"{arm} ({'pre' if ck == 's1' else 'released'})")
        ax.set_title(c); ax.grid(alpha=0.3); ax.set_xlabel("mean token log-prob")
    for ax in axes.flat[len(cohorts):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("ECDF of per-document mean log-prob: inject vs holdout, pre-treatment (solid) and released (dashed)", y=1.0)
    fig.tight_layout(); fig.savefig(out / "figures" / "ecdf_mean_lp.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # per-cohort scatter pre vs released, colour by arm
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    for ax, c in zip(axes.flat, cohorts):
        for arm in ("inject", "holdout"):
            s = plot_df[(plot_df.cohort == c) & (plot_df.arm == arm)]
            ax.scatter(s["s1_mean_lp"], s["main_mean_lp"], s=6, alpha=0.5, color=colors[arm], label=arm)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k:", lw=1)
        ax.set_title(c); ax.set_xlabel("pre-treatment mean lp"); ax.set_ylabel("released mean lp"); ax.grid(alpha=0.3)
    for ax in axes.flat[len(cohorts):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "scatter_pre_vs_released.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # standardized mean differences summary
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(cohorts))
    for k, (ck, lab) in enumerate((("s1", "pre-treatment"), ("main", "released"), ("d", "shift"))):
        lookup = {c: (c if c != "F" else "F (LaTeX form, both arms)") for c in cohorts}
        dvals = [bal.loc[bal.cohort == lookup[c], f"{ck}_mean_lp__cohen_d"].values[0] for c in cohorts]
        ax.bar(x + (k - 1) * 0.27, dvals, width=0.27, label=f"{lab}: mean_lp")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(0.1, color="grey", ls=":"); ax.axhline(-0.1, color="grey", ls=":")
    ax.set_xticks(x); ax.set_xticklabels(cohorts, rotation=45, ha="right"); ax.set_ylabel("Cohen's d (inject − holdout)")
    ax.set_title("Standardized mean difference between arms (|d| < 0.1 dotted = negligible)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "figures" / "cohens_d_mean_lp.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # strata figure for cohorts with manipulations
    strata_cohorts = [c for c in ("D", "L", "E", "X", "M") if c in cohorts]
    fig, axes = plt.subplots(1, len(strata_cohorts), figsize=(5 * len(strata_cohorts), 5))
    for ax, c in zip(np.atleast_1d(axes), strata_cohorts):
        sub = plot_df[plot_df.cohort == c]
        strata = sorted(sub.stratum.unique(), key=lambda s: (len(s), s))
        data, pos = [], []
        for i, s in enumerate(strata):
            for j, arm in enumerate(("inject", "holdout")):
                data.append(sub[(sub.stratum == s) & (sub.arm == arm)]["s1_mean_lp"].dropna().to_numpy()); pos.append(i * 3 + j)
        bp = ax.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True)
        for k, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors["inject"] if k % 2 == 0 else colors["holdout"]); patch.set_alpha(0.6)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(strata))]); ax.set_xticklabels(strata, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"{c} by stratum (pre-treatment mean lp)"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "figures" / "strata_pre_mean_lp.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---- report -------------------------------------------------------------
    lines = ["# Arm balance of the injection set before injection\n",
             "Nothing has been injected yet: inject and holdout arms of every cohort must be exchangeable under "
             "both the pre-treatment checkpoint (`stage1-step1907359-tokens4001B`) and the released model (`main`, "
             "the control), and their pre→released shift must be equal. Per-document statistics come from the "
             "scorer's per-token arrays (NaN chunk starts dropped). `per_document.csv.gz` has every number; "
             "`arm_balance.csv` and `arm_balance_by_stratum.csv` the comparisons; figures in `figures/`.\n",
             "Columns: d = Cohen's d (inject − holdout; |d| < 0.1 negligible), MWU = Mann–Whitney p, KS = "
             "Kolmogorov–Smirnov p, Δpair = mean within-pair difference inject − holdout ± 95% CI.\n"]
    for ck, lab in (("s1", "Pre-treatment checkpoint"), ("main", "Released model (control)"), ("d", "Shift released − pre-treatment")):
        lines.append(f"\n## {lab}\n")
        lines.append("| cohort | n/arm | stat | inject mean | holdout mean | d | MWU p | KS p | Δpair ± CI |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in bal.iterrows():
            for k in PRIMARY:
                col = f"{ck}_{k}"
                if f"{col}__inj_mean" not in r or pd.isna(r[f"{col}__inj_mean"]):
                    continue
                lines.append(f"| {r['cohort']} | {int(r['n_inject'])}/{int(r['n_holdout'])} | {k} | {r[f'{col}__inj_mean']:.4f} | "
                             f"{r[f'{col}__hold_mean']:.4f} | {r[f'{col}__cohen_d']:+.3f} | {r[f'{col}__mwu_p']:.2f} | "
                             f"{r[f'{col}__ks_p']:.2f} | {r[f'{col}__pair_diff_mean']:+.4f} ± {r[f'{col}__pair_diff_ci95']:.4f} |")
    # flags
    flagged = []
    for _, r in bal.iterrows():
        if "expected to differ" in r["cohort"]:
            continue
        for ck in ("s1", "main", "d"):
            for k in PRIMARY:
                col = f"{ck}_{k}"
                if f"{col}__cohen_d" in r and not pd.isna(r[f"{col}__cohen_d"]):
                    ci_excl = abs(r[f"{col}__pair_diff_mean"]) > r[f"{col}__pair_diff_ci95"]
                    if ci_excl or r[f"{col}__ks_p"] < 0.01:
                        flagged.append(f"- {r['cohort']} / {ck}_{k}: d = {r[f'{col}__cohen_d']:+.3f} "
                                       f"(randomization SD of d ≈ {r['noise_sd_d']:.3f}), KS p = {r[f'{col}__ks_p']:.3f}, "
                                       f"Δpair = {r[f'{col}__pair_diff_mean']:+.4f} ± {r[f'{col}__pair_diff_ci95']:.4f}")
    lines.append("\n## Flags (within-pair 95% CI excludes 0, or KS p < 0.01)\n")
    lines.append("Cohen's d under pure randomization has SD ≈ sqrt(1/n₁ + 1/n₂): 0.032 for 1,986 pairs, "
                 "0.089 for 250, 0.22 for 40. |d| of that size is noise, not drift.\n")
    lines += flagged or ["- none"]
    lines.append("\n## Figures\n")
    for f in sorted((out / "figures").glob("*.png")):
        lines.append(f"- `figures/{f.name}`")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    json.dump({"documents": len(df), "cohorts": cohorts, "flags": flagged, "missing": dict(missing)},
              open(out / "summary.json", "w"), indent=1)
    print("\n".join(lines[:3]))
    print(f"flags: {len(flagged)}")
    for f in flagged:
        print(f)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
