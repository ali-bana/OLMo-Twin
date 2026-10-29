"""Shared helpers of the analysis scripts and the figure generators.

Reading the per-document attack table (with the post-training and cohort-F
exclusions), rank AUC, ROC, TPR at fixed FPR, bootstrap intervals, and one
plotting style (palette, grid, fonts) for the figures.

Palette roles (fixed, never cycled):
    injected / member      blue    #2a78d6
    control / non-member   orange  #eb6834
    pre-treatment          aqua    #1baf7a
    null pool              yellow  #eda100
    neutral / annotation   grey    #52514e
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {
    "injected": "#2a78d6",
    "holdout": "#eb6834",
    "pretrained": "#1baf7a",
    "null": "#eda100",
    "neutral": "#52514e",
    "grid": "#d8d6d2",
    "accent": "#8b5cf6",
    "bad": "#c2352b",
    "good": "#1baf7a",
}

# ordered, never cycled: one colour per dose level
DOSE_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
# ordered categorical for cohorts (13) -- perceptually spaced, CVD-checked pairs
COHORT_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5cf6", "#c2352b", "#0e7490",
    "#b45309", "#4d7c0f", "#be185d", "#525252", "#0369a1", "#7c2d12",
]


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8783",
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.color": "#4a4845",
        "ytick.color": "#4a4845",
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def finish(fig, out_dir, name: str, caption: str = "") -> Path:
    """Save a figure and its caption next to it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    if caption:
        (out_dir / f"{name}.md").write_text(caption.strip() + "\n")
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------- statistics
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney), ties at 1/2. Higher score = more member-like."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    pooled = np.concatenate([pos, neg])
    order = pooled.argsort(kind="mergesort")
    ranks = np.empty(pooled.size, float)
    ranks[order] = np.arange(1, pooled.size + 1)
    sv = pooled[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    u = ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def roc(pos: np.ndarray, neg: np.ndarray, n_points: int = 4000):
    """Empirical ROC as (fpr, tpr) arrays, thresholds swept over pooled values."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.array([0, 1.0]), np.array([0, 1.0])
    thr = np.unique(np.concatenate([pos, neg]))
    if thr.size > n_points:
        thr = np.quantile(thr, np.linspace(0, 1, n_points))
    thr = np.sort(thr)[::-1]
    tpr = np.searchsorted(np.sort(pos), thr, side="left")
    tpr = 1.0 - tpr / pos.size
    fpr = np.searchsorted(np.sort(neg), thr, side="left")
    fpr = 1.0 - fpr / neg.size
    return np.concatenate([[0], fpr, [1]]), np.concatenate([[0], tpr, [1]])


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, target_fpr: float) -> float:
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - target_fpr)
    return float((pos > thr).mean())


def bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 7, alpha: float = 0.05):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n, values.size))
    boots = stat(values[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def auc_ci(pos, neg, n: int = 500, seed: int = 11, alpha: float = 0.05):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if pos.size < 2 or neg.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        p = pos[rng.integers(0, pos.size, pos.size)]
        q = neg[rng.integers(0, neg.size, neg.size)]
        vals.append(auc(p, q))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


EXCLUSIONS = paths.ANALYSIS / "posttrain_exclusions.json"


def posttrain_exclusions():
    """Pair ids and document ids found in OLMo 2's post-training data.

    The 2026-09-29 scan searched all four post-training datasets of
    OLMo-2-0425-1B-Instruct with the same 8-word matcher and the same
    fingerprint table as the 13 pre/mid-training components. 16 documents in use
    were flagged, touching 10 pairs (0.13%) and 5 null-pool documents. They are
    dropped from EVERY analysis, not just the post-training stages: a document
    flagged here is only genuinely contaminated after SFT/DPO, since the
    mid-training models never saw those mixes, but excluding globally costs
    0.13% of the estimator and means one document set is used throughout.

    The file is required: if it is missing this raises FileNotFoundError
    instead of silently analysing the unfiltered table. It is written by
    collection/make_posttrain_exclusions.py from posttrain_hits.csv.gz.
    """
    import json
    if not EXCLUSIONS.exists():
        raise FileNotFoundError(
            f"{EXCLUSIONS} is missing; create it with collection/make_posttrain_exclusions.py "
            "(from posttrain_overlap/posttrain_hits.csv.gz) or copy it from the data release")
    d = json.loads(EXCLUSIONS.read_text())
    return set(d.get("exclude_pair_ids", [])), set(d.get("exclude_doc_ids", []))


COHORTF_EXCL = paths.ANALYSIS / "cohortF_data_exclusions.json"


def cohortF_exclusions():
    """Cohort-F pairs and documents with data-quality defects, not contamination.

    Written by analysis/audit_cohortF_data.py from two rules: a LaTeX
    rendering that is a fragment, and a paper whose two renderings share no
    content. Root cause is that F's 1,024-32,768 token window was checked
    against the pre-treatment PDF rendering while the form actually injected was
    the LaTeX one -- F is the only cohort where those differ.
    """
    import json
    if not COHORTF_EXCL.exists():
        return set(), set()
    d = json.loads(COHORTF_EXCL.read_text())
    return set(d.get("exclude_pair_ids", [])), set(d.get("exclude_doc_ids", []))


def load_table(path, drop_posttrain_flagged=True):
    """Read the attack table into a dict of numpy arrays (no pandas dependency).

    Rows flagged by the post-training overlap scan are dropped by default --
    both arms of an affected pair, and flagged null-pool documents. Pass
    drop_posttrain_flagged=False to reproduce the unfiltered numbers (used for
    the robustness comparison).
    """
    import csv, gzip, sys
    csv.field_size_limit(sys.maxsize)
    with gzip.open(path, "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if drop_posttrain_flagged and rows:
        bad_pairs, bad_docs = posttrain_exclusions()
        if bad_pairs or bad_docs:
            before = len(rows)
            rows = [r for r in rows
                    if (r.get("pair_id") or "") not in bad_pairs
                    and r.get("doc_id") not in bad_docs]
            n = before - len(rows)
            if n:
                print(f"  [posttrain exclusions] dropped {n} of {before} rows "
                      f"({len(bad_pairs)} pairs, {len(bad_docs)} flagged docs)")
    cols = {}
    if not rows:
        return cols, rows
    for k in rows[0]:
        raw = [r.get(k, "") for r in rows]
        try:
            cols[k] = np.array([float(x) if x not in ("", None) else np.nan for x in raw])
        except (TypeError, ValueError):
            cols[k] = np.array(raw, dtype=object)
    return cols, rows
