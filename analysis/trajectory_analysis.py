"""Event-aligned forgetting analysis on the twin-ladder trajectory.

Shared by the figure generators (figures/paper_figures.py and
figures/published_figures.py) so the two cannot drift apart.

Why event-aligned. The first version of the trajectory analysis grouped
single-copy documents by a 4,000-update placement window, averaged their
within-pair effect at every checkpoint, and read the "peak" off the mean curve
at the window's upper edge. Documents placed early in a window have already
decayed for up to 4,000 updates by then, so that peak understates the
per-document peak and overstates the fraction surviving, by an amount that
differs between windows. Here every document is aligned to
the first checkpoint at or after its own placement step: its peak is the value
at that checkpoint, its endpoint is the value at step 23,852, and the last
checkpoint BEFORE placement is its own placebo.

What this still cannot do. Checkpoints are 1,000 updates apart, so the first
post-placement checkpoint is on average ~500 updates after the copy was seen,
and the true peak -- immediately after the gradient step -- is higher than the
value we read. Every "surviving" fraction reported from this module is
therefore an UPPER bound on retention.
"""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path

import numpy as np

ENDPOINT = 23852
WINDOWS = [(0, 4000), (4000, 8000), (8000, 12000), (12000, 16000),
           (16000, 20000), (20000, 23000)]


def load_ladder(path, pair_filter=None):
    """Return (steps, TAU[n_pairs, n_steps], FU[n_pairs], LR[n_steps], meta).

    TAU is the within-pair difference (injected minus matched holdout) of
    treated-minus-control mean log-probability, for every single-copy pair with
    a known placement step. pair_filter(row_of_injected_doc) -> bool restricts
    the population (e.g. to the verified-unseen cohorts).
    """
    with gzip.open(path, "rt", newline="") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        cols = rd.fieldnames
    steps = np.array(sorted(int(m.group(1)) for m in
                            (re.fullmatch(r"delta_s(\d+)", c) for c in cols) if m))

    def col(c):
        return np.array([float(x[c]) if x.get(c) not in ("", None, "nan") else np.nan
                         for x in rows])

    arm = np.array([x.get("arm", "") for x in rows])
    pair = np.array([x.get("pair_id", "") for x in rows])
    r_, fu = col("r"), col("first_update")
    D = np.column_stack([col(f"delta_s{s}") for s in steps])
    lr = np.array([np.nanmean(col(f"lr_s{s}")) for s in steps])

    single = (arm == "inject") & (r_ == 1) & np.isfinite(fu)
    # the same post-training-overlap exclusions every other analysis applies
    try:
        import injection_v2_plotstyle as ps
        bad_pairs, bad_docs = ps.posttrain_exclusions()
        single &= ~np.isin(pair, list(bad_pairs))
    except FileNotFoundError:
        raise  # posttrain_exclusions.json missing: fail closed, as every other analysis does
    except Exception:
        pass
    if pair_filter is not None:
        single &= np.array([bool(pair_filter(x)) for x in rows])
    idx = {}
    for i, (a, p) in enumerate(zip(arm, pair)):
        if a in ("inject", "holdout") and p:
            idx.setdefault(p, {})[a] = i
    PI, PH = [], []
    for v in idx.values():
        if "inject" in v and "holdout" in v and single[v["inject"]]:
            PI.append(v["inject"]); PH.append(v["holdout"])
    PI, PH = np.array(PI), np.array(PH)
    TAU = D[PI] - D[PH]
    FU = fu[PI]
    return steps, TAU, FU, lr, {"n_pairs": int(PI.size), "rows": rows, "PI": PI, "PH": PH}


def event_align(steps, TAU, FU):
    """Per-document peak / pre / end, aligned to each document's own placement.

    Returns dict of arrays over pairs: k0 (index of first checkpoint >= FU),
    peak, pre (value at the last checkpoint before placement; nan if none),
    end, lag (steps[k0] - FU), valid (k0 points at a checkpoint before the
    endpoint, so there is a post-placement reading that is not the endpoint).
    """
    k0 = np.searchsorted(steps, FU, side="left")
    k0c = np.minimum(k0, len(steps) - 1)
    n = np.arange(TAU.shape[0])
    peak = TAU[n, k0c]
    end = TAU[:, -1]
    pre = np.where(k0 > 0, TAU[n, np.maximum(k0 - 1, 0)], np.nan)
    lag = steps[k0c] - FU
    valid = k0 < len(steps) - 1
    return dict(k0=k0, peak=peak, pre=pre, end=end, lag=lag, valid=valid)


def by_window(steps, TAU, FU, windows=WINDOWS, min_n=30, nboot=1000, seed=3):
    """Window summaries of the event-aligned quantities, with bootstrap CIs on
    the survival fraction end/peak (ratio of means)."""
    ev = event_align(steps, TAU, FU)
    rng = np.random.default_rng(seed)
    out = []
    for a, b in windows:
        g = ev["valid"] & (FU >= a) & (FU < b)
        if g.sum() < min_n:
            continue
        pk, en, pr = ev["peak"][g], ev["end"][g], ev["pre"][g]
        surv = float(np.nanmean(en) / np.nanmean(pk))
        bs = []
        ii = np.nonzero(g)[0]
        for _ in range(nboot):
            j = rng.choice(ii, ii.size, replace=True)
            bs.append(np.nanmean(ev["end"][j]) / np.nanmean(ev["peak"][j]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(dict(a=a, b=b, n=int(g.sum()),
                        pre=float(np.nanmean(pr)) if np.isfinite(pr).any() else float("nan"),
                        pre_se=float(np.nanstd(pr, ddof=1) / np.sqrt(np.isfinite(pr).sum()))
                        if np.isfinite(pr).sum() > 1 else float("nan"),
                        peak=float(np.nanmean(pk)), peak_se=float(np.nanstd(pk, ddof=1) / np.sqrt(g.sum())),
                        end=float(np.nanmean(en)), end_se=float(np.nanstd(en, ddof=1) / np.sqrt(g.sum())),
                        survive=surv, survive_lo=float(lo), survive_hi=float(hi),
                        lag=float(np.nanmean(ev["lag"][g]))))
    return out


def decay_curve(steps, TAU, FU, bins=(0, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000),
                min_peak=0.005):
    """Mean within-pair effect as a function of time since the document's own
    placement, pooled over all documents (each contributes every checkpoint at
    or after its placement). Also the mean of tau/peak over documents whose
    peak exceeds min_peak, which is the retention curve."""
    ev = event_align(steps, TAU, FU)
    rel = []
    for i in range(TAU.shape[0]):
        if not ev["valid"][i]:
            continue
        pk = ev["peak"][i]
        for k in range(ev["k0"][i], len(steps)):
            rel.append((steps[k] - FU[i], TAU[i, k], TAU[i, k] / pk if pk > min_peak else np.nan))
    rel = np.array(rel)
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (rel[:, 0] >= a) & (rel[:, 0] < b)
        if m.sum() == 0:
            continue
        out.append(dict(a=a, b=b, n=int(m.sum()), tau=float(np.nanmean(rel[m, 1])),
                        tau_se=float(np.nanstd(rel[m, 1], ddof=1) / np.sqrt(m.sum())),
                        ratio=float(np.nanmean(rel[m, 2]))))
    return out


def window_curves(steps, TAU, FU, windows=WINDOWS[:5], min_n=30):
    """Mean curve and SE per window (for the staircase panel)."""
    out = []
    for a, b in windows:
        g = (FU >= a) & (FU < b)
        if g.sum() < min_n:
            continue
        mu = np.nanmean(TAU[g], axis=0)
        se = np.nanstd(TAU[g], axis=0, ddof=1) / np.sqrt(g.sum())
        out.append(dict(a=a, b=b, n=int(g.sum()), mu=mu, se=se))
    return out
