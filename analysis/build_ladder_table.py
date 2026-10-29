#!/usr/bin/env python3
"""Assemble the twin-ladder trajectory table.

scoring/run_scoring.sh (ladder labels) writes one scores.csv.gz per (ladder, step). This joins
them into a single tidy table, one row per document, with the counterfactual
membership trace at every step:

    delta_s<STEP> = mean_logprob(treated, STEP) - mean_logprob(control, STEP)

plus the design columns needed to interpret it (cohort, arm, pair_id, r,
first_update / last_update / mean_update from the injection placement) and,
per (document, step), the learning rate at that step and the learning rate
integrated from the document's first injected copy to that step -- which is
what the identification below actually needs.

Why the learning rate is here: the run uses `linear_with_warmup` with
t_warmup = 0 and alpha_f = 0, so

    lr(s) = PEAK_LR * (1 - s / TOTAL_STEPS)

exactly, and the integral between two steps is analytic. Note the denominator:
the config leaves scheduler.t_max null, so OLMo derived max_steps from
max_duration and the training log shows it used 23,842 for essentially the
whole run (24,932 of 24,942 logged steps), reaching 23,852 only in the last ten
updates when the epoch ran out before the token budget. Using 23,852 as the
denominator would overstate the learning rate by 1.1% at step 23,000 -- the
error concentrated exactly where the annealing argument lives -- so LR_TOTAL
is 23,842 while ENDPOINT_STEP stays 23,852.

In the cross-section (the endpoint alone) elapsed-updates and the learning rate at
injection are perfectly collinear -- a document injected late is both recent
and trained at low LR. Along a trajectory they separate, because two documents
injected at different steps can be observed over the same number of elapsed
updates at different integrated LR. That is the identification this table
exists to support, and it is why ilr_s<STEP> is emitted per document.

THREE ladders, not two, and the reason is a provenance defect in AI2's
control. Its rungs do not come from one launch:

    steps  1000-19000   run_name peteish1-augusta-anneal-20250319_191614
    steps 20000-23852   run_name peteish1-augusta-anneal-20250319_164508

AI2 relaunched the anneal into the same remote folder with save_overwrite:
true, so the ladder mixes two realizations of the same recipe with a seam at
19000->20000, and the published endpoint sits on the earlier trajectory while
19 of the 23 intermediate rungs sit on the later one. Unless the relaunch was
bitwise reproducible -- OLMo at this scale is not -- delta inherits a step
change there that is not a training-dynamics effect. ingredient1 (seed 42069)
carries one run_name across all 24 rungs, so delta_sib_s<STEP> (treated minus
ingredient1) is a seam-free reference for the trajectory SHAPE, and
sibdrift_s<STEP> (ingredient3 minus ingredient1, both non-members of the
injection) measures the seed-to-seed spread the seam must be judged against.
seam_report() prints the null-arm discontinuity test.

The step-23852 columns are taken from the existing full-set scores
(scores/{injected-step23852, official-seed42-step23852,
sibling-seed42069-step23852}), produced by the same scorers with the same
settings; the ladder sweep deliberately does not re-score them.

Inputs (under the data root, see olmo_twin/paths.py): run/scores/<label>/ and
run/scores_ladder/<label>/ (scores.csv.gz and scoring_meta.rank*.json),
bundle/evaluation_manifest.csv.gz and bundle/stream_plan/placement.jsonl.gz.
Output: analysis/ladder_trajectory.csv.gz (--out) and ladder_meta.json next to it.
Pure CPU + IO.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

RUN = paths.RUN
BUNDLE = paths.BUNDLE
ANA = paths.ANALYSIS

# The schedule, read off the run config (scheduler: linear_with_warmup,
# t_warmup: 0, alpha_f: 0.0, learning_rate: 7.4487e-05, 23,852 updates).
PEAK_LR = 7.4487e-05
LR_TOTAL = 23842        # the schedule's denominator, from the training log
ENDPOINT_STEP = 23852   # the last saved checkpoint
LADDER_STEPS = list(range(1000, 23001, 1000)) + [ENDPOINT_STEP]

# The control ladder's provenance seam (see the module docstring).
SEAM_BEFORE, SEAM_AFTER = 19000, 20000

ARMS = ("treated", "control", "sibling")
# endpoint dirs live in the original scores tree under their own labels;
# every intermediate step lives in scores_ladder.
ENDPOINT_DIRS = {
    "treated": RUN / "scores/injected-step23852",
    "control": RUN / "scores/official-seed42-step23852",
    "sibling": RUN / "scores/sibling-seed42069-step23852",
}
LADDER_LABEL = {"treated": "injected", "control": "seed42", "sibling": "sibling"}
# what scoring_meta.rank000.json must point at, per arm (nothing otherwise
# cross-checks that a directory came from the checkpoint it claims). Each entry
# lists accepted path fragments: the treated run's checkpoints are
# injection_v2_runs/olmo2-1b-injection-v2/checkpoints in the score metadata of
# the original run and run/checkpoints in the data layout of this repository.
EXPECT_CKPT = {
    "treated": ("injection_v2_runs/olmo2-1b-injection-v2/checkpoints", "run/checkpoints"),
    "control": ("official_stage2_ingredient3_seed42",),
    "sibling": ("official_stage2_ingredient1",),
}


def score_dir(arm: str, step: int) -> Path:
    if step == ENDPOINT_STEP:
        return ENDPOINT_DIRS[arm]
    return RUN / f"scores_ladder/{LADDER_LABEL[arm]}-step{step}"


def check_provenance(arm: str, step: int, d: Path) -> str:
    """Assert the scored directory really came from this arm's checkpoint.

    An arm swap would invert the sign of every result, so this is cheap
    insurance. Returns a note describing what was verified, or raises.
    """
    metas = sorted(d.glob("scoring_meta.rank*.json"))
    if not metas:
        return f"{arm} step {step}: no scoring_meta, provenance UNVERIFIED"
    meta = json.loads(metas[0].read_text())
    ckpt = str(meta.get("checkpoint", ""))
    want = EXPECT_CKPT[arm]
    if not any(w in ckpt for w in want):
        raise SystemExit(f"PROVENANCE FAILURE: {d} was scored from {ckpt!r}, which does "
                         f"not contain any of {want!r} as arm {arm!r} requires. Refusing to "
                         f"build a table that may have the arms swapped.")
    if not ckpt.rstrip("/").endswith(f"step{step}-unsharded"):
        raise SystemExit(f"PROVENANCE FAILURE: {d} claims step {step} but was scored from "
                         f"{ckpt!r}")
    return f"{arm} step {step}: {Path(ckpt).parent.name}/{Path(ckpt).name}"


def lr_at(step: float) -> float:
    """Learning rate at an update index, from the declared linear schedule."""
    return PEAK_LR * max(0.0, 1.0 - step / LR_TOTAL)


def integrated_lr(a: float, b: float) -> float:
    """Integral of lr(s) ds over [a, b], in units of lr * updates.

    lr is clamped to 0 beyond LR_TOTAL (alpha_f = 0), so the integral is taken
    over the clipped interval.
    """
    if b <= a:
        return 0.0
    a = max(0.0, min(a, LR_TOTAL))
    b = max(0.0, min(b, LR_TOTAL))
    # integral of PEAK*(1 - s/T) = PEAK*(s - s^2/(2T))
    f = lambda s: PEAK_LR * (s - s * s / (2.0 * LR_TOTAL))
    return f(b) - f(a)


def read_scores(d: Path) -> dict[str, float]:
    """doc_id -> mean_logprob for one scored checkpoint, ok rows only."""
    f = d / "scores.csv.gz"
    if not f.exists():
        raise FileNotFoundError(f)
    out: dict[str, float] = {}
    with gzip.open(f, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            v = row.get("mean_logprob", "")
            if v == "" or v is None:
                continue
            out[row["doc_id"]] = float(v)
    return out


def load_placement() -> dict[str, dict[str, float]]:
    """doc_id -> first/last/mean update over its injected copies.

    NOTE a deliberate difference from build_injection_v2_attack_table.py: that
    table's mean_update is the mean of each copy's first_update, this one is
    the mean of each copy's midpoint 0.5*(first+last). Only 21 of 12,414 copies
    span more than one update and the max span is 1, so the two differ by at
    most 0.5 for 21 copies -- but do not join the two tables on this column
    without knowing that.
    """
    p = BUNDLE / "stream_plan/placement.jsonl.gz"
    acc: dict[str, list[tuple[int, int]]] = {}
    with gzip.open(p, "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            acc.setdefault(rec["doc_id"], []).append(
                (int(rec["first_update"]), int(rec["last_update"])))
    out = {}
    for doc, rows in acc.items():
        firsts = [a for a, _ in rows]
        lasts = [b for _, b in rows]
        mids = [0.5 * (a + b) for a, b in rows]
        out[doc] = {"first_update": float(min(firsts)),
                    "last_update": float(max(lasts)),
                    "mean_update": float(np.mean(mids)),
                    "n_copies_placed": float(len(rows))}
    return out


def seam_report(per_step, steps, manifest) -> dict:
    """Localise the control ladder's provenance seam using the null arm.

    Null documents have no injected copies, so treated-minus-control for them
    should be a smooth baseline. A jump between SEAM_BEFORE and SEAM_AFTER that
    is large against the step-to-step variation elsewhere is the seam showing
    up in the data. The sibling reference, whose ladder is one launch, gives
    the same quantity without a seam, so the two together say whether the seam
    matters at the size of the effect being measured.
    """
    nulls = [m["doc_id"] for m in manifest if m.get("arm") == "null"]
    out = {"steps": steps, "null_n": len(nulls)}
    for ref in ("control", "sibling"):
        if ref not in per_step.get(steps[0], {}):
            continue
        series = []
        for s in steps:
            arms = per_step[s]
            vals = [arms["treated"][d] - arms[ref][d]
                    for d in nulls if d in arms["treated"] and d in arms[ref]]
            series.append(float(np.mean(vals)) if vals else float("nan"))
        diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]
        out[f"null_delta_vs_{ref}"] = [round(v, 6) for v in series]
        out[f"null_step_diffs_vs_{ref}"] = [round(v, 6) for v in diffs]
        if SEAM_BEFORE in steps and SEAM_AFTER in steps:
            i = steps.index(SEAM_BEFORE)
            if i + 1 < len(steps) and steps[i + 1] == SEAM_AFTER:
                seam = diffs[i]
                others = [abs(d) for j, d in enumerate(diffs) if j != i]
                med = float(np.median(others)) if others else float("nan")
                out[f"seam_jump_vs_{ref}"] = round(float(seam), 6)
                out[f"median_other_step_jump_vs_{ref}"] = round(med, 6)
                out[f"seam_ratio_vs_{ref}"] = (round(abs(seam) / med, 2)
                                               if med and np.isfinite(med) else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ANA / "ladder_trajectory.csv.gz")
    ap.add_argument("--arms", default="treated,control,sibling",
                    help="comma-separated subset of treated,control,sibling")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write a table from whatever steps are present instead of "
                         "refusing. The trajectory analysis needs a complete ladder, "
                         "so this is for inspecting a sweep still in flight.")
    args = ap.parse_args()
    csv.field_size_limit(sys.maxsize)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {ARMS}")
    if "treated" not in arms:
        raise SystemExit("the treated arm is required -- every delta is relative to it")

    with gzip.open(BUNDLE / "evaluation_manifest.csv.gz", "rt", newline="") as fh:
        manifest = list(csv.DictReader(fh))
    print(f"manifest: {len(manifest):,} documents")
    print(f"arms: {', '.join(arms)}")

    have, missing = [], []
    for step in LADDER_STEPS:
        if all((score_dir(a, step) / "scores.csv.gz").exists() for a in arms):
            have.append(step)
        else:
            missing.append(step)
    print(f"complete steps: {len(have)}/{len(LADDER_STEPS)}")
    if missing:
        print(f"  missing: {missing}")
        if not args.allow_partial:
            print("refusing to write a partial ladder; pass --allow-partial to override")
            return 1
    if not have:
        print("nothing to do")
        return 1

    per_step: dict[int, dict[str, dict[str, float]]] = {}
    for step in have:
        arms_scores = {}
        notes = []
        for a in arms:
            d = score_dir(a, step)
            notes.append(check_provenance(a, step, d))
            arms_scores[a] = read_scores(d)
        common = set.intersection(*(set(v) for v in arms_scores.values()))
        sizes = "  ".join(f"{a} {len(arms_scores[a]):,}" for a in arms)
        print(f"  step {step:>6,}: {sizes}  joined {len(common):,}")
        for n in notes:
            print(f"      {n}")
        if len(common) < 22_000:
            raise SystemExit(f"step {step}: only {len(common):,} documents joined across "
                             f"{len(arms)} ladders -- refusing to build on an incomplete step")
        per_step[step] = arms_scores

    place = load_placement()
    print(f"placement: {len(place):,} injected documents")

    refs = [a for a in arms if a != "treated"]
    fields = (["doc_id", "cohort", "stratum", "arm", "pair_id", "split", "r", "group",
               "n_tokens", "n_bytes",
               "first_update", "last_update", "mean_update", "n_copies_placed"]
              + [f"{a}_s{s}" for a in arms for s in have]
              + [f"delta_s{s}" for s in have]
              + ([f"delta_sib_s{s}" for s in have] if "sibling" in refs else [])
              + ([f"sibdrift_s{s}" for s in have]
                 if ("sibling" in refs and "control" in refs) else [])
              + [f"lr_s{s}" for s in have]
              + [f"ilr_s{s}" for s in have])

    n_written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for m in manifest:
            doc = m["doc_id"]
            row = {k: m.get(k, "") for k in
                   ("doc_id", "cohort", "stratum", "arm", "pair_id", "split", "r",
                    "group", "n_tokens", "n_bytes")}
            pl = place.get(doc)
            for k in ("first_update", "last_update", "mean_update", "n_copies_placed"):
                row[k] = f"{pl[k]:.1f}" if pl else ""
            first = pl["first_update"] if pl else None
            ok_any = False
            for s in have:
                a_s = per_step[s]
                t = a_s["treated"].get(doc)
                for a in arms:
                    v = a_s[a].get(doc)
                    row[f"{a}_s{s}"] = "" if v is None else f"{v:.6f}"
                c = a_s["control"].get(doc) if "control" in arms else None
                sib = a_s["sibling"].get(doc) if "sibling" in arms else None
                row[f"delta_s{s}"] = "" if (t is None or c is None) else f"{t - c:.6f}"
                if "sibling" in refs:
                    row[f"delta_sib_s{s}"] = "" if (t is None or sib is None) else f"{t - sib:.6f}"
                if "sibling" in refs and "control" in refs:
                    row[f"sibdrift_s{s}"] = "" if (c is None or sib is None) else f"{c - sib:.6f}"
                # lr at this step, and lr integrated from the first injected
                # copy to this step -- the per-(document, step) quantity that
                # decouples elapsed updates from the annealing schedule.
                row[f"lr_s{s}"] = f"{lr_at(s):.9e}"
                row[f"ilr_s{s}"] = f"{integrated_lr(first, s):.6f}" if first is not None else ""
                if t is not None:
                    ok_any = True
            if ok_any:
                w.writerow(row)
                n_written += 1
    print(f"wrote {args.out}  ({n_written:,} documents x {len(have)} steps x {len(arms)} arms)")

    seam = seam_report(per_step, have, manifest) if len(have) > 2 else {}
    if seam:
        for ref in ("control", "sibling"):
            k = f"seam_ratio_vs_{ref}"
            if seam.get(k) is not None:
                print(f"  seam test vs {ref}: jump at {SEAM_BEFORE}->{SEAM_AFTER} is "
                      f"{seam[f'seam_jump_vs_{ref}']:+.6f} against a median step jump of "
                      f"{seam[f'median_other_step_jump_vs_{ref}']:.6f} "
                      f"({seam[k]}x)")

    meta = {
        "arms": arms,
        "steps": have,
        "missing_steps": missing,
        "peak_lr": PEAK_LR,
        "lr_total_steps": LR_TOTAL,
        "endpoint_step": ENDPOINT_STEP,
        "schedule": ("linear_with_warmup, t_warmup=0, alpha_f=0.0; denominator 23842 "
                     "from the training log, not the 23852 final checkpoint index"),
        "lr_at_step": {str(s): lr_at(s) for s in have},
        "integrated_lr_from_step_to_end": {str(s): integrated_lr(s, ENDPOINT_STEP) for s in have},
        "control_ladder_seam": {
            "between": [SEAM_BEFORE, SEAM_AFTER],
            "run_name_before": "peteish1-augusta-anneal-20250319_191614",
            "run_name_after": "peteish1-augusta-anneal-20250319_164508",
            "why": ("AI2 relaunched the anneal into the same remote folder with "
                    "save_overwrite: true, so the seed-42 ladder mixes two launches. "
                    "The sibling (ingredient1) ladder carries one run_name throughout "
                    "and is seam-free."),
        },
        "seam_test": seam,
        "documents": n_written,
        "note": ("delta_s<STEP> = treated - control (exact counterfactual, seam at 19000); "
                 "delta_sib_s<STEP> = treated - sibling (seed 42069, seam-free, but the "
                 "reference is a different run so it carries seed-to-seed drift); "
                 "sibdrift_s<STEP> = control - sibling, both non-members, which measures "
                 "that drift directly. ilr_s<STEP> is lr integrated from the document's "
                 "first injected copy to that step."),
    }
    (args.out.parent / "ladder_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out.parent / 'ladder_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
