#!/usr/bin/env python3
"""Sensitivity of the headline results to the analysis exclusions.

One command:
    python3 -u analysis/exclusion_sensitivity.py

Writes analysis/exclusion_sensitivity.json and exclusion_sensitivity.md (with a
booktabs LaTeX fragment) under the data root (see olmo_twin/paths.py). Nothing
in the paper output directory is touched. Also reads
bundle/stream_plan/placement.jsonl.gz, analysis/posttrain_exclusions.json,
posttrain_overlap/posttrain_hits.csv.gz, exclusions/repetition_filter_masked_chunks.json
(optional cross-check of the embedded MASKED list) and analysis/paper_numbers.json
(the primary rows are asserted against it, so run figures/paper_figures.py
--first-pass before this script on a fresh data root, see analysis/README.md).

Every variant is defined relative to the paper's primary analysis, which is
exactly what figures/paper_figures.py computes: pf.load() (post-training
exclusions applied, complete inject/holdout pairs only, standardized-ratio
columns merged) and the populations NAT1 (4,832 pairs) and ALL (7,803 pairs).
ALL here is every complete analysed pair, pf.population(..., "ALLPAIRS"); the
generator's own pooled population pf.population(..., "ALL") excludes cohorts F
and X and is variant d3 below (7,326 pairs), whose headline is asserted against
paper_numbers.json. figures/paper_figures.py reads the ALL rows with exactly
these meanings (sensall* macros, tables/sensitivity.tex).

Variants
  (a) repetition-filter masking (OLMo's instance filter set the labels of a whole
      4,096-token training instance to -100, olmo/train.py:727-728, so a masked
      chunk contributes no gradient for any of its tokens):
        a1  drop the two pairs whose injected document received no gradient at all
        a2  drop the five pairs the paper names (a1 + the three multi-copy arXiv
            documents with one chunk masked in every copy)
        a3  drop every pair whose injected document had at least one masked chunk
  (b) the scoring-length cap: reported as a check, not a variant (the orphan
      partner is not in any pair, see the JSON's "orphan" block)
  (c) post-training exclusions NOT applied (ps.load_table(..., drop_posttrain_flagged=False),
      pairs rebuilt exactly as pf.load does)
  (d) ALL only: drop cohort X, drop cohort F, drop both

Statistics per (population, variant, score), score in {ratio_oracle, z_oracle}:
  within-pair difference d_j = s(member_j) - s(partner_j)
  tau     = mean(d);  SE = SD(d)/sqrt(n);  d_z = mean(d)/SD(d)  (SD with ddof=1)
  tau CI  = percentile 95% interval, pair bootstrap (resampling d), 2,000 draws,
            seed 7 -- this is ps.bootstrap_ci as called by pf.headline, so the
            primary row reproduces the paper's interval
  pos     = fraction of pairs with d > 0
  AUC     = rank AUC of members vs matched non-members, ties 1/2 (ps.auc)
  AUC CI  = (i) the paper's interval (pf.headline -> pf.auc_ci_pairs: pairs
            resampled, 400 draws, paper_figures' shared generator RNG) and
            (ii) a pair-bootstrap interval (both documents of a pair move
            together), N_BOOT_AUC draws, numpy default_rng(SEED)
  TPR@f   = pf.tpr_at: threshold = the score that admits k = floor(f * n_neg)
            matched non-members, member counted when score > threshold (in-sample)
  FDR     = pf.fdr_named(a, b, 0.05): Benjamini-Hochberg with the (v+1)/(n+1)
            empirical-null envelope, matched non-members as the null sample
  n80     = pf.n80(d_z) = ceil(((1.6449 + 0.8416)/d_z)^2)
  z_sf    = mean(d) / (sqrt(sum d^2)/n)   (normal approximation of the sign-flip
            randomization null of the mean; pf.signflip_p's z_analytic)
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from olmo_twin import paths  # noqa: E402

sys.path.insert(0, str(HERE))
paths.add_code_dirs("figures")
import paper_figures as pf  # noqa: E402
import injection_v2_plotstyle as ps  # noqa: E402

SEED = 20261013
N_BOOT_AUC = 1000
SCORES = ("ratio_oracle", "z_oracle")
OUT_JSON = pf.ANA / "exclusion_sensitivity.json"
OUT_MD = pf.ANA / "exclusion_sensitivity.md"
PLACEMENT = pf.BUNDLE / "stream_plan/placement.jsonl.gz"
AUDIT_MASKED = paths.EXCLUSIONS / "repetition_filter_masked_chunks.json"   # optional cross-check
CHUNK = 4096

# The 43 injected training instances whose loss OLMo's instance filter masked
# (re-derived from the injected stream with OLMo's instance filter, see
# training/checks/check_instance_filter.py; the same 43 tuples are released as
# the "rows" of exclusions/repetition_filter_masked_chunks.json):
# (doc_id, cohort, copy, chunk index k, chunks per copy, trigger inside the document's own tokens)
MASKED = [
    ("2505.12480", "D", 14, 2, 3, 1), ("2602.11728", "D32", 10, 1, 2, 0), ("wikiA:81660217", "W-new", 0, 0, 1, 0),
    ("2506.14493", "D", 1, 2, 4, 1), ("2602.16463", "D", 1, 0, 3, 1), ("2601.16669", "M", 0, 3, 4, 1),
    ("2506.09272", "M", 0, 4, 5, 1), ("2505.12480", "D", 15, 2, 3, 1), ("2602.16463", "D", 0, 0, 3, 1),
    ("2508.00241", "D", 1, 1, 2, 0), ("2505.12480", "D", 8, 2, 3, 1), ("2505.12480", "D", 2, 2, 3, 1),
    ("2509.08770", "D", 3, 1, 2, 0), ("2506.14493", "D", 2, 2, 4, 1), ("wikiB:56249738", "E", 0, 0, 1, 0),
    ("2505.12480", "D", 10, 2, 3, 1), ("2505.12480", "D", 12, 2, 3, 1), ("2602.15115", "M", 0, 1, 2, 0),
    ("2602.16463", "D", 3, 0, 3, 1), ("2510.16932", "M", 0, 1, 2, 1), ("2506.14493", "D", 7, 2, 4, 1),
    ("2602.16463", "D", 2, 0, 3, 1), ("2505.12480", "D", 4, 2, 3, 1), ("2504.19202", "M", 0, 1, 3, 1),
    ("2505.12480", "D", 7, 2, 3, 1), ("2505.12480", "D", 13, 2, 3, 1), ("2505.12480", "D", 0, 2, 3, 1),
    ("2506.14493", "D", 5, 2, 4, 1), ("2505.09493", "D32", 29, 4, 5, 0), ("2506.14493", "D", 4, 2, 4, 1),
    ("2505.12480", "D", 11, 2, 3, 1), ("2505.12480", "D", 3, 2, 3, 1), ("2602.13891", "M", 0, 3, 5, 1),
    ("2509.26059", "D", 0, 3, 4, 0), ("2505.12480", "D", 1, 2, 3, 1), ("2505.12480", "D", 6, 2, 3, 1),
    ("2506.14493", "D", 6, 2, 4, 1), ("2505.12480", "D", 5, 2, 3, 1), ("2506.14493", "D", 0, 2, 4, 1),
    ("2506.14493", "D", 3, 2, 4, 1), ("2602.07709", "M", 0, 2, 3, 1), ("2505.12480", "D", 9, 2, 3, 1),
    ("wikiB:9879892", "E0-D", 3, 0, 1, 0),
]
ZERO_DOSE = ["wikiA:81660217", "wikiB:56249738"]
EVERY_COPY = ["2505.12480", "2506.14493", "2602.16463"]
ORPHAN = ("2510.26874", "2512.22751", "F-03010")   # unscored injected doc, its partner, pair id


# ------------------------------------------------------------------ helpers
def fast_auc(pos, neg):
    """Same definition as ps.auc (Mann-Whitney, ties 1/2), vectorised ranks."""
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def build(drop_posttrain_flagged=True):
    """pf.load(), with the post-training filter switchable. Same join: a pair
    enters only if both its inject and holdout rows are present."""
    cols, rows = ps.load_table(pf.TABLE, drop_posttrain_flagged=drop_posttrain_flagged)
    pf.merge_calibrated(cols)
    arm = cols["arm"]; pid = cols["pair_id"]
    byp = defaultdict(dict)
    for i, p in enumerate(pid):
        if isinstance(p, str) and p:
            byp[p][arm[i]] = i
    pairs = [(v["inject"], v["holdout"]) for v in byp.values() if "inject" in v and "holdout" in v]
    return cols, rows, arm, pairs, byp


def stats(cols, sel, key, rng):
    h = pf.headline(cols, sel, key)            # the paper's own function
    a, b = pf.arrays(cols, key, sel)
    assert np.isfinite(a).all() and np.isfinite(b).all(), f"non-finite {key}"
    d = a - b
    assert abs(fast_auc(a, b) - h["auc"]) < 1e-12
    boots = np.empty(N_BOOT_AUC)
    for t in range(N_BOOT_AUC):
        j = rng.integers(0, a.size, a.size)
        boots[t] = fast_auc(a[j], b[j])
    h.update(n_pos=int(a.size), fdr05=pf.fdr_named(a, b, 0.05), n80=pf.n80(h["dz"]),
             z_signflip=float(d.mean() / (math.sqrt((d ** 2).sum()) / d.size)),
             auc_pair_lo=float(np.percentile(boots, 2.5)), auc_pair_hi=float(np.percentile(boots, 97.5)))
    return h


def masking_table(cols, pairs):
    plan = {}
    with gzip.open(PLACEMENT, "rt") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                plan[(r["doc_id"], r["copy"])] = r
    n_chunks_total = sum(r["n_chunks"] for r in plan.values())
    cross = None
    if AUDIT_MASKED.exists():
        aud = json.loads(AUDIT_MASKED.read_text())
        if isinstance(aud, dict):   # the released file wraps the list of six-tuples in "rows"
            aud = aud["rows"]
        aud = sorted(tuple(x) for x in aud)
        cross = aud == sorted(MASKED)
        assert cross, f"embedded MASKED list disagrees with {AUDIT_MASKED}"
    docidx = {d: i for i, d in enumerate(cols["doc_id"])}
    inpair = {i for p in pairs for i in p}
    per = defaultdict(lambda: {"masked_chunks": 0, "masked_tokens": 0, "copies": set(), "chunks_by_copy": defaultdict(set), "trigger_in_doc": 0})
    for d, coh, cp, k, nc, ind in MASKED:
        r = plan[(d, cp)]
        assert r["n_chunks"] == nc and r["cohort"] == coh
        ntok = CHUNK if k < nc - 1 else r["tail_tokens"]
        e = per[d]
        e["masked_chunks"] += 1; e["masked_tokens"] += ntok; e["copies"].add(cp)
        e["chunks_by_copy"][cp].add(k); e["trigger_in_doc"] += ind
    out = []
    for d, e in sorted(per.items()):
        cps = [r for (dd, _), r in plan.items() if dd == d]
        r0 = cps[0]
        ntok, nc, rcp = r0["n_tokens_with_eos"], r0["n_chunks"], len(cps)
        assert all(c["n_tokens_with_eos"] == ntok and c["n_chunks"] == nc for c in cps)
        never = set.intersection(*[e["chunks_by_copy"].get(c["copy"], set()) for c in cps])
        never_tok = sum(CHUNK if k < nc - 1 else r0["tail_tokens"] for k in never)
        i = docidx.get(d)
        out.append(dict(doc_id=d, cohort=r0["cohort"], pair_id=r0["pair_id"], r=int(r0["r"]), copies=rcp,
                        tokens_per_copy=int(ntok), chunks_per_copy=int(nc),
                        masked_chunks=e["masked_chunks"], copies_affected=len(e["copies"]),
                        trigger_inside_document=e["trigger_in_doc"],
                        token_exposures_lost=round(e["masked_tokens"] / (ntok * rcp), 4),
                        tokens_never_trained=round(never_tok / ntok, 4),
                        in_table=i is not None, in_analysed_pair=(i is not None and i in inpair),
                        in_NAT1=bool(i is not None and cols["r"][i] == 1 and cols["cohort"][i] in pf.NAT and i in inpair)))
    return out, dict(chunks_injected=n_chunks_total, chunks_masked=len(MASKED), docs=len(out),
                     audit_file_crosscheck=("agrees" if cross else "audit file absent; embedded list used"))


def orphan_check(cols, arm, pairs, byp):
    inj, part, pid = ORPHAN
    docid = list(cols["doc_id"])
    present = {d: d in docid for d in (inj, part)}
    pidx = docid.index(part) if present[part] else None
    in_pair = any(pidx in p for p in pairs) if pidx is not None else False
    hold = (arm == "holdout")
    v = cols["ratio_oracle"]
    m_with = float(np.nanmean(v[hold])); n_with = int(np.isfinite(v[hold]).sum())
    m_wo = float(np.nanmean(v[hold & (cols["doc_id"] != part)]))
    return dict(injected_doc=inj, injected_row_present=present[inj], partner=part, partner_row_present=present[part],
                partner_arm=str(arm[pidx]) if pidx is not None else None,
                partner_cohort=str(cols["cohort"][pidx]) if pidx is not None else None,
                pair_id=pid, pair_in_pf_load_pairs=bool(pid in byp and "inject" in byp[pid]), partner_in_any_pair=in_pair,
                uses_outside_pairs=("paper_figures.fig_signal panel (a) histograms every row with arm == 'holdout' "
                                    "(all documents, no pairing), so the orphan is one of the plotted held-out "
                                    "documents and enters that panel's dashed arm-mean line; no macro, table or "
                                    "prose number is derived from it. fig_tiers(b) and fig_drift select held-out rows "
                                    "by cohort in NAT/{M,L,W-new,W-gap,S-se,E,E0,E0-D}, which excludes F."),
                holdout_arm_mean_ratio_oracle_with_orphan=m_with, holdout_arm_n_with_orphan=n_with,
                holdout_arm_mean_ratio_oracle_without_orphan=m_wo,
                orphan_ratio_oracle=float(v[pidx]) if pidx is not None else None)


def posttrain_stages(cols, sels):
    """Mid-training, SFT and DPO within-pair statistics of the ratio to the
    control twin, computed exactly as pf.fig_posttrain does (score = treated
    minus control mean log-probability at that stage; pairs with a missing
    score at that stage dropped)."""
    def load_scores(p):
        out = {}
        with gzip.open(p, "rt", newline="") as fh:
            import csv
            for row in csv.DictReader(fh):
                try:
                    out[row["doc_id"]] = float(row["mean_logprob"])
                except (KeyError, ValueError):
                    pass
        return out
    docid = cols["doc_id"]
    stages = {"mid": None}
    for st, (pt, pc) in pf.PT.items():
        if pt.exists() and pc.exists():
            stages[st] = (load_scores(pt), load_scores(pc))
    res = {}
    for pop, sel in sels.items():
        res[pop] = {}
        for st in stages:
            if st == "mid":
                a, b = pf.arrays(cols, "ratio_oracle", sel)
            else:
                T, Cc = stages[st]
                a = np.array([T.get(docid[i], np.nan) - Cc.get(docid[i], np.nan) for i, _ in sel])
                b = np.array([T.get(docid[h], np.nan) - Cc.get(docid[h], np.nan) for _, h in sel])
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            d = a - b
            t1, k, n = pf.tpr_at(a, b, 0.01)
            res[pop][st] = dict(n=int(d.size), tau=float(d.mean()), dz=float(d.mean() / d.std(ddof=1)),
                                auc=float(ps.auc(a, b)), tpr1=t1, k1=k, nneg=n)
            print(f"  [stages] {pop:20s} {st}: n={d.size} dz={res[pop][st]['dz']:.3f} auc={res[pop][st]['auc']:.4f} "
                  f"tpr1={100*t1:.2f}% ({k}/{n})", flush=True)
    return res


# ------------------------------------------------------------------ main
def main() -> int:
    rng = np.random.default_rng(SEED)
    cols, rows, arm, pairs, byp = build(True)
    # identical to the paper's loader
    c2, _, _, p2 = pf.load()
    assert sorted(p2) == sorted(pairs) and len(c2["doc_id"]) == len(cols["doc_id"])
    # ALL = every complete pair (the generator's "ALLPAIRS"; its "ALL" excludes F and X = variant d3)
    selN = pf.population(cols, pairs, "NAT1"); selA = pf.population(cols, pairs, "ALLPAIRS")
    assert len(selN) == 4832 and len(selA) == 7803, (len(selN), len(selA))
    pid, coh = cols["pair_id"], cols["cohort"]

    mask_rows, mask_meta = masking_table(cols, pairs)
    pairs_zero = {pid[cols["doc_id"].tolist().index(d)] for d in ZERO_DOSE}
    pairs_five = pairs_zero | {pid[cols["doc_id"].tolist().index(d)] for d in EVERY_COPY}
    pairs_all = {m["pair_id"] for m in mask_rows}
    assert pairs_zero == {"W-new-03723", "E-07259"}

    def drop(sel, bad):
        return [(i, h) for i, h in sel if pid[i] not in bad]

    colsU, _, armU, pairsU, _ = build(False)
    selNU = pf.population(colsU, pairsU, "NAT1"); selAU = pf.population(colsU, pairsU, "ALLPAIRS")
    added = sorted({colsU["pair_id"][i] for i, _ in selAU} - {pid[i] for i, _ in selA})

    variants = {
        "NAT1": [("primary", "primary analysis (paper)", cols, selN),
                 ("a1", "drop pairs of the 2 zero-dose documents", cols, drop(selN, pairs_zero)),
                 ("a2", "drop pairs of the 5 documents named in the paper", cols, drop(selN, pairs_five)),
                 ("a3", "drop pairs of all 18 masking-affected documents", cols, drop(selN, pairs_all)),
                 ("c", "keep post-training-flagged pairs", colsU, selNU)],
        "ALL": [("primary", "primary analysis (paper)", cols, selA),
                ("a1", "drop pairs of the 2 zero-dose documents", cols, drop(selA, pairs_zero)),
                ("a2", "drop pairs of the 5 documents named in the paper", cols, drop(selA, pairs_five)),
                ("a3", "drop pairs of all 18 masking-affected documents", cols, drop(selA, pairs_all)),
                ("c", "keep post-training-flagged pairs", colsU, selAU),
                ("d1", "drop cohort X", cols, [(i, h) for i, h in selA if coh[i] != "X"]),
                ("d2", "drop cohort F", cols, [(i, h) for i, h in selA if coh[i] != "F"]),
                ("d3", "drop cohorts F and X", cols, [(i, h) for i, h in selA if coh[i] not in ("F", "X")])],
    }
    res = {}
    for pop, vs in variants.items():
        res[pop] = []
        for tag, lab, cc, sel in vs:
            row = dict(tag=tag, label=lab, n_pairs=len(sel), pairs_removed_vs_primary=None, scores={})
            for key in SCORES:
                row["scores"][key] = stats(cc, sel, key, rng)
            res[pop].append(row)
            print(f"{pop:4s} {tag:7s} n={len(sel):5d} " + " | ".join(
                f"{k}: tau={s['tau']:+.5f} dz={s['dz']:.3f} auc={s['auc']:.4f} tpr1={100*s['tpr1']:.2f}% "
                f"tpr01={100*s['tpr01']:.2f}% fdr={s['fdr05']} n80={s['n80']}" for k, s in row["scores"].items()), flush=True)
        base = res[pop][0]
        for row in res[pop]:
            row["delta_vs_primary"] = {k: {m: row["scores"][k][m] - base["scores"][k][m]
                                           for m in ("tau", "se", "dz", "pos", "auc", "tpr1", "tpr01", "fdr05", "n80", "z_signflip")}
                                       for k in SCORES}
            row["n_pairs_delta"] = row["n_pairs"] - base["n_pairs"]
    for pop, vs in variants.items():
        base_ids = {pid[i] for i, _ in vs[0][3]}
        for row, (tag, _, cc, sel) in zip(res[pop], vs):
            ids = {cc["pair_id"][i] for i, _ in sel}
            row["pairs_removed_vs_primary"] = sorted(base_ids - ids) if len(base_ids - ids) <= 20 else len(base_ids - ids)
            row["pairs_added_vs_primary"] = sorted(ids - base_ids)

    # sanity: primary NAT1 reproduces the paper
    P = json.loads((pf.ANA / "paper_numbers.json").read_text())["NAT1"]
    ro = res["NAT1"][0]["scores"]["ratio_oracle"]; zo = res["NAT1"][0]["scores"]["z_oracle"]
    for k in ("n", "tau", "se", "lo", "hi", "dz", "pos", "auc", "auc_lo", "auc_hi", "tpr1", "k1", "tpr01", "k01", "nneg"):
        assert abs(ro[k] - P[k]) < 1e-12, (k, ro[k], P[k])
    assert ro["n80"] == 13 and ro["fdr05"] == 0 and round(ro["auc"], 3) == 0.767 and round(100 * ro["tpr1"], 1) == 3.8
    assert round(zo["auc"], 3) == 0.778 and round(100 * zo["tpr1"], 1) == 8.8 and zo["fdr05"] == 162
    # the generator's pooled population (ALL without F and X) is variant d3 here
    PA = json.loads((pf.ANA / "paper_numbers.json").read_text())["ALL"]
    d3 = next(r for r in res["ALL"] if r["tag"] == "d3")
    for k in ("n", "tau", "dz", "auc", "tpr1", "tpr01"):
        assert abs(d3["scores"]["ratio_oracle"][k] - PA[k]) < 1e-12, k

    # post-training stages, replicating pf.fig_posttrain: does the pooled (ALL) comparison the
    # paper cites (d_z and TPR@1% mid -> DPO) depend on keeping cohorts F and X in the negatives?
    stage_res = posttrain_stages(cols, {"NAT1": selN, "ALL": selA,
                                        "ALL without F and X": [(i, h) for i, h in selA if coh[i] not in ("F", "X")]})

    # cohort composition of ALL
    comp = defaultdict(int)
    for i, _ in selA:
        comp[str(coh[i])] += 1
    pt = json.loads(ps.EXCLUSIONS.read_text())
    hits = []
    with gzip.open(paths.POSTTRAIN_OVERLAP / "posttrain_hits.csv.gz", "rt") as fh:
        import csv
        for r in csv.DictReader(fh):
            hits.append({k: r[k] for k in ("doc_id", "cohort", "arm", "pair_id", "reasons", "shingles", "component", "ref", "run_words")})
    out = dict(
        meta=dict(script="analysis/exclusion_sensitivity.py", table=str(pf.TABLE.relative_to(paths.DATA)),
                  seed_pair_bootstrap_auc=SEED, n_boot_auc_pair=N_BOOT_AUC,
                  tau_ci="ps.bootstrap_ci on per-pair differences, 2000 draws, seed 7 (as in pf.headline)",
                  auc_ci_paper="pf.auc_ci_pairs: pairs resampled with replacement, 400 draws, paper_figures' shared RNG default_rng(20261012) (as in pf.headline)",
                  auc_ci_pair=f"pairs resampled with replacement, {N_BOOT_AUC} draws, numpy default_rng({SEED}), percentile 2.5/97.5",
                  formulas=dict(d="s(member) - s(matched partner)", tau="mean(d)", se="SD(d)/sqrt(n), ddof=1",
                                dz="mean(d)/SD(d)", tpr="pf.tpr_at: k=floor(f*n_neg) negatives admitted, member counted if score > threshold",
                                fdr="pf.fdr_named(q=0.05): BH with (v+1)/(n_neg+1) envelope",
                                n80="ceil(((1.6449+0.8416)/dz)^2)", z_signflip="mean(d)/(sqrt(sum d^2)/n)"),
                  populations=dict(NAT1="r==1 and cohort in {M,L,W-new,W-gap,S-se}", ALL="every complete analysed pair (pf.population 'ALLPAIRS')"),
                  ALL_composition=dict(sorted(comp.items()))),
        repetition_masking=dict(summary=mask_meta, documents=mask_rows,
                                pairs_zero_dose=sorted(pairs_zero), pairs_named_in_paper=sorted(pairs_five),
                                pairs_all_affected=sorted(pairs_all)),
        orphan=orphan_check(cols, arm, pairs, byp),
        posttrain=dict(exclusions_file=str(ps.EXCLUSIONS.relative_to(paths.DATA)), audit_date=pt.get("audit_date"),
                       rationale=pt.get("rationale"), pair_ids=pt["exclude_pair_ids"], doc_ids=pt["exclude_doc_ids"],
                       hits=hits, pairs_restored_when_unfiltered=added,
                       NAT1_pairs_restored=sorted({colsU["pair_id"][i] for i, _ in selNU} - {pid[i] for i, _ in selN})),
        paraphrase=dict(tested=False, reason="requires a matcher looser than the exact 40-word fingerprint test; not attempted",
                        existing_bound="revision-2 Stage G record (collection/README.md): on the 15,233 Wikipedia-B pages "
                                       "with a trained ancestor under their own page id, the n-gram tests detected 14,619 (96.0%); "
                                       "the 614 missed (4.0%) were caught only by the page-id ancestor rule. No equivalent "
                                       "ancestor key exists for arXiv or StackExchange."),
        posttrain_stages=stage_res,
        results=res,
    )
    OUT_JSON.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {OUT_JSON}")
    OUT_MD.write_text(markdown(out))
    print(f"wrote {OUT_MD}")
    return 0


# ------------------------------------------------------------------ report
def _pc(x, nd=1):
    return f"{100 * x:.{nd}f}"


def markdown(o):
    L = []
    A = L.append
    A("# Sensitivity of the headline results to the analysis exclusions\n")
    A(f"Generated by `analysis/exclusion_sensitivity.py` from `{o['meta']['table']}` via `paper_figures.load()`. "
      f"Pair-bootstrap AUC intervals: {o['meta']['auc_ci_pair']}. Within-pair effect intervals: {o['meta']['tau_ci']}. "
      "TPR is in-sample (paper rule: the threshold admits k = floor(f n_neg) matched non-members). "
      "FDR: number of members nameable at Benjamini-Hochberg FDR <= 0.05 with matched non-members as the null sample. "
      "Changes are variant minus primary.\n")
    for pop in ("NAT1", "ALL"):
        for key, unit in (("ratio_oracle", "nats/token"), ("z_oracle", "z units")):
            A(f"## {pop}, {key}\n")
            A(f"| variant | pairs | tau ({unit}) [95% CI] | SE | d_z | pos. | AUC [paper CI] [pair CI] | TPR@1% (k/n_neg) | TPR@0.1% (k/n_neg) | FDR-0.05 named | n80 | z_sf |")
            A("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for row in o["results"][pop]:
                s = row["scores"][key]; dl = row["delta_vs_primary"][key]
                prim = row["tag"] == "primary"
                def ch(v, fmt):
                    return "" if prim else f" ({fmt.format(v)})"
                A(f"| {row['tag']}: {row['label']} | {row['n_pairs']:,}{'' if prim else ' ({:+d})'.format(row['n_pairs_delta'])} "
                  f"| {s['tau']:+.5f} [{s['lo']:+.5f}, {s['hi']:+.5f}]{ch(dl['tau'], '{:+.5f}')} "
                  f"| {s['se']:.5f} | {s['dz']:.3f}{ch(dl['dz'], '{:+.3f}')} | {_pc(s['pos'])}%{ch(100*dl['pos'], '{:+.2f}')} "
                  f"| {s['auc']:.4f} [{s['auc_lo']:.3f}, {s['auc_hi']:.3f}] [{s['auc_pair_lo']:.3f}, {s['auc_pair_hi']:.3f}]{ch(dl['auc'], '{:+.4f}')} "
                  f"| {_pc(s['tpr1'], 2)}% ({s['k1']}/{s['nneg']}){ch(100*dl['tpr1'], '{:+.2f}')} "
                  f"| {_pc(s['tpr01'], 2)}% ({s['k01']}/{s['nneg']}){ch(100*dl['tpr01'], '{:+.2f}')} "
                  f"| {s['fdr05']}{ch(dl['fdr05'], '{:+d}')} | {s['n80']}{ch(dl['n80'], '{:+d}')} | {s['z_signflip']:.2f} |")
            A("")
    A("Changes in parentheses: pairs, tau, d_z and AUC as differences; pos., TPR as percentage-point differences; FDR and n80 as count differences. "
      "TPR is given to two decimals because the same count of detected members can round differently at one decimal when n_pos changes by one "
      "(NAT1 ratio_oracle: {} and {}).\n".format(*[
          f"{round(r['scores']['ratio_oracle']['tpr1'] * r['n_pairs'])}/{r['n_pairs']:,} = {100 * r['scores']['ratio_oracle']['tpr1']:.3f}%"
          for r in o["results"]["NAT1"][:2]]))
    # masking
    M = o["repetition_masking"]
    A("## (a) Repetition-filter masking\n")
    A(f"{M['summary']['chunks_masked']} of {M['summary']['chunks_injected']:,} injected chunks masked, {M['summary']['docs']} documents "
      f"(embedded list vs. audit file: {M['summary']['audit_file_crosscheck']}). OLMo sets the labels of a masked instance to -100 "
      "(olmo/train.py:727-728), so no token of that 4,096-token window contributes gradient. "
      "'Trigger in doc' counts masked chunks whose periodic repeat lies inside the document's own tokens; otherwise it lies in the "
      "official tokens that complete the document's final partial chunk. 'Token-exposures lost' = masked document tokens / "
      "(r x document tokens); 'tokens never trained' = share of the document's tokens masked in every copy.\n\n"
      "Masking is observed only for the injected member of each pair; whether the held-out partner would have been masked had it "
      "been injected is unobserved. Dropping pairs by masking therefore conditions on a post-assignment variable (a per-protocol "
      "analysis), whereas the primary analysis, which keeps them, estimates the effect of assignment to injection (intention to "
      "treat) and is unbiased for it under the coin flip. The post-training and cohort exclusions, by contrast, are functions of "
      "both documents or of the cohort and do not depend on the coin.\n")
    A("| doc | cohort | pair | r | chunks/copy | masked chunks | copies affected | trigger in doc | token-exposures lost | tokens never trained | NAT1 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in M["documents"]:
        A(f"| {m['doc_id']} | {m['cohort']} | {m['pair_id']} | {m['r']} | {m['chunks_per_copy']} | {m['masked_chunks']} | {m['copies_affected']} | "
          f"{m['trigger_inside_document']}/{m['masked_chunks']} | {_pc(m['token_exposures_lost'])}% | {_pc(m['tokens_never_trained'])}% | {'yes' if m['in_NAT1'] else 'no'} |")
    A("")
    # orphan
    Or = o["orphan"]
    A("## (b) Scoring-length cap: the orphan partner\n")
    A(f"- injected document {Or['injected_doc']} row present in the table: {Or['injected_row_present']}; partner {Or['partner']} present: "
      f"{Or['partner_row_present']} (arm {Or['partner_arm']}, cohort {Or['partner_cohort']}).")
    A(f"- pair {Or['pair_id']} formed by pf.load(): {Or['pair_in_pf_load_pairs']}; partner in any analysed pair: {Or['partner_in_any_pair']}.")
    A(f"- {Or['uses_outside_pairs']}")
    A(f"- held-out-arm mean ratio_oracle in that panel: {Or['holdout_arm_mean_ratio_oracle_with_orphan']:+.6f} with the orphan (n = {Or['holdout_arm_n_with_orphan']:,}), "
      f"{Or['holdout_arm_mean_ratio_oracle_without_orphan']:+.6f} without; orphan's own value {Or['orphan_ratio_oracle']:+.5f}.\n")
    # posttrain
    T = o["posttrain"]
    A("## (c) Post-training exclusions\n")
    A(f"Source: `{T['exclusions_file']}` (scan of {T['audit_date']}). {T['rationale']}\n")
    A("| doc | cohort | arm | pair | match | shingles | longest run (words) | component |")
    A("|---|---|---|---|---|---|---|---|")
    for h in T["hits"]:
        A(f"| {h['doc_id']} | {h['cohort']} | {h['arm']} | {h['pair_id'] or '-'} | {h['reasons']} | {h['shingles']} | {h['run_words']} | {h['component']} |")
    A(f"\nPairs restored when the filter is off: {', '.join(T['pairs_restored_when_unfiltered'])} (NAT1: {', '.join(T['NAT1_pairs_restored'])}).\n")
    A("## (d) Composition of ALL\n")
    A(", ".join(f"{k} {v:,}" for k, v in o["meta"]["ALL_composition"].items()) + "\n")
    A("ALL as defined here (`pf.population(cols, pairs, 'ALLPAIRS')`) is every complete analysed pair, so cohort X and cohort F "
      "pairs, and their held-out partners as negatives, are included in every ALL statistic (pf.headline, fig_posttrain).\n")
    A("### Post-training stages (replicating pf.fig_posttrain), ratio to the control twin\n")
    A("| population | stage | pairs | tau | d_z | AUC | TPR@1% (k/n_neg) |")
    A("|---|---|---|---|---|---|---|")
    for pop, sts in o["posttrain_stages"].items():
        for st, v in sts.items():
            A(f"| {pop} | {st} | {v['n']:,} | {v['tau']:+.5f} | {v['dz']:.3f} | {v['auc']:.4f} | {_pc(v['tpr1'], 2)}% ({v['k1']}/{v['nneg']}) |")
    A("")
    A("## (e) Paraphrases\n")
    A(f"Not tested ({o['paraphrase']['reason']}). Existing bound: {o['paraphrase']['existing_bound']}\n")
    A("## LaTeX fragment (booktabs)\n")
    A("```latex")
    A(latex(o))
    A("```\n")
    return "\n".join(L)


def latex(o):
    L = [r"% generated by analysis/exclusion_sensitivity.py -- do not edit by hand",
         r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r" & & \multicolumn{5}{c}{Ratio to the exact counterfactual} & \multicolumn{4}{c}{Standardized ratio} \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-11}",
         r"Variant & Pairs & $\hat\tau$ & $d_z$ & AUC & TPR@1\% & Named & $d_z$ & AUC & TPR@1\% & Named \\", r"\midrule"]
    names = {"primary": "Primary analysis", "a1": "Without zero-dose documents", "a2": "Without the five named documents",
             "a3": "Without any masked document", "c": "With post-training-flagged pairs",
             "d1": "Without cohort X", "d2": "Without cohort F", "d3": "Without cohorts F and X"}
    for pop, title in (("NAT1", "Verified-unseen single-copy pairs"), ("ALL", "All analysed pairs")):
        L.append(rf"\multicolumn{{11}}{{l}}{{\emph{{{title}}}}} \\")
        for row in o["results"][pop]:
            r_, z_ = row["scores"]["ratio_oracle"], row["scores"]["z_oracle"]
            L.append(f"\\quad {names[row['tag']]} & {row['n_pairs']:,}".replace(",", "{,}") +
                     f" & {r_['tau']:+.4f} & {r_['dz']:.3f} & {r_['auc']:.3f} & {100*r_['tpr1']:.2f} & {r_['fdr05']:,}".replace(",", "{,}") +
                     f" & {z_['dz']:.3f} & {z_['auc']:.3f} & {100*z_['tpr1']:.2f} & {z_['fdr05']:,} \\\\".replace(",", "{,}"))
        if pop == "NAT1":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
