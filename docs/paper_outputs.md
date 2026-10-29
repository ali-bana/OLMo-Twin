# The paper's figures, tables and numbers

This document maps every figure and table of the paper to the command that
produces it, reports how closely the output matches the published version,
and lists what was edited by hand. All commands run on a CPU in the analysis
environment (`env/analysis/requirements.txt`) with `OLMO_TWIN_DATA` pointing
at the data root (`docs/data.md`). [REPRODUCING.md](REPRODUCING.md) describes
the whole pipeline that leads up to these steps.

`NAT1` is the name that the code gives to the paper's main population: the
4,832 verified-unseen single-exposure pairs. In `figures/paper_figures.py`,
`ALL` is the pooled set of 7,326 pairs (all cohorts except F and X, after
exclusions); [analysis/README.md](../analysis/README.md#populations-exclusions-and-seeds)
lists the population names that the other scripts use.

## Commands

```bash
# Tables, numbers and figures from the analysis outputs in the data root.
bash figures/make_all.sh                 # default output: $OLMO_TWIN_DATA/paper/{tables,figures}

# Table 4 on all 4,832 NAT1 pairs, and the calib* macros. PAPER_TABLES is a directory
# with the paper's numbers.tex and calibrated.tex (see "Tables" below).
python analysis/recompute_table4_all_pairs.py --paper-tables "$PAPER_TABLES"   # -> analysis/table4_all_pairs/
python analysis/format_table4.py --numbers "$PAPER_TABLES/numbers.tex"

# RLVR rows of the stages table and the RLVR macros.
python analysis/rlvr/recompute.py                    # -> rlvr/work/
python analysis/rlvr/rl_minus_dpo.py
python analysis/rlvr/finalize.py                     # -> rlvr/ (stages_extended.csv, proposed_macros.tex, ...)

# Comparison with the published figures and macros.
python figures/compare_with_paper.py --generated "$OLMO_TWIN_DATA/paper" --paper <paper source directory>
```

`figures/make_all.sh` runs three scripts in this order:

1. `figures/paper_figures.py` writes the tables (`tables/*.tex`), the macro
   file `tables/numbers.tex` and `analysis/paper_numbers.json`, and draws its
   own version of every data figure; its `attacks_auc.pdf` is the published
   one;
2. `figures/published_figures.py` redraws the other data figures exactly as
   published and overwrites those of step 1: `intro`, `signal_distributions`,
   `roc`, `collection`, `reference_tiers`, `trajectory`, `dose`,
   `domain_transfer` and `cohort`, plus `posttrain.pdf`, which is drawn but
   commented out in the paper; it writes no tables;
3. `figures/paper_design_figure.py` draws `design.pdf` (it reads no data).

On a fresh data root, where the analysis outputs do not yet exist, run
`analysis/run_analysis.sh` instead. `figures/paper_figures.py` reads the
outputs of five analysis scripts (screening, collection by domain, exclusion
sensitivity, cohort F, length) before it writes `paper_numbers.json`, while
two of these scripts read `paper_numbers.json`. The analysis script therefore
runs `figures/paper_figures.py --first-pass` (which skips the tables and
macros that need those outputs but writes `paper_numbers.json`), then the
analysis scripts, then the full `figures/paper_figures.py`;
`analysis/README.md` gives the order.

## Why there are two figure generators

The published data figures and the published numbers were produced by
different revisions of the generator. The figures were drawn by an earlier
revision with later figure-only changes; the tables and numbers come from the
final revision, whose random-number stream and population definitions the
analysis scripts depend on (they import it as a library). No single revision
reproduces both. `figures/paper_figures.py` is the final revision and the
generator of record for tables and numbers; `figures/published_figures.py` is
the revision that reproduces the published figures. Its own tables and macros
use earlier definitions and are not those of the paper, so it does not write
them.

## Figures

| Paper figure (label) | File | Produced by |
|---|---|---|
| Introduction figure (`fig:intro`) | `figures/intro.pdf` | `figures/published_figures.py` |
| The testbed (`fig:design`) | `figures/design.pdf` | `figures/paper_design_figure.py` |
| Signal distributions (`fig:signal`) | `figures/signal_distributions.pdf` | `figures/published_figures.py` |
| ROC curves (`fig:roc`) | `figures/roc.pdf` | `figures/published_figures.py` |
| Attack AUCs (`fig:attacks`) | `figures/attacks_auc.pdf` | `figures/paper_figures.py` |
| Collection-level inference (`fig:collection`) | `figures/collection.pdf` | `figures/published_figures.py` |
| Reference tiers (`fig:tiers`) | `figures/reference_tiers.pdf` | `figures/published_figures.py` |
| Trajectory over training (`fig:trajectory`) | `figures/trajectory.pdf` | `figures/published_figures.py` |
| Dose (`fig:dose`) | `figures/dose.pdf` | `figures/published_figures.py` |
| Never-injected documents (`fig:drift`) | `figures/domain_transfer.pdf` | `figures/published_figures.py` |
| Per-cohort effect (`fig:cohort`) | `figures/cohort.pdf` | `figures/published_figures.py` |
| Post-training (`fig:posttrain`, commented out in the paper) | `figures/posttrain.pdf` | `figures/published_figures.py` |

Measured agreement (rendered at 150 dpi, exact pixel comparison): the 10
figures of `figures/published_figures.py`, `attacks_auc.pdf` of
`figures/paper_figures.py` and `design.pdf` are identical to the published
files, given Matplotlib 3.10.8 and the same font resolution
(`env/analysis/requirements.txt`). The paper's source directory also holds
`timing.pdf`, which the paper does not include and no script draws;
`compare_with_paper.py` reports it as missing from the generated figures. The
figures that `figures/paper_figures.py`
draws itself are identical for `attacks_auc`, `cohort` and
`signal_distributions`; the others differ in 0.3% to 29% of their pixels
(collection 5.76%,
domain_transfer 0.30%, intro 1.16%, posttrain 0.51%, reference_tiers 1.93%,
roc 22.39%, trajectory 28.74%), and `dose.pdf` has a different page width.

## Tables

| Paper table (label) | File | Produced by |
|---|---|---|
| Table 1, cohorts as built (`tab:cohorts-design`) | written in the paper source | counts from `bundle/summary.json` |
| Table 2, post-training stages (`tab:stages`) | `tables/stages.tex` | mid-training, SFT and DPO rows: `figures/paper_figures.py`; the three RLVR rows: `analysis/rlvr/finalize.py` (`stages_extended.csv`), pasted by hand |
| Table 3, attack AUC and TPR (`tab:attacks`) | `tables/attacks.tex` | `figures/paper_figures.py` |
| Table 4, reference ratio and standardized ratio (`tab:calibrated`) | `tables/calibrated.tex` | `analysis/recompute_table4_all_pairs.py` and `analysis/format_table4.py` (`calibrated_allNAT1_textCIs.tex`) |
| Table 5, per-cohort effect (`tab:cohorts`) | `tables/cohorts.tex` | `figures/paper_figures.py` |
| Table 6, training settings (`tab:hyperparameters`) | written in the paper source | the configurations in `training/` |
| Numbers cited in the text | `tables/numbers.tex` | `figures/paper_figures.py`, with the additions below |

`figures/paper_figures.py` also writes `tables/collection.tex` and
`tables/sensitivity.tex`, which the current paper does not include.

Table 4 is recomputed on all 4,832 NAT1 pairs (instead of the 3,886-pair
holdout split) with the unmodified functions of `analysis/calibrated_attacks.py`
and `figures/paper_figures.py`; the first step of
`analysis/recompute_table4_all_pairs.py` reproduces the holdout values exactly.
`analysis/format_table4.py` writes a variant whose confidence intervals for
the exact-counterfactual rows and the standardized-ratio rows are the
whole-NAT1 intervals printed in the text; with the paper's `numbers.tex`
(`--numbers`), this variant equals the paper's `tables/calibrated.tex` except
for its comment lines. Without `--numbers`, the script reads the generator's
own `paper/tables/numbers.tex`, and two interval bounds of the
standardized-ratio rows differ in the last digit (7.1 instead of 7.3, and 1.7
instead of 1.6). These bounds are among the interval macros that the paper
took from the run that drew the published figures, whose random stream
differs (see the list of macros below). `--paper-tables` of
`recompute_table4_all_pairs.py` is optional: it adds cross-checks against the
given macros and carries the commented RMIA rows of `calibrated.tex` into the
table. `analysis/run_analysis.sh` passes both options when `PAPER_TABLES` is
set.

## Hand edits in the paper

The paper's table and macro files are assembled from the outputs above with
the following edits; the repository reproduces each ingredient, not the final
`.tex` files byte for byte.

**Table files.** The generated tables were reformatted to `siunitx` `S`
columns (no percent signs or `{,}` in the cells, cohort names with subscripts,
`K{=}20` instead of `k=20`); the values are unchanged. The RMIA rows of Table 4
are commented out. The three RLVR rows of `stages.tex` were pasted by hand
from `rlvr/stages_extended.csv`; re-running `figures/paper_figures.py` drops
them.

**Macros.** The paper's `numbers.tex` is the generator's file reformatted to
bare values (no `\xspace`, no `{,}`). Of the 868 macros that the generator
emits and the paper also defines, 765 are identical and 103 differ:

- 64 `calib*` macros: Table 4 recomputed on all NAT1 pairs (from
  `analysis/format_table4.py`);
- 19 `coll*` macros and 10 interval bounds (9 `tpr*` bounds and `\xsecci`):
  taken from the run that drew the published figures, whose random stream
  differs;
- 9 rounding formats (6 `\dosetpr*`, 3 `\screenz*`);
- `\npairs`: the paper's 7,803 complete pairs, while the generator's population
  ALL is the 7,326-pair pooled set.

The paper defines 26 macros that the generator does not emit:

- 15 RLVR macros from `analysis/rlvr/finalize.py` (`proposed_macros.tex`),
  with one renaming: the paper's `\ptinstructdzall` (0.50) and
  `\ptinstructtprall` (9.1) are finalize's `\ptinstructdzpooled` and
  `\ptinstructtprpooled` (the 7,326 pairs without cohorts F and X), not its
  `*all` values (7,803 pairs: 0.51 and 7.2), because the paper's pooled
  population is the generator's ALL;
- 11 macros entered by hand, rounded from analysis outputs under the data
  root:

  | Macro | Value | Source |
  |---|---|---|
  | `\npairspooled` | 7326 | `analysis/paper_numbers.json`, `ALL/n` |
  | `\cohfnform` | 271 | `analysis/cohortF_2x2_check.json`, `accounting/pairs_analysed` |
  | `\cohfplacebo` | 0.48 | `analysis/cohortF_2x2_check.json`, `placebos/P1_native_control_level/auc` |
  | `\cohfplacebopdf` | 0.47 | `analysis/cohortF_2x2_check.json`, `placebos/P1_transfer_control_level/auc` |
  | `\lenaucfourk` | 0.726 | `analysis/length_truncation.json`, `fixed_curve` at 4,096 tokens |
  | `\lendiff`, `\lendifflo`, `\lendiffhi` | +0.14, 0.12, 0.16 | `analysis/length_truncation.json`, `paired_auc_differences/4096_minus_16` |
  | `\cohlaucfivetwelve`, `\cohlaucfourk` | 0.61, 0.80 | `analysis/length_truncation.json`, `L_by_length` at 512 and 4,096 tokens |
  | `\cohmbestfree` | 0.636 | `analysis/cohort_reference.json`, `M/best_solo` (the largest AUC of a reference-free attack on cohort M) |

  `cohort_reference.json` was written by an exploratory script that is not
  part of this repository; it is a released data file.

`figures/compare_with_paper.py` reproduces this comparison: it renders every
PDF in both `figures/` directories and reports identical files, the share of
differing pixels or a size mismatch; it parses both `numbers.tex` files
(ignoring `\num{}`, `\xspace` and `{,}`) and reports common, differing and
missing macros.

## Notes on exact reproduction

- The analysis scripts use fixed seeds (20261012 for `figures/paper_figures.py`
  and `analysis/calibrated_attacks.py`, 7 for the intervals of mean effects, 3
  for the trajectory windows, 20261013 for the other analysis scripts,
  20261028 for the RLVR analysis), so their outputs are deterministic for a
  given environment.
- The JSON and Markdown outputs record the generating script and their input
  files; these strings name the layout of this repository and therefore differ
  from the files of the original working copy, while every number is the same.
- `analysis/rlvr/` was run with the Python of the post-training image
  (Python 3.10, numpy 1.26.4, pandas 2.2.3; `env/posttrain/pip-freeze.txt`).
- With `--paper-dir`, `analysis/rlvr/finalize.py` also writes
  `reproduction_check.csv` and annotates in `proposed_macros.tex` which macros
  the given paper already defines; the macro values do not depend on it.
