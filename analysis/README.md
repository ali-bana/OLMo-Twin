# Analysis

The scripts in this directory turn the scores of the scored models into every
statistic reported in the paper: the per-document attack table, the
standardized reference ratio, the checkpoint-ladder trajectories, the
operating points and screening results, the collection-level tests, the
sensitivity and cohort-specific checks, Table 4 on all pairs, and the
post-training and RLVR results. They run on a CPU; no GPU is needed.

All inputs and outputs live under the data root, which is resolved by
`olmo_twin/paths.py`: the environment variable `OLMO_TWIN_DATA`, or
`<repository>/data` when it is not set (see `docs/data.md` for the layout and
for the data release). Paths below are relative to the data root. Use the
analysis environment (`env/analysis/requirements.txt`). Run the scripts from a
checkout of the repository (or after `pip install -e .`); each script finds the
`olmo_twin` package and the modules of `figures/` relative to its own location.

## Running everything

```bash
export OLMO_TWIN_DATA=/path/to/olmo-twin-data
bash analysis/run_analysis.sh
```

`run_analysis.sh` runs the steps below in order. Environment variables:
`PYTHON` (interpreter, default `python3`), `WORKERS` (processes of the parallel
steps, default 32; the results do not depend on it), `SKIP` (step names to
leave out, for example `SKIP="attack_table token_neff"`), `FORCE=1` (run steps
whose inputs look absent, see below), `PAPER_TABLES` (a directory with the
paper's `numbers.tex` and `calibrated.tex`, used only for the Table 4
cross-checks) and `PAPER_DIR` (paper sources with `tables/numbers.tex` and
`tables/stages.tex`, used only for the reproduction check of the RLVR macros).
Every step overwrites its outputs in place; work on a copy of the data root if
you want to keep the released files.

## Order of the steps

| step (`SKIP` name) | script | main inputs | outputs |
|---|---|---|---|
| `attack_table` | `build_injection_v2_attack_table.py` | `bundle/` (evaluation manifest, manifest, placement, texts index and texts); per-token files of `run/scores_ext/ext-injected`, `ext-seed42`, `ext-stage1-pretreatment` and `run/scores/sibling-seed42069-step23852` | `analysis/per_document_attacks_prelim.csv.gz` |
| `token_neff` | `compute_token_neff.py` (`NWORKERS`) | per-token files of `run/scores/{injected,official-seed42}-step23852` | `analysis/token_neff.npz` |
| `token_cache` | `precompute_token_cache.py` (`WORKERS`) | the same per-token files (log-probabilities only) | `analysis/cache/token_cache.npz` |
| `ladder` | `build_ladder_table.py` | `run/scores_ladder/`, `run/scores/` (69 ladder checkpoints and the three endpoints), placement | `analysis/ladder_trajectory.csv.gz`, `analysis/ladder_meta.json` |
| `calibrated` | `calibrated_attacks.py [--recompute]` | per-token files of the four models, attack table | `analysis/per_document_attacks_calibrated.csv.gz`, `analysis/calibrated_attacks_summary.json` (cache `analysis/cache/doc_stats.npz`) |
| `first_pass` | `figures/paper_figures.py --first-pass` | the outputs above | `analysis/paper_numbers.json` |
| `screening` | `screening_analysis.py` | attack table (through `paper_figures.load()`) | `analysis/screening_analysis.{json,md}` |
| `collection` | `collection_by_domain.py` | attack table, `analysis/paper_numbers.json` | `analysis/collection_by_domain.{json,md}` |
| `sensitivity` | `exclusion_sensitivity.py` | attack table, placement, `posttrain_overlap/posttrain_hits.csv.gz`, `exclusions/repetition_filter_masked_chunks.json`, `analysis/paper_numbers.json` | `analysis/exclusion_sensitivity.{json,md}` |
| `length` | `length_truncation_analysis.py` | `analysis/cache/token_cache.npz`, per-token log-probabilities | `analysis/length_truncation.json` |
| `cohortF_audit` | `audit_cohortF_data.py` | bundle manifest and texts, `cohortF_reparse/`, scores | `analysis/cohortF_data_exclusions.json` |
| `cohortF_check` | `cohortF_2x2_check.py` | attack table, `run/scores_cohortF_2x2/`, `run/cohortF_2x2/tokens/`, bundle tokens of cohort F, `bundle/logprobs_extra_stage1/scores.csv.gz` | `analysis/cohortF_2x2_check.json` |
| `full_pass` | `figures/paper_figures.py --out paper` | all outputs above | `paper/tables/`, `paper/figures/`, `analysis/paper_numbers.json` (overwritten) |
| `published_figures` | `figures/published_figures.py`, `figures/paper_design_figure.py` | as the full pass | the other data figures as published (all but `attacks_auc.pdf`, which the full pass draws) and `design.pdf` in `paper/figures/` |
| `table4` | `recompute_table4_all_pairs.py`, then `format_table4.py` | calibrated outputs, `paper/tables/numbers.tex` | `analysis/table4_all_pairs/` (values, macros, `calibrated_allNAT1_textCIs.tex`) |
| `rlvr` | `rlvr/recompute.py`, `rlvr/rl_minus_dpo.py`, `rlvr/finalize.py` | attack table, `run/scores_posttrain/`, `rlvr/overlap/rlvr_overlap_per_doc.csv.gz` | `rlvr/work/` (intermediate), `rlvr/` (final tables and `proposed_macros.tex`) |

`injection_v2_plotstyle.py` (table loading with the exclusions, rank AUC,
bootstrap intervals, plotting style) and `trajectory_analysis.py`
(event-aligned forgetting analysis over the ladders) are libraries used by the
scripts above and by the figure generators.

### Why `figures/paper_figures.py` runs twice

`figures/paper_figures.py` is both the generator of the tables and macros and
the library that the analysis scripts import (`load()`, `population()`,
`headline()`, ...). Its full run reads the outputs of five analysis scripts
(screening, collection by domain, exclusion sensitivity, cohort-F check and
length truncation) and writes `analysis/paper_numbers.json` only at the end,
while `exclusion_sensitivity.py` asserts its primary rows against
`paper_numbers.json` and `collection_by_domain.py` records its `coll` entry. On
a fresh data root neither side could run first. The first pass
(`--first-pass`) skips the part that reads the five JSON files and the table
writer and writes `paper_numbers.json` from the headline computations, which
draw the same random numbers as in the full run, so the NAT1, ALL, E and `coll`
entries are identical to those of the full run. The full run afterwards
overwrites the file.

### Inputs that are not produced here

- `analysis/posttrain_exclusions.json`: the pairs and documents found in the
  post-training data. It is written by `collection/make_posttrain_exclusions.py`
  from `posttrain_overlap/posttrain_hits.csv.gz` and is also part of the data
  release. Loading the attack table fails if the file is missing, so that no
  analysis silently runs without the exclusions.
- `analysis/cohortF_2x2_stats.json`: summary statistics of the cohort-F 2x2
  analysis written by an exploratory figure script that is not part of this
  repository; it is released with the analysis outputs. `cohortF_2x2_check.py`
  compares its re-derivation against it (step 10) and records the comparison as
  skipped when the file is absent; nothing else depends on it.
- `rlvr/overlap/rlvr_overlap_per_doc.csv.gz`: the scan of the RLVR datasets
  for evaluation documents (`collection/rlvr_overlap_scan.py` and
  `collection/rlvr_overlap_per_doc.py`).
- `exclusions/repetition_filter_masked_chunks.json`: the 43 injected training
  chunks masked by OLMo's repetition filter (`training/checks/`); used only as
  a cross-check of the list embedded in `exclusion_sensitivity.py`.

### Steps that need token ids or document texts

The document texts and token ids are third-party content and are not
redistributed (see `docs/data.md`). The following steps need them and can only
run after the texts and tokens have been rebuilt with `collection/` and
`design/`: the attack table (token ids of the per-token files, zlib on the exact
text), `compute_token_neff.py` (token-id alignment check),
`calibrated_attacks.py --recompute` (token-id alignment check; without
`--recompute` it reuses `analysis/cache/doc_stats.npz`), `audit_cohortF_data.py`
(texts) and `cohortF_2x2_check.py` (token arrays). `run_analysis.sh` checks for
these inputs and skips such a step with a message, keeping the released
outputs, unless `FORCE=1`. With the released analysis tables, everything from
`precompute_token_cache.py` onward that does not appear in this list can be
re-run.

## Populations, exclusions and seeds

- **NAT1**: the 4,832 verified-unseen single-copy pairs (one injected copy;
  cohorts M, L, W-new, W-gap and S-se). It is the primary population.
- **Pooled**: the 7,326 pairs of all cohorts except F and X;
  `paper_figures.population(..., "ALL")`.
- **Every complete pair**: 7,803 pairs, `population(..., "ALLPAIRS")`. In
  `screening_analysis.py` and `exclusion_sensitivity.py` the label `ALL` denotes
  this set, and the pooled set appears as `ALL_noXF` and as variant `d3`
  respectively; `calibrated_attacks.py` and `rlvr/` name their populations
  explicitly (`ALL7326`, `ALLPAIRS7803`).
- **Exclusions**: the 10 pairs touched by documents found in the post-training
  data (16 documents, including 5 null-pool documents) are removed from every
  analysis; one injected cohort-F document exceeds the scorer's 65,536-token
  limit, is not scored, and its pair is incomplete.
- **Seeds**: 20261012 (pair bootstrap of `calibrated_attacks.py` and the shared
  random generator of `figures/paper_figures.py`), 20261013 (screening,
  collection by domain, exclusion sensitivity, length truncation, cohort-F
  check), 7 (bootstrap intervals of the within-pair effect,
  `injection_v2_plotstyle.bootstrap_ci`), 3 (window bootstrap of
  `trajectory_analysis.py`) and 20261028 (post-training and RLVR pair
  bootstrap).

Where a JSON output carries metadata (a `meta`, `script`, `table` or
`generated_by` field), it names the generating script and the input files
relative to the data root; the Markdown reports state the script in their first
lines.

## Table 4 on all pairs

Table 4 of the paper (`tables/calibrated.tex`) reports the reference ratio and
the standardized ratio on all 4,832 NAT1 pairs. `calibrated_attacks.py`
evaluates them on the holdout split fixed in the design (3,886 pairs);
`recompute_table4_all_pairs.py` re-runs its evaluation with the unmodified
functions of `calibrated_attacks.py` and `figures/paper_figures.py` on all NAT1
pairs. Its step 0 reproduces the 141 holdout values of
`calibrated_attacks_summary.json` exactly. `format_table4.py` then formats the
macros and writes `calibrated_allNAT1_textCIs.tex`, the table whose intervals
for the exact-counterfactual rows and the standardized-ratio rows are the
whole-NAT1 intervals printed in the text. With the paper's `numbers.tex`
(`--numbers`) its body equals the paper's table; with the generator's own
`numbers.tex` two interval bounds of the standardized-ratio rows differ in the
last digit, for the reason given in `docs/paper_outputs.md`.

## Post-training and RLVR

`rlvr/recompute.py` computes every post-training and RLVR number independently
of `figures/paper_figures.py`: the within-pair contrasts at mid-training, SFT,
DPO, RLVR1, RLVR2 step 2000 (the analogue of the released Instruct model) and
the RLVR2 endpoint, the RL checkpoint trajectory, and the robustness rows with
AI2's released post-trained models as control. `rlvr/rl_minus_dpo.py` computes
the paired change from DPO to each RL stage on the same resampled pairs, and
`rlvr/finalize.py` writes the final tables and `proposed_macros.tex`. With
`--paper-dir` it also writes `reproduction_check.csv` and marks the macros that
the paper already defines; the macro values do not depend on it.
