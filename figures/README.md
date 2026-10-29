# Figures, tables and macros

This directory produces the paper's figures, its table fragments
(`tables/*.tex`) and its macro file (`tables/numbers.tex`, one `\newcommand`
per number cited in the text). Everything here runs on a CPU in the analysis
environment (`env/analysis/requirements.txt`) and reads only the analysis
outputs under the data root (see `olmo_twin/paths.py` and `docs/data.md`).

| File | Purpose |
| --- | --- |
| `paper_figures.py` | Generator of record for the tables, `numbers.tex` and `analysis/paper_numbers.json`. Also the library that the analysis scripts import (`load`, `population`, `headline`, `tpr_at`, ...). |
| `published_figures.py` | Redraws the data figures exactly as published, except `attacks_auc`, which `paper_figures.py` draws. Writes no tables or macros. |
| `paper_design_figure.py` | Draws the design schematic, `design.pdf` (reads no data). |
| `make_all.sh` | Runs the three scripts above in the required order. |
| `compare_with_paper.py` | Compares a set of regenerated figures and macros with the paper's. |

## Why there are two generators

The paper's numbers and its data figures were produced by different
revisions of the same generator, so no single revision reproduces both:

* `paper_figures.py` is the current revision. Its pooled population `ALL`
  excludes cohorts F and X (7,326 pairs). The five checklist analyses
  (`analysis/screening_analysis.py`, `collection_by_domain.py`,
  `exclusion_sensitivity.py`, `cohortF_2x2_check.py`,
  `length_truncation_analysis.py`) import it for their populations and
  statistics, and `collection_by_domain.py` and `exclusion_sensitivity.py`
  check their results against the `analysis/paper_numbers.json` it writes.
  It writes the tables, the macros and `paper_numbers.json`. It also draws
  its own versions of the data figures; two of them (`cohort.pdf`,
  `signal_distributions.pdf`) equal the published ones.
* `published_figures.py` is the earlier revision (12 October 2026) that drew
  the published data figures, together with the figure-only corrections made
  before publication and the one-column cohort figure of the current revision.
  Its pooled population is every complete pair (7,803) and it consumes the
  shared random stream in a different order, so its own numbers are not the
  paper's; it therefore stops after the last figure. The header of the file
  lists the differences.

The 11 data figures are `intro`, `signal_distributions`, `roc`,
`attacks_auc`, `reference_tiers`, `dose`, `domain_transfer`, `cohort`,
`trajectory`, `posttrain` and `collection`. The paper includes ten of them;
`posttrain.pdf` is drawn but not included. `attacks_auc` comes from
`paper_figures.py` and the other ten from `published_figures.py`: on 29
October 2026 the paper's `attacks_auc` was redrawn with pair-bootstrap
intervals, as the paper describes, replacing an earlier version whose
intervals resampled members and non-members independently.

## Running

```bash
pip install -r env/analysis/requirements.txt
pip install -e .                      # or run from the checkout without installing
export OLMO_TWIN_DATA=/path/to/data   # default: <repository>/data
figures/make_all.sh                   # or: figures/make_all.sh --out DIR
```

`make_all.sh` runs, with the same output directory (default
`$OLMO_TWIN_DATA/paper`):

1. `python figures/paper_figures.py --out DIR` writes `DIR/tables/` and
   `analysis/paper_numbers.json`, and draws its versions of the data figures
   into `DIR/figures/`;
2. `python figures/published_figures.py --out DIR` overwrites the 11 data
   figures in `DIR/figures/` with the published versions;
3. `python figures/paper_design_figure.py --out DIR/figures` writes
   `DIR/figures/design.pdf`. For this script `--out` names the directory that
   receives the PDF, not its parent.

Inputs, all under the data root:

* `analysis/per_document_attacks_prelim.csv.gz`, `posttrain_exclusions.json`
  (required: the table is never analysed without the post-training
  exclusions), `per_document_attacks_calibrated.csv.gz` and
  `calibrated_attacks_summary.json`, `ladder_trajectory.csv.gz`,
  `ladder_meta.json`, `token_neff.npz`;
* for the full run of `paper_figures.py` only: `analysis/screening_analysis.json`,
  `collection_by_domain.json`, `exclusion_sensitivity.json`,
  `cohortF_2x2_check.json` and `length_truncation.json`;
* `bundle/summary.json` (injected tokens per cohort);
* `run/scores_posttrain/{sft,dpo}-{injected,baseline}/scores.csv.gz` for the
  post-training figure and table (a stage whose scores are absent is left out).

### Fresh data root

`analysis/collection_by_domain.py` and `analysis/exclusion_sensitivity.py`
read `analysis/paper_numbers.json`, which `paper_figures.py` writes, while the
full run of `paper_figures.py` reads their outputs. On a data root without
these files, run `python figures/paper_figures.py --first-pass` first: it skips
the checklist macros and the table files (except `tables/calibrated.tex`) and
writes `paper_numbers.json` with the same headline entries (`NAT1`, `ALL`, `E`,
`coll`) as the full run. Then run the five analyses and `make_all.sh`.
`analysis/run_analysis.sh` does all of this in order (see
`analysis/README.md`). Without the first pass, the full run stops with
`FileNotFoundError` on `analysis/screening_analysis.json`.

## Agreement with the paper

Measured on 28 October 2026 with Python 3.13.12, numpy 2.5.1 and
matplotlib 3.10.8. Figures were compared with `compare_with_paper.py`
(rendered at 150 dpi with pypdfium2, exact pixel equality); macros after
normalizing `\num{}`, `\xspace` and thousands separators.

* `make_all.sh` (`paper_figures.py`, then `published_figures.py` and
  `paper_design_figure.py`): all 12 PDFs (the 11 data figures and
  `design.pdf`) are pixel-identical to the published ones.
* `paper_figures.py`: `analysis/paper_numbers.json` is byte-identical to the
  file the analysis steps used. Of its figures, `cohort.pdf` and
  `signal_distributions.pdf` are identical to the published ones; the others
  differ in 0.3 to 29% of pixels, and `dose.pdf` has another width.
* `numbers.tex`: the generator emits 868 of the paper's 894 macros, 765 of
  them with the published value. The 103 that differ are 64 `calib*` macros
  (Table 4 was recomputed on all 4,832 NAT1 pairs by
  `analysis/recompute_table4_all_pairs.py` and `analysis/format_table4.py`),
  19 `coll*` macros and 10 interval bounds (9 `tpr*` and `\xsecci`) taken
  from the run that drew the published figures, 9 rounding formats
  (6 `\dosetpr*`, 3 `\screenz*`) and `\npairs` (the paper states the 7,803
  complete pairs; the generator's pooled population has 7,326). The 26
  macros the generator does not emit are 15 post-training macros from
  `analysis/rlvr/finalize.py` and 11 entered by hand. The generator also
  defines 1,084 macros that the paper does not use.

The paper's `numbers.tex` and table files are the generator's output edited
by hand: values reformatted for `siunitx`, the corrections above, rows
commented out, and the RLVR rows of the stages table pasted from
`analysis/rlvr/finalize.py`. `docs/paper_outputs.md` maps every figure and
table to its command and lists these edits.

Pixel identity also depends on the fonts that matplotlib finds. The
generators ask for Nimbus Roman No9 L, then Times New Roman, then DejaVu Serif;
the published figures were rendered with DejaVu Serif (shipped with
matplotlib) because matplotlib found neither of the first two. Where
matplotlib finds one of them (as a TrueType or OpenType file in its font
cache), the figures show the same content but differ in pixels.

To compare your own output with the paper's sources (a directory with
`figures/` and `tables/`):

```bash
python figures/compare_with_paper.py --generated $OLMO_TWIN_DATA/paper --paper /path/to/paper --verbose
```

The paper's source directory also holds `timing.pdf`, which the paper does
not include and no script here draws; the comparison lists it as missing.
