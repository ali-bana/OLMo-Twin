# Reproducing the experiment

This guide goes through the pipeline that produced the paper, from document
collection to the figures. For each stage it lists the inputs, the outputs
and the commands, and it ends with the parts that cannot be reproduced from
this repository alone. The README of each directory has the details of its
stage. [paper_outputs.md](paper_outputs.md) maps each figure and table
of the paper to the command that produces it. [data.md](data.md) describes
the data layout and the data release.

## Overview

All paths below are relative to the data root (`$OLMO_TWIN_DATA`, see
[Before you start](#before-you-start)).

| Stage | Directory | Reads | Writes | Compute |
|---|---|---|---|---|
| 1. Collect candidates and search the training data | `collection/` | arXiv, Wikipedia and StackExchange sources; the OLMo 2 training corpora | `collection/` | CPU, network, GROBID; 8.3 TiB of corpora; one GPU for the embedding search |
| 2. Pre-treatment scores | `scoring/` | `collection/audit_v2/stageG/scoring_manifest.csv.gz` | `collection/audit_v2/logprobs_{stage1,main}/` | one GPU |
| 3. Freeze the design | `design/` | `collection/` | `bundle/` | CPU; one GPU for the cohort-X samples and the extra pre-treatment scores |
| 4. Download AI2's inputs | `training/midtrain/` | public URLs | `assets/` | network, about 0.5 TB of storage |
| 5. Build the training stream | `training/midtrain/` | `bundle/`, `assets/` | `stream/` | CPU, about 200 GB |
| 6. Mid-training | `training/midtrain/` | `stream/`, `assets/` | `run/checkpoints/` | 16 A100 80GB GPUs |
| 7. Check the run | `training/checks/` | `run/checkpoints/`, `stream/`, `assets/`, `bundle/` | printed reports | CPU |
| 8. Post-training | `training/posttrain/` | `run/checkpoints/`, `assets/` | `posttrain/runs/` | 8 GPUs (SFT, DPO); 16 GPUs (RLVR) |
| 9. Scoring | `scoring/` | `bundle/`, model checkpoints | `run/scores*/` | one GPU per model |
| 10. Scans of the post-training data | `collection/` | post-training datasets, `bundle/` | `posttrain_overlap/`, `analysis/posttrain_exclusions.json`, `rlvr/overlap/` | CPU |
| 11. Analysis | `analysis/` | `run/scores*/`, `bundle/`, exclusion lists | `analysis/`, `rlvr/` | CPU |
| 12. Tables, numbers and figures | `figures/` | `analysis/`, `bundle/summary.json`, `run/scores_posttrain/` | `paper/` | CPU |

Stages 1 to 3 produced the frozen design, and stages 4 to 9 the models and
their scores. Stages 10 to 12 turn the scores into the paper's numbers.

## Where to start

- **From the data release** (once it is available): stages 11 and 12 run on
  a CPU with the released analysis tables and scores. Steps that need
  document texts or token ids are skipped (see [data.md](data.md#steps-that-need-token-ids-or-document-text));
  the release contains their outputs.
- **From rebuilt texts and token ids**: rebuild the texts with stages 1 and 3
  and compare them with the released checksums. Then every CPU step of
  stages 11 and 12 can run, and scoring (stage 9) can be repeated for the
  released models.
- **From scratch**: all stages. This needs the GPUs listed above, local
  copies of the OLMo 2 training corpora, and access to the public sources,
  which change over time (see [Known gaps](#known-gaps)).

## Before you start

**Environments.** [env/README.md](../env/README.md) describes the analysis
and collection environments (requirements files) and the three GPU images
(Dockerfiles for mid-training, post-training and RLVR).

**Data root.** Every script resolves its inputs and outputs through
`olmo_twin/paths.py`. A script's own command-line options win. Otherwise the
data root is `$OLMO_TWIN_DATA`, or `data/` in the repository if the variable
is not set. Python scripts import the module; shell scripts evaluate its
shell form:

```bash
export OLMO_TWIN_DATA=/path/to/olmo-twin-data
python3 -m olmo_twin.paths                       # resolved layout
eval "$(python3 -m olmo_twin.paths --shell)"     # OLMO_TWIN_BUNDLE, OLMO_TWIN_ASSETS, ... (from the repository root)
```

**Upstream checkouts.** Two upstream repositories are used at pinned commits
and passed to the scripts by environment variable:

```bash
git clone https://github.com/allenai/OLMo /path/to/OLMo
git -C /path/to/OLMo checkout 090253dac6688f2532509daa7aa2eb5fae50e956
export OLMO_REPO=/path/to/OLMo

git clone https://github.com/allenai/open-instruct /path/to/open-instruct
git -C /path/to/open-instruct checkout d19de7d27ac2af2cd1dcbf856dba1a2d077b37f4
git -C /path/to/open-instruct apply "$PWD/training/posttrain/open-instruct.patch"
export OPEN_INSTRUCT_DIR=/path/to/open-instruct
```

The OLMo patch (`training/midtrain/olmo-shared-fs-checkpoint.patch`) is
needed only when several nodes write checkpoints to one shared file system
([training/README.md](../training/README.md#shared-file-systems)).

**No scheduler.** The scripts contain no scheduler or container commands.
Multi-node steps are run once per node, with the rendezvous given by
environment variables (`NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`
for `torchrun`; a Ray head and worker for RLVR). Start them by hand, or from
the job scripts of the scheduler you use.

**Outputs are overwritten in place.** Each step writes into the directories
that later steps read. To keep the released files unchanged, work on a copy
of the data root. Directories that a step only reads can be symbolic links.

**Scripts without options.** Some scripts are configured only through the
data root and environment variables. They ignore command-line arguments, so
`--help` starts the computation and overwrites the outputs:

- in `analysis/`: `screening_analysis.py`, `collection_by_domain.py`,
  `exclusion_sensitivity.py`, `length_truncation_analysis.py`,
  `audit_cohortF_data.py`, `cohortF_2x2_check.py`, `compute_token_neff.py`,
  `precompute_token_cache.py` and `rlvr/rl_minus_dpo.py`;
- `analysis/rlvr/recompute.py`, which takes only an optional number of
  worker processes;
- in `collection/`: `rlvr_overlap_scan.py`, `rlvr_overlap_per_doc.py` and
  `rlvr_overlap_positive_control.py`.

Read the docstring at the top of each of these files instead. The other
Python scripts accept `--help`, except `training/checks/check_lr_schedule.py`,
which takes one Weights & Biases log file as its only argument and only
reads it. The shell scripts describe their usage in
their first comment lines. Called with `--help` or with missing arguments,
they print their usage or an error and do not start any work.

## Stage 1: collect candidates and search the training data

Directory: `collection/` ([README](../collection/README.md), Stages A to G).
Environment: collection (`numba` is required; fingerprints made without it
are not compatible with scans made with it).

**Reads.** A local mirror of the arXiv bulk data (LaTeX sources and PDFs),
the Wikipedia dump `enwiki-20260801` and a list of page creations, the
StackExchange API, and the OLMo 2 training corpora `allenai/olmo-mix-1124`
and `allenai/dolmino-mix-1124` at the revisions given in
[data.md](data.md#public-ai2-inputs). It also needs a GROBID 0.8.1 service.

**Writes** (under `collection/`): the arXiv pools (`postcutoff/`), the
Wikipedia pools (`wikipools_v2/`), the StackExchange pool (`sepool_api/`),
the candidate manifest and fingerprints (`audit_v2/combined.csv.gz`,
`audit_v2/fp/`), one scan directory per corpus component, the embedding
matches, the Stage-G evidence and clean set (`audit_v2/stageG/`,
`audit_v2/stageG_13/`), the edited-Wikipedia group (`wiki_edited/`) and the
scoring manifest `audit_v2/stageG/scoring_manifest.csv.gz`.

**Run.**

1. Sample and parse the post-cutoff arXiv papers in both text forms
   (`build_postcutoff_arxiv_pool.py sample`, `parse`, `rawtext`;
   `merge_arxiv_pools.py`).
2. Build the Wikipedia pools and verify each page's creation date
   (`build_wikipedia_pools.py`, `verify_wikipedia_creation_dates.py`).
3. Fetch the StackExchange question/answer pairs
   (`fetch_stackexchange_api.py`).
4. Merge the candidates and fingerprint them (`build_combined_candidates.py`,
   `olmo2_nonmember_audit.py fingerprints`).
5. Scan the 13 components of the training corpora. The driver resumes after
   an interruption and fails if any target fails:

   ```bash
   WORK=$OLMO_TWIN_DATA/collection/audit_v2 CORPUS_ROOT=/path/to/olmo2-corpora WORKERS=11 \
       bash collection/run_overlap_scan.sh
   ```

6. Run the embedding search on one GPU (`olmo2_nonmember_audit.py embed`).
7. Apply the exclusion rule and build the edited-Wikipedia group
   (`assemble_injection_candidates.py`, `build_edited_wikipedia_group.py
   ancestors` and `align`), then write the scoring manifest:

   ```bash
   python collection/make_scoring_manifest.py    # -> collection/audit_v2/stageG/scoring_manifest.csv.gz
   ```

The full commands, with every option, are in the
[collection README](../collection/README.md#pipeline). For the paper, the
stage-1 `dclm` component (6.6 TiB) was scanned after the design had been
built. The pairs it affected were refilled in stage 3.

## Stage 2: pre-treatment scores

Directory: `scoring/` ([README](../scoring/README.md#pre-treatment-scores-for-pairing)).
Environment: collection (`torch`, `transformers`); one GPU.

**Reads** `collection/audit_v2/stageG/scoring_manifest.csv.gz` and the raw
texts it points to. **Writes** `collection/audit_v2/logprobs_stage1/` and
`logprobs_main/` (bits per byte under the stage-1 endpoint and under the
released model).

```bash
C="$OLMO_TWIN_DATA/collection"
python scoring/score_pretreatment_logprobs.py \
    --manifest "$C/audit_v2/stageG/scoring_manifest.csv.gz" --out-dir "$C/audit_v2/logprobs_stage1"
python scoring/score_pretreatment_logprobs.py --revision main \
    --manifest "$C/audit_v2/stageG/scoring_manifest.csv.gz" --out-dir "$C/audit_v2/logprobs_main"
```

The edited-Wikipedia group was scored in the same way (`wiki_edited/`).

## Stage 3: freeze the design

Directory: `design/` ([README](../design/README.md#commands)).
Environment: collection; one GPU for `gen-samples` and for the extra
pre-treatment scores.

**Reads** the clean candidates, their pre-treatment scores and the
edited-Wikipedia group under `collection/`. **Writes** `bundle/`:

- `manifest.csv.gz`, `copies.csv.gz`, `null_pool.csv.gz` and
  `summary.json`;
- the texts and token files (`texts/`, `tokens/`, `null_tokens/`);
- the placement of every copy in the training stream
  (`stream_plan/placement.jsonl.gz`);
- the list of documents to score (`evaluation_manifest.csv.gz`).

```bash
python design/build_injection_manifest.py gen-samples      # cohort-X model samples (GPU)
python design/build_injection_manifest.py build            # first build; keep its manifest
cp "$OLMO_TWIN_DATA/bundle/manifest.csv.gz" "$OLMO_TWIN_DATA/bundle/manifest_v1a_prepairfix.csv.gz"
# score the as-injected L, F and X texts (scoring_manifest_extra.csv.gz, see design/README.md), then:
python design/build_injection_manifest.py build --pair-scores "$OLMO_TWIN_DATA/bundle/logprobs_extra_stage1/scores.csv.gz"
python design/build_injection_manifest.py refill --version-tag dclm1 \
    --exclusions "$OLMO_TWIN_DATA/bundle/dclm_stage1_exclusions.csv.gz" \
    --clean "$OLMO_TWIN_DATA/collection/audit_v2/stageG_13/candidates_clean.csv.gz"
python design/verify_injection_set.py                     # exits non-zero on any violation
python design/injection_balance_report.py --out arm_balance
python training/midtrain/build_injection_stream_v2.py plan --injection-dir "$OLMO_TWIN_DATA/bundle"
python training/midtrain/build_injection_stream_v2.py make-eval-manifest \
    --injection-dir "$OLMO_TWIN_DATA/bundle" --output "$OLMO_TWIN_DATA/bundle/evaluation_manifest.csv.gz"
```

`build` is deterministic for a given `--seed` (20260922 for the paper),
because every random draw comes from a named random stream. The refill ran in
three passes; [design/README.md](../design/README.md#refill-after-the-stage-1-dclm-scan)
lists them. That README also describes the finishing steps of the bundle
that were done without a script: the extra pre-treatment scoring,
`texts_all/`, `texts_index.csv.gz` and `SHA256SUMS`. To check a rebuilt
bundle against the released checksums, run
`cd "$OLMO_TWIN_DATA/bundle" && sha256sum --check --ignore-missing SHA256SUMS`.

## Stage 4: download AI2's inputs

Directory: `training/midtrain/` ([README](../training/README.md#1-ai2s-inputs)).

**Writes** `assets/`:

- AI2's stage-1 endpoint with its optimizer state (17.8 GB);
- the 576 stage-2 token files (206 GB) with a token-file manifest and a
  localized configuration;
- AI2's seed-42 and seed-42069 checkpoint ladders (24 checkpoints each);
- for post-training and scoring, the Hugging Face tokenizer files.

```bash
CFG=$OLMO_REPO/configs/official-0425/OLMo2-1B-stage2-seed42.yaml
A=$OLMO_TWIN_DATA/assets
bash training/midtrain/download_stage1_checkpoint.sh
python training/midtrain/prepare_stage2_assets.py plan --official-config $CFG \
    --data-root $A/stage2_tokenized --manifest $A/configs/stage2_tokenized_manifest.json \
    --localized-config $A/configs/OLMo2-1B-stage2-seed42.local.yaml
python training/midtrain/prepare_stage2_assets.py download --manifest $A/configs/stage2_tokenized_manifest.json
python training/midtrain/prepare_stage2_assets.py verify --olmo-repo $OLMO_REPO --official-config $CFG \
    --localized-config $A/configs/OLMo2-1B-stage2-seed42.local.yaml \
    --manifest $A/configs/stage2_tokenized_manifest.json \
    --checkpoint-root $A/checkpoints/official_olmo_stage1_4001B \
    --checkpoint-alias $A/checkpoints/official_olmo_stage1_4001B-unsharded \
    --output $A/asset_verification.json
bash training/midtrain/download_ai2_stage2_checkpoints.sh ingredient3   # seed 42, the control
bash training/midtrain/download_ai2_stage2_checkpoints.sh ingredient1   # seed 42069, the retrained reference
```

The downloads are restartable and check the pinned sizes and SHA-256 values.

## Stage 5: build the training stream

Directory: `training/midtrain/` ([README](../training/README.md#2-build-and-verify-the-injected-stream)).
Environment: mid-training image (or Python with `numpy` and `pyyaml`); CPU.

**Reads** the complete `bundle/` and the stage-2 token files. **Writes**
`stream/`: 64 data files (about 200 GB), `stream_manifest.json`, and
`config/olmo2-1b-injection-v2.yaml`. The configuration is AI2's file with
only the data paths, run name, output folder, checkpoint intervals and
tracking keys changed; the script asserts this.

```bash
OLMO_REPO=$OLMO_REPO bash training/midtrain/build_stream.sh
```

The script first checks every file listed in `bundle/SHA256SUMS`, and checks
the placement file and AI2's configuration against the SHA-256 values pinned
in `training/midtrain/versions.env`. It then builds the stream and verifies:

- every shard;
- all 24,081 injected chunks against the bundle's token files;
- 20,000 sampled background chunks against AI2's files;
- the configuration diff.

## Stage 6: mid-training

Directory: `training/midtrain/` ([README](../training/README.md#3-launch)).
Environment: mid-training image; 2 nodes with 8 GPUs each in our run.

**Reads** `stream/`, `assets/checkpoints/official_olmo_stage1_4001B-unsharded`.
**Writes** `run/checkpoints/`: unsharded checkpoints every 1,000 updates,
`config.yaml`, `train_data/global_indices.npy`, `data-indices/`, and an
offline Weights & Biases log.

```bash
# on every node; node 0 runs the preflight checks first
OLMO_REPO=$OLMO_REPO NNODES=2 NODE_RANK=<0 or 1> MASTER_ADDR=<address of node 0> MASTER_PORT=29500 \
    bash training/midtrain/train.sh
```

`train.sh` runs `torchrun` with OLMo's `scripts/train.py`. Restarting the
same command resumes from the latest checkpoint. The global batch (512
sequences of 4,096 tokens) and the data order do not depend on the number
of GPUs. Our run used 16 GPUs where AI2 used 256; the
[training README](../training/README.md#differences-from-ai2s-run) lists
every difference from AI2's run.

## Stage 7: check the run

Directory: `training/checks/` ([README](../training/README.md#checks)).
Environment: analysis (`torch` for two checks, `wandb` for one).

```bash
python training/checks/check_data_order.py
python training/checks/check_injected_copies.py           # all 12,414 copies
python training/checks/check_background_chunks.py
python training/checks/check_background_chunks.py --seed 777 --draw 1300 --keep 1200
python training/checks/check_instance_filter.py --olmo-repo $OLMO_REPO
python training/checks/compare_configs.py --olmo-repo $OLMO_REPO
python training/checks/compare_hf_branches.py              # network access; --main also compares the main revision
python training/checks/check_lr_schedule.py <offline W&B log under run/checkpoints/wandb/>
```

These checks reproduce the numbers in the paper's appendix "Implementation
Details":

- the data order: 93,632 of 93,632 per-rank index rows match;
- all 12,414 injected copies are at their planned positions;
- 3,700 background positions are identical to AI2's chunks;
- 43 injected chunks from 18 documents are masked by OLMo's instance filter;
- the configuration differences;
- bitwise identity of the Hugging Face branches with the downloaded
  checkpoints;
- the learning-rate schedule.

## Stage 8: post-training

Directory: `training/posttrain/` ([README](../training/README.md#posttrain)).
Environment: post-training image (SFT, DPO, conversion) and RLVR image.

**Reads** the two mid-training endpoints. **Writes**
`assets/hf_models/` (the converted start models) and
`posttrain/runs/{sft,dpo,rlvr1,rlvr2}_{baseline,injected}/`.

```bash
OLMO_REPO=$OLMO_REPO bash training/posttrain/convert_to_hf.sh \
    $OLMO_TWIN_DATA/assets/checkpoints/official_stage2_ingredient3_seed42/step23852-unsharded \
    $OLMO_TWIN_DATA/assets/hf_models/OLMo-2-0425-1B-seed42-hf            # control
OLMO_REPO=$OLMO_REPO bash training/posttrain/convert_to_hf.sh \
    $OLMO_TWIN_DATA/run/checkpoints/step23852-unsharded \
    $OLMO_TWIN_DATA/assets/hf_models/olmo2-1b-injection-v2-final-hf      # treated
for ARM in baseline injected; do
    OPEN_INSTRUCT_DIR=$OPEN_INSTRUCT_DIR ARM=$ARM bash training/posttrain/run_sft.sh
    OPEN_INSTRUCT_DIR=$OPEN_INSTRUCT_DIR ARM=$ARM bash training/posttrain/run_dpo.sh
done
# RLVR: for STAGE in 1 2 and ARM in baseline injected, on two nodes
#   first node:   OPEN_INSTRUCT_DIR=... RAY_PORT=6379 bash training/posttrain/run_rlvr.sh head STAGE ARM
#   second node:  OPEN_INSTRUCT_DIR=... RAY_HEAD_IP=<first node> RAY_PORT=6379 bash training/posttrain/run_rlvr.sh worker
```

The tokenizer files must be downloaded first; the training README lists the
`hf download` commands. On GPU nodes without internet access, run
`training/posttrain/cache_datasets.sh` on a machine that has access (sharing
the data root) and set the Hugging Face offline variables for the training
runs. Both arms use the same recipe, with only the start model, names and
paths changed. One setting differs from AI2's logged SFT run: our SFT
averages the loss per micro-batch instead of summing it. The
[training README](../training/README.md#recipe-and-the-known-deviation)
explains this.

## Stage 9: scoring

Directory: `scoring/` ([README](../scoring/README.md)).
Environment: mid-training image for the OLMo-native scorers, post-training
image for the Hugging Face scorer; one GPU per process.

**Reads** `bundle/evaluation_manifest.csv.gz` and the token files it lists,
and the model checkpoints. **Writes** one directory per model under
`run/scores/`, `run/scores_ext/`, `run/scores_ladder/`,
`run/scores_posttrain/` and `run/scores_cohortF_2x2/`. Each holds
`scores.csv.gz`, the per-token files `tokens/*.npz` (except for the
ladders) and `scoring_meta.rank*.json`.

```bash
scoring/run_scoring.sh --list                    # the 142 labels in scoring/models.tsv
scoring/run_scoring.sh injected-step23852        # one GPU: score, then merge
for r in 0 1 2 3 4 5 6 7; do                     # eight GPUs: one rank each, then merge
    CUDA_VISIBLE_DEVICES=$r scoring/run_scoring.sh ext-injected "$r" 8 &
done; wait
scoring/run_scoring.sh ext-injected merge
```

`run_scoring.sh` looks up the scorer, model and output directory of a label.
Before it loads an OLMo-native checkpoint, it checks the seed that the
checkpoint's configuration declares. Cohort F's second text form is scored
after `collection/cohortF_reparse.py` and `scoring/cohortF_2x2_tokenise.py`
(labels `F2x2-*`). AI2's released post-trained models are scored under the
labels `ai2-*`; their weight files are identified by SHA-256 in
[data.md](data.md#public-ai2-inputs).

## Stage 10: scans of the post-training data

Directory: `collection/` ([README](../collection/README.md#post-training-datasets)).

The post-training datasets were scanned for evaluation documents with the
same matcher. The scan found 16 evaluation documents, 11 of them in 10 pairs.
Both members of these pairs and the 5 flagged never-inserted documents are
excluded from every analysis.

```bash
python collection/posttrain_impact.py --out-dir "$OLMO_TWIN_DATA/posttrain_overlap"
python collection/make_posttrain_exclusions.py   # -> analysis/posttrain_exclusions.json
python collection/rlvr_overlap_scan.py           # -> rlvr/overlap/ (needs bundle texts and token files)
python collection/rlvr_overlap_per_doc.py
python collection/rlvr_overlap_positive_control.py
```

`posttrain_impact.py` reads the scans of the four datasets, which are made
with `build_posttrain_corpus.py` and `olmo2_nonmember_audit.py scan`.
`analysis/posttrain_exclusions.json` and `rlvr/overlap/` are part of the data
release, so this stage is needed only to rebuild them.

## Stage 11: analysis

Directory: `analysis/` ([README](../analysis/README.md)).
Environment: analysis; CPU.

```bash
PYTHON=python3 WORKERS=16 bash analysis/run_analysis.sh
```

The script runs 16 steps in order:

- the per-document attack table, the token statistics, the ladder table and
  the calibrated attacks;
- a first pass of the table generator;
- the screening, collection, sensitivity, length and cohort-F analyses;
- the full generator and the published figures;
- Table 4 on all pairs;
- the post-training and RLVR analysis.

`SKIP` leaves steps out. Steps whose inputs (token ids, document texts) are
absent are skipped with a message unless `FORCE=1`. Set `PAPER_TABLES` to a
directory with the paper's `numbers.tex` and `calibrated.tex` (from the
paper's LaTeX sources) to reproduce Table 4 exactly; see
[How closely a re-run matches](#how-closely-a-re-run-matches).

## Stage 12: tables, numbers and figures

Directory: `figures/` ([README](../figures/README.md)).

```bash
bash figures/make_all.sh                         # or: bash figures/make_all.sh --out DIR
python figures/compare_with_paper.py --generated "$OLMO_TWIN_DATA/paper" --paper /path/to/paper-sources
```

`make_all.sh` runs `paper_figures.py`, which writes the tables, the macros
and `analysis/paper_numbers.json`. It then runs `published_figures.py`,
which redraws the other data figures exactly as published (`attacks_auc.pdf`
is the version of `paper_figures.py`), and
`paper_design_figure.py`. [paper_outputs.md](paper_outputs.md) explains
why there are two generators and lists the parts of the paper that were
edited by hand.

## How closely a re-run matches

**What was checked.** The CPU steps of this repository were re-run on the
original data, with all their outputs written to a separate data root. In
the original software environments, they reproduce the original outputs:

- `analysis/paper_numbers.json` is byte-identical;
- the 11 data figures and `design.pdf` are pixel-identical to the published
  ones;
- the generated tables are identical except for their comment line;
- the Table 4 values and the 570 RLVR macro values are identical;
- the checks of stage 7 give the numbers of the paper;
- the RLVR overlap scan and `make_posttrain_exclusions.py` reproduce the
  released files.

The JSON and Markdown outputs name the generating script and their inputs
with the paths of this repository, so they differ from the original files
in these strings only.

The GPU stages (mid-training, post-training, scoring) were not re-run from
this repository. Their launchers pass the same arguments as the commands of
our runs (compared flag by flag), without the scheduler and container
commands.

**Numerical environment.** Some outputs differ in the last digits under a
different NumPy version:

- the attack table, `token_neff.npz` and the RLVR drift files: in the
  analysis environment of `env/analysis/` (NumPy 2.5) they differ at the
  level of floating-point round-off from the files computed in the GPU
  images (NumPy 1.26). In those images they are reproduced exactly;
- downstream of a rebuilt attack table, `paper_numbers.json` then differs at
  the same level, while the figures, tables and macros are unchanged;
- the `mink_delta` and `maxk_delta` fields of `analysis/cache/token_cache.npz`
  depend on the order in which `np.argsort` places ties. No script in this
  repository reads them.

**Known differences from the released analysis outputs.**

- *Table 4 needs the paper's macro file.* `analysis/format_table4.py`
  copies the confidence intervals that the text gives for all 4,832
  verified-unseen single-exposure pairs (the population `NAT1` of the code)
  into Table 4.
  With the generator's own `numbers.tex`, two bounds of the
  standardized-ratio rows differ in the last digit: 7.1 instead of 7.3, and
  1.7 instead of 1.6. These bounds in the paper come from an earlier run with
  a different random stream (see [paper_outputs.md](paper_outputs.md)).
  Set `PAPER_TABLES` as described in stage 11.
- *Exclusion sensitivity.* All point estimates and pair-bootstrap intervals
  in `analysis/exclusion_sensitivity.json` are reproduced. The released file
  also holds `auc_lo`/`auc_hi` "paper interval" fields for the oracle rows.
  These were computed with an earlier revision of the generator, and a re-run
  gives slightly different values for them. No number in the paper is
  taken from these fields.
- *Fresh data root.* On a data root without earlier outputs,
  `screening_analysis.json` has no `reproduction.paper_macros` block, because
  `paper/tables/numbers.tex` exists only after the full pass of the
  generator. Running the screening step again after the full pass adds it.
- *Fonts.* Pixel identity of the figures needs Matplotlib 3.10.8 and the same
  font resolution. The published figures were drawn with Matplotlib's bundled
  DejaVu Serif ([figures/README.md](../figures/README.md#agreement-with-the-paper)).

## Known gaps

The following parts of the experiment cannot be reproduced from this
repository alone, or are recorded only partly.

1. **Collection records.** These records are planned for the data release
   ([data.md](data.md#contents)); this repository does not contain them:
   - the candidate pools;
   - the fingerprint metadata;
   - the scan summaries and hit records of the 13 corpus components;
   - the Stage-G evidence;
   - the edited-Wikipedia group;
   - the pre-treatment scores.

   Without them, stage 1 must be re-run from the public sources. These
   change over time: arXiv papers get new versions, and Wikipedia pages are
   edited or deleted. A rebuilt candidate set will therefore not be
   identical. The unigram model of the PDF parser can be rebuilt with
   `pes2o_style_pdf_parser.py build-lm`, but the exact file used is part of
   the records. The list of Wikipedia page creations that
   `build_wikipedia_pools.py sample` reads has no producer in this
   repository.
2. **Steps whose drivers were not kept.** Three sets of steps were run with
   short commands that are not part of the repository:
   - the three refill passes after the stage-1 `dclm` scan;
   - the scan of the post-training datasets;
   - the finishing steps of the bundle: the extra pre-treatment scoring,
     `texts_all/`, `texts_index.csv.gz` and `SHA256SUMS`.

   The [collection](../collection/README.md) and
   [design](../design/README.md) READMEs reconstruct them from the recorded
   outputs and mark the reconstructed options. The exclusion list
   `analysis/posttrain_exclusions.json` was originally assembled by hand;
   `collection/make_posttrain_exclusions.py` reproduces it.
3. **The stream needs an exact rebuild of the bundle.** `build_stream.sh`
   refuses to run unless every file listed in `bundle/SHA256SUMS` matches.
   Its pinned values also fix the placement file and AI2's configuration. A
   document whose text cannot be rebuilt exactly therefore prevents an exact
   rebuild of the training stream.
4. **Checks reported in the paper without code here.**
   - The audit of the training text displaced by injected copies (appendix
     "The Thirteen Cohorts": 49 evaluation documents share at least 50
     document-specific 13-grams with displaced text).
   - The rescoring comparison (appendix "Statistical Details", paragraph
     "Rescoring"): 180 documents in fp32 under flash attention and PyTorch's
     attention kernel, and the agreement of the native and Hugging Face
     scorers.
   - Two byte-identity statements of the appendix "Implementation Details":
     the converted control start model against the Hugging Face branch
     `stage2-ingredient3-step23852-tokens51B`, and the tokenized
     post-training datasets of the two arms.

   The comparison of four tensors on each of five Hugging Face branches is
   repeated by `training/checks/compare_hf_branches.py`. The other checks of
   that appendix are in `training/checks/`. They were rewritten from the
   one-off scripts of the original audit and reproduce its results.
5. **Hand edits in the paper.** The paper's table and macro files were
   assembled from the generators' outputs with some edits by hand. The
   repository reproduces each ingredient, but not the final `.tex` files
   byte for byte. [paper_outputs.md](paper_outputs.md#hand-edits-in-the-paper)
   lists the edits.
6. **Environments are recorded only partly.**
   - No package list was recorded for the mid-training image: OLMo's
     training dependencies were installed unpinned on the base image, whose
     digest is recorded.
   - The package versions of the collection environment were not recorded.
   - The Dockerfiles in `env/` are conversions of the container recipes
     that were built for our runs, and have not been built themselves.
   - The RLVR analysis ran with the Python of the post-training image, not
     with the analysis environment.
7. **Revisions of AI2's post-trained models.** The revisions of the four
   released post-trained models that were scored were not recorded when they
   were downloaded. The SHA-256 of their weight files identifies them
   ([data.md](data.md#public-ai2-inputs)).
8. **Run directories with timestamps.** The RLVR run directories carry their
   launch time, because the GRPO script is not patched. `scoring/models.tsv`
   therefore names them with a pattern.
9. **Third-party text.** Without document texts and token ids, the analysis
   can be re-run only from the released attack table onward
   ([data.md](data.md#steps-that-need-token-ids-or-document-text)).
