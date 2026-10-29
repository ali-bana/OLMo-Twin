# Data

This document describes four things:

- where the scripts expect their data;
- what the planned data release contains;
- which data are not redistributed, and why;
- the licences, and the public AI2 inputs with the versions we used.

The data and models are to be released; this repository contains only code.

## The data root

Every script resolves its inputs and outputs through `olmo_twin/paths.py`:

1. a script's own command-line options (for example `--out`,
   `--evaluation-manifest`) always win;
2. otherwise the data root is `$OLMO_TWIN_DATA`;
3. if the variable is not set, the data root is `data/` in the repository
   (ignored by git; it can be a symbolic link).

`python -m olmo_twin.paths` prints the resolved layout and which directories
exist. Shell scripts obtain the same paths as `OLMO_TWIN_<NAME>` variables
through `eval "$(python3 -m olmo_twin.paths --shell)"`, run from the
repository root or after `pip install -e .`.

A step writes into the same directories that later steps read, so re-running
a step overwrites its earlier output in place. To keep the released files
unchanged, work on a copy of the data root. Directories that a step only
reads can be symbolic links to the released copy.

### Layout

| Directory | Contents | Written by | Read by |
|---|---|---|---|
| `bundle/` | The frozen design: pairs, cohorts, arm assignment, copies and their positions, the never-injected background arm, evaluation manifest, checksums; document texts (`texts_all/`, `texts/`) and token ids (`tokens/`, `null_tokens/`) | `design/build_injection_manifest.py` (`build`, `refill`); `training/midtrain/build_injection_stream_v2.py` (`plan`: `stream_plan/`, `null_tokens/`; `make-eval-manifest`); `scoring/score_pretreatment_logprobs.py` (`logprobs_extra*/`); `texts_all/`, `texts_index.csv.gz` and `SHA256SUMS` were written without a script (see `design/README.md`) | `design/`, `training/midtrain/`, `training/checks/`, `scoring/`, `analysis/`, `figures/`, the RLVR overlap scan |
| `analysis/` | Per-document attack tables and the outputs of every analysis script (JSON, Markdown reports), `paper_numbers.json`, `cache/` | `analysis/*.py`, `figures/paper_figures.py` (`paper_numbers.json`), `collection/make_posttrain_exclusions.py` (`posttrain_exclusions.json`) | `analysis/`, `figures/` |
| `run/` | Treated run: `checkpoints/` (with `train_data/global_indices.npy` and `data-indices/`), scores under `scores/`, `scores_ext/`, `scores_ladder/`, `scores_posttrain/`, `scores_cohortF_2x2/` (one directory per scored model), and `cohortF_2x2/` | mid-training (`training/midtrain/train.sh`), `scoring/` (including `scoring/cohortF_2x2_tokenise.py` for `cohortF_2x2/`) | `training/checks/`, `analysis/`, `figures/` |
| `stream/` | The injected training stream: 64 data files, `stream_manifest.json`, `config/` | `training/midtrain/build_stream.sh` | mid-training, `training/checks/` |
| `assets/` | Public AI2 inputs: `configs/`, `stage2_tokenized/`, `checkpoints/`, `hf_models/`, `posttrain_datasets/`, `hf_cache_posttrain/` | `training/midtrain/` download scripts, `training/posttrain/convert_to_hf.sh`, `collection/download_posttrain_datasets.py`, `hf download` | `training/`, `scoring/`, `training/checks/`, the post-training and RLVR overlap scans |
| `posttrain/` | Post-training runs (`runs/sft_<arm>`, `dpo_<arm>`, `rlvr1_<arm>`, `rlvr2_<arm>`) and open-instruct's tokenized dataset cache (`dataset_cache/`) | `training/posttrain/` | `scoring/` |
| `posttrain_overlap/` | Scan of the post-training datasets for evaluation documents: `posttrain_hits.csv.gz`, `posttrain_impact.json` | `collection/posttrain_impact.py` | `collection/make_posttrain_exclusions.py`, `analysis/exclusion_sensitivity.py` |
| `exclusions/` | `repetition_filter_masked_chunks.json` (injected chunks masked by OLMo's instance filter) | released file; `training/checks/check_instance_filter.py` recomputes the masked chunks | `analysis/exclusion_sensitivity.py` |
| `rlvr/` | Final RLVR tables and macros; `overlap/` (RLVR dataset scan); `work/` (intermediate files) | `analysis/rlvr/*.py`, `collection/rlvr_overlap_*.py` | `analysis/rlvr/` |
| `collection/` | Working directory of the document collection and of the overlap search (candidate pools, fingerprints, scan results and hit records, edited-Wikipedia group, pre-treatment scores) | `collection/` | `collection/`, `design/` |
| `cohortF_reparse/` | Second text form of the cohort-F papers (PDF form of the injected papers, LaTeX form of the held-out ones, a 20-paper control) | `collection/cohortF_reparse.py` | `scoring/cohortF_2x2_tokenise.py`, `analysis/audit_cohortF_data.py`, `analysis/cohortF_2x2_check.py` |
| `paper/` | Output only: `tables/` and `figures/` | `figures/` | `analysis/format_table4.py` (`tables/numbers.tex`) |

### Bundle files read by the code

| File | Contents |
|---|---|
| `manifest.csv.gz` | One row per paired document: cohort, stratum, arm, pair, split, dose, burst, target length, pre-treatment bits per byte, text form, novelty stratum |
| `copies.csv.gz` | One row per injected copy: training decile and target position |
| `null_pool.csv.gz` | The never-injected background arm (7,053 documents) |
| `evaluation_manifest.csv.gz` | The 22,681 documents to score, with their token files (relative to `bundle/`); one injected cohort-F paper is longer than the scorers' limit of 65,536 tokens and has no scores |
| `summary.json` | Per-cohort counts and injected tokens |
| `texts_index.csv.gz` | Document id -> text file and text form |
| `stream_plan/placement.jsonl.gz`, `stream_plan/placement_summary.json` | Training positions and data-file slots of every injected chunk |
| `SHA256SUMS` | SHA-256 of the design files, token files and texts |
| `manifest_v1a_prepairfix.csv.gz` | An earlier build of the manifest, before a correction of the pairing; `analysis/cohortF_2x2_check.py` uses its arm column for a randomization check of cohort F |
| `scoring_manifest_extra.csv.gz` | Extra texts scored before training (LaTeX forms of cohort F, cohort-L prefixes, cohort-X anchors) with their text paths |
| `logprobs_extra_stage1/scores.csv.gz` | Pre-treatment per-document scores of these extra texts under the stage-1 checkpoint (read by `analysis/cohortF_2x2_check.py`) |
| `tokens/`, `null_tokens/` | Token ids of every document (int32 `.npy`); redistributed only where the licence permits, see below |
| `texts_all/` | The exact text of every scored document; redistributed only where the licence permits, see below |

`analysis/cohortF_2x2_check.py` also reads `analysis/cohortF_2x2_stats.json`.
This file is part of the data release; it was written by an exploratory
script that is not part of this repository. Without it, the script skips
the comparison with the stored values.

## The planned data release

The data release is planned to be organised as a data root. Unpacking it
and setting `OLMO_TWIN_DATA` is then enough to run the CPU steps.

### Contents

- **Design metadata** (`bundle/`): the files in the table above except
  texts and token ids. It also includes the records of the refill after the
  stage-1 exclusions: `dclm_stage1_exclusions*.csv.gz`,
  `refill_dclm1*.csv.gz` and the manifests before each refill pass. Titles
  are arXiv metadata (CC0) and Wikipedia or StackExchange page titles.
- **Analysis tables** (`analysis/`):
  - the attack tables `per_document_attacks_prelim.csv.gz` and
    `per_document_attacks_calibrated.csv.gz`;
  - the ladder tables `ladder_trajectory.csv.gz` and `ladder_meta.json`;
  - `token_neff.npz` and `cache/token_cache.npz`;
  - the cached per-document statistics of `analysis/calibrated_attacks.py`
    (`cache/doc_stats.npz`);
  - `posttrain_exclusions.json`, `cohortF_2x2_stats.json` and
    `cohort_reference.json`;
  - the outputs of the analysis scripts.
- **Per-document scores** of every model and checkpoint
  (`run/scores*/<label>/scores.csv.gz` and `scoring_meta*.json`):
  - the treated model, the control, the retrained reference and the
    pre-stage checkpoint;
  - the 69 intermediate checkpoints of the three stage-2 runs;
  - the post-trained models of both arms: SFT, DPO, both RLVR stages, and
    the RLVR checkpoints every 200 steps;
  - AI2's released post-trained models;
  - the pre-treatment scores in `bundle/logprobs_extra*/scores.csv.gz`.
- **Per-token log-probabilities** (`run/scores*/<label>/tokens/*.npz`): the
  arrays `logprobs`, and `mu` and `sigma` for the extended and Hugging Face
  scorers, without the array `token_ids` (see [Token ids](#token-ids)).
- **Run metadata**:
  - the configuration saved with the treated run;
  - the stream manifest and the verification records;
  - the token-file manifest of AI2's stage-2 data with its SHA-256 values;
  - the RLVR results (`rlvr/`).
- **Exclusions**: `exclusions/repetition_filter_masked_chunks.json`, and
  `posttrain_overlap/posttrain_hits.csv.gz` with `posttrain_impact.json`.
- **Models**: the treated model's checkpoints every 1,000 updates
  (OLMo-native format) and the post-trained models of both arms (Hugging
  Face format).
- **Collection records**: the records of the non-membership verification.
  These are the candidate manifests, the fingerprint scan summaries and hit
  records of the 13 corpus components, and the evidence of the final
  selection step (the `collection/` directory of the data root).
- **Document texts and token ids** of the documents whose licences permit
  redistribution, with attribution (see [Licences](#licences)).

### Form of the released files

- Several files recorded absolute paths of the machines on which they were
  made. In the release, these are given relative to the data root or
  removed:
  - path columns of the manifests (`text_path`, `inject_text_path`,
    `alt_text_path`, `tokens_path`, `mask_path` in `manifest*.csv.gz` and
    `null_pool*.csv.gz`; `raw_text_path` in
    `scoring_manifest_extra*.csv.gz`);
  - the token-file manifest of AI2's data;
  - the run configurations and the verification records.

  `analysis/cohortF_2x2_check.py` infers the text form of a cohort-F entry
  from the `/raw/latex/` component of `raw_text_path`, so the relative form
  keeps that component.
- `verify_injection_set.py` and `injection_balance_report.py` open the text
  and token paths of the manifests. On another machine they need these paths
  to point at local copies (see `design/README.md`).
- `bundle/SHA256SUMS` also lists files that are not redistributed (texts and
  token ids), so that a rebuilt bundle can be checked against it. Check a
  partial copy with
  `cd "$OLMO_TWIN_DATA/bundle" && sha256sum --check --ignore-missing SHA256SUMS`.

### Not redistributed

- Texts and token ids of documents whose licences do not permit
  redistribution (see [Licences](#licences)). This covers the corresponding
  files in `bundle/texts_all/`, `bundle/texts/`, `bundle/tokens/`,
  `bundle/null_tokens/` and the `tokens/` subdirectories of
  `bundle/logprobs_extra*/`. For these documents, the release gives the
  identifier, the text form, and the SHA-256 of the text and of the token
  file.
- `cohortF_reparse/`, the full texts of the cohort-F papers in their second
  form. The arXiv identifiers of these papers are in
  `collection/cohortF_reparse_ids/`.
- The `token_ids` array of every per-token file.
- `posttrain_overlap/matched_examples.txt`, which quotes excerpts of the
  post-training datasets.
- AI2's inputs (checkpoints, training data, post-training datasets, models),
  which the scripts download from AI2 at the versions listed below.

## Licences

### Code

The code in this repository is released under the Apache License, Version
2.0 ([LICENSE](../LICENSE)). [NOTICE](../NOTICE) lists the third-party code
and configurations that it contains or modifies.

### Documents

Every evaluation document is third-party text, except the synthetic anchors
of cohort X. Whether its text may be redistributed depends on its source.

- **arXiv.** arXiv metadata (identifiers, titles, abstracts) is available
  under CC0. Each paper's full text is under the licence its authors chose
  when submitting. This is either arXiv's non-exclusive distribution licence
  or one of several Creative Commons licences.
  - Under the non-exclusive licence, the rights are granted to arXiv only, so
    third parties may not redistribute the text.
  - Under a no-derivatives licence (such as CC BY-NC-ND), modified forms may
    not be shared. The cleaned and parsed texts used here are modified forms.

  Texts of such papers will not be redistributed; the release gives their
  identifiers. Texts of papers under licences that permit redistribution of
  modified text will be released with the attribution and licence notice
  that their licence requires.
- **Wikipedia and StackExchange.** Their text is available under the
  Creative Commons Attribution-ShareAlike licence (CC BY-SA). The release is
  planned to give, for each redistributed text, the attribution that the
  licence requires:
  - the source page or post (the page title, or the site with the question
    and answer ids) and a link to it;
  - a link to its authors (the page history or the post);
  - the licence;
  - a note that the text was converted to plain text in the format of
    OLMo's training data.

  The derived texts are shared under the same licence.
- **Synthetic anchors of cohort X.** The sentence- and word-shuffled anchors
  are derived from arXiv papers and follow the licence of their source paper.
  The model samples and the random token sequences are generated: the
  samples by the stage-1 checkpoint of OLMo 2 1B, the random sequences
  directly as token ids.

### Token ids

Token ids encode the document text exactly: the public OLMo 2 tokenizer
decodes them back to the text. They are therefore treated like the text and
will be released only for documents whose text is released, in
`bundle/tokens/` and `bundle/null_tokens/`. The per-token files of the scored
models will be released without the `token_ids` array.

### Data derived by us

The manifests, scores, per-token log-probabilities, analysis tables and the
treated and post-trained models will be released under a licence stated
with the data release. The models are derived from AI2's OLMo 2 1B
(Apache-2.0).

### AI2's inputs

AI2's models, checkpoints, code and datasets are under AI2's licences:
Apache-2.0 for the models and code, and ODC-BY together with the terms of
the individual sources for the datasets. They are referenced, not
redistributed.

## Rebuilding texts and token ids

Texts are rebuilt from public sources with the collection steps
([collection/README.md](../collection/README.md)):

- arXiv: the LaTeX source through the RedPajama cleaner, and the PDF through
  GROBID 0.8.1;
- Wikipedia: through WikiExtractor 3.0.8;
- StackExchange: from the public API.

The design steps ([design/README.md](../design/README.md)) then make the
prefixes, the LaTeX forms and the shuffled anchors, and tokenize the texts
with the Hugging Face tokenizer of `allenai/OLMo-2-0425-1B` (no special
tokens, one end-of-text token). The arXiv identifiers of the cohort-F
re-parse are in `collection/cohortF_reparse_ids/`. Compare the result with
the released checksums:

```bash
cd "$OLMO_TWIN_DATA/bundle" && sha256sum --check --ignore-missing SHA256SUMS
```

Sources can change after collection: an arXiv paper can get a new version,
and a Wikipedia page can be edited or deleted. The checksums show which
documents were rebuilt exactly.

### Steps that need token ids or document text

These steps cannot run on the redistributed data alone. They need the
rebuilt `bundle/tokens/`, `bundle/texts_all/`, or the per-token files with
`token_ids`:

| Step | Needs |
|---|---|
| `analysis/build_injection_v2_attack_table.py` | `token_ids` of the per-token files; zlib statistic on the exact text in `bundle/texts_all/` |
| `analysis/calibrated_attacks.py --recompute` | `token_ids` (without `--recompute` it uses the released `cache/doc_stats.npz`) |
| `analysis/compute_token_neff.py` | `token_ids` (alignment check of the two models' files) |
| `analysis/audit_cohortF_data.py` | `bundle/texts_all/` and `cohortF_reparse/` |
| `analysis/cohortF_2x2_check.py` | `token_ids` of the per-token files, `bundle/tokens/F/`, `run/cohortF_2x2/tokens/`, `cohortF_reparse/` |
| `scoring/cohortF_2x2_tokenise.py` | `cohortF_reparse/` texts |
| `collection/rlvr_overlap_scan.py` | `bundle/tokens/` and `bundle/texts_all/` |
| `training/midtrain/build_injection_stream_v2.py`, `training/midtrain/build_stream.sh` | `bundle/tokens/` (the injected chunks); `build_stream.sh` checks every file in `bundle/SHA256SUMS` |
| `scoring/score_injection_v2*.py` | `bundle/tokens/` |
| `scoring/score_pretreatment_logprobs.py` | the document texts (it tokenizes them itself) |

The other analysis steps, the figures and tables, Table 4 and the RLVR
analysis run on the released data. For the steps above, the release
contains their outputs. `analysis/run_analysis.sh` checks for these inputs
and skips a step whose inputs are absent.

## Public AI2 inputs

| Input | Source and version |
|---|---|
| OLMo trainer | `allenai/OLMo` at commit `090253dac6688f2532509daa7aa2eb5fae50e956` |
| Stage-2 configuration | `configs/official-0425/OLMo2-1B-stage2-seed42.yaml` at that commit, SHA-256 `095dabb7b20c9b6710abad3717a789c9d3109e070069b8643805bb407d2a2fd1` (checked by `training/midtrain/`) |
| Stage-1 endpoint (start of mid-training) | `https://olmo-checkpoints.org/ai2-llm/peteish1/step1907359-unsharded/` (step 1,907,359; 4,000,021,741,568 tokens). SHA-256: `config.yaml` `ec6f24df31443597a5a281b09b03d46aba7f21f942d572bdff8d64fff18d21a3`, `train.pt` `7f5b2e840fa256f589dab6ae206e667e1ef4891d9c26ba1d4db364d6bdf45939`, `model.pt` `7b5a229bde2dc24a48fc269411e27c2bf3020446483a1e99a611d6985633b80b`, `optim.pt` `c5baee99495d0af7596cb58b69fd2ef108282bcf6348f02b8958ef9c6e41772a` (`training/midtrain/download_stage1_checkpoint.sh`) |
| Stage-2 data | The 576 pre-tokenized files listed in `data.paths` of the configuration (under `http://olmo-data.org/preprocessed/`), 51,491,007,838 tokens, 205,964,031,352 bytes (`training/midtrain/prepare_stage2_assets.py` downloads them and records their SHA-256) |
| Control (seed 42) | `https://olmo-checkpoints.org/ai2-llm/peteish1/stage2/olmo2-1b-0425_stage2_ingredient3/`, `config.yaml` and `model.pt` of steps 1000 to 23000 every 1,000 and 23852 |
| Retrained reference (seed 42069) | `https://olmo-checkpoints.org/ai2-llm/peteish1/stage2/olmo2-1b-0425_stage2_ingredient1/`, same steps (`training/midtrain/download_ai2_stage2_checkpoints.sh`) |
| Tokenizer and Hugging Face branches | `allenai/OLMo-2-0425-1B`, downloaded at commit `a1847dff35000b4271fa70afc5db10fd29fedbdf`; its tokenizer files are copied into both converted start models; branch `stage1-step1907359-tokens4001B` for the pre-treatment scores |
| RLVR tokenizer | `allenai/OLMo-2-1124-7B-DPO` at commit `e34ea60adff2e575f4fe7569eaffd1b28509b6fd` (tokenizer files only) |
| SFT data | `allenai/tulu-3-sft-olmo-2-mixture-0225` at commit `d91a0785ade02942520280fb484866fce41e448f` |
| DPO data | `allenai/olmo-2-0425-1b-preference-mix` at commit `c6454f3d364622718a8ecd3d307e524febbba569` |
| RLVR data | `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` at commit `7dbd180f5440c0b90f2944e6efea934b85437a95` (stage 1); `allenai/RLVR-MATH` at commit `bd2a93551b503a395fadd1a740d957559cfe6f3c` (stage 2) |
| Training corpora (overlap search) | `allenai/olmo-mix-1124` at commit `99ee6aaace88779d1ef099d36251b91101c1679b`; `allenai/dolmino-mix-1124` at commit `a319f19eef1e257417b11ea8c30da266ae175557`; 13 components (`collection/run_overlap_scan.sh`) |
| AI2's post-trained models | `allenai/OLMo-2-0425-1B-SFT`, `-DPO`, `-RLVR1`, `-Instruct`; see below |

The revisions of AI2's four post-trained models were not recorded when they
were downloaded. The Hugging Face Hub stores weight files with Git LFS, which
identifies a file by the SHA-256 of its content, so these values identify the
weights we scored:

| Model | File | SHA-256 |
|---|---|---|
| `allenai/OLMo-2-0425-1B-SFT` | `pytorch_model.bin` | `338aeb8b3049cc9b19ba1f3e6df4b5280f3e002b5ae9e38328a353d8f072d534` |
| `allenai/OLMo-2-0425-1B-DPO` | `pytorch_model.bin` | `2262a62b6caa094aeea3996b054a080370d59f349436d5e9788792063d6acda1` |
| `allenai/OLMo-2-0425-1B-RLVR1` | `model.safetensors` | `c556b74da82a3f3d72a4a07515c1703ae87c06eeefb1c726628090535ac1e2ad` |
| `allenai/OLMo-2-0425-1B-Instruct` | `model.safetensors` | `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` |

These inputs are referenced, not redistributed; see [AI2's inputs](#ai2s-inputs)
for their licences.
