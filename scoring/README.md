# Scoring

This directory computes the per-token log-probabilities of the evaluation
documents under every model that the paper compares: the treated mid-training
run, AI2's seed-42 run (the control) and seed-42069 run (an independent
reference), the published stage-1 endpoint, the intermediate checkpoints of
these runs, the post-trained models of both arms and AI2's released
post-trained models. All membership scores in `analysis/` are computed from
these files. A second script computes the pre-treatment bits per byte that
were used to form the matched pairs before the design was frozen.

All paths below are relative to the data root, `$OLMO_TWIN_DATA` (default
`<repository>/data`; see `olmo_twin/paths.py` and `docs/data.md`).

## Conventions

The three scorers of the evaluation set share the following conventions, so
that a document's score is computed in the same way at every training stage.

- **Input.** The stored token IDs listed in `bundle/evaluation_manifest.csv.gz`
  (column `tokens_relpath`, relative to `bundle/`; int32 `.npy` files under
  `tokens/<cohort>/` and `null_tokens/`). Nothing is re-tokenized. The IDs were
  produced once with the Hugging Face tokenizer of `allenai/OLMo-2-0425-1B`
  without special tokens; every document ends with one end-of-text token
  (ID 100257) and no beginning-of-sequence token is added.
- **Windows.** Non-overlapping windows of 4,096 tokens with no context across
  windows, one window per forward pass and no padding, which matches how
  training saw long documents. The first position of each window has no
  prediction and is stored as NaN, so a document of *n* tokens has
  *n* − ⌈*n*/4096⌉ scored positions.
- **Precision.** Weights in bf16; the log-softmax is taken in fp32. The
  OLMo-native scorers use the attention kernel set in each checkpoint's
  configuration (flash-attn for the treated checkpoints, PyTorch's
  scaled-dot-product attention for AI2's checkpoints); the Hugging Face scorer
  uses scaled-dot-product attention. The scorers draw no random numbers.
- **Length limit.** Documents longer than 65,536 tokens are not scored; they
  get a row with status `too_long`. This applies to one document of the
  evaluation set (a cohort-F member of 69,157 tokens) and to two documents of
  the cohort-F re-parse.
- **Per-document values.** `nll_sum` (negative log-likelihood over the scored
  positions, nats), `mean_logprob` (the token-weighted mean over the scored
  positions), `bpb` (`nll_sum / ln 2 / n_bytes`, with the UTF-8 byte count of
  the document from the manifest) and `n_chunks` (number of windows).
- **Vocabulary statistics.** The extended and Hugging Face scorers also store,
  at every scored position *t*, the mean μ<sub>t</sub> and the standard
  deviation σ<sub>t</sub> of the log-probability over the full output layer
  (100,352 rows: the tokenizer's vocabulary and 74 padding rows). Min-K%++ and
  the standardized ratio need them; they cannot be recovered from the
  log-probability of the observed token alone.
- **Raw text for post-trained models.** The post-trained models score the raw
  document, not a chat-formatted version, so that scores are comparable across
  stages.

The plain and the extended OLMo-native scorer give identical per-document
scores: for the treated and the control endpoint, `mean_logprob` in
`run/scores/` and `run/scores_ext/` agrees exactly on all 22,680 scored
documents.

## Scripts

| Script | Loads | Used for |
|---|---|---|
| `score_injection_v2.py` | OLMo-native checkpoint (`model.pt` + `config.yaml`, unsharded) | treated and control endpoints, the 69 ladder checkpoints (`--no-token-dump`: document-level scores only), the cohort-F re-parse |
| `score_injection_v2_extended.py` | OLMo-native checkpoint | the same plus μ<sub>t</sub>, σ<sub>t</sub>: treated, control and stage-1 endpoints (`scores_ext/`), seed-42069 endpoint |
| `score_injection_v2_hf.py` | Hugging Face model directory | SFT, DPO and RLVR models of both arms, the RLVR intermediate checkpoints, AI2's released post-trained models |
| `score_pretreatment_logprobs.py` | `allenai/OLMo-2-0425-1B` from the Hugging Face Hub, at a branch | pre-treatment bits per byte for pairing (see below) |
| `cohortF_2x2_tokenise.py` | the OLMo 2 tokenizer | tokenizes the re-parsed cohort-F texts (see below) |
| `run_scoring.sh`, `models.tsv` | | runs one of the first three scripts for a model listed by label |

The OLMo-native scorers need the `olmo` package of OLMo at commit `090253d`
(the mid-training environment, `env/midtrain/`); the Hugging Face scorer needs
`transformers` (the post-training environment, `env/posttrain/`). Each scoring
process uses one GPU; an A100 processes about 34,000 tokens per second with
the plain scorer, so one model on the 22,681-document evaluation set (93.8M
tokens) takes about 46 GPU-minutes.

### Outputs

Each model is scored into its own output directory, which contains

- `scores.rank<r>.csv.gz`: one row per document of rank *r*, written as the
  scoring proceeds (a restarted rank skips the documents already in its file,
  and the plain scorer also recovers a file left truncated by a killed process);
- `scores.csv.gz`: all ranks merged by `--merge`, with columns `doc_id`,
  `revision` (the label), `n_tokens`, `n_bytes`, `nll_sum`, `bpb`,
  `mean_logprob`, `n_chunks`, `status`, and for the extended and Hugging Face
  scorers also `mean_mu` and `mean_sigma`;
- `tokens/<doc_id>.npz` (with `/` and `:` in the ID replaced by `_`): arrays
  `token_ids` (int32), `logprobs` (float32, NaN where unscored), and `mu`,
  `sigma` for the extended and Hugging Face scorers;
- `scoring_meta.rank<r>.json`: model path, label, settings and number of
  documents of each rank.

The per-token files contain the token IDs of third-party documents; the data
release ships them without `token_ids` (see `docs/data.md`).

### Sharding

The documents are split by position in the manifest: rank *r* of *W* scores
documents *r*, *r* + *W*, *r* + 2*W*, ... (`--rank r --world-size W`). Since
every document is scored on its own, the split does not change any score.
After all ranks have finished, a run with `--merge` (same arguments otherwise)
concatenates the rank files into `scores.csv.gz`. The paper's models were
scored with 8 ranks, except the ladder checkpoints and the cohort-F re-parse
(1 rank).

## Labels and output directories

`models.tsv` lists every model scored for the paper, one row per label, with
the scorer, the model directory, the output directory, the token directory
the manifest refers to, the seed that the checkpoint's `config.yaml` must
declare, and extra flags. The label is written into `scores.csv.gz` and is the
name of the output directory. The analysis scripts read the directories by
these names.

| Directory | Labels | Models |
|---|---|---|
| `run/scores/` | `injected-step23852`, `official-seed42-step23852`, `sibling-seed42069-step23852` | final checkpoints (update 23,852) of the treated run, AI2's seed-42 run and AI2's seed-42069 run |
| `run/scores_ext/` | `ext-injected`, `ext-seed42`, `ext-stage1-pretreatment` | treated and seed-42 endpoints and the stage-1 endpoint, with μ<sub>t</sub>, σ<sub>t</sub>; read by the attack table and `analysis/calibrated_attacks.py` |
| `run/scores_ladder/` | `{injected,seed42,sibling}-step{1000,...,23000}` | the checkpoints saved every 1,000 updates by the three runs (69 in all), document-level only; read by `analysis/build_ladder_table.py` |
| `run/scores_posttrain/` | `{sft,dpo,rlvr1,rlvr2}-{injected,baseline}`, `rlvr{1,2}-step{200,...,2600}-{injected,baseline}`, `ai2-{sft,dpo,rlvr1,instruct}` | post-trained models of the treated (`injected`) and control (`baseline`) arm, the RLVR checkpoints saved every 200 steps, AI2's released models |
| `run/scores_cohortF_2x2/` | `F2x2-{injected,seed42,stage1}` | treated, seed-42 and stage-1 endpoints on the re-parsed cohort-F documents; read by `analysis/cohortF_2x2_check.py` |

Where the models come from: `run/checkpoints/` is written by the mid-training
run (`training/midtrain/`); AI2's stage-1 endpoint and the seed-42
(`official_stage2_ingredient3_seed42`) and seed-42069
(`official_stage2_ingredient1`) checkpoint ladders under `assets/checkpoints/`
are obtained as described in `training/midtrain/README.md`; `posttrain/runs/` is written by
`training/posttrain/`. AI2's released post-trained models were downloaded from
the Hugging Face Hub into `assets/hf_models/ai2_released/<name>`, for example

```bash
hf download allenai/OLMo-2-0425-1B-SFT --revision 0d85a3d037876ce6ac7d4311d994400fc66ac27f \
    --local-dir "$OLMO_TWIN_DATA/assets/hf_models/ai2_released/OLMo-2-0425-1B-SFT"
```

The revisions were not recorded at download; the comment block of
`models.tsv` gives the SHA-256 of each weight file and the commits that `main`
pointed to on the day of the download.

## Running a model with `run_scoring.sh`

`run_scoring.sh` looks a label up in `models.tsv`, resolves the paths under
the data root and calls the matching scorer with the arguments listed above.
It contains no scheduler directives; call it directly, from a shell loop, or
from the job script of whatever scheduler you use.

```bash
export OLMO_TWIN_DATA=/path/to/olmo-twin-data
export PYTHON=python3                     # interpreter of the matching environment
export OLMO_REPO=/path/to/OLMo            # only if the olmo package is not installed

scoring/run_scoring.sh --list             # labels, scorers, output directories

# one GPU: score all documents, then merge
scoring/run_scoring.sh injected-step23852

# eight GPUs of one machine: one rank per GPU, then merge
for r in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$r scoring/run_scoring.sh ext-injected "$r" 8 &
done
wait
scoring/run_scoring.sh ext-injected merge
```

Before loading an OLMo-native checkpoint, the script checks that its
`config.yaml` declares the seed given in `models.tsv` (42 for the treated run
and AI2's seed-42 run, 42069 for AI2's seed-42069 run): scoring a checkpoint
as the wrong arm would invert the comparison. The model path of a row may be a
glob, because the RLVR run directories end in their launch time; it must match
exactly one directory, otherwise set `MODEL_PATH` to the directory to score.
A label whose `scores.csv.gz` already has one row per document is skipped. A
shorter merged file is reported and removed, and the scorer then continues
from the per-rank files; a merge that yields fewer rows than documents is
reported, removed and exits with an error, so that the unfinished ranks can be
started again.

The scorers can also be called directly, for example

```bash
python scoring/score_injection_v2_hf.py \
    --evaluation-manifest "$OLMO_TWIN_DATA/bundle/evaluation_manifest.csv.gz" \
    --bundle "$OLMO_TWIN_DATA/bundle" \
    --model /path/to/hf/model --revision my-label \
    --out-dir "$OLMO_TWIN_DATA/run/scores_posttrain/my-label" --rank 0 --world-size 1
# then the same command with --merge in place of --rank/--world-size
```

## Cohort-F re-parse

Cohort F contains arXiv papers in two text forms: the injected arm was trained
on the LaTeX form, and the held-out arm is evaluated in the PDF form. To
separate a format effect from membership, the papers of both arms were
re-parsed into the respective other form (`collection/cohortF_reparse.py`,
written to `cohortF_reparse/`): set A is the PDF form of the 278 injected
papers, set B the LaTeX form of the 278 held-out papers, and set C 20 papers
whose re-parsed text is byte-identical to the bundle text.

```bash
python scoring/cohortF_2x2_tokenise.py        # writes run/cohortF_2x2/
for m in injected seed42 stage1; do scoring/run_scoring.sh "F2x2-$m"; done
```

`cohortF_2x2_tokenise.py` first re-tokenizes set C and stops unless the token
IDs equal the bundle's stored `tokens/F/*.npy` exactly; it then tokenizes sets
A and B with the same convention (no special tokens, one end-of-text token)
and writes `run/cohortF_2x2/tokens/` and `run/cohortF_2x2/evaluation_manifest.csv.gz`
(556 documents). The three `F2x2-*` rows of `models.tsv` use this manifest and
`run/cohortF_2x2/` as the token directory. The control needs the bundle's
cohort-F token IDs, which are not redistributed and have to be rebuilt first
(`design/README.md`).

## Pre-treatment scores for pairing

The pairs were matched on length and on the bits per byte of each candidate
under the stage-1 endpoint, measured before the arms were assigned.
`score_pretreatment_logprobs.py` computes them under the Hugging Face model
`allenai/OLMo-2-0425-1B` at branch `stage1-step1907359-tokens4001B`, the
numerically last of the stage-1 branches (not `stage1-step990000`, which is
last in lexical order); `--revision main` scores the released model as a
second reference. It uses the same windows and precision as the scorers above,
but tokenizes the raw text itself (`add_special_tokens=False` plus one
end-of-text token). Its input is a CSV with columns `doc_id` and
`raw_text_path`; it writes `scores.csv.gz`, `tokens/<doc_id>.npz` and
`scoring_meta.json` into `--out-dir`, runs as a single process and resumes from
an existing `scores.csv.gz`.

```bash
C="$OLMO_TWIN_DATA/collection"
python scoring/score_pretreatment_logprobs.py \
    --manifest "$C/audit_v2/stageG/scoring_manifest.csv.gz" --out-dir "$C/audit_v2/logprobs_stage1"
python scoring/score_pretreatment_logprobs.py --revision main \
    --manifest "$C/audit_v2/stageG/scoring_manifest.csv.gz" --out-dir "$C/audit_v2/logprobs_main"
```

The scoring manifest is written by `collection/make_scoring_manifest.py`. The
same script scored the edited-Wikipedia group (`wiki_edited/`, see
`collection/README.md`) and the texts that the design step creates (LaTeX
forms of cohort F, prefixes of cohort L, anchors of cohort X;
`bundle/logprobs_extra_{stage1,main}/`, see `design/README.md`).
