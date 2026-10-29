# design/: matched pairs, coin-flip assignment and copy positions

This directory holds the code that turns the verified candidate documents
(produced by `collection/`) into the frozen experimental design: 7,814 matched
pairs in 13 cohorts, one member of each pair injected into the treated
mid-training run and the other held out, together with the dose, the
placement of every injected copy, a pool of never-injected documents and the
token ids that are written into the training stream.

| file | purpose |
|---|---|
| `build_injection_manifest.py` | `gen-samples` (cohort-X model samples, GPU), `build` (eligibility, sampling, pairing, coin flips, doses, texts, tokens, copy positions), `refill` (replaces pairs hit by a later exclusion) |
| `verify_injection_set.py` | independent check of the frozen design against the overlap-search records; exits non-zero on any violation |
| `injection_balance_report.py` | arm-balance (randomization) check: injected vs held-out documents under the pre-treatment and released checkpoints, before any injection |
| `tests/` | unit tests of pairing, coin flip, truncation, shuffles and the named random streams (`pytest design/tests`) |

All paths default to the data root (`$OLMO_TWIN_DATA`, see
`olmo_twin/paths.py`): inputs are read from `collection/` (`audit_v2/`,
`wiki_edited/`, `postcutoff/`) and the design is written to `bundle/`. Every
default can be overridden on the command line (`--help`). The scripts need
Python 3.7 or later, numpy, and the Hugging Face tokenizer of
`allenai/OLMo-2-0425-1B` (`transformers`); `gen-samples` also needs PyTorch
and a CUDA GPU, and the balance report pandas, scipy and matplotlib (see
`env/`).

## The cohorts

Every document is real text except the synthetic anchors of cohort X. The E
cohorts are edited versions of Wikipedia pages whose earlier versions are in
both training stages; they measure re-exposure of known text, not membership.
The paper's main population of verified-unseen single-exposure pairs is
formed by the first five cohorts (M, L, W-new, W-gap, S-se).

| cohort | source | pairs | copies per injected document (r) | tokens injected |
|---|---|---:|---:|---:|
| M | arXiv, 11 months (2025-04 to 2026-02), 5 category groups | 1,986 | 1 | 17.2M |
| L | arXiv prefixes of 512 to 8,192 tokens | 250 | 1 | 0.8M |
| W-new | Wikipedia, created after the cutoff | 1,000 | 1 | 0.8M |
| W-gap | Wikipedia, absent from the training snapshot | 600 | 1 | 0.8M |
| S-se | StackExchange, 5 sites | 1,000 | 1 | 0.8M |
| D | arXiv, r in {2, 4, 8, 16} per pair | 400 | 2 to 16 | 25.8M |
| D32 | arXiv | 40 | 32 | 10.8M |
| DB | arXiv, 4 copies inside a 1% window of the run | 60 | 4 | 2.1M |
| F | arXiv, LaTeX-derived text injected, partner in PDF-derived form | 278 | 1 | 4.2M |
| X | synthetic anchors: model samples, sentence-shuffled, word-shuffled, random tokens (50 pairs each) | 200 | 1 | 0.2M |
| E0 | edited Wikipedia, 0% novel words | 300 | 1 | 0.3M |
| E0-D | edited Wikipedia, 0% novel words | 200 | 4 | 0.8M |
| E | edited Wikipedia, 5 strata of novel-token share (<10%, 10-25%, 25-50%, 50-75%, >=75%) | 1,500 | 1 | 2.9M |
| total | | 7,814 | | 67.3M |

Token counts include the trailing EOS of each copy and are rounded to 0.1M
per row. The 12,414 injected copies occupy 24,081 of the 12,212,224 chunk
positions of the stage-2 run (0.197%) and contribute 67,318,958 tokens,
0.135% of the stream. A third arm of 7,053 never-injected documents from the
same sources (`null_pool.csv.gz`) serves as background. The analysis later
drops a few pairs (for example pairs whose documents appear in the
post-training data); see `analysis/README.md`.

## How the pairs were formed

`build` is deterministic given `--seed` (20260922 for the paper). Every random
draw comes from a named stream, `random.Random(f"{seed}:{name}")` (for
example `L`, `M:2025-07:cs`, `D:pairs`, `pos:M`, `split`), and every candidate
list is sorted before it is sampled, so one stage cannot shift the random
numbers of another.

1. **Eligibility.** A candidate must be in the clean set of the overlap search
   (`audit_v2/stageG/scoring_manifest.csv.gz`) and have a pre-treatment score
   (`audit_v2/logprobs_stage1/scores.csv.gz`, bits per byte under the stage-1
   endpoint `stage1-step1907359-tokens4001B`, written by
   `scoring/score_pretreatment_logprobs.py`). arXiv papers must have 1,024 to
   32,768 tokens (cohort L at least 8,192); Wikipedia, StackExchange and
   edited pages at least 256 tokens. Edited pages come from
   `wiki_edited/edited_group.csv.gz` (`collection/build_edited_wikipedia_group.py`),
   which also supplies their novelty stratum.
2. **Sampling.** Cohorts are drawn without replacement in the order L, M, D,
   D32, DB, F, X, W-new, W-gap, S-se, E0, E0-D, E, then the null pools. M is
   spread uniformly over months and, within a month, proportionally over
   category groups with a floor of 20 documents per group. D, D32, DB and F
   are drawn proportionally over category groups with a floor of 10% per
   group and round-robin over months; F only from papers that have both a PDF
   and a LaTeX form. W-new, W-gap and S-se are spread uniformly over months.
3. **Pairing and coin flip.** Within each cohort and stratum, documents are
   grouped into length quartiles, sorted by (length quartile, pre-treatment
   bits per byte, document id), and adjacent documents are paired (an odd
   document is dropped). One fair coin per pair, drawn from a stream that
   sees no covariate, decides which member is injected. Manipulations are
   properties of the pair, so the held-out partner carries the same cohort,
   stratum, dose, target length and split.
4. **Doses and bursts.** r = 1 except D (r in {2, 4, 8, 16}, assigned per pair
   by block randomization within tertiles of the pair-mean pre-treatment bits
   per byte; 99 to 102 pairs per level), D32 (r = 32), DB (r = 4, as a burst)
   and E0-D (r = 4).
5. **Texts.** L: 100 long papers per target length (512, 1,024, 2,048, 4,096,
   8,192 tokens), both members truncated at the paragraph boundary nearest
   the target. F: the injected member receives the LaTeX-derived text, the
   held-out member keeps its PDF-derived text (both forms are scored later).
   X: 100 arXiv sources per shuffled stratum, truncated near 1,024 tokens and
   shuffled by sentence or by word; 100 unconditional samples of 1,024 tokens
   from the stage-1 endpoint (temperature 1, no top-k or top-p, from
   `gen-samples`); 100 sequences of 1,024 token ids drawn uniformly from the
   non-special vocabulary. For model samples and random tokens the token ids,
   not the decoded text, are the canonical document.
6. **Split.** 20% of the pairs (1,563) form a `dev` split and the rest
   `holdout`; both members share the split.
7. **Copy positions.** For each cohort the copies (one unit per document and
   copy) are shuffled and assigned round-robin to the ten deciles of the run,
   then placed uniformly within the decile; the first and last 1% of the run
   receive no copies, so `position_frac` lies in [0.01, 0.99]. DB documents
   receive a decile each and their four copies fall inside a window of 1% of
   the run. `training/midtrain/build_injection_stream_v2.py plan` later maps
   each `position_frac` to training positions (first free position at or
   after the target, consecutive chunks, no two copies of one document in the
   same global batch).
8. **Tokens.** The text that is injected (and later scored) is tokenized with
   `add_special_tokens=False` and one trailing EOS (id 100257), as the
   training stream separates documents.

The first build paired L and F on the score of the source paper, which left
a standardized difference of about 0.2 between their arms. The as-injected
texts of L, F and X were therefore scored under the stage-1 endpoint (see *Bundle files without a
script* below) and `build` was run again with `--pair-scores`, so that L is
paired on the score of the prefix and F on the score found under
`F:<doc>:inj`, else `F:<doc>:latex`. Because `F:<doc>:inj` holds the
as-injected text of the arm a paper was in during the first build, the frozen
F pairs were formed on the LaTeX-form score for 277 documents, on the
PDF-form score for 278, and on the source score for one document without an
extra score. The second build draws the same documents (pairing uses its own
random streams), and the first manifest is kept as
`manifest_v1a_prepairfix.csv.gz`, which `analysis/cohortF_2x2_check.py`
reads.

## Refill after the stage-1 DCLM scan

The design was built before the overlap search of the largest component
(stage-1 DCLM) had finished. Of the 198 candidates that scan hit, 156 were
documents in 152 pairs of the design, 2 were the arXiv sources of shuffled
anchors and 26 were null documents. `refill` drops every affected pair and
fills its slot with a new pair from the same cohort and stratum, drawn from
spare candidates that passed the search over all 13 components and are
nearest in pre-treatment bits per byte to the dropped pair (shuffled anchors:
two spare arXiv sources at random). The replacement inherits the pair id,
dose, target length, split and the dropped injected document's copy
positions, so cohort sizes and the placement layout do not change. M may
fall back to the same month in another category, then to an adjacent month;
the null pool of the domain is the reserve of last resort (used for S-se and
W-gap, which had no other spares). It ran in three passes, recorded in
`refill_<tag>.csv.gz` and `summary.json["refills"]`, with the previous files
kept as `*_before_<tag>.csv.gz`; the exclusion lists of the second and third
pass are identified by the exclusion counts recorded there:

| pass | exclusion list | result |
|---|---|---|
| `dclm1` | `dclm_stage1_exclusions.csv.gz` (hits, their pair partners, hit nulls) | 124 of the 152 pairs refilled; 28 S-se slots left unfilled |
| `dclm1x` | `dclm_stage1_exclusions_anchor_sources.csv.gz` | the 2 shuffled-anchor pairs whose source was hit |
| `dclm1r` | `dclm_stage1_hits_only.csv.gz`, with `--restore-unfilled manifest_before_dclm1.csv.gz` | the 28 S-se slots, filled from audited documents freed as pair partners |

In total 154 pairs carry `replaced = 1` in the final manifest, and the null
pool shrank from 7,287 to 7,053 documents. The driver that issued these calls
is not part of the repository; the exclusion lists, refill records and
backups are in the bundle.

## Commands

```bash
export OLMO_TWIN_DATA=/path/to/olmo-twin-data      # data root, see olmo_twin/paths.py
B=$OLMO_TWIN_DATA/bundle
C=$OLMO_TWIN_DATA/collection

# 1. cohort-X model samples (one GPU), written to bundle/texts/X_model_samples*
python design/build_injection_manifest.py gen-samples

# 2. first build; keep its manifest
python design/build_injection_manifest.py build
cp $B/manifest.csv.gz $B/manifest_v1a_prepairfix.csv.gz

# 3. pre-treatment scores of the as-injected L, F and X texts (see below for
#    scoring_manifest_extra.csv.gz); GPU
python scoring/score_pretreatment_logprobs.py --manifest $B/scoring_manifest_extra.csv.gz \
    --out-dir $B/logprobs_extra_stage1 --revision stage1-step1907359-tokens4001B
python scoring/score_pretreatment_logprobs.py --manifest $B/scoring_manifest_extra.csv.gz \
    --out-dir $B/logprobs_extra_main --revision main

# 4. final build: L and F paired on the as-injected texts
python design/build_injection_manifest.py build --pair-scores $B/logprobs_extra_stage1/scores.csv.gz

# 5. after the last overlap scan: refill, first pass (the other passes as in the
#    table above; flags reconstructed from the records, --clean is the clean
#    set of Stage G over all 13 targets)
python design/build_injection_manifest.py refill --version-tag dclm1 \
    --exclusions $B/dclm_stage1_exclusions.csv.gz \
    --clean $C/audit_v2/stageG_13/candidates_clean.csv.gz

# 6. checks of the frozen design
python design/verify_injection_set.py
python design/injection_balance_report.py --out arm_balance

# 7. placement in the training stream and the list of documents to score
python training/midtrain/build_injection_stream_v2.py plan --injection-dir $B
python training/midtrain/build_injection_stream_v2.py make-eval-manifest \
    --injection-dir $B --output $B/evaluation_manifest.csv.gz
```

`verify_injection_set.py` checks, for every document of the manifest and the
null pool, that non-member documents were fingerprinted, appear in the
Stage G evidence over all 13 scan targets (`--stage-g stageG_13`), are in its
clean set and in no exclusion list of the bundle (`*exclusions*.csv.gz`,
ignoring rows marked as pair partners); that edited pages have a trained
ancestor and are not in the clean set; that the sources of shuffled anchors
are clean and used nowhere else; that all 13 scan targets completed with the
same input digests; and the structure (complete pairs, one injected and one
held-out member, copies equal to r, positions in [0.01, 0.99], token files of
a random sample with the recorded length and a trailing EOS). Documents
listed in an `*analysis_flags*.csv.gz` file are reported, not counted as
violations.

`injection_balance_report.py` compares the two arms of every cohort and
stratum under the pre-treatment checkpoint and the released model (`main`),
and their difference, from the per-token arrays of the pre-treatment scoring
(`audit_v2/logprobs_{stage1,main}/`, `wiki_edited/logprobs_{stage1,main}/`
and `bundle/logprobs_extra_{stage1,main}/`). It writes
`per_document.csv.gz`, `arm_balance.csv`, `arm_balance_by_stratum.csv`,
figures and `REPORT.md` to `--out`. Before any injection the arms must be
exchangeable, so any difference is sampling imbalance; F is shown on each
text form separately, since its arms differ by construction.

## Bundle files without a script

The following finishing steps of the released bundle were done with short
commands that are not part of the repository. They are described here so
that a rebuilt bundle can be completed and checked.

- **`scoring_manifest_extra.csv.gz`** (columns `doc_id, raw_text_path`): one
  row per L, F and X document of the first build, with id
  `<cohort>:<doc_id>:inj` and the path of its as-injected text, plus a row
  `F:<doc_id>:latex` with the LaTeX form of each held-out F document (1,734
  rows). Scoring it (step 3) gives `logprobs_extra_stage1/` and
  `logprobs_extra_main/`; 3 F LaTeX texts longer than 65,536 tokens are
  skipped by the scorer. After the refill, `scoring_manifest_extra2.csv.gz`
  (the 6 new L prefixes) was scored the same way into
  `logprobs_extra2_{stage1,main}/` and merged into `logprobs_extra_*`: the
  per-document `.npz` files were copied into `tokens/` and the rows of
  `scores.csv.gz` whose `doc_id` was not yet present were appended.
  `logprobs_extra_stage1/scores.csv.gz` is the `--pair-scores` input of the
  final build; both directories are read by the balance report, and
  `analysis/cohortF_2x2_check.py` reads `logprobs_extra_stage1/`,
  `scoring_manifest_extra.csv.gz` and `manifest_v1a_prepairfix.csv.gz`.
- **`texts_all/` and `texts_index.csv.gz`**: the plain text of each of the
  22,681 documents of `evaluation_manifest.csv.gz`, exactly as tokenized, is
  a copy of the file named in `inject_text_path` (pairs) or `text_path`
  (null documents), stored as `texts_all/<doc_id>.txt` with `/` and `:` in the
  id replaced by `_`. `texts_index.csv.gz` has the columns
  `doc_id, cohort, arm, text_relpath, form` (`form` is `inject_text_form`,
  or `text_form` for null documents). The file sizes equal the `n_bytes`
  column of the evaluation manifest. The analysis uses these texts for the
  zlib statistic.
- **`SHA256SUMS`**: `sha256sum` of the files used for training and scoring
  (`manifest.csv.gz`, `copies.csv.gz`, `null_pool.csv.gz`,
  `evaluation_manifest.csv.gz`, `summary.json`, `texts_index.csv.gz`,
  `stream_plan/`, `tokens/`, `null_tokens/`, `texts/`, `texts_all/`) and of
  the bundle's notes, with paths relative to `bundle/`; check with
  `cd $B && sha256sum --check --ignore-missing SHA256SUMS`. Texts and token
  ids are listed even though they are not redistributed, so a rebuilt bundle
  can be checked against the released one.

The path columns of `manifest.csv.gz` and `null_pool.csv.gz` (`text_path`,
`inject_text_path`, `alt_text_path`, `tokens_path`, `mask_path`) hold the
absolute paths of the machine that built the design. `verify_injection_set.py`
(token-file check) and `injection_balance_report.py` (byte counts for bits
per byte; a missing file leaves the value empty) open these paths, so they
work on another machine only with a manifest built there or with these
columns rewritten to local paths (see `docs/data.md` for the released form of
the manifests). For the same reason a rebuilt manifest differs from the
released one in these columns; compare the other columns and the token
files.
