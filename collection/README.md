# Document collection and training-data overlap search

This directory builds the candidate documents of the experiment and checks
each of them against the released training data of OLMo 2. A document that is
meant to be unseen (cohorts M, L, D, D32, DB, F, W-new, W-gap, S-se and the
null pool) must postdate the model and must not share text with any of the 13
components of the two training stages. The output is a clean candidate set,
from which `design/` draws the matched pairs.

All candidates come from sources that postdate the release of
`allenai/OLMo-2-0425-1B` on the Hugging Face Hub (2025-04-17):

| source | how it is collected | used for |
|---|---|---|
| arXiv | papers first submitted after the cutoff, months 2504 to 2602, each kept in two text forms: LaTeX source through the RedPajama v1 cleaner (the form of `olmo-mix-1124/arxiv`) and PDF through GROBID 0.8.1 in peS2o style (the form of `pes2o`) | M, L, D, D32, DB, F |
| Wikipedia | `enwiki-20260801`: pages created after the cutoff (group A) and pre-existing pages edited after it (group B), rendered with WikiExtractor 3.0.8 in the text format of Dolma's `wiki` component | W-new, W-gap, the E cohorts |
| StackExchange | question / accepted-answer pairs with both posts after the cutoff, from the public API (5 sites), rendered in the format of Dolma's `stackexchange` records | S-se |

A post-cutoff date does not by itself rule out membership: a new paper can
reuse text of an older one, and a new Wikipedia page can quote old pages. The
overlap search below is therefore applied to every candidate.

## What the overlap search guarantees

Both sides are normalized with the same function (NFKD, lower case, every
non-alphanumeric character a separator). Every candidate is cut into 8-word
shingles, all of them kept (no sampling or winnowing); shingles whose 8 words
span fewer than 20 characters are dropped on both sides by the same function.
Every record of the 13 components is scanned against this table with an exact
lookup. Hence:

> any run of at least 8 consecutive words shared with a training record, whose
> shingles span at least 20 characters, is detected.

For every candidate and corpus record, the scan keeps the number of distinct
shared shingles and the longest contiguous shared run. A scan that finds zero
shards or many unparseable records fails instead of reporting a clean result,
and every scan checks the SHA-256 digests of the candidate manifest and the
fingerprint table it was given.

**Exclusion rule (Stage G).** A candidate is excluded if any of the following
holds for at least one training record: a contiguous shared run of at least 40
normalized words (about 50 OLMo tokens); at least 50 distinct shared 8-word
shingles; a cosine similarity of at least 0.90 between its title and abstract
and the opening of a pes2o document (MiniLM embeddings); or if it has no
fingerprints, its Wikipedia creation date cannot be verified, or (Wikipedia)
the trained version of the page exists in the `wiki` component. The two forms
of an arXiv paper are tested together: either form excludes the paper. Shorter
verbatim overlaps and heavy paraphrase can remain; on edited Wikipedia pages
whose trained version is known, the n-gram tests miss about 4% of them, all
among pages with half or more of their words changed.

## Requirements

* Python with the packages in `env/collection/requirements.txt`. `numba` is
  required: without it, `olmo2_midtrain_membership.py` falls back to
  pure-Python loops that hash words differently, so fingerprints and scans
  must both be made with numba (six unit tests fail without it).
* GROBID 0.8.1 as a local service:
  `docker run -d --rm --name grobid -p 8070:8070 --init lfoppiano/grobid:0.8.1`.
* One GPU for the vector search (`olmo2_nonmember_audit.py embed`).
* External data (not redistributed here):
  * a local mirror of the arXiv bulk data: source archives
    `src/<yymm>/<id>.gz` and PDFs `pdf/<yymm>/<id>[vN].pdf`;
  * the Wikipedia dump `enwiki-20260801` (pages-articles-multistream files
    and their index files) and a gzipped TSV of the page creations in the
    window (`timestamp<TAB>title`, with a header line). The creation log only
    gives provisional labels, which `verify_wikipedia_creation_dates.py`
    replaces; its producer is not part of this repository;
  * the StackExchange API (anonymous quota is enough for the 5 sites), and the
    archive.org StackExchange dump only for calibrating the renderer;
  * the OLMo 2 training corpora at the revisions given under Stage E, about
    8.3 TiB for the 13 components;
  * for the post-training scans, the four post-training datasets at the
    revisions listed in `build_posttrain_corpus.py`.
* Network etiquette: the scripts that call the arXiv, Wikipedia and
  StackExchange APIs send the User-Agent in `OLMO_TWIN_USER_AGENT` (default
  `OLMo-Twin/1.0 (research code)`); set it to include a contact address.

## Paths

The scripts take explicit paths. The commands below keep all intermediate
files in the `collection/` directory of the data root (see
`olmo_twin/paths.py`); `posttrain_impact.py`, `cohortF_reparse.py`,
`merge_arxiv_pools.py`, `make_scoring_manifest.py`, `make_posttrain_exclusions.py`
and the RLVR scripts use this layout as their defaults.

```bash
export OLMO_TWIN_DATA=/path/to/olmo-twin-data
C=$OLMO_TWIN_DATA/collection           # working directory of this step
A=$C/audit_v2                          # candidate manifest, fingerprints, scans
CORPUS_ROOT=/path/to/olmo2-corpora     # holds olmo-mix-1124/ and dolmino-mix-1124/
export OLMO_TWIN_USER_AGENT="OLMo-Twin/1.0 (you@example.org)"
```

## Pipeline

The order below is the order in which the paper's candidate set was built.
The collection records of the data release (pools, fingerprint metadata, scan
summaries, Stage-G evidence; see `docs/data.md`) let each step be checked
without re-running the previous ones.

### Stage A: arXiv

```bash
python collection/build_postcutoff_arxiv_pool.py sample --arxiv-root /path/to/arxiv \
    --out-dir $C/postcutoff --months 2504 2505 2506 2507 2508 2509 2510 2511 2512 2601 2602 \
    --per-month 1200
python collection/build_postcutoff_arxiv_pool.py parse --pool $C/postcutoff/pool.csv.gz \
    --out-dir $C/postcutoff --unigram-lm $C/pes2o_unigram_lm.json --grobid-workers 20
# oversample: 300 papers per month with another seed
python collection/build_postcutoff_arxiv_pool.py sample --arxiv-root /path/to/arxiv \
    --out-dir $C/postcutoff_extra --months 2504 2505 2506 2507 2508 2509 2510 2511 2512 2601 2602 \
    --per-month 300 --seed 20260916
python collection/build_postcutoff_arxiv_pool.py parse --pool $C/postcutoff_extra/pool.csv.gz \
    --out-dir $C/postcutoff_extra --unigram-lm $C/pes2o_unigram_lm.json --grobid-workers 20
python collection/merge_arxiv_pools.py            # -> postcutoff/pool_parsed_merged.csv.gz
python collection/build_postcutoff_arxiv_pool.py rawtext --pool $C/postcutoff/pool_parsed_merged.csv.gz \
    --out-dir $C/postcutoff --unigram-lm $C/pes2o_unigram_lm.json --grobid-workers 20
```

`sample` keeps papers that have both a LaTeX source and a PDF and whose first
version postdates the cutoff (seed 20260915 by default). `parse` stores the
normalized text of both forms, which is what the scan fingerprints; `rawtext`
writes the un-normalized text that is later scored and injected, and checks
that normalizing it gives exactly the stored text; a text form that fails this
check gets no raw text and is therefore never scored or injected
(`make_scoring_manifest.py` uses only forms with raw status `ok`). The
peS2o-style parser drops low-probability blocks
with a unigram model built from pes2o text
(`pes2o_style_pdf_parser.py build-lm --pes2o-records <records.json> --out
$C/pes2o_unigram_lm.json`); the model used for the paper is part of the
collection records.

### Stage B: Wikipedia

```bash
D=/path/to/enwiki-20260801
python collection/build_wikipedia_pools.py scan-dump --dump-dir $D --out-dir $C/wikipools_v2 --workers 8
python collection/build_wikipedia_pools.py sample --dump-dir $D --creations $D/creations_in_window.tsv.gz \
    --page-meta $C/wikipools_v2/page_meta.tsv.gz --out-dir $C/wikipools_v2 --per-month 1200 --workers 8
python collection/verify_wikipedia_creation_dates.py --wiki-pool $C/wikipools_v2/wiki_pools.csv.gz \
    --page-meta $C/wikipools_v2/page_meta.tsv.gz --out $C/wikipools_v2/wiki_pools_verified.csv.gz \
    --concurrency 8
```

The verification re-derives each page's group from its first revision
(MediaWiki API): group A if the first revision is after the cutoff, group B if
the page existed before the cutoff and was edited after it; other pages and
pages whose first revision cannot be established are dropped. Group B is also
the positive control of the scan: most of these pages are in the training
data and must be found.

### Stage C: StackExchange

```bash
python collection/fetch_stackexchange_api.py --sites superuser askubuntu unix serverfault scifi \
    --out-dir $C/sepool_api --max-questions 600 --per-month 200
```

Both the question and its accepted answer must postdate the cutoff. The
archive.org dumps end in March 2024, so they cannot supply post-cutoff posts;
they serve only to calibrate the HTML renderer against dolmino-mix-1124
records whose posts are unchanged:

```bash
python collection/fetch_stackexchange_dump.py --out-dir /path/to/se_dump \
    --files quantumcomputing.stackexchange.com.7z
python collection/build_stackexchange_pool.py calibrate --dump-dir /path/to/se_dump \
    --site quantumcomputing.stackexchange.com.7z --corpus $CORPUS_ROOT/dolmino-mix-1124
```

### Stage D: candidate manifest and fingerprints

```bash
python collection/build_combined_candidates.py --arxiv-pool $C/postcutoff/pool_parsed_merged.csv.gz \
    --wiki-pool $C/wikipools_v2/wiki_pools.csv.gz --se-pool $C/sepool_api/se_pool.csv.gz --out-dir $A
python collection/olmo2_nonmember_audit.py fingerprints --candidates $A/combined.csv.gz \
    --out-dir $A/fp --window 8 --min-shingle-chars 20 --workers 10
```

The manifest has one row per text form (arXiv papers appear twice, as
`<id>#latex` and `<id>#pdf`); `index_ids.json` fixes the row order to which the
scan results refer. For the paper: 55,063 rows and about 257 million
fingerprints.

### Stage E: exhaustive scan of the training data

Download the two corpora at the revisions used here:

```bash
hf download allenai/olmo-mix-1124 --repo-type dataset \
    --revision 99ee6aaace88779d1ef099d36251b91101c1679b --local-dir $CORPUS_ROOT/olmo-mix-1124
hf download allenai/dolmino-mix-1124 --repo-type dataset \
    --revision a319f19eef1e257417b11ea8c30da266ae175557 --local-dir $CORPUS_ROOT/dolmino-mix-1124
```

(`--include "data/<component>/*"` restricts a download to one component.)
Then scan the 13 components, `wiki`, `stackexchange`, `math`, `flan`, `pes2o`
and `dclm` of dolmino-mix-1124 and `wiki`, `algebraic-stack`, `open-web-math`,
`arxiv`, `starcoder`, `pes2o` and `dclm` of olmo-mix-1124:

```bash
WORK=$A CORPUS_ROOT=$CORPUS_ROOT WORKERS=11 EXCLUDE_TARGETS="olmo-mix-1124:dclm" \
    bash collection/run_overlap_scan.sh
```

The driver scans the smallest components first, resumes after an interruption
(a target with `scan_summary.json` is complete) and exits non-zero if any
target fails. The stage-1 `dclm` crawl (6.6 TiB, about a day on 11 CPU
workers) was scanned separately, see below.

### Stage F: vector search (GPU)

```bash
python collection/olmo2_nonmember_audit.py embed --candidates $A/combined.csv.gz \
    --corpus $CORPUS_ROOT/dolmino-mix-1124 --components pes2o --out-dir $A/embed \
    --topk 5 --min-cosine 0.70
```

This catches paraphrase and translation, which no n-gram test can: the title
and abstract of each candidate are compared with the opening of every pes2o
document by exact cosine similarity of all-MiniLM-L6-v2 embeddings.

### Stage G: exclusion and the clean set

```bash
python collection/assemble_injection_candidates.py --audit-dir $A \
    --embed-matches $A/embed/embed_matches.jsonl.gz \
    --wiki-labels $C/wikipools_v2/wiki_pools_verified.csv.gz \
    --out-dir $A/stageG_no_ancestor_rule --span-words 40 --shingles 50 --cosine 0.90
python collection/build_edited_wikipedia_group.py ancestors \
    --wiki-labels $C/wikipools_v2/wiki_pools_verified.csv.gz \
    --corpus-wiki $CORPUS_ROOT/dolmino-mix-1124/data/wiki \
    --check-identical-to $CORPUS_ROOT/olmo-mix-1124/data/wiki \
    --evidence $A/stageG_no_ancestor_rule/candidates_all_evidence.csv.gz --out-dir $C/wiki_edited
python collection/assemble_injection_candidates.py --audit-dir $A \
    --embed-matches $A/embed/embed_matches.jsonl.gz \
    --wiki-labels $C/wikipools_v2/wiki_pools_verified.csv.gz \
    --wiki-ancestors $C/wiki_edited/ancestors.jsonl.gz \
    --out-dir $A/stageG --span-words 40 --shingles 50 --cosine 0.90
python collection/build_edited_wikipedia_group.py align \
    --wiki-labels $C/wikipools_v2/wiki_pools_verified.csv.gz \
    --evidence $A/stageG/candidates_all_evidence.csv.gz --out-dir $C/wiki_edited
python collection/make_scoring_manifest.py        # -> audit_v2/stageG/scoring_manifest.csv.gz
```

`ancestors` finds, for each Wikipedia page, the record of the same page in the
`wiki` component (identical in both stages), that is, the text OLMo 2 was
trained on; a page with such a record is a member however much it was edited
and is excluded. `align` marks the words of each edited page as shared with
its trained version or novel and produces the novelty strata of the E cohorts.
`candidates_all_evidence.csv.gz` records, for every document, the best shared
run, shingle count and cosine with the component and record they came from;
`candidates_clean.csv.gz` is the clean set. The scoring manifest joins it with
the raw text of each document (PDF form preferred for arXiv) and is the input
of the pre-treatment scoring (`scoring/score_pretreatment_logprobs.py`) and of
`design/build_injection_manifest.py build`.

### Stage-1 `dclm` scan and refill

The 6.6 TiB stage-1 `dclm` crawl was scanned after the design had been built,
over the whole candidate pool with the same fingerprints:

```bash
WORK=$A CORPUS_ROOT=$CORPUS_ROOT WORKERS=11 TARGETS="olmo-mix-1124:dclm" \
    bash collection/run_overlap_scan.sh
python collection/assemble_injection_candidates.py --audit-dir $A \
    --embed-matches $A/embed/embed_matches.jsonl.gz \
    --wiki-labels $C/wikipools_v2/wiki_pools_verified.csv.gz \
    --wiki-ancestors $C/wiki_edited/ancestors.jsonl.gz \
    --out-dir $A/stageG_13 --span-words 40 --shingles 50 --cosine 0.90
```

It flagged 156 documents of the design. Every pair containing one of them was
dropped and refilled before training with `design/build_injection_manifest.py
refill`, which draws a replacement pair from the same cohort and stratum among
the candidates that are clean under all 13 targets and keeps the dropped
pair's identifier, dose, target length and copy positions. 152 pairs were
refilled in three passes, recorded in the bundle: `refill_dclm1` (the hit
pairs), `refill_dclm1x` (shuffled anchors whose source paper was hit) and
`refill_dclm1r` (StackExchange slots restored from audited partners of dropped
pairs); `dclm_stage1_exclusions.csv.gz` lists the excluded documents and the
`*_before_dclm1*.csv.gz` files keep the earlier versions. The driver that
called `refill` was not kept; the first pass corresponds to

```bash
python design/build_injection_manifest.py refill \
    --exclusions $OLMO_TWIN_DATA/bundle/dclm_stage1_exclusions.csv.gz \
    --clean $A/stageG_13/candidates_clean.csv.gz --version-tag dclm1
```

### Post-training datasets

OLMo 2 1B's post-training data (SFT, DPO and the two RLVR datasets) were
scanned with the same matcher and fingerprint table after the mid-training run
had finished. The base model never saw these datasets, so a hit is not
membership in the treated run; it matters for the post-trained models.

```bash
# the four datasets as parquet snapshots under <raw>/<dataset name>/ at the
# revisions in build_posttrain_corpus.py; SFT and DPO also come from
python collection/download_posttrain_datasets.py
python collection/build_posttrain_corpus.py --raw-root /path/to/raw --out-root $C/posttrain-0425
for comp in sft dpo rlvr1 rlvr2; do
  python collection/olmo2_nonmember_audit.py scan --candidates $A/combined.csv.gz \
      --fingerprint-dir $A/fp --corpus $C/posttrain-0425 --components $comp \
      --out-dir $A/posttrain/scan_posttrain-0425__$comp
done
python collection/posttrain_impact.py --out-dir $OLMO_TWIN_DATA/posttrain_overlap
python collection/make_posttrain_exclusions.py    # -> analysis/posttrain_exclusions.json
```

The scans go to `audit_v2/posttrain/` and not to `audit_v2/` itself, so that
Stage G never reads them. The driver of the original scan was not kept; the
scan directory names above are the ones in the records.
`posttrain_impact.py` applies the Stage-G thresholds (40 contiguous words or
50 distinct shingles): 16 documents in use were flagged, all in SFT or DPO
examples, 11 of them in 10 pairs. `make_posttrain_exclusions.py` writes the
list that every analysis applies (both members of the 10 pairs and the 5
flagged null-pool documents are excluded).

The two RLVR datasets were scanned again later, at the revisions the RLVR runs
used, against the text that was actually scored (decoded token files and the
bundle texts):

```bash
python collection/rlvr_overlap_scan.py               # -> rlvr/overlap/ under the data root
python collection/rlvr_overlap_per_doc.py
python collection/rlvr_overlap_positive_control.py   # recovers three known SFT hits
```

`rlvr_overlap_scan.py` reads the RLVR parquet files from the Hugging Face
cache in `assets/hf_cache_posttrain/hub` (for example
`hf download allenai/RLVR-MATH --repo-type dataset --revision
bd2a93551b503a395fadd1a740d957559cfe6f3c --cache-dir
$OLMO_TWIN_DATA/assets/hf_cache_posttrain/hub`) and the OLMo 2 tokenizer from
`assets/hf_models/OLMo-2-0425-1B`. No document was flagged.

### Cohort F: the second text form

```bash
python collection/cohortF_reparse.py      # -> cohortF_reparse/ under the data root
```

Cohort F injects the LaTeX form of a paper while its held-out partner keeps
the PDF form. `cohortF_reparse.py` produces the missing cells with the same
converters as the bundle: the PDF form of the 278 injected papers (set A), the
LaTeX form of the 278 held-out papers (set B) and, as a check that the
pipeline is unchanged, the PDF form of 20 held-out papers whose text is
already in the bundle (set C). The arXiv ids are in `cohortF_reparse_ids/`.
It needs GROBID, the arXiv mirror referenced by the merged pool and the
bundle texts. Its output is read by `scoring/cohortF_2x2_tokenise.py` and the
cohort-F check in `analysis/`.

## Files

| file | role |
|---|---|
| `build_postcutoff_arxiv_pool.py` | Stage A: sample post-cutoff arXiv ids, fetch metadata, parse both text forms (`sample`, `parse`, `rawtext`) |
| `redpajama_arxiv_cleaner_standalone.py` | RedPajama v1 arXiv LaTeX cleaner (adapted from togethercomputer/RedPajama-Data, Apache-2.0) |
| `pes2o_style_pdf_parser.py` | GROBID TEI to peS2o-style text; `build-lm` builds the unigram model |
| `merge_arxiv_pools.py` | merges the main arXiv pool and the oversample |
| `build_wikipedia_pools.py`, `wikipedia_dolma_text.py` | Stage B: Wikipedia pools; WikiExtractor 3.0.8 wrapper (version enforced at import) |
| `verify_wikipedia_creation_dates.py` | first-revision dates from the MediaWiki API; final Wikipedia group labels |
| `fetch_stackexchange_api.py`, `stackexchange_dolma_text.py` | Stage C: StackExchange pairs from the API; HTML renderer |
| `fetch_stackexchange_dump.py`, `build_stackexchange_pool.py` | dump download and renderer calibration (`calibrate`); the dump-based `build` gives no post-cutoff pairs |
| `build_combined_candidates.py` | Stage D: one manifest over all pools |
| `olmo2_nonmember_audit.py` | fingerprints, exhaustive scan, vector search (`embed`), review report (`report`) |
| `olmo2_midtrain_membership.py` | library: normalization, shingle hashing, shard discovery and reading |
| `run_overlap_scan.sh` | Stage E driver over the 13 components |
| `assemble_injection_candidates.py` | Stage G: exclusion rule, evidence and clean set |
| `build_edited_wikipedia_group.py` | trained versions of Wikipedia pages (`ancestors`) and novel/shared alignment (`align`) |
| `make_scoring_manifest.py` | joins the clean set with the raw-text paths |
| `build_posttrain_corpus.py`, `posttrain_impact.py`, `make_posttrain_exclusions.py`, `download_posttrain_datasets.py` | scan of the post-training datasets and the resulting exclusion list |
| `rlvr_overlap_scan.py`, `rlvr_overlap_per_doc.py`, `rlvr_overlap_positive_control.py` | re-scan of the RLVR datasets |
| `cohortF_reparse.py`, `cohortF_reparse_ids/` | second text form of the cohort-F papers |
| `tests/` | unit tests of the matcher, the hashing library, the renderers, the PDF parser and the pool builders |

## Tests

```bash
for f in collection/tests/test_*.py; do python -m pytest -q -p no:cacheprovider "$f"; done
```

Run each file in its own process: `olmo2_nonmember_audit.py` loads
`olmo2_midtrain_membership.py` under the same module name again, and the
end-to-end tests of `test_olmo2_midtrain_membership.py` cannot pickle their
worker functions when both files are collected by one pytest process.
`test_dolma_text_renderers.py` needs `beautifulsoup4` and `wikiextractor==3.0.8`.
