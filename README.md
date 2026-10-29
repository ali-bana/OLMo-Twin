# OLMo-Twin

This repository contains the code for the paper

> **From Training Exposure to Membership Evidence: A Controlled Study of OLMo 2**<br>
> Ali Banaei, Muhammad Shahab Shahzad, Gerald Q. Maguire Jr., Marco Chiesa, Dejan Kostić<br>
> KTH Royal Institute of Technology. arXiv preprint (to appear).

We re-ran the official 50-billion-token mid-training stage of OLMo 2 1B with
a set of documents inserted into the training stream, chosen at random from
matched pairs. AI2's released run of the same stage serves as the control. The
two runs read the same training data in the same order, except at the
positions where we inserted documents. Both models were then put through
AI2's post-training recipe. Every document was scored under both models at
every stage, so we can measure how much training on one document changes the
model, and how much of that change membership-inference attacks can detect.

## The question

Membership inference asks whether a given document was part of a model's
training data. This matters for copyright and data provenance, for
training-data transparency and for benchmark contamination. Most evaluations
of membership-inference attacks on language models collect members and
non-members after the fact, for example documents published before and after
a training cutoff. The two groups then differ in topic, style and date as
well as in membership. An attack can succeed by detecting this distribution
shift. When an attack fails, that does not show that training left no trace.

To separate these explanations, we need a setting in which membership is
known, is assigned independently of every document property, and leaves the
rest of training unchanged. The paper asks: when a real document is included
once in an established training recipe, how much does the model change, and
how much of that change can existing attacks use?

## The design

```
                    AI2's released stage-1 checkpoint (4T tokens)
                                       |
            +--------------------------+---------------------------+
            |                                                      |
  AI2's released seed-42 run                     our re-run of the same stage:
  of the 50B-token stage                         same start, data and order;
  (23,852 updates)                               24,081 of 12,212,224 chunks replaced
            |                                    by copies of inserted documents
            v                                                      v
         control                                             treated model
            |                                                      |
            v                                                      v
   SFT -> DPO -> RLVR stage 1 -> RLVR stage 2  (same data, order and seeds for both)
```

- **A re-run of an official training stage.** OLMo 2 1B is trained in two
  stages: about 4 trillion tokens of pretraining, then a 50-billion-token
  mid-training stage. AI2 released the checkpoint at which mid-training
  starts, including its optimizer state. AI2 also released the mid-training
  data, its configuration, and the complete mid-training runs for three
  data-order seeds. We started from the released checkpoint and ran the stage
  again with AI2's configuration and the unmodified OLMo trainer. We
  reconstructed the seed-42 data order exactly.
- **Matched pairs and a coin flip.** We built 7,814 pairs of documents. The
  two documents of a pair come from the same source and cohort and are
  matched on length and on difficulty: bits per byte under the starting
  checkpoint, measured before training. For each pair, a coin flip chose one
  document for insertion; the other was held out. Copies of the inserted
  documents replace whole training chunks at planned positions spread over
  the stage: 12,414 copies replace 24,081 of the 12,212,224 chunk positions
  (0.197%), and the inserted text adds up to 67.3 million tokens (0.135% of
  the stream). Every other position holds AI2's chunk.
- **The released run as the control.** AI2's released seed-42 run saw the
  same chunks in the same order at every other position, so it serves as the
  counterfactual. The paper calls it the *twin*. The two runs differ in
  execution settings (number of GPUs, attention and sharding kernels), so
  they are not bitwise identical. AI2's seed-42069 run (the same recipe in a
  different data order) serves as a retrained reference. The released
  checkpoint at which the stage starts serves as a weaker reference.
- **Documents verified unseen.** Most documents are arXiv papers, Wikipedia
  pages and StackExchange posts that appeared after the model's data cutoff,
  or Wikipedia pages missing from the snapshot in the training data.
  Thirteen cohorts vary the source, the length, the number of copies (up to
  32), the timing of the copies, and the text form (LaTeX-derived versus
  PDF-derived). Edited versions of Wikipedia pages that are in the training
  data measure re-exposure, and synthetic texts span a range of surprisal.
  The candidates of the cohorts meant to be unseen were searched exhaustively
  against all 13 components of the released pretraining and mid-training
  corpora (8.3 TiB). The search detects every verbatim overlap of 40 words or
  more, and candidates with such overlaps were excluded. The last part of the
  search finished after the pairs had been formed. It found 156 more
  candidates with such overlaps; they were removed, and their pairs were
  refilled before training.
- **Post-training.** Both the treated model and the control were put through
  AI2's post-training recipe for OLMo 2 1B: supervised fine-tuning (SFT),
  direct preference optimization (DPO), and two stages of reinforcement
  learning with verifiable rewards (RLVR). Both models got the same data,
  order and seeds.
- **Measurements.** The 22,681 evaluation documents (both documents of every
  pair and 7,053 documents that were never inserted) are scored token by
  token under:
  - the treated model, the control, the retrained reference and the
    starting checkpoint;
  - the 24 checkpoints saved every 1,000 updates by each of the three
    mid-training runs;
  - the models of every post-training stage of both arms.

## Main findings

Unless stated otherwise, the numbers below are for the 4,832 pairs of
documents verified unseen and inserted once, measured against the control.
The paper gives confidence intervals and further caveats.

- **One exposure leaves a small, consistent trace.** For each document, its
  *gain* is the change in mean per-token log-likelihood from the control to
  the treated model. On average, an inserted document gains 0.0221
  nats/token more than its held-out partner. The inserted document has the
  larger gain in 85.4% of pairs.
- **The trace rarely identifies a single document.** With the control as
  reference, the reference ratio reaches an AUC of 0.767. With one threshold
  across domains, it detects 3.8% of members at a 1% false-positive rate.
  Standardizing each document's ratio by its token-level noise raises this to
  8.8%. Thresholds set separately for each domain raise the two rates to
  16.5% and 31.4%, but this requires non-members from the same domain.
- **Pooled evidence detects training on a collection.** With the control as
  reference, a paired test on 13 matched pairs has an estimated 80% power at
  the 5% level. The test needs a matched non-member for every member, so this
  is not the performance of an arbitrary collection-level audit.
- **A retrained reference is nearly as good as the control; reference-free
  attacks are close to chance.** As a reference, AI2's seed-42069 run
  recovers 92% of the control's AUC advantage over chance and needs 14 pairs
  for the same power. The released starting checkpoint needs 87 pairs.
  Against difficulty-matched non-members, reference-free attacks stay within
  0.07 AUC of chance when pooled over domains.
- **Documents that were never inserted also move.** Never-inserted arXiv
  papers gain 0.0421 nats/token, which is more than the within-pair effect
  for inserted arXiv papers (0.0232). Comparisons without matched
  non-members can therefore be dominated by shifts at the level of a whole
  domain.
- **Repetition, forgetting and post-training.** The effect grows nearly
  linearly with the number of copies (fitted exponent 1.00, up to 32 copies),
  and most members become detectable at a 1% false-positive rate with four
  to eight copies. Documents inserted in the first 4,000 updates retain at
  most 21% of their measured peak effect at the end of mid-training. After
  the full post-training recipe, at the checkpoint that corresponds to the
  released OLMo 2 1B Instruct, 89% of the within-pair effect and 74% of the
  AUC advantage over chance remain.

These results concern one 1B-parameter model, one mid-training stage in
which 0.135% of the tokens were replaced, and the attack statistics we
evaluated. The reference ratio was specified before the analysis; the other
analyses were added later and are exploratory.

## Repository layout

```
OLMo-Twin/
├── olmo_twin/        small shared package; paths.py resolves the data root
├── collection/       candidate documents and their search against the OLMo 2 training data
├── design/           matched pairs, coin flips, copy positions: the frozen design
├── training/
│   ├── midtrain/     AI2's inputs, the training stream with the inserted documents, the launcher
│   ├── checks/       checks of the finished run (data order, inserted copies, configuration)
│   └── posttrain/    SFT, DPO and RLVR with AI2's open-instruct, for both models
├── scoring/          per-token log-probabilities under every model and checkpoint
├── analysis/         membership statistics, attacks, trajectories, post-training analysis
├── figures/          the paper's tables, numbers and figures
├── env/              software environments (requirements files, Dockerfiles)
└── docs/             reproduction guide and data documentation
```

Each directory has a README that describes its step in detail.
[docs/REPRODUCING.md](docs/REPRODUCING.md) puts the steps in order, with the
inputs and outputs of each. [docs/data.md](docs/data.md) describes the data
layout, the planned data release and the licences.
[docs/paper_outputs.md](docs/paper_outputs.md) maps each figure and table
of the paper to the command that produces it.

## Getting started

### Installation

The analysis, the figures and the checks run on a CPU. The pinned versions
in `env/analysis/requirements.txt` need Python 3.12 or later.

```bash
git clone https://github.com/ali-bana/OLMo-Twin.git
cd OLMo-Twin
python3 -m venv .venv && . .venv/bin/activate
pip install -r env/analysis/requirements.txt
pip install -e .
```

Use the repository as a checkout. You can run the scripts without installing
anything, or install the package in editable mode (`pip install -e .`). A
regular `pip install .` is not supported, because the package derives the
repository root, and with it the default data root, from its own location.
The document collection, the training runs and the scoring use other
environments; [env/README.md](env/README.md) describes them, including
Dockerfiles for the GPU images.

### Where the data live

No path is hard-coded. Every script finds its inputs and outputs under one
*data root*, resolved in this order:

1. the script's own command-line options (for example `--out`), where it has
   them;
2. the environment variable `OLMO_TWIN_DATA`;
3. `data/` in the repository (ignored by git; it may be a symbolic link).

```bash
export OLMO_TWIN_DATA=/path/to/olmo-twin-data
python -m olmo_twin.paths              # prints the resolved layout and which directories exist
```

The data root has a fixed layout: `bundle/` for the frozen design,
`analysis/` for the analysis tables, `run/` for the scores and checkpoints,
`assets/` for AI2's public inputs, and so on. Shell scripts get the same
paths as `OLMO_TWIN_<NAME>` variables from
`eval "$(python3 -m olmo_twin.paths --shell)"` (run from the repository root,
or after `pip install -e .`). [docs/data.md](docs/data.md)
lists each directory, the step that writes it and the steps that read it.

### Regenerating the paper's tables and figures

With the analysis tables and scores of the data release (once it is
available) in the data root, either command regenerates the paper's tables,
macros and figures:

```bash
bash figures/make_all.sh               # from the analysis outputs -> $OLMO_TWIN_DATA/paper/
bash analysis/run_analysis.sh          # re-runs the analysis first, then the tables and figures,
                                       # Table 4 on all pairs and the RLVR results
```

Both scripts write into the data root and overwrite earlier outputs in place.
Work on a copy of the data release if you want to keep the original files.
`run_analysis.sh` skips, with a message, the steps that need document texts
or token ids, which are not redistributed for every document. The other
steps run on the released data.

### Tests

```bash
python -m pytest -p no:cacheprovider design/tests training/midtrain/tests
# collection/: in the collection environment (numba, beautifulsoup4, wikiextractor 3.0.8)
# with pytest added, one test file per process
for f in collection/tests/test_*.py; do python -m pytest -q -p no:cacheprovider "$f"; done
```

## Reproducing the experiment

The full pipeline runs from document collection to the paper's figures:
collection, design, mid-training, checks, post-training, scoring, analysis
and figures. [docs/REPRODUCING.md](docs/REPRODUCING.md) describes each stage,
its inputs and outputs, the commands, and the parts that cannot be
reproduced from this repository alone. The scripts contain no scheduler or
cluster commands. The multi-node launchers are plain `torchrun`,
`accelerate launch` and `ray` commands that take the node rank and the
address of the first node from environment variables, so they can be started
by any scheduler or by hand.

| Stage | Compute used for the paper |
|---|---|
| Collection and overlap search | CPU, network access, a GROBID server; local copies of the OLMo 2 training corpora (8.3 TiB); one GPU for an embedding search |
| Mid-training | 16 A100 80GB GPUs (two nodes of eight) |
| SFT, DPO | 8 A100 80GB GPUs per model |
| RLVR | 16 A100 GPUs (two nodes) per stage and model |
| Scoring | one GPU per model and checkpoint (about 46 minutes per model on one A100) |
| Analysis, figures, checks | CPU |

Upstream code is used at pinned commits and is not copied into this
repository. We use the OLMo trainer at commit `090253d` and open-instruct at
commit `d19de7d`. Our changes to them are two small patches. One makes
checkpoint writing safe when several nodes share one file system, and is
optional. The other keeps intermediate post-training checkpoints and gives
all processes the same run name. No loss, optimizer or data code is changed.

## Data and models

The data and models are to be released. The release is planned to contain:

- the frozen design (pairs, cohorts, arm assignment, the position of every
  copy, checksums);
- the records of the training-data overlap search;
- per-document scores under every scored model and checkpoint;
- per-token log-probabilities;
- the analysis tables;
- the treated model's checkpoints every 1,000 updates;
- the post-trained models of both arms.

Document texts, and the token ids that encode them, will be released only
where the documents' licences permit. Wikipedia and StackExchange texts are
under CC BY-SA and will be released with attribution. For other documents,
such as arXiv papers distributed only under arXiv's non-exclusive licence,
the release gives identifiers and checksums, and the texts can be rebuilt
with `collection/` and `design/`. AI2's public inputs (checkpoints, training
data, post-training datasets) are not redistributed; the scripts download
them at pinned versions. See [docs/data.md](docs/data.md).

## Citation

If you use this code or the data, please cite the paper:

```bibtex
@misc{banaei2026olmotwin,
  title  = {From Training Exposure to Membership Evidence: A Controlled Study of {OLMo} 2},
  author = {Banaei, Ali and Shahzad, Muhammad Shahab and Maguire, Jr., Gerald Q. and
            Chiesa, Marco and Kosti{\'c}, Dejan},
  year   = {2026},
  note   = {arXiv preprint, to appear}
}
```

[CITATION.cff](CITATION.cff) contains the same information in machine-readable
form.

## License

The code is released under the Apache License, Version 2.0
([LICENSE](LICENSE)). [NOTICE](NOTICE) lists the third-party code and
configurations that this repository contains or modifies, with their
licences. The data, the documents and AI2's inputs have their own licences,
described in [docs/data.md](docs/data.md).
