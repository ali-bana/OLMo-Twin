"""Where the OLMo-Twin scripts look for their data.

Every script in this repository that reads or writes data finds it through this
module, so the code contains no machine-specific paths. Locations are resolved
in this order:

1. Command-line options of the individual script (for example ``--out`` or
   ``--evaluation-manifest``). Where a script has such an option, it always
   wins over the defaults below.
2. The environment variable ``OLMO_TWIN_DATA``: the *data root*, a directory
   that holds the released data (or the data you produce yourself) in the
   layout described below. A relative value is made absolute against the
   current working directory when this module is imported; symbolic links are
   kept as they are, not resolved.
3. If ``OLMO_TWIN_DATA`` is not set: ``<repository root>/data``. This
   directory is ignored by git; you can unpack the data release there or make
   it a symbolic link to wherever the data actually live.

The variable is read once, when this module is first imported, so set it
before starting Python::

    export OLMO_TWIN_DATA=/path/to/olmo-twin-data
    python -m olmo_twin.paths        # prints the resolved layout

Shell scripts use the same layout through
``eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"``, which
defines ``OLMO_TWIN_REPO``, ``OLMO_TWIN_DATA`` and one ``OLMO_TWIN_<NAME>``
variable per directory below (``OLMO_TWIN_BUNDLE``, ``OLMO_TWIN_ANALYSIS``,
...). All of them are absolute paths, so a script may change its working
directory afterwards. The module needs only the Python standard library
(Python 3.6 or later).

The repository root is derived from the location of this file. Use the package
from a checkout of the repository: either do not install it at all (every
script adds the repository root to ``sys.path`` itself) or install it in
editable mode (``pip install -e .``). A regular, non-editable install copies
the package into ``site-packages``, where :data:`REPO_ROOT`, :func:`code_dir`,
:func:`add_code_dirs` and the ``<repository root>/data`` default would point to
the wrong place.

Layout of the data root
-----------------------
Each name below is a module-level constant (``paths.BUNDLE``, ...). A pipeline
step writes into the same directories that later steps read, so re-running a
step overwrites its earlier output in place; work on a copy of the data root if
you want to keep the released files untouched. ``docs/data.md`` lists which
step writes and which step reads each directory.

``bundle/``
    The frozen experimental design: ``manifest.csv.gz`` (pairs, cohorts, arm
    assignment), ``copies.csv.gz``, ``null_pool.csv.gz``,
    ``evaluation_manifest.csv.gz``, ``summary.json``, ``texts_index.csv.gz``,
    ``stream_plan/placement.jsonl.gz`` (with ``placement_summary.json``),
    ``SHA256SUMS``, ``manifest_v1a_prepairfix.csv.gz``,
    ``scoring_manifest_extra.csv.gz`` and ``logprobs_extra_stage1/scores.csv.gz``.
    The document texts (``texts_all/``, ``texts/``) and token ids
    (``tokens/``, ``null_tokens/``, the ``tokens/`` subdirectories of
    ``logprobs_extra*/``) are third-party text and are not redistributed;
    rebuild them with the collection and design steps and check them against
    ``SHA256SUMS``.
``analysis/``
    Per-document attack tables and every analysis output that the figure and
    table generators read (``per_document_attacks_prelim.csv.gz``,
    ``per_document_attacks_calibrated.csv.gz``, ``ladder_trajectory.csv.gz``,
    ``token_neff.npz``, ``posttrain_exclusions.json``,
    ``cohortF_2x2_stats.json``, ``*.json`` of the analysis scripts,
    ``paper_numbers.json``).
``run/``
    Outputs of the injected (treated) mid-training run and its scoring:
    ``scores/<label>/``, ``scores_ext/<label>/``, ``scores_ladder/<label>/``,
    ``scores_posttrain/<label>/``, ``scores_cohortF_2x2/<label>/``,
    ``cohortF_2x2/`` and ``checkpoints/`` (``train_data/global_indices.npy``,
    ``data-indices/``, ``step<N>-unsharded/``).
``stream/``
    The injected training stream built from AI2's stage-2 data:
    ``data/train-*-of-00064.npy``, ``stream_manifest.json``, ``config/``.
``assets/``
    Public inputs from AI2: ``configs/`` (stage-2 YAML and the token-file
    manifest), ``stage2_tokenized/`` (the 576 token files),
    ``checkpoints/`` (stage-1 endpoint and the seed-42 / seed-42069 stage-2
    checkpoints), ``hf_models/``, ``posttrain_datasets/`` and
    ``hf_cache_posttrain/``.
``posttrain/``
    Post-training runs (``runs/sft_<arm>``, ``dpo_<arm>``, ``rlvr1_<arm>``,
    ``rlvr2_<arm>``) and open-instruct's tokenized dataset cache
    (``dataset_cache/``).
``posttrain_overlap/``
    Results of the scan of the post-training datasets for evaluation
    documents (``posttrain_hits.csv.gz``, ``posttrain_impact.json``; the
    excerpt file ``matched_examples.txt`` quotes third-party training text and
    is not redistributed).
``exclusions/``
    Exclusion lists that are not produced by the analysis scripts
    (``repetition_filter_masked_chunks.json``).
``rlvr/``
    RLVR analysis: final tables in the directory itself, the overlap scan of
    the RLVR datasets in ``overlap/`` and intermediate files in ``work/``.
``collection/``
    Working directory of document collection and the training-data overlap
    search (candidate pools, fingerprints, scan results, edited-Wikipedia
    group, pre-treatment scores).
``cohortF_reparse/``
    The re-parsed second text form of the cohort-F papers (third-party text,
    not redistributed; rebuild with ``collection/cohortF_reparse.py``).
``paper/``
    Default output directory of the figure and table generators
    (``figures/`` and ``tables/``).
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

ENV_VAR = "OLMO_TWIN_DATA"

#: Root of this repository (the directory that contains ``olmo_twin/``).
#: Valid only when the package is used from a checkout (see the module
#: docstring): no install, or ``pip install -e .``.
REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return the data root: ``$OLMO_TWIN_DATA`` if set, else ``<repo>/data``.

    A relative ``$OLMO_TWIN_DATA`` is made absolute against the current
    working directory (``~`` is expanded); symbolic links are not resolved, so
    a data root assembled from symbolic links keeps working.
    """
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(os.path.abspath(os.path.expanduser(value))) if value else REPO_ROOT / "data"


#: The data root, resolved when this module is imported.
DATA = data_root()

BUNDLE = DATA / "bundle"
ANALYSIS = DATA / "analysis"
RUN = DATA / "run"
STREAM = DATA / "stream"
ASSETS = DATA / "assets"
POSTTRAIN = DATA / "posttrain"
POSTTRAIN_OVERLAP = DATA / "posttrain_overlap"
EXCLUSIONS = DATA / "exclusions"
RLVR = DATA / "rlvr"
COLLECTION = DATA / "collection"
COHORTF_REPARSE = DATA / "cohortF_reparse"
PAPER_OUT = DATA / "paper"

#: Name -> directory, in the order of the docstring above.
LAYOUT = {
    "bundle": BUNDLE,
    "analysis": ANALYSIS,
    "run": RUN,
    "stream": STREAM,
    "assets": ASSETS,
    "posttrain": POSTTRAIN,
    "posttrain_overlap": POSTTRAIN_OVERLAP,
    "exclusions": EXCLUSIONS,
    "rlvr": RLVR,
    "collection": COLLECTION,
    "cohortF_reparse": COHORTF_REPARSE,
    "paper": PAPER_OUT,
}


def code_dir(name: str) -> Path:
    """Return a top-level code directory of the repository (e.g. ``"analysis"``)."""
    path = REPO_ROOT / name
    if not path.is_dir():
        raise FileNotFoundError(f"no code directory {name!r} under {REPO_ROOT}")
    return path


def add_code_dirs(*names: str) -> None:
    """Make the modules in the given code directories importable.

    The figure generator imports helper modules from ``analysis/`` and several
    analysis scripts import the generator from ``figures/``; this puts those
    directories on ``sys.path`` without installing anything.
    """
    for name in names:
        entry = str(code_dir(name))
        if entry not in sys.path:
            sys.path.insert(0, entry)


def shell_assignments() -> str:
    """The layout as shell variable assignments, for the shell scripts.

    ``eval "$(python3 -m olmo_twin.paths --shell)"`` defines ``OLMO_TWIN_REPO``,
    ``OLMO_TWIN_DATA`` and one ``OLMO_TWIN_<NAME>`` variable per layout entry
    (for example ``OLMO_TWIN_BUNDLE``), so shell and Python scripts resolve the
    same directories.
    """
    import shlex

    lines = [f"OLMO_TWIN_REPO={shlex.quote(str(REPO_ROOT))}", f"OLMO_TWIN_DATA={shlex.quote(str(DATA))}"]
    lines += [f"OLMO_TWIN_{name.upper()}={shlex.quote(str(path))}" for name, path in LAYOUT.items()]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--shell"]:
        print(shell_assignments())
        return 0
    if argv:
        print("usage: python -m olmo_twin.paths [--shell]", file=sys.stderr)
        return 2
    source = f"${ENV_VAR}" if os.environ.get(ENV_VAR, "").strip() else f"default (${ENV_VAR} not set)"
    print(f"data root: {DATA}  [{source}]")
    for name, path in LAYOUT.items():
        state = "present" if path.exists() else "missing"
        print(f"  {name + '/':20s} {state:8s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
