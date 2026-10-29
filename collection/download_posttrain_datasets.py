#!/usr/bin/env python3
"""Download the raw SFT and DPO dataset snapshots.

They are used by the overlap scan of the post-training data
(build_posttrain_corpus.py) and by rlvr_overlap_positive_control.py. Each
dataset is downloaded with huggingface_hub.snapshot_download into
<data root>/assets/posttrain_datasets/<name>/ at the commit that the
post-training runs used:

  allenai/tulu-3-sft-olmo-2-mixture-0225   d91a0785ade02942520280fb484866fce41e448f
  allenai/olmo-2-0425-1b-preference-mix    c6454f3d364622718a8ecd3d307e524febbba569

The original download (28 Sep 2026) requested revision ``main``, which resolved
to exactly these commits, so the files are the same. ``--revision REPO=COMMIT``
replaces the commit of one dataset (for example ``--revision
allenai/olmo-2-0425-1b-preference-mix=main``).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

DATASETS = {
    "allenai/tulu-3-sft-olmo-2-mixture-0225": "d91a0785ade02942520280fb484866fce41e448f",
    "allenai/olmo-2-0425-1b-preference-mix": "c6454f3d364622718a8ecd3d307e524febbba569",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=paths.ASSETS / "posttrain_datasets",
                    help="target directory (default: assets/posttrain_datasets under the data root)")
    ap.add_argument("--revision", action="append", default=[], metavar="REPO=COMMIT",
                    help="download REPO at COMMIT instead of the pinned commit (repeatable)")
    args = ap.parse_args()
    revisions = dict(DATASETS)
    for item in args.revision:
        repo, sep, commit = item.partition("=")
        if not sep or not commit or repo not in revisions:
            ap.error(f"--revision expects REPO=COMMIT with REPO one of {sorted(revisions)}; got {item!r}")
        revisions[repo] = commit

    from huggingface_hub import snapshot_download

    dest = args.dest
    for repo, revision in revisions.items():
        name = repo.split("/")[-1]
        print(f"=== downloading {repo} @ {revision}", flush=True)
        p = snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, local_dir=f"{dest}/{name}")
        print(f"=== done {repo} -> {p}", flush=True)
    print("ALL-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
