#!/usr/bin/env python3
"""Find the injected chunks whose loss OLMo's instance filter masked.

AI2's stage-2 configuration enables OLMo's instance filter
(data.instance_filter: repetition_min_period 1, repetition_max_period 13,
repetition_max_count 32): the loss of a training chunk is masked when a
sequence of 1 to 13 tokens repeats at least 32 times in a row. This script
re-applies the filter function of the pinned trainer
(olmo/data/util.py: find_periodic_sequences, executed from an OLMo checkout)
to every injected chunk, read from the built stream through the saved global
index file at the copies' training positions.

Appendix D reports 43 masked chunks from 18 documents (of 24,081 injected
chunks), including the only chunk of two single-copy documents.

Inputs (defaults from the data root, see olmo_twin/paths.py):
  stream/stream_manifest.json, stream/data/      the built stream
  run/checkpoints/train_data/global_indices.npy  saved by the trainer
  bundle/stream_plan/placement.jsonl.gz          the injected copies

Usage:
    python training/checks/check_instance_filter.py --olmo-repo OLMO_CHECKOUT
        [--out masked.json] [--checkpoint-dir DIR]

--olmo-repo (or the environment variable OLMO_REPO) is an OLMo checkout at
commit 090253d (training/midtrain/versions.env). With --out, the masked chunks
are written as a JSON list of [doc_id, copy, training_position]. Needs numpy
and torch (imported by olmo/data/util.py); about two minutes on one CPU.
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from olmo_twin import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--olmo-repo", type=Path, default=os.environ.get("OLMO_REPO") or None,
                    help="OLMo checkout at the pinned commit (default: $OLMO_REPO)")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON file for the masked chunks")
    ap.add_argument("--checkpoint-dir", type=Path, default=paths.RUN / "checkpoints",
                    help="save folder of the run, holding train_data/global_indices.npy (default: %(default)s)")
    args = ap.parse_args()
    if args.olmo_repo is None:
        ap.error("--olmo-repo (or $OLMO_REPO) is required")

    ns = {}
    exec(open(args.olmo_repo / "olmo" / "data" / "util.py").read(), ns)
    fps = ns["find_periodic_sequences"]
    S = 4096
    st = json.load(open(paths.STREAM / "stream_manifest.json")); per = int(st["shards"][0]["chunks"])
    T = np.memmap(args.checkpoint_dir / "train_data" / "global_indices.npy", dtype=np.uint32, mode="r")
    rows = [json.loads(l) for l in gzip.open(paths.BUNDLE / "stream_plan" / "placement.jsonl.gz", "rt")]
    fh = {}

    def tr_chunk(slot):
        sh = slot // per; loc = slot % per
        p = paths.STREAM / "data" / st["shards"][sh]["path"]
        if p not in fh: fh[p] = open(p, "rb")
        f = fh[p]; f.seek(loc * S * 4); return np.frombuffer(f.read(S * 4), dtype="<u4").astype(np.int64)

    masked = []; t = time.time(); n = 0
    for r in rows:
        for p in r["training_positions"]:
            x = tr_chunk(int(T[p])); n += 1
            if any(m.times >= 32 for m in fps(x, max_period=13, min_period=1)): masked.append((r["doc_id"], r["copy"], p))
    print("injected chunks scanned", n, "masked", len(masked), "docs", len(set(m[0] for m in masked)), "secs", round(time.time() - t))
    if args.out is not None:
        json.dump(masked, open(args.out, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
