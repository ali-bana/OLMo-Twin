#!/usr/bin/env python3
"""Check that non-injected training positions hold AI2's original chunks.

Draws random training positions of the injected run, keeps those that do not
hold an injected chunk, and compares, for each, the chunk the run read there
(built stream, read through the saved global index file) with the official
chunk that AI2's seed-42 run read at the same position (AI2's token files at
position p of the PCG64(42) permutation of the official chunks). It then
checks, for about 500 injected positions spread over the placement file, that
the chunk read there differs from the official one.

Appendix D reports 3,700 byte-identical non-injected positions in two
independent samples:
    --seed 20261028 --draw 3000 --keep 2500   (the defaults)
    --seed 777 --draw 1300 --keep 1200
In both runs every sampled injected position differed from the official chunk.

Inputs (defaults from the data root, see olmo_twin/paths.py):
  assets/configs/stage2_tokenized_manifest.json  AI2's token files (local_path
      of each entry, written by training/midtrain/prepare_stage2_assets.py)
  stream/stream_manifest.json, stream/data/      the built stream
  run/checkpoints/train_data/global_indices.npy  saved by the trainer
  bundle/stream_plan/placement.jsonl.gz          the injected positions

Usage:
    python training/checks/check_background_chunks.py [--seed S] [--draw N] [--keep K]
        [--checkpoint-dir DIR]

Needs numpy. Exits with status 1 if a background chunk differs or a sampled
injected chunk equals the official one.
"""

import argparse
import bisect
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from olmo_twin import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261028, help="seed of numpy.random.default_rng (default: %(default)s)")
    ap.add_argument("--draw", type=int, default=3000, help="positions drawn without replacement (default: %(default)s)")
    ap.add_argument("--keep", type=int, default=2500,
                    help="non-injected positions kept, in drawing order (default: %(default)s)")
    ap.add_argument("--checkpoint-dir", type=Path, default=paths.RUN / "checkpoints",
                    help="save folder of the run, holding train_data/global_indices.npy (default: %(default)s)")
    args = ap.parse_args()

    S = 4096
    man = json.load(open(paths.ASSETS / "configs" / "stage2_tokenized_manifest.json"))["entries"]
    man = sorted(man, key=lambda e: int(e["index"]))
    offc = []; ends = []; o = 0
    for e in man:
        c = int(e["tokens"]) // S; offc.append(o); o += c; ends.append(o)
    N_off = o
    st = json.load(open(paths.STREAM / "stream_manifest.json"))
    N = int(st["target_chunks"]); per = int(st["shards"][0]["chunks"])
    print("official chunks", N_off, "treated chunks", N, "per shard", per)
    base = np.arange(N_off, dtype=np.uint32); np.random.Generator(np.random.PCG64(42)).shuffle(base)
    T = np.memmap(args.checkpoint_dir / "train_data" / "global_indices.npy", dtype=np.uint32, mode="r")
    inj = {}
    with gzip.open(paths.BUNDLE / "stream_plan" / "placement.jsonl.gz", "rt") as f:
        for line in f:
            x = json.loads(line)
            for p in x["training_positions"]: inj[p] = x["doc_id"]
    print("injected positions in placement", len(inj))
    rng = np.random.default_rng(args.seed)
    pos = rng.choice(N, args.draw, replace=False)
    pos = [int(p) for p in pos if int(p) not in inj][:args.keep]

    def off_chunk(g):
        i = bisect.bisect_right(ends, g); loc = g - offc[i]
        with open(man[i]["local_path"], "rb") as f:
            f.seek(loc * S * 4); return np.frombuffer(f.read(S * 4), dtype="<u4")

    def tr_chunk(slot):
        sh = slot // per; loc = slot % per
        with open(paths.STREAM / "data" / st["shards"][sh]["path"], "rb") as f:
            f.seek(loc * S * 4); return np.frombuffer(f.read(S * 4), dtype="<u4")

    eq = ne = 0; t = time.time()
    for p in pos:
        if np.array_equal(off_chunk(int(base[p])), tr_chunk(int(T[p]))): eq += 1
        else: ne += 1
    print("background positions checked", len(pos), "equal", eq, "differ", ne, "secs", round(time.time() - t, 1))
    # injected positions: all differ from official?
    ip = list(inj)[:: max(1, len(inj) // 500)]
    d = sum(1 for p in ip if not np.array_equal(off_chunk(int(base[p])), tr_chunk(int(T[p]))))
    print("injected positions sampled", len(ip), "differ from official", d)
    return 0 if (ne == 0 and d == len(ip)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
