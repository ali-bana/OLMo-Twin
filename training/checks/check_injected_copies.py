#!/usr/bin/env python3
"""Check that every injected copy was read at its planned training positions.

For each injected copy in the placement file (bundle/stream_plan/
placement.jsonl.gz), the script reads the chunks at the copy's training
positions from the built stream the way OLMo's loader read them, through the
global index file that the trainer saved (position p -> physical slot
global_indices[p] -> shard and offset in stream/data/), concatenates them and
checks that they hold the document's tokens (bundle token file) followed, in
the last partial chunk, by the leading tokens of the official chunk that the
copy replaced (read from AI2's token files at the official position, i.e. the
PCG64(42) permutation of the official chunks). It also prints summary
statistics of the placement.

Appendix D reports that all 12,414 injected copies hold their documents'
tokens at their planned positions; this is the default (--stride 1). A larger
--stride checks every n-th copy only (our first check used --stride 60, 207
copies).

Inputs (defaults from the data root, see olmo_twin/paths.py):
  assets/configs/stage2_tokenized_manifest.json  AI2's token files; each entry's
      local_path, as written by training/midtrain/prepare_stage2_assets.py
  stream/stream_manifest.json, stream/data/      the built stream
  run/checkpoints/train_data/global_indices.npy  saved by the trainer
  bundle/stream_plan/placement.jsonl.gz and the bundle token files
      (bundle/tokens/, rebuilt with the design step; not redistributed)

Usage:
    python training/checks/check_injected_copies.py [--stride N] [--checkpoint-dir DIR]

Needs numpy; reads about 100 MB of injected chunks plus one official chunk
per copy. Exits with status 1 if any checked copy differs.
"""

import argparse
import bisect
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from olmo_twin import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=1,
                    help="check every n-th copy of the placement file (default: %(default)s, all copies)")
    ap.add_argument("--checkpoint-dir", type=Path, default=paths.RUN / "checkpoints",
                    help="save folder of the run, holding train_data/global_indices.npy (default: %(default)s)")
    args = ap.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be positive")

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

    def off_chunk(g):
        i = bisect.bisect_right(ends, g); loc = g - offc[i]
        with open(man[i]["local_path"], "rb") as f:
            f.seek(loc * S * 4); return np.frombuffer(f.read(S * 4), dtype="<u4")

    def tr_chunk(slot):
        sh = slot // per; loc = slot % per
        with open(paths.STREAM / "data" / st["shards"][sh]["path"], "rb") as f:
            f.seek(loc * S * 4); return np.frombuffer(f.read(S * 4), dtype="<u4")

    BUN = paths.BUNDLE
    rows = [json.loads(l) for l in gzip.open(BUN / "stream_plan" / "placement.jsonl.gz", "rt")]
    print("rows", len(rows), "chunks", sum(r["n_chunks"] for r in rows), "tokens_with_eos", sum(r["n_tokens_with_eos"] for r in rows))
    print("first_update", min(r["first_update"] for r in rows), "last_update", max(r["last_update"] for r in rows))
    print("n_chunks dist", collections.Counter(min(r["n_chunks"], 9) for r in rows).most_common())
    print("copies with tail==4096 (no filler)", sum(1 for r in rows if r["tail_tokens"] == 4096))
    ok = bad = 0; eos_ok = 0
    for r in rows[::args.stride]:
        ids = np.load(BUN / r["tokens_relpath"]).astype("<u4")
        eos_ok += int(ids[-1] == 100257)
        got = np.concatenate([tr_chunk(int(T[p])) for p in r["training_positions"]])
        n = len(ids); filler = len(got) - n
        lastoff = off_chunk(int(base[r["training_positions"][-1]]))
        good = np.array_equal(got[:n], ids) and (filler == 0 or np.array_equal(got[n:], lastoff[:filler]))
        ok += good; bad += (not good)
    print("copies checked", ok + bad, "ok", ok, "bad", bad, "ending in EOS", eos_ok)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
