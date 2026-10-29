#!/usr/bin/env python3
"""Check the order in which the injected mid-training run read its data.

OLMo's loader permutes the chunk indices of the training data once with
NumPy's PCG64 generator seeded with the training seed (42), and update u reads
positions [512 (u-1), 512 u) of that order, whatever the number of GPUs. The
trainer saves the order as train_data/global_indices.npy in its save folder and
logs the indices each rank read as data-indices/rank<r>.tsv.gz (one line per
update: the update number, then the rank's indices).

This script

1. recomputes the permutation for the number of chunks in the saved file and
   checks that it equals the saved order (and is a permutation);
2. checks every logged line of the 16 ranks against the saved order: rank r
   must have read every 16th index of the update's 512 positions, starting at
   r (the update number in the log may be offset by one, both offsets are
   tried and the ones found are printed);
3. checks for four updates that 16 and 256 ranks would read the same set of
   chunks, i.e. that the order of the data does not depend on the number of
   GPUs.

Our run kept the per-rank logs of updates 18001-23852 only; Appendix D
reports that all 93,632 rows (16 ranks x 5,852 updates) match.

Usage:
    python training/checks/check_data_order.py [--checkpoint-dir DIR]

DIR is the run's save folder (default: $OLMO_TWIN_DATA/run/checkpoints, see
olmo_twin/paths.py). Needs numpy only; a few minutes on one CPU. Exits with
status 1 if any check fails.
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from olmo_twin import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, default=paths.RUN / "checkpoints",
                    help="save folder of the run, holding train_data/ and data-indices/ (default: %(default)s)")
    args = ap.parse_args()
    ckpt = args.checkpoint_dir

    print(np.__version__)
    g = np.memmap(ckpt / "train_data" / "global_indices.npy", dtype=np.uint32, mode="r"); N = len(g); print("N", N, N % 512, N // 512, N % 16, N % 256)
    idx = np.arange(N, dtype=np.uint32); np.random.Generator(np.random.PCG64(seed=42)).shuffle(idx)
    same = np.array_equal(idx, g); perm = len(np.unique(g)) == N
    print("recomputed PCG64(42) equals saved:", same, "is permutation:", perm)
    ga = np.asarray(g)
    bad = 0; tot = 0; offs = set(); steps = set()
    for r in range(16):
        with gzip.open(ckpt / "data-indices" / f"rank{r}.tsv.gz", "rt") as f:
            for line in f:
                p = line.rstrip("\n").split("\t"); s = int(p[0]); v = np.array(p[1:], dtype=np.int64)
                tot += 1; steps.add(s)
                ok = False
                for off in (0, -1):
                    st = 512 * (s + off)
                    if np.array_equal(ga[st:st + 512][r::16], v): offs.add(off); ok = True; break
                if not ok: bad += 1
    print("rank-step lines", tot, "mismatches", bad, "offsets", offs, "steps", min(steps), max(steps), len(steps))

    def step_sets(W, s):
        dev = 512 // W
        return set(np.concatenate([ga[r::W][s * dev:(s + 1) * dev] for r in range(W)]).tolist())

    sets_ok = True
    for s in (0, 1, 12345, 23851):
        a = step_sets(16, s); b = step_sets(256, s); c = set(ga[512 * s:512 * (s + 1)].tolist())
        print(s, a == b == c, len(a))
        sets_ok = sets_ok and a == b == c
    return 0 if (same and perm and bad == 0 and sets_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
