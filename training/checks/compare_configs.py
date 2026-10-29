#!/usr/bin/env python3
"""Compare the configuration of the injected run with AI2's configurations.

Flattens five OLMo configurations to dotted keys and prints the keys in which
they differ:

  ours  the configuration saved with the injected run (run/checkpoints/config.yaml)
  pub   AI2's published stage-2 file, configs/official-0425/OLMo2-1B-stage2-seed42.yaml
        of the OLMo checkout (the file our run was built from)
  ai2a  the configuration stored in AI2's seed-42 checkpoint at step 1000
  ai2b  the same at step 23852 (AI2's seed-42 run consists of two launches)
  sib   the configuration stored in AI2's seed-42069 checkpoint at step 23852

It prints whether the data paths agree (the last four path components, since
AI2's checkpoints record their own storage locations), then the keys in which
ours differs from AI2's seed-42 run (excluding data.paths and the in-loop
evaluators), the keys in which AI2's two seed-42 launches differ, and the keys
in which the seed-42 and seed-42069 runs differ. Appendix D summarises these
differences (execution keys such as fused loss, flash attention, compilation,
sharding, per-GPU batch and number of GPUs; logging, checkpointing, file
locations and run identity; no optimizer, scheduler, data, duration,
loss-weight or precision key).

Inputs (defaults from the data root, see olmo_twin/paths.py):
  run/checkpoints/config.yaml
  assets/checkpoints/official_stage2_ingredient3_seed42/step{1000,23852}-unsharded/config.yaml
  assets/checkpoints/official_stage2_ingredient1/step23852-unsharded/config.yaml
      (training/midtrain/download_ai2_stage2_checkpoints.sh)

Usage:
    python training/checks/compare_configs.py --olmo-repo OLMO_CHECKOUT [--checkpoint-dir DIR]

--olmo-repo (or the environment variable OLMO_REPO) is an OLMo checkout at
commit 090253d. Needs pyyaml.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from olmo_twin import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--olmo-repo", type=Path, default=os.environ.get("OLMO_REPO") or None,
                    help="OLMo checkout at the pinned commit (default: $OLMO_REPO)")
    ap.add_argument("--checkpoint-dir", type=Path, default=paths.RUN / "checkpoints",
                    help="save folder of the injected run, holding config.yaml (default: %(default)s)")
    args = ap.parse_args()
    if args.olmo_repo is None:
        ap.error("--olmo-repo (or $OLMO_REPO) is required")

    A = paths.ASSETS / "checkpoints"
    ours = yaml.safe_load(open(args.checkpoint_dir / "config.yaml"))
    pub = yaml.safe_load(open(args.olmo_repo / "configs" / "official-0425" / "OLMo2-1B-stage2-seed42.yaml"))
    ai2a = yaml.safe_load(open(A / "official_stage2_ingredient3_seed42" / "step1000-unsharded" / "config.yaml"))
    ai2b = yaml.safe_load(open(A / "official_stage2_ingredient3_seed42" / "step23852-unsharded" / "config.yaml"))
    sib = yaml.safe_load(open(A / "official_stage2_ingredient1" / "step23852-unsharded" / "config.yaml"))

    def flat(d, p=""):
        o = {}
        if isinstance(d, dict):
            for k, v in d.items(): o.update(flat(v, p + k + "."))
        else: o[p[:-1]] = d
        return o

    F = {n: flat(x) for n, x in dict(ours=ours, pub=pub, ai2a=ai2a, ai2b=ai2b, sib=sib).items()}

    def tail(ps): return ["/".join(str(x).split("/")[-4:]) for x in ps]

    for n in F: print(n, "n_paths", len(F[n].get("data.paths", [])))
    print("ai2a paths == pub paths (last 4 comps):", tail(F["ai2a"]["data.paths"]) == tail(F["pub"]["data.paths"]))
    print("ai2a paths == sib paths:", tail(F["ai2a"]["data.paths"]) == tail(F["sib"]["data.paths"]))
    print("ai2a paths == ai2b paths:", F["ai2a"]["data.paths"] == F["ai2b"]["data.paths"])
    print("ai2a first path:", F["ai2a"]["data.paths"][0])
    skip = {"data.paths"}
    keys = set(F["ours"]) | set(F["ai2a"])
    diff = [k for k in sorted(keys) if k not in skip and F["ours"].get(k, "<absent>") != F["ai2a"].get(k, "<absent>")]
    print("\n#keys ours", len(F["ours"]), "ai2a", len(F["ai2a"]), "common", len(set(F["ours"]) & set(F["ai2a"])))
    print("keys differing ours vs AI2(step1000):")
    for k in diff:
        if k.startswith("evaluators"): continue
        print(" ", k, "| ours:", str(F["ours"].get(k, "<absent>"))[:80], "| ai2:", str(F["ai2a"].get(k, "<absent>"))[:80])
    print("\nkeys differing AI2 step1000 vs step23852 (two launches):")
    for k in sorted(set(F["ai2a"]) | set(F["ai2b"])):
        if F["ai2a"].get(k) != F["ai2b"].get(k) and k != "data.paths": print(" ", k, F["ai2a"].get(k), "|", F["ai2b"].get(k))
    print("\nkeys differing AI2 seed42 step23852 vs seed42069 step23852:")
    for k in sorted(set(F["sib"]) | set(F["ai2b"])):
        if F["sib"].get(k) != F["ai2b"].get(k) and k != "data.paths" and not k.startswith("evaluators"): print(" ", k, str(F["ai2b"].get(k))[:70], "|", str(F["sib"].get(k))[:70])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
