#!/usr/bin/env python3
"""Compare Hugging Face branches of allenai/OLMo-2-0425-1B with AI2's OLMo-native checkpoints.

The reference models of the paper are loaded from AI2's OLMo-native
checkpoints (training/midtrain/download_stage1_checkpoint.sh and
download_ai2_stage2_checkpoints.sh). This script checks which Hugging Face
branches hold the same weights (Appendix D, "Reference models").

Default mode: for each (branch, local checkpoint) pair below, it reads four
tensors from the branch's safetensors files over HTTP range requests (the
final norm, the key norm of block 0, the down projection of block 8 and the
attention output projection of block 15) and compares them with the same
tensors of the local model.pt. It prints, per tensor, the dtype on the Hub,
the local dtype, whether the tensors are bitwise identical (same values and
same dtype) and the relative difference ||local - hub|| / ||hub||.

  stage1-step1907359-tokens4001B          official_olmo_stage1_4001B
  stage2-ingredient3-step23852-tokens51B  official_stage2_ingredient3_seed42/step23852-unsharded
  stage2-ingredient3-step12000-tokens26B  official_stage2_ingredient3_seed42/step12000-unsharded
  main                                    official_stage2_ingredient3_seed42/step23852-unsharded
  main                                    official_olmo_stage1_4001B
  stage2-ingredient1-step23852-tokens51B  official_stage2_ingredient1/step23852-unsharded
  stage2-ingredient3-step19000-tokens40B  official_stage2_ingredient3_seed42/step19000-unsharded
  stage2-ingredient3-step20000-tokens42B  official_stage2_ingredient3_seed42/step20000-unsharded
  stage2-ingredient3-step1000-tokens3B    official_stage2_ingredient3_seed42/step1000-unsharded

AI2 labels its seed-42 stage-2 run ingredient 3 and its seed-42069 run
ingredient 1 (ingredient 2 is seed 666).

--main mode: reads the final norm and the down projection of block 8 from the
Hub only, for the revisions main, stage1-step1907359-tokens4001B and the
stage-2 endpoints of ingredients 3, 2 and 1, and prints for each revision its
relative difference and cosine similarity to main, and the same for the
stage-1 endpoint against the ingredient-3 endpoint. It shows that main is
none of these checkpoints, so every reference is loaded by branch or
checkpoint path.

Usage:
    python training/checks/compare_hf_branches.py [--main] [--checkpoints DIR]

DIR defaults to $OLMO_TWIN_DATA/assets/checkpoints (see olmo_twin/paths.py).
Needs network access to huggingface.co and torch (2.1 or later); the
User-Agent of the requests is $OLMO_TWIN_USER_AGENT if set.
"""

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from olmo_twin import paths  # noqa: E402

USER_AGENT = os.environ.get("OLMO_TWIN_USER_AGENT", "OLMo-Twin/1.0 (research code)")
REPO_URL = "https://huggingface.co/allenai/OLMo-2-0425-1B/resolve/{rev}/"

# (Hugging Face branch, local checkpoint directory relative to --checkpoints)
PAIRS = [("stage1-step1907359-tokens4001B", "official_olmo_stage1_4001B"),
         ("stage2-ingredient3-step23852-tokens51B", "official_stage2_ingredient3_seed42/step23852-unsharded"),
         ("stage2-ingredient3-step12000-tokens26B", "official_stage2_ingredient3_seed42/step12000-unsharded"),
         ("main", "official_stage2_ingredient3_seed42/step23852-unsharded"),
         ("main", "official_olmo_stage1_4001B"),
         ("stage2-ingredient1-step23852-tokens51B", "official_stage2_ingredient1/step23852-unsharded"),
         ("stage2-ingredient3-step19000-tokens40B", "official_stage2_ingredient3_seed42/step19000-unsharded"),
         ("stage2-ingredient3-step20000-tokens42B", "official_stage2_ingredient3_seed42/step20000-unsharded"),
         ("stage2-ingredient3-step1000-tokens3B", "official_stage2_ingredient3_seed42/step1000-unsharded"),
         ]
# (Hugging Face tensor name, OLMo-native tensor name)
NAMES = [("model.norm.weight", "transformer.ln_f.weight"), ("model.layers.0.self_attn.k_norm.weight", "transformer.blocks.0.k_norm.weight"),
         ("model.layers.8.mlp.down_proj.weight", "transformer.blocks.8.ff_out.weight"), ("model.layers.15.self_attn.o_proj.weight", "transformer.blocks.15.attn_out.weight")]
MAIN_REVS = ["main", "stage1-step1907359-tokens4001B", "stage2-ingredient3-step23852-tokens51B", "stage2-ingredient2-step23852-tokens51B", "stage2-ingredient1-step23852-tokens51B"]
MAIN_NAMES = ["model.norm.weight", "model.layers.8.mlp.down_proj.weight"]


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if rng: req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=300) as r: return r.read()


def compare_branches(A: Path) -> None:
    for rev, loc in PAIRS:
        base = REPO_URL.format(rev=rev)
        try: idx = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
        except Exception as e:
            # single file
            idx = None
        sd = torch.load(A / loc / "model.pt", map_location="cpu", mmap=True, weights_only=True)
        res = []
        for hn, on in NAMES:
            u = base + (idx[hn] if idx else "model.safetensors"); n = struct.unpack("<Q", get(u, (0, 7)))[0]; hdr = json.loads(get(u, (8, 8 + n - 1))); m = hdr[hn]; s, e = m["data_offsets"]
            raw = get(u, (8 + n + s, 8 + n + e - 1))
            dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[m["dtype"]]
            arr = torch.frombuffer(bytearray(raw), dtype=dt).reshape(m["shape"])
            loc_t = sd[on]
            eq = bool(torch.equal(loc_t.float(), arr.float())) and loc_t.dtype == arr.dtype
            rel = float((loc_t.float() - arr.float()).norm() / arr.float().norm())
            res.append((hn.replace("model.", ""), m["dtype"], str(loc_t.dtype), eq, round(rel, 4)))
        print(rev, "vs", loc, res, flush=True)
        del sd


def fetch(rev, hn):
    base = REPO_URL.format(rev=rev)
    try: f = json.loads(get(base + "model.safetensors.index.json"))["weight_map"][hn]
    except Exception: f = "model.safetensors"
    u = base + f; n = struct.unpack("<Q", get(u, (0, 7)))[0]; hdr = json.loads(get(u, (8, 8 + n - 1))); m = hdr[hn]; s, e = m["data_offsets"]
    dt = {"BF16": torch.bfloat16, "F32": torch.float32}[m["dtype"]]
    return torch.frombuffer(bytearray(get(u, (8 + n + s, 8 + n + e - 1))), dtype=dt).reshape(m["shape"]).float()


def compare_main() -> None:
    revs = MAIN_REVS
    for hn in MAIN_NAMES:
        T = {r: fetch(r, hn) for r in revs}
        for r in revs[1:]:
            a, b = T["main"], T[r]
            print(hn, "main vs", r, "||a-b||/||main||=%.4f" % float((a - b).norm() / a.norm()), "||a-b||/||other||=%.4f" % float((a - b).norm() / b.norm()), "cos=%.4f" % float((a.flatten() @ b.flatten()) / (a.norm() * b.norm())))
        a, b = T["stage1-step1907359-tokens4001B"], T["stage2-ingredient3-step23852-tokens51B"]
        print(hn, "stage1 vs ing3 end rel=%.4f cos=%.4f" % (float((a - b).norm() / b.norm()), float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", action="store_true", help="compare the main revision with the stage-1 and stage-2 endpoints (Hub only)")
    ap.add_argument("--checkpoints", type=Path, default=paths.ASSETS / "checkpoints",
                    help="directory of AI2's OLMo-native checkpoints (default: %(default)s)")
    args = ap.parse_args()
    if args.main:
        compare_main()
    else:
        compare_branches(args.checkpoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
