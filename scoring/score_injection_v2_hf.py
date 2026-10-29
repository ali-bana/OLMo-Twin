#!/usr/bin/env python3
"""Per-token log-probs of the injection-v2 evaluation set under a HuggingFace model.

The post-training runs (SFT, later DPO) emit HuggingFace checkpoints, while
score_injection_v2{,_extended}.py load OLMo-native ones. This scorer is the HF
counterpart and is deliberately compatible with them (bitwise identical in fp32; per-document
means within ~5e-3 nats/token in bf16, audited 2026-10-12): same stored token
ids (nothing is re-tokenised), same non-overlapping 4,096-token chunks with no
cross-chunk context, NaN at the first position of every chunk, bf16 forward
with an fp32 log-softmax, same output schema.

That compatibility is the whole point: it lets us put the membership signal
before and after post-training on one axis. A document's score is computed the
same way at every stage, so

    delta_stage(d) = mean_logprob(d | injected model at stage)
                   - mean_logprob(d | control model at same stage)

is comparable across stages, and the within-pair contrast can be tracked
through mid-training -> SFT -> DPO.

Note on chat templates: the evaluation documents are scored as RAW TEXT, not
wrapped in the tulu chat format. That is intentional. We are asking what the
model assigns to the document itself, exactly as the pre-SFT models were asked,
so the comparison across stages is apples to apples.

Also emits mu (E[log p] over the vocabulary) and sigma (its sd), so Min-K%++
is available at every stage.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SEQ_LEN = 4096


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace(":", "_")


def score_ids(model, ids: np.ndarray, device, seq_len: int, stat_chunk: int, want_stats: bool):
    import torch

    n = len(ids)
    ids_t = torch.tensor(ids.astype(np.int64), dtype=torch.long)
    lp = np.full(n, np.nan, dtype=np.float32)
    mu = np.full(n, np.nan, dtype=np.float32)
    sd = np.full(n, np.nan, dtype=np.float32)

    for s in range(0, n, seq_len):
        e = min(s + seq_len, n)
        if e - s < 2:
            continue
        x = ids_t[s:e].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x).logits[:, :-1]
        tgt = x[:, 1:]
        L = logits.shape[1]
        out_lp = torch.empty(L, dtype=torch.float32, device=device)
        out_mu = torch.empty(L, dtype=torch.float32, device=device)
        out_sd = torch.empty(L, dtype=torch.float32, device=device)
        for a in range(0, L, stat_chunk):
            b = min(a + stat_chunk, L)
            block = logits[0, a:b].float()
            lsm = torch.log_softmax(block, dim=-1)
            out_lp[a:b] = lsm.gather(-1, tgt[0, a:b].unsqueeze(-1)).squeeze(-1)
            if want_stats:
                p = lsm.exp()
                m1 = (p * lsm).sum(-1)
                m2 = (p * lsm * lsm).sum(-1)
                out_mu[a:b] = m1
                out_sd[a:b] = (m2 - m1 * m1).clamp_min(0).sqrt()
                del p
            del block, lsm
        lp[s + 1 : e] = out_lp.cpu().numpy()
        if want_stats:
            mu[s + 1 : e] = out_mu.cpu().numpy()
            sd[s + 1 : e] = out_sd.cpu().numpy()
        del logits
    return lp, mu, sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluation-manifest", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="HF model directory")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--stat-chunk", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=65_536)
    ap.add_argument("--no-vocab-stats", action="store_true",
                    help="skip mu/sigma (faster; Min-K%%++ then unavailable at this stage)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out = args.out_dir
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    with gzip.open(args.evaluation_manifest, "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = ["doc_id", "revision", "n_tokens", "n_bytes", "nll_sum", "bpb",
              "mean_logprob", "mean_mu", "mean_sigma", "n_chunks", "status"]

    if args.merge:
        merged = []
        parts = sorted(out.glob("scores.rank*.csv.gz"))
        for p in parts:
            with gzip.open(p, "rt", newline="") as fh:
                merged += list(csv.DictReader(fh))
        with gzip.open(out / "scores.csv.gz", "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader(); w.writerows(merged)
        log(f"merged {len(merged):,} rows from {len(parts)} ranks")
        return 0

    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading HF model {args.model} on {device}")
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=torch.bfloat16)
    model.eval().to(device)
    log(f"vocab {model.config.vocab_size:,} | params {sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    want_stats = not args.no_vocab_stats
    mine = rows[args.rank :: args.world_size]
    log(f"rank {args.rank}/{args.world_size}: {len(mine):,} documents")
    part = out / f"scores.rank{args.rank:03d}.csv.gz"
    done = set()
    if part.exists():
        with gzip.open(part, "rt", newline="") as fh:
            done = {r["doc_id"] for r in csv.DictReader(fh)}
        log(f"resuming: {len(done):,} already scored")

    started, n_new = time.time(), 0
    with gzip.open(part, "at", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not done:
            w.writeheader()
        for i, r in enumerate(mine, 1):
            if r["doc_id"] in done:
                continue
            ids = np.load(args.bundle / r["tokens_relpath"]).astype(np.int32)
            rec = {"doc_id": r["doc_id"], "revision": args.revision,
                   "n_tokens": int(len(ids)), "n_bytes": r.get("n_bytes", "") or "",
                   "status": "ok"}
            if len(ids) > args.max_tokens:
                rec.update(status="too_long", nll_sum="", bpb="", mean_logprob="",
                           mean_mu="", mean_sigma="", n_chunks="")
                w.writerow(rec); continue
            lp, mu, sd = score_ids(model, ids, device, args.seq_len, args.stat_chunk, want_stats)
            valid = ~np.isnan(lp)
            nll = float(-lp[valid].sum())
            rec.update(nll_sum=round(nll, 4),
                       mean_logprob=round(float(lp[valid].mean()), 6),
                       mean_mu=round(float(mu[valid].mean()), 6) if want_stats else "",
                       mean_sigma=round(float(sd[valid].mean()), 6) if want_stats else "",
                       n_chunks=int(math.ceil(len(ids) / args.seq_len)),
                       bpb=round(nll / math.log(2) / int(rec["n_bytes"]), 6) if rec["n_bytes"] else "")
            payload = {"token_ids": ids, "logprobs": lp}
            if want_stats:
                payload.update(mu=mu, sigma=sd)
            np.savez_compressed(out / "tokens" / f"{safe(r['doc_id'])}.npz", **payload)
            w.writerow(rec); fh.flush()
            n_new += 1
            if n_new % 200 == 0:
                log(f"  {i:,}/{len(mine):,}  {n_new/(time.time()-started):.2f} docs/s")

    json.dump({"model": str(args.model), "revision": args.revision, "format": "huggingface",
               "seq_len": args.seq_len, "vocab_stats": want_stats,
               "context": "non-overlapping chunks, no cross-chunk context, raw text (no chat template)",
               "dtype": "bfloat16 forward, fp32 log_softmax",
               "rank": args.rank, "world_size": args.world_size, "documents": len(mine)},
              open(out / f"scoring_meta.rank{args.rank:03d}.json", "w"), indent=1)
    log(f"rank {args.rank} done: {n_new:,} new documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
