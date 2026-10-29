#!/usr/bin/env python3
"""Per-token log-probs PLUS next-token distribution statistics.

Same conventions as score_injection_v2.py (stored token ids, non-overlapping
4096-token chunks, no cross-chunk context, first token of a chunk is NaN,
bf16 forward with fp32 log-softmax) but additionally stores, for every scored
position t:

    mu[t]    = E_{z ~ p(.|x_<t)} [ log p(z|x_<t) ]      (= -entropy)
    sigma[t] = std of log p(z|x_<t) under the same p

These are exactly the normalisers Min-K%++ needs
(score_t = (logprob_t - mu_t) / sigma_t); they cannot be recovered from the
target-token log-prob alone, which is why this second pass exists. Everything
else downstream (LOSS, perplexity, bits-per-byte, classic Min-K%, zlib ratio,
reference ratios) is computable from the log-probs already stored.

The vocabulary reduction is chunked over positions so a 4096-token window
never materialises more than `--stat-chunk` rows of the |V| = 100k
distribution at once.

Outputs <out>/tokens/<doc>.npz with token_ids, logprobs, mu, sigma and
<out>/scores.rank*.csv.gz with the same columns as score_injection_v2.py.
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


def score_ids_extended(model, ids, device, seq_len: int, stat_chunk: int):
    """Return (logprobs, mu, sigma), each float32 [len(ids)] with NaN where unscored."""
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
            logits = model(x).logits[:, :-1]          # [1, L-1, V], bf16
        tgt = x[:, 1:]                                 # [1, L-1]
        L = logits.shape[1]
        out_lp = torch.empty(L, dtype=torch.float32, device=device)
        out_mu = torch.empty(L, dtype=torch.float32, device=device)
        out_sd = torch.empty(L, dtype=torch.float32, device=device)
        for a in range(0, L, stat_chunk):
            b = min(a + stat_chunk, L)
            block = logits[0, a:b].float()              # [m, V] fp32
            lsm = torch.log_softmax(block, dim=-1)
            p = lsm.exp()
            m1 = (p * lsm).sum(-1)                      # E[log p]  (= -entropy)
            m2 = (p * lsm * lsm).sum(-1)                # E[(log p)^2]
            out_mu[a:b] = m1
            out_sd[a:b] = (m2 - m1 * m1).clamp_min(0).sqrt()
            out_lp[a:b] = lsm.gather(-1, tgt[0, a:b].unsqueeze(-1)).squeeze(-1)
            del block, lsm, p
        lp[s + 1 : e] = out_lp.cpu().numpy()
        mu[s + 1 : e] = out_mu.cpu().numpy()
        sd[s + 1 : e] = out_sd.cpu().numpy()
        del logits
    return lp, mu, sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluation-manifest", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--stat-chunk", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=65_536)
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
    from olmo.model import OLMo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading {args.checkpoint} on {device}")
    model = OLMo.from_checkpoint(str(args.checkpoint), device="cpu")
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)

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
            lp, mu, sd = score_ids_extended(model, ids, device, args.seq_len, args.stat_chunk)
            valid = ~np.isnan(lp)
            nll = float(-lp[valid].sum())
            rec.update(nll_sum=round(nll, 4),
                       mean_logprob=round(float(lp[valid].mean()), 6),
                       mean_mu=round(float(mu[valid].mean()), 6),
                       mean_sigma=round(float(sd[valid].mean()), 6),
                       n_chunks=int(math.ceil(len(ids) / args.seq_len)),
                       bpb=round(nll / math.log(2) / int(rec["n_bytes"]), 6) if rec["n_bytes"] else "")
            np.savez_compressed(out / "tokens" / f"{safe(r['doc_id'])}.npz",
                                token_ids=ids, logprobs=lp, mu=mu, sigma=sd)
            w.writerow(rec); fh.flush()
            n_new += 1
            if n_new % 200 == 0:
                log(f"  {i:,}/{len(mine):,}  {n_new/(time.time()-started):.2f} docs/s")

    json.dump({"checkpoint": str(args.checkpoint), "revision": args.revision,
               "seq_len": args.seq_len, "stat_chunk": args.stat_chunk,
               "context": "non-overlapping chunks, no cross-chunk context",
               "dtype": "bfloat16 forward, fp32 log_softmax",
               "extras": "mu = E[log p] over vocab, sigma = sd of log p over vocab (Min-K%++)",
               "rank": args.rank, "world_size": args.world_size, "documents": len(mine)},
              open(out / f"scoring_meta.rank{args.rank:03d}.json", "w"), indent=1)
    log(f"rank {args.rank} done: {n_new:,} new documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
