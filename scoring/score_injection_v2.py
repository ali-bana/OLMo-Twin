#!/usr/bin/env python3
"""Per-token log-probs of the injection-v2 evaluation set under an OLMo checkpoint.

Same conventions as scoring/score_pretreatment_logprobs.py so the
pre-treatment, control (released) and injected-model scores are directly
comparable and the existing balance / AUC tools read them unchanged:
  * input = the stored token ids (tokens/<cohort>/<doc>.npy, null_tokens/),
    which already end with EOS -- nothing is re-tokenized;
  * non-overlapping 4096-token chunks, no cross-chunk context; the first
    token of every chunk gets NaN; bf16 forward, fp32 log-softmax;
  * outputs <out>/scores.csv.gz (doc_id, revision, n_tokens, n_bytes, nll_sum,
    bpb, mean_logprob, n_chunks, status) and <out>/tokens/<doc>.npz
    (token_ids int32, logprobs float32).
  n_bytes is taken from the evaluation manifest when present (UTF-8 bytes of
  the injected text); for documents without it bpb is left empty.

The checkpoint is an OLMo (pinned commit) checkpoint directory with
model.pt/config.yaml (an unsharded save, or the official released
checkpoint converted the same way). Shard the work across processes or
GPUs with --rank/--world-size and merge with --merge.
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


def score_ids(model, ids: np.ndarray, device, seq_len: int) -> np.ndarray:
    import torch
    ids_t = torch.tensor(ids.astype(np.int64), dtype=torch.long)
    lp = np.full(len(ids), np.nan, dtype=np.float32)
    for s in range(0, len(ids), seq_len):
        e = min(s + seq_len, len(ids))
        if e - s < 2:
            continue
        x = ids_t[s:e].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x).logits.float()
        lsm = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = x[:, 1:]
        lp[s + 1:e] = lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0].cpu().numpy()
    return lp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluation-manifest", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True, help="directory the manifest's tokens_relpath are relative to")
    ap.add_argument("--checkpoint", type=Path, required=True, help="OLMo checkpoint dir (model.pt + config.yaml)")
    ap.add_argument("--revision", required=True, help="label written into scores.csv.gz, e.g. injected-final")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--max-tokens", type=int, default=65_536)
    ap.add_argument("--no-token-dump", action="store_true",
                    help="write only scores.rank*.csv.gz, not tokens/<doc>.npz. The "
                         "ladder sweep scores 22,681 documents under 69 checkpoints and "
                         "needs document-level mean log-probability only; keeping the "
                         "per-token dumps would cost ~33 GB for nothing.")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="merge per-rank score files into scores.csv.gz")
    args = ap.parse_args()
    out = args.out_dir
    if args.no_token_dump:
        out.mkdir(parents=True, exist_ok=True)
    else:
        (out / "tokens").mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    with gzip.open(args.evaluation_manifest, "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = ["doc_id", "revision", "n_tokens", "n_bytes", "nll_sum", "bpb", "mean_logprob", "n_chunks", "status"]
    if args.merge:
        merged = []
        for p in sorted(out.glob("scores.rank*.csv.gz")):
            with gzip.open(p, "rt", newline="") as fh:
                merged += list(csv.DictReader(fh))
        with gzip.open(out / "scores.csv.gz", "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(merged)
        log(f"merged {len(merged):,} rows from {len(list(out.glob('scores.rank*.csv.gz')))} ranks")
        return 0

    import torch
    from olmo.model import OLMo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading OLMo checkpoint {args.checkpoint} on {device}")
    model = OLMo.from_checkpoint(str(args.checkpoint), device="cpu")
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    mine = rows[args.rank::args.world_size]
    log(f"rank {args.rank}/{args.world_size}: {len(mine):,} documents")
    part = out / f"scores.rank{args.rank:03d}.csv.gz"
    done: set[str] = set()
    if part.exists():
        # A walltime kill (SIGTERM then SIGKILL) stops the interpreter without
        # running GzipFile.__exit__, so the final member has no trailer and a
        # plain read raises EOFError -- which would make every resubmission of
        # this task die before it even loads the model. Read what is there,
        # keep the rows that decoded, then REWRITE the file so it is a
        # well-formed gzip again before reopening it in append mode.
        kept = []
        try:
            with gzip.open(part, "rt", newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("doc_id") and r["doc_id"] != "doc_id":
                        kept.append(r)
        except (EOFError, OSError) as exc:
            log(f"partial rank file was truncated ({type(exc).__name__}); "
                f"recovered {len(kept):,} complete rows")
        done = {r["doc_id"] for r in kept}
        with gzip.open(part, "wt", newline="") as fh:
            w0 = csv.DictWriter(fh, fieldnames=fields)
            w0.writeheader()
            for r in kept:
                w0.writerow({k: r.get(k, "") for k in fields})
        log(f"resuming: {len(done):,} already scored")
    started = time.time()
    n_new = 0
    # Decide about the header BEFORE opening in append mode, which would itself
    # create the file. Gating on the file rather than on `done` matters because
    # a rank file holding only a header (a crash between writeheader and the
    # first row) would otherwise get a second header, which merges in as a
    # bogus doc_id="doc_id" row.
    need_header = not part.exists() or part.stat().st_size == 0
    with gzip.open(part, "at", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if need_header:
            w.writeheader()
        for i, r in enumerate(mine, 1):
            if r["doc_id"] in done:
                continue
            ids = np.load(args.bundle / r["tokens_relpath"]).astype(np.int32)
            rec = {"doc_id": r["doc_id"], "revision": args.revision, "n_tokens": int(len(ids)),
                   "n_bytes": r.get("n_bytes", "") or "", "status": "ok"}
            if len(ids) > args.max_tokens:
                rec.update(status="too_long", nll_sum="", bpb="", mean_logprob="", n_chunks="")
                w.writerow(rec); continue
            lp = score_ids(model, ids, device, args.seq_len)
            valid = lp[~np.isnan(lp)]
            nll = float(-valid.sum())
            rec.update(nll_sum=round(nll, 4), mean_logprob=round(float(valid.mean()), 6),
                       n_chunks=int(math.ceil(len(ids) / args.seq_len)),
                       bpb=round(nll / math.log(2) / int(rec["n_bytes"]), 6) if rec["n_bytes"] else "")
            if not args.no_token_dump:
                np.savez_compressed(out / "tokens" / f"{safe(r['doc_id'])}.npz",
                                    token_ids=ids, logprobs=lp)
            w.writerow(rec); fh.flush()
            n_new += 1
            if n_new % 200 == 0:
                rate = n_new / (time.time() - started)
                log(f"  {i:,}/{len(mine):,}  {rate:.1f} docs/s")
    json.dump({"checkpoint": str(args.checkpoint), "revision": args.revision, "seq_len": args.seq_len,
               "context": "non-overlapping chunks, no cross-chunk context", "dtype": "bfloat16",
               "token_dump": not args.no_token_dump,
               "rank": args.rank, "world_size": args.world_size, "documents": len(mine)},
              open(out / f"scoring_meta.rank{args.rank:03d}.json", "w"), indent=1)
    log(f"rank {args.rank} done: {n_new:,} new documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
