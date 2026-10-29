#!/usr/bin/env python3
"""Per-token log-probabilities of candidate documents under a fixed OLMo 2
checkpoint -- the design-time measurement called bpb_pre (pre-treatment bits
per byte; see scoring/README.md and design/README.md).

Why per token and why now. Pairing needs bpb_pre for matched-pair
assignment BEFORE the manifest is frozen, scored in non-overlapping seq_len
chunks with no cross-chunk context, exactly as training sees a long document.
Per-token values are stored rather than per-document aggregates because
chunks and aggregates are recomputable from tokens and not the other way
round; a question asked later (where in the document does the signal live?)
must not require re-running the model.

Checkpoints. --revision selects the HF branch. The pre-treatment model is the
final stage-1 checkpoint, `stage1-step1907359-tokens4001B` (the numerically
last of 195 stage-1 branches -- NOT `stage1-step990000`, which is merely the
lexically last). `main` is the released model, scored as a second reference.

Tokenization. text -> tokenizer(add_special_tokens=False) + [eos], mirroring
how Dolma concatenates documents with an EOS separator. Scoring is teacher-
forced: logprob[i] is the model's log-probability of token i given tokens
0..i-1 within its chunk; the first token of every chunk has no prediction and
is recorded as NaN.

Outputs. <out>/<doc_id>.npz with token_ids (int32) and logprobs (float32),
plus <out>/scores.csv.gz with n_tokens, n_bytes, nll_sum, bpb, mean_logprob
and the checkpoint used. Deterministic given the checkpoint; bf16 on GPU.
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
import torch

MODEL = "allenai/OLMo-2-0425-1B"
PRETREATMENT_REVISION = "stage1-step1907359-tokens4001B"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_manifest(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(gzip.open(path, "rt", newline="")))
    if not rows or "doc_id" not in rows[0] or "raw_text_path" not in rows[0]:
        raise SystemExit("manifest must have doc_id and raw_text_path columns")
    return rows


@torch.inference_mode()
def score_document(model, tok, text: str, seq_len: int, device) -> tuple[np.ndarray, np.ndarray]:
    ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids_t = torch.tensor(ids, dtype=torch.long)
    logprobs = np.full(len(ids), np.nan, dtype=np.float32)
    # non-overlapping chunks, no cross-chunk context (see scoring/README.md)
    chunks = [(s, min(s + seq_len, len(ids))) for s in range(0, len(ids), seq_len)]
    # batch chunks of equal length together for throughput
    for s, e in chunks:
        x = ids_t[s:e].unsqueeze(0).to(device)
        logits = model(x).logits.float()                       # [1, L, V]
        lp = torch.log_softmax(logits[:, :-1], dim=-1)          # predicts tokens 1..L-1
        tgt = x[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        logprobs[s + 1:e] = tok_lp[0].cpu().numpy()
    return np.asarray(ids, dtype=np.int32), logprobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="csv.gz with doc_id and raw_text_path (Stage G output joined with raw paths)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--revision", default=PRETREATMENT_REVISION)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=65536,
                    help="skip documents longer than this after tokenization")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = args.out_dir / "tokens"
    npz_dir.mkdir(exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {MODEL}@{args.revision} on {device}")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=args.revision, torch_dtype=torch.bfloat16
    ).to(device).eval()
    if tok.eos_token_id is None:
        raise SystemExit("tokenizer has no eos token")
    log(f"eos_token_id={tok.eos_token_id}  vocab={len(tok)}")

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    log(f"documents to score: {len(rows):,}")

    out_csv = args.out_dir / "scores.csv.gz"
    done: set[str] = set()
    if out_csv.exists():
        for r in csv.DictReader(gzip.open(out_csv, "rt", newline="")):
            done.add(r["doc_id"])
        log(f"resuming: {len(done):,} already scored")
    fields = ["doc_id", "revision", "n_tokens", "n_bytes", "nll_sum", "bpb",
              "mean_logprob", "n_chunks", "status"]
    mode = "at" if out_csv.exists() else "wt"
    started = time.time()
    n_ok = 0
    with gzip.open(out_csv, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if mode == "wt":
            w.writeheader()
        for i, r in enumerate(rows, 1):
            if r["doc_id"] in done:
                continue
            rec = {"doc_id": r["doc_id"], "revision": args.revision, "status": "ok"}
            try:
                text = Path(r["raw_text_path"]).read_text(encoding="utf-8")
                n_bytes = len(text.encode("utf-8"))
                ids_probe = tok(text, add_special_tokens=False)["input_ids"]
                if len(ids_probe) + 1 > args.max_tokens:
                    rec.update(status="too_long", n_tokens=len(ids_probe) + 1, n_bytes=n_bytes)
                    w.writerow(rec)
                    continue
                ids, lp = score_document(model, tok, text, args.seq_len, device)
                valid = lp[~np.isnan(lp)]
                nll = float(-valid.sum())
                rec.update(
                    n_tokens=int(len(ids)), n_bytes=n_bytes, nll_sum=round(nll, 4),
                    bpb=round(nll / math.log(2) / max(n_bytes, 1), 6),
                    mean_logprob=round(float(valid.mean()), 6),
                    n_chunks=math.ceil(len(ids) / args.seq_len),
                )
                safe = r["doc_id"].replace("/", "_").replace(":", "_")
                np.savez_compressed(npz_dir / f"{safe}.npz", token_ids=ids, logprobs=lp)
                n_ok += 1
            except Exception as exc:
                rec.update(status=f"error_{type(exc).__name__}")
            w.writerow(rec)
            if i % 200 == 0:
                rate = i / max(time.time() - started, 1e-9)
                log(f"  {i:,}/{len(rows):,}  {rate:.1f} docs/s  eta {(len(rows)-i)/max(rate,1e-9)/60:.0f} min")
    (args.out_dir / "scoring_meta.json").write_text(json.dumps({
        "model": MODEL, "revision": args.revision, "seq_len": args.seq_len,
        "dtype": "bfloat16", "context": "non-overlapping chunks, no cross-chunk context",
        "tokenization": "add_special_tokens=False, trailing eos appended",
        "documents_scored": n_ok, "manifest": str(args.manifest),
    }, indent=1))
    log(f"scored {n_ok:,} documents -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
