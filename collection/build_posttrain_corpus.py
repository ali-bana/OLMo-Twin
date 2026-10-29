#!/usr/bin/env python3
"""OLMo 2 1B post-training datasets -> scannable Dolma-style shards.

The four post-training stages of `allenai/OLMo-2-0425-1B-Instruct` each train
on one public dataset:

  sft    allenai/tulu-3-sft-olmo-2-mixture-0225        866,138 examples
  dpo    allenai/olmo-2-0425-1b-preference-mix         378,301 pairs
  rlvr1  allenai/RLVR-GSM-MATH-IF-Mixed-Constraints     29,946 examples
  rlvr2  allenai/RLVR-MATH                               7,500 examples

They are NOT part of the base model's pre-training or mid-training, so a
candidate found here is not a member of the model this experiment injects
into. It would become one for the SFT/DPO/Instruct variants, so the audit
records it separately.

Conversion rules (conservative for a membership audit):
  * one output record per text sequence the model was actually trained on:
    SFT one per example, DPO TWO per example (chosen and rejected -- the DPO
    loss consumes both), RLVR one per example;
  * `text` is every message's `content` joined with a blank line, plus
    `ground_truth` and `constraint` where the schema has them. A shared span
    inside any one message therefore survives in the record, so the matcher
    cannot miss it; spans that only appear ACROSS a join are false positives,
    which is the safe direction;
  * `id` carries the dataset's own id (or `<dataset>-<row>`), so a hit points
    back at the exact training example.

Output: <out>/data/{sft,dpo,rlvr1,rlvr2}/part-*.jsonl.gz, a `manifest.json`
with per-component record/byte counts and the pinned HF revisions, and the
same layout `olmo2_nonmember_audit.py scan --corpus ... --components ...`
already reads for the two pre-training mixes.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

# dataset directory name -> (component, pinned revision)
SOURCES = {
    "tulu-3-sft-olmo-2-mixture-0225": ("sft", "d91a0785ade02942520280fb484866fce41e448f"),
    "olmo-2-0425-1b-preference-mix": ("dpo", "c6454f3d364622718a8ecd3d307e524febbba569"),
    "RLVR-GSM-MATH-IF-Mixed-Constraints": ("rlvr1", "7dbd180f5440c0b90f2944e6efea934b85437a95"),
    "RLVR-MATH": ("rlvr2", "bd2a93551b503a395fadd1a740d957559cfe6f3c"),
}
SHARDS = {"sft": 12, "dpo": 12, "rlvr1": 2, "rlvr2": 1}
JOIN = "\n\n"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def messages_text(messages: Any) -> str:
    """Join every message's content. Non-string or empty contents are skipped."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str) and c.strip():
            parts.append(c)
    return JOIN.join(parts)


def records_from_row(component: str, row: dict, index: int) -> Iterator[dict]:
    """One record per trained-on sequence (see module docstring)."""
    base_id = row.get("id") or f"{component}-{index}"
    if component == "dpo":
        for side in ("chosen", "rejected"):
            text = messages_text(row.get(side))
            if text:
                yield {"id": f"{base_id}#{side}", "text": text,
                       "metadata": {"source": row.get("source"), "side": side,
                                    "model": row.get(f"{side}_model")}}
        return
    text = messages_text(row.get("messages"))
    extra = [row.get(k) for k in ("ground_truth", "constraint") if isinstance(row.get(k), str) and row.get(k).strip()]
    if extra:
        text = JOIN.join([text, *extra]) if text else JOIN.join(extra)
    if text:
        meta = {k: row.get(k) for k in ("source", "dataset", "constraint_type") if row.get(k)}
        yield {"id": str(base_id), "text": text, "metadata": meta}


def convert(raw_root: Path, out_root: Path, only: list[str] | None) -> dict:
    out_data = out_root / "data"
    components = {}
    for name, (component, revision) in SOURCES.items():
        if only and component not in only:
            continue
        files = sorted(glob.glob(str(raw_root / name / "**" / "*.parquet"), recursive=True))
        if not files:
            raise SystemExit(f"no parquet files for {name} under {raw_root}")
        comp_dir = out_data / component
        comp_dir.mkdir(parents=True, exist_ok=True)
        n_shards = SHARDS[component]
        handles = [gzip.open(comp_dir / f"part-{i:03d}-of-{n_shards:03d}.jsonl.gz", "wt", compresslevel=4)
                   for i in range(n_shards)]
        n_rows = n_records = n_chars = 0
        try:
            for f in files:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=512):
                    for row in batch.to_pylist():
                        for rec in records_from_row(component, row, n_rows):
                            handles[n_records % n_shards].write(json.dumps(rec, ensure_ascii=False) + "\n")
                            n_records += 1
                            n_chars += len(rec["text"])
                        n_rows += 1
                log(f"  {component}: {os.path.basename(f)} done ({n_rows:,} rows, {n_records:,} records)")
        finally:
            for h in handles:
                h.close()
        shards = sorted(comp_dir.glob("part-*.jsonl.gz"))
        components[component] = {
            "dataset": f"allenai/{name}", "revision": revision, "rows": n_rows, "records": n_records,
            "text_chars": n_chars, "shards": len(shards),
            "compressed_bytes": sum(p.stat().st_size for p in shards),
            "shard_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shards},
        }
        log(f"{component}: {n_rows:,} rows -> {n_records:,} records, {n_chars / 1e9:.2f} G chars, "
            f"{components[component]['compressed_bytes'] / 2**20:.0f} MiB in {len(shards)} shards")
    return components


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, required=True, help="directory holding the downloaded HF snapshots")
    ap.add_argument("--out-root", type=Path, required=True, help="corpus root: <out>/data/<component>/*.jsonl.gz")
    ap.add_argument("--components", nargs="*", default=None, help="subset of sft dpo rlvr1 rlvr2")
    args = ap.parse_args()
    components = convert(args.raw_root, args.out_root, args.components)
    manifest_path = args.out_root / "manifest.json"
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": "allenai/OLMo-2-0425-1B-Instruct",
                "note": "post-training stages; not part of the base model's pre/mid-training",
                "join": "blank line between message contents; one record per trained sequence "
                        "(DPO: chosen and rejected separately)",
                "components": components}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        previous["components"].update(components)
        manifest["components"] = previous["components"]
    manifest_path.write_text(json.dumps(manifest, indent=1))
    log(f"wrote {manifest_path}: " + ", ".join(f"{k} {v['records']:,}" for k, v in manifest["components"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
