#!/usr/bin/env python3
"""Injection set v1 -> the OLMo 2 1B stage-2 training stream (v2 mechanics).

Reproduces the official seed-42 stage-2 run exactly, except that chosen
training positions are replaced by the injected documents. The physical
memmap layout is arranged so the UNMODIFIED OLMo trainer (pinned commit,
official YAML, data.paths swapped) consumes the intended stream:

  * OLMo's IterableDataset shuffles arange(N) once with PCG64(seed=42) and
    global batch k is positions [512k, 512k+512) of that order, whatever the
    rank interleaving. With N = target_updates * 512 physical chunks, position
    p reads physical slot train_order[p] = shuffle(arange(N))[p].
  * The official run over the 576 official memmaps consumed, at position p,
    official chunk baseline_global[p] = shuffle(arange(N_official))[p].
  * We write physical slot train_order[p] := official chunk baseline_global[p]
    for every untouched position, and := the injected chunk for replaced ones.

Per-copy placement (new in v2). Every injected COPY (a document x dose r
gives r copies) has a target training position from copies.csv.gz
(`position_frac` in [0.01, 0.99], decile-balanced per cohort). A copy of n
tokens (incl. EOS) occupies ceil(n/4096) CONSECUTIVE positions starting at
the first free position >= floor(position_frac * N); two copies of one
document never share a global batch. The last partial chunk is completed with
the leading tokens of the official chunk it replaces, so the token budget is
exactly the official one and every background chunk is either kept verbatim
or replaced whole.

Subcommands
  plan          resolve copies -> positions/slots; write the
                bundle: stream_plan/placement.jsonl.gz, placement_summary.json,
                null_tokens/, bundle SHA256SUMS. Needs only the injection set.
  build         (CPU) materialize the shards from the official
                memmap manifest + bundle; resumable per shard.
  verify        shard hashes, background chunks vs official
                (sampled), every injected chunk vs bundle tokens, config diff.
  make-config   custom YAML: only data.paths and run identity/checkpointing
                keys differ from the official config (asserted).
  make-eval-manifest  documents to score after training (both arms, nulls,
                edited cohorts), token files relative to the bundle.
  simulate      (tests) read the built shards the way the OLMo loader does and
                return the chunk at a training position.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OLMO_COMMIT = "090253dac6688f2532509daa7aa2eb5fae50e956"
OFFICIAL_CONFIG_SHA256 = "095dabb7b20c9b6710abad3717a789c9d3109e070069b8643805bb407d2a2fd1"
OFFICIAL_CONFIG_RELATIVE = "configs/official-0425/OLMo2-1B-stage2-seed42.yaml"
SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 512
DEFAULT_TARGET_UPDATES = 23_852          # official stop_at
LOADER_SEED = 42
EOS_TOKEN_ID = 100_257
VOCAB_SIZE = 100_278
MEMMAP_DTYPE = np.dtype("<u4")
DEFAULT_OUTPUT_SHARDS = 64
PLAN_SCHEMA_VERSION = 2
STREAM_SCHEMA_VERSION = 2
ALLOWED_CONFIG_CHANGES = {
    "run_name", "wandb.name", "wandb.project", "wandb.group", "save_folder",
    "save_interval", "save_interval_ephemeral", "save_num_checkpoints_to_keep",
    "save_interval_unsharded", "data.paths",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def rd_csv(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with gzip.open(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def rd_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def safe(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace(":", "_")


def train_order(n: int, seed: int = LOADER_SEED) -> np.ndarray:
    """Physical slot consumed at each training position by OLMo's loader."""
    order = np.arange(n, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed)).shuffle(order)
    return order


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def allocate_positions(copies: list[dict], n_chunks_of: dict[str, int], target_chunks: int,
                       trim: float = 0.0) -> list[dict]:
    """Give every copy ceil(n/4096) consecutive positions at/after its target.
    Copies are processed in target order; a copy is shifted right past any
    occupied position and past any batch already holding another copy of the
    same document. Returns the copies with 'positions' filled in."""
    lo, hi = int(math.ceil(trim * target_chunks)), int(math.floor((1 - trim) * target_chunks))
    occupied: list[tuple[int, int]] = []          # sorted disjoint [start, end)
    steps_of_doc: dict[str, set[int]] = defaultdict(set)
    out = []
    for c in sorted(copies, key=lambda c: (float(c["position_frac"]), c["doc_id"], int(c["copy"]))):
        n = n_chunks_of[c["doc_id"]]
        p = max(lo, int(math.floor(float(c["position_frac"]) * target_chunks)))
        while True:
            # skip occupied intervals
            i = bisect.bisect_right(occupied, (p, 10**18)) - 1
            if i >= 0 and occupied[i][1] > p:
                p = occupied[i][1]
                continue
            if i + 1 < len(occupied) and occupied[i + 1][0] < p + n:
                p = occupied[i + 1][1]
                continue
            steps = {q // GLOBAL_BATCH_SIZE for q in range(p, p + n)}
            clash = steps & steps_of_doc[c["doc_id"]]
            if clash:
                p = (max(clash) + 1) * GLOBAL_BATCH_SIZE
                continue
            break
        if p + n > hi:
            raise ValueError(f"no room for {c['doc_id']} copy {c['copy']} before the trimmed tail")
        bisect.insort(occupied, (p, p + n))
        steps_of_doc[c["doc_id"]] |= {q // GLOBAL_BATCH_SIZE for q in range(p, p + n)}
        out.append({**c, "positions": list(range(p, p + n)), "shift": p - int(math.floor(float(c["position_frac"]) * target_chunks))})
    return out


def cmd_plan(args: argparse.Namespace) -> int:
    I = args.injection_dir
    out = args.out_dir
    (out / "stream_plan").mkdir(parents=True, exist_ok=True)
    target_chunks = args.target_updates * GLOBAL_BATCH_SIZE
    man = rd_csv(I / "manifest.csv.gz")
    copies = rd_csv(I / "copies.csv.gz")
    inject = {r["doc_id"]: r for r in man if r["arm"] == "inject"}
    # tokens per injected document (stored ids include EOS)
    n_tokens, n_chunks, tok_sha = {}, {}, {}
    for d, r in inject.items():
        ids = np.load(I / os.path.relpath(r["tokens_path"], I) if os.path.isabs(r["tokens_path"]) else I / r["tokens_path"])
        ids = np.asarray(ids)
        if ids.dtype != np.int32 and ids.dtype != np.uint32:
            raise ValueError(f"{d}: token dtype {ids.dtype}")
        if ids.min() < 0 or ids.max() >= VOCAB_SIZE or ids[-1] != EOS_TOKEN_ID:
            raise ValueError(f"{d}: token ids out of vocabulary or missing EOS")
        n_tokens[d] = int(len(ids))
        n_chunks[d] = int(math.ceil(len(ids) / SEQUENCE_LENGTH))
        tok_sha[d] = hashlib.sha256(ids.astype(MEMMAP_DTYPE).tobytes()).hexdigest()
    cc = Counter(c["doc_id"] for c in copies)
    for d, r in inject.items():
        if cc[d] != int(r["r"]):
            raise ValueError(f"{d}: {cc[d]} copies but r = {r['r']}")
    placed = allocate_positions(copies, n_chunks, target_chunks, trim=args.trim)
    order = train_order(target_chunks)
    rows = []
    for c in placed:
        r = inject[c["doc_id"]]
        pos = c["positions"]
        rows.append({
            "doc_id": c["doc_id"], "copy": int(c["copy"]), "cohort": r["cohort"], "stratum": r["stratum"],
            "pair_id": r["pair_id"], "r": int(r["r"]), "burst": int(c["burst"]), "decile": int(c["decile"]),
            "position_frac": float(c["position_frac"]), "shift_positions": int(c["shift"]),
            "n_tokens_with_eos": n_tokens[c["doc_id"]], "n_chunks": n_chunks[c["doc_id"]],
            "tail_tokens": n_tokens[c["doc_id"]] - (n_chunks[c["doc_id"]] - 1) * SEQUENCE_LENGTH,
            "training_positions": pos, "physical_slots": [int(order[p]) for p in pos],
            "first_update": pos[0] // GLOBAL_BATCH_SIZE + 1, "last_update": pos[-1] // GLOBAL_BATCH_SIZE + 1,
            "tokens_relpath": os.path.relpath(r["tokens_path"], I) if os.path.isabs(r["tokens_path"]) else r["tokens_path"],
            "tokens_sha256": tok_sha[c["doc_id"]],
        })
    rows.sort(key=lambda x: x["training_positions"][0])
    # invariants
    allpos = [p for x in rows for p in x["training_positions"]]
    if len(allpos) != len(set(allpos)):
        raise AssertionError("overlapping positions")
    by_doc_steps: dict[str, list[int]] = defaultdict(list)
    for x in rows:
        by_doc_steps[x["doc_id"]] += list(range(x["first_update"], x["last_update"] + 1))
    if any(len(v) != len(set(v)) for v in by_doc_steps.values()):
        raise AssertionError("a document has two copies in one update")
    # mtime=0: the gzip header must not carry a timestamp, or the file's SHA-256
    # (pinned in versions.env and checked on the cluster) changes on every
    # regeneration even when the placement is identical.
    payload = "".join(json.dumps(x) + "\n" for x in rows).encode()
    with open(out / "stream_plan" / "placement.jsonl.gz", "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
            fh.write(payload)
    # null pool + holdout tokens for post-training scoring live in the bundle too
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    nt_dir = out / "null_tokens"
    nt_dir.mkdir(exist_ok=True)
    null = rd_csv(I / "null_pool.csv.gz")
    n_null = 0
    for r in null:
        p = nt_dir / f"{safe(r['doc_id'])}.npy"
        if p.exists():
            n_null += 1
            continue
        text = Path(r["text_path"]).read_text(encoding="utf-8")
        ids = np.asarray(tok(text, add_special_tokens=False)["input_ids"] + [EOS_TOKEN_ID], dtype=np.int32)
        np.save(p, ids)
        n_null += 1
    inj_chunks = sum(x["n_chunks"] for x in rows)
    summary = {
        "schema_version": PLAN_SCHEMA_VERSION, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "olmo_commit": OLMO_COMMIT, "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "loader_seed": LOADER_SEED, "sequence_length": SEQUENCE_LENGTH, "global_batch_size": GLOBAL_BATCH_SIZE,
        "target_updates": args.target_updates, "target_chunks": target_chunks,
        "target_tokens": target_chunks * SEQUENCE_LENGTH, "trim": args.trim,
        "copies": len(rows), "documents": len(inject), "injected_chunks": inj_chunks,
        "injected_tokens_with_eos": sum(x["n_tokens_with_eos"] for x in rows),
        "share_of_chunks": inj_chunks / target_chunks,
        "first_update": min(x["first_update"] for x in rows), "last_update": max(x["last_update"] for x in rows),
        "max_shift_positions": max(x["shift_positions"] for x in rows),
        "copies_shifted": sum(1 for x in rows if x["shift_positions"]),
        "chunks_by_cohort": dict(sorted(Counter({}).items())),
        "manifest_sha256": sha256_file(I / "manifest.csv.gz"), "copies_sha256": sha256_file(I / "copies.csv.gz"),
        "null_pool_sha256": sha256_file(I / "null_pool.csv.gz"), "null_documents_tokenized": n_null,
        "tokenizer": args.model,
    }
    cbc: Counter = Counter()
    for x in rows:
        cbc[x["cohort"]] += x["n_chunks"]
    summary["chunks_by_cohort"] = dict(sorted(cbc.items()))
    atomic_json(out / "stream_plan" / "placement_summary.json", summary)
    log(json.dumps({k: v for k, v in summary.items() if k not in ("chunks_by_cohort",)}, indent=1))
    log(f"chunks by cohort: {summary['chunks_by_cohort']}")
    return 0


# ---------------------------------------------------------------------------
# official memmaps
# ---------------------------------------------------------------------------
def official_inventory(manifest_path: Path) -> tuple[list[dict], int]:
    m = json.loads(manifest_path.read_text())
    if m.get("olmo_commit") != OLMO_COMMIT or m.get("official_config_sha256") != OFFICIAL_CONFIG_SHA256:
        raise ValueError("official data manifest records another OLMo commit/config")
    entries, offset = [], 0
    for i, raw in enumerate(m["entries"]):
        if int(raw["index"]) != i:
            raise ValueError("manifest indices are not sequential")
        p = Path(raw["local_path"])
        if raw.get("status") != "complete" or not p.is_file() or p.stat().st_size != int(raw["bytes"]):
            raise ValueError(f"official shard missing/incomplete: {p}")
        tokens = int(raw["tokens"])
        if tokens * 4 != p.stat().st_size:
            raise ValueError(f"token count inconsistent: {p}")
        chunks = tokens // SEQUENCE_LENGTH
        entries.append({**raw, "path": p, "chunks": chunks, "chunk_start": offset, "chunk_end": offset + chunks})
        offset += chunks
    return entries, offset


def map_global_chunk(g: int, entries: list[dict], ends: list[int]) -> tuple[int, int]:
    i = bisect.bisect_right(ends, g)
    return i, g - entries[i]["chunk_start"]


def read_official_chunk(entries: list[dict], ends: list[int], g: int) -> np.ndarray:
    i, local = map_global_chunk(g, entries, ends)
    e = entries[i]
    mm = np.memmap(e["path"], dtype=MEMMAP_DTYPE, mode="r", shape=(e["chunks"], SEQUENCE_LENGTH))
    v = np.asarray(mm[local]).copy()
    del mm
    return v


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def injected_chunks_for(row: dict, ids: np.ndarray, background_last: np.ndarray) -> np.ndarray:
    """(n_chunks, 4096) matrix: the document's tokens, last chunk completed
    with the leading tokens of the official chunk it replaces."""
    n = row["n_chunks"]
    buf = np.empty(n * SEQUENCE_LENGTH, dtype=MEMMAP_DTYPE)
    buf[: len(ids)] = ids.astype(MEMMAP_DTYPE)
    filler = n * SEQUENCE_LENGTH - len(ids)
    if filler:
        buf[len(ids):] = background_last[:filler]
    return buf.reshape(n, SEQUENCE_LENGTH)


def cmd_build(args: argparse.Namespace) -> int:
    bundle = args.bundle
    plan = rd_jsonl(bundle / "stream_plan" / "placement.jsonl.gz")
    summary = json.loads((bundle / "stream_plan" / "placement_summary.json").read_text())
    target_chunks = int(summary["target_chunks"])
    entries, n_official = official_inventory(args.official_data_manifest)
    ends = [e["chunk_end"] for e in entries]
    if target_chunks > n_official:
        raise ValueError(f"official data has {n_official:,} chunks < target {target_chunks:,}")
    log(f"official: {len(entries)} shards, {n_official:,} chunks; target {target_chunks:,} chunks; plan {len(plan):,} copies")
    order = train_order(target_chunks)                            # position -> physical slot
    baseline = train_order(n_official)[:target_chunks]            # position -> official chunk
    slot_to_position = np.empty(target_chunks, dtype=np.uint32)
    slot_to_position[order] = np.arange(target_chunks, dtype=np.uint32)
    # injected slots
    slot_payload: dict[int, tuple[int, int]] = {}                 # physical slot -> (copy index, chunk index)
    for ci, row in enumerate(plan):
        for k, (p, s) in enumerate(zip(row["training_positions"], row["physical_slots"])):
            if int(order[p]) != s:
                raise ValueError(f"plan slot mismatch at position {p}")
            slot_payload[s] = (ci, k)
    tokens_cache: dict[str, np.ndarray] = {}

    def doc_tokens(row: dict) -> np.ndarray:
        d = row["doc_id"]
        if d not in tokens_cache:
            ids = np.load(bundle / row["tokens_relpath"])
            if hashlib.sha256(np.asarray(ids).astype(MEMMAP_DTYPE).tobytes()).hexdigest() != row["tokens_sha256"]:
                raise ValueError(f"token file changed: {d}")
            tokens_cache[d] = np.asarray(ids)
        return tokens_cache[d]

    out_dir = args.output_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = args.output_shards
    if target_chunks % shards:
        raise ValueError("target chunks not divisible by output shards")
    per = target_chunks // shards
    records = []
    for si in range(shards):
        first, last = si * per, (si + 1) * per
        path = out_dir / f"train-{si:05d}-of-{shards:05d}.npy"
        marker = path.with_suffix(".npy.json")
        expected_bytes = per * SEQUENCE_LENGTH * 4
        if path.is_file() and marker.is_file():
            mk = json.loads(marker.read_text())
            if path.stat().st_size == expected_bytes and mk.get("sha256") == sha256_file(path) and mk.get("chunk_start") == first:
                records.append(mk); log(f"shard {si + 1}/{shards} verified (existing)"); continue
            raise ValueError(f"inconsistent existing shard {path}")
        partial = path.with_suffix(".npy.partial")
        partial.unlink(missing_ok=True)
        outm = np.memmap(partial, dtype=MEMMAP_DTYPE, mode="w+", shape=(per, SEQUENCE_LENGTH))
        # background: group by official entry for locality
        slots = np.arange(first, last)
        positions = slot_to_position[slots]
        globals_ = baseline[positions]
        by_entry: dict[int, list[tuple[int, int]]] = defaultdict(list)
        n_inj = 0
        for s, g in zip(slots, globals_):
            if int(s) in slot_payload:
                n_inj += 1
                continue
            i, local = map_global_chunk(int(g), entries, ends)
            by_entry[i].append((int(s) - first, local))
        for i, pairs in by_entry.items():
            e = entries[i]
            src = np.memmap(e["path"], dtype=MEMMAP_DTYPE, mode="r", shape=(e["chunks"], SEQUENCE_LENGTH))
            pairs.sort(key=lambda t: t[1])
            for b in range(0, len(pairs), 1024):
                batch = pairs[b:b + 1024]
                outm[np.asarray([o for o, _ in batch])] = src[np.asarray([l for _, l in batch])]
            del src
        # injected
        for s in slots:
            s = int(s)
            if s not in slot_payload:
                continue
            ci, k = slot_payload[s]
            row = plan[ci]
            ids = doc_tokens(row)
            bg_last = read_official_chunk(entries, ends, int(baseline[row["training_positions"][-1]]))
            mat = injected_chunks_for(row, ids, bg_last)
            outm[s - first] = mat[k]
        outm.flush(); del outm
        if partial.stat().st_size != expected_bytes:
            raise ValueError("partial shard has wrong size")
        os.replace(partial, path)
        mk = {"path": path.name, "bytes": expected_bytes, "sha256": sha256_file(path), "chunk_start": first,
              "chunk_end": last, "chunks": per, "tokens": per * SEQUENCE_LENGTH, "injected_chunks": n_inj}
        atomic_json(marker, mk)
        records.append(mk)
        log(f"shard {si + 1}/{shards} written ({n_inj} injected chunks)")
    stream = {
        "schema_version": STREAM_SCHEMA_VERSION, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok", "olmo_commit": OLMO_COMMIT, "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "official_data_manifest_sha256": sha256_file(args.official_data_manifest),
        "official_total_chunks": n_official, "target_updates": int(summary["target_updates"]),
        "target_chunks": target_chunks, "target_tokens": target_chunks * SEQUENCE_LENGTH,
        "placement_sha256": sha256_file(bundle / "stream_plan" / "placement.jsonl.gz"),
        "placement_summary_sha256": sha256_file(bundle / "stream_plan" / "placement_summary.json"),
        "copies": len(plan), "injected_chunks": len(slot_payload),
        "official_chunks_replaced": len(slot_payload),
        "background_chunks": target_chunks - len(slot_payload),
        "shards": records, "sequence_length": SEQUENCE_LENGTH, "global_batch_size": GLOBAL_BATCH_SIZE,
        "loader_seed": LOADER_SEED, "dtype": "uint32", "eos_token_id": EOS_TOKEN_ID,
    }
    atomic_json(args.output_dir / "stream_manifest.json", stream)
    log(f"stream written: {shards} shards, {stream['target_tokens']:,} tokens, {len(slot_payload):,} injected chunks")
    return 0


# ---------------------------------------------------------------------------
# verify / simulate
# ---------------------------------------------------------------------------
class StreamReader:
    def __init__(self, training_dir: Path, stream: dict):
        self.per = int(stream["shards"][0]["chunks"])
        self.maps = {}
        self.dir = training_dir
        self.stream = stream
        self.order = train_order(int(stream["target_chunks"]))

    def slot(self, s: int) -> np.ndarray:
        si, local = divmod(int(s), self.per)
        if si not in self.maps:
            sh = self.stream["shards"][si]
            self.maps[si] = np.memmap(self.dir / "data" / sh["path"], dtype=MEMMAP_DTYPE, mode="r",
                                      shape=(int(sh["chunks"]), SEQUENCE_LENGTH))
        return np.asarray(self.maps[si][local]).copy()

    def at_position(self, p: int) -> np.ndarray:
        """What the OLMo loader feeds at training position p."""
        return self.slot(int(self.order[p]))


def cmd_verify(args: argparse.Namespace) -> int:
    errors: list[str] = []
    bundle = args.bundle
    stream = json.loads((args.training_dir / "stream_manifest.json").read_text())
    plan = rd_jsonl(bundle / "stream_plan" / "placement.jsonl.gz")
    summary = json.loads((bundle / "stream_plan" / "placement_summary.json").read_text())
    if stream.get("status") != "ok":
        errors.append("stream status not ok")
    if stream.get("placement_sha256") != sha256_file(bundle / "stream_plan" / "placement.jsonl.gz"):
        errors.append("stream was built from another placement")
    if int(stream["target_tokens"]) != int(summary["target_chunks"]) * SEQUENCE_LENGTH:
        errors.append("target tokens differ from plan")
    total = 0
    for sh in stream["shards"]:
        p = args.training_dir / "data" / sh["path"]
        if not p.is_file() or p.stat().st_size != int(sh["bytes"]):
            errors.append(f"shard missing/size: {p}"); continue
        if not args.skip_hashes and sha256_file(p) != sh["sha256"]:
            errors.append(f"shard hash changed: {p}")
        total += int(sh["tokens"])
    if total != int(stream["target_tokens"]):
        errors.append(f"stream exposes {total} tokens, expected {stream['target_tokens']}")
    reader = StreamReader(args.training_dir, stream)
    entries, n_official = official_inventory(args.official_data_manifest)
    ends = [e["chunk_end"] for e in entries]
    baseline = train_order(n_official)[: int(stream["target_chunks"])]
    injected_positions = {p for row in plan for p in row["training_positions"]}
    # background sample
    rng = np.random.default_rng(args.seed)
    candidates = rng.choice(int(stream["target_chunks"]), size=min(args.background_samples * 2, int(stream["target_chunks"])), replace=False)
    checked = mism = 0
    for p in candidates:
        p = int(p)
        if p in injected_positions:
            continue
        if not np.array_equal(reader.at_position(p), read_official_chunk(entries, ends, int(baseline[p]))):
            mism += 1
        checked += 1
        if checked >= args.background_samples:
            break
    if mism:
        errors.append(f"{mism} of {checked} sampled background chunks differ from the official stream")
    # every injected chunk
    inj_checked = inj_bad = 0
    for row in plan:
        ids = np.load(bundle / row["tokens_relpath"])
        if hashlib.sha256(np.asarray(ids).astype(MEMMAP_DTYPE).tobytes()).hexdigest() != row["tokens_sha256"]:
            errors.append(f"bundle token file changed: {row['doc_id']}"); continue
        bg_last = read_official_chunk(entries, ends, int(baseline[row["training_positions"][-1]]))
        mat = injected_chunks_for(row, np.asarray(ids), bg_last)
        for k, p in enumerate(row["training_positions"]):
            inj_checked += 1
            if not np.array_equal(reader.at_position(p), mat[k]):
                inj_bad += 1
    if inj_bad:
        errors.append(f"{inj_bad} of {inj_checked} injected chunks do not match the bundle")
    # config
    if args.custom_config:
        errors += config_diff_errors(args.official_config, args.custom_config, args.training_dir, stream)
    report = {
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "ok" if not errors else "error",
        "errors": errors, "background_chunks_checked": checked, "background_mismatches": mism,
        "injected_chunks_checked": inj_checked, "injected_mismatches": inj_bad,
        "stream_manifest_sha256": sha256_file(args.training_dir / "stream_manifest.json"),
        "placement_sha256": sha256_file(bundle / "stream_plan" / "placement.jsonl.gz"),
        "olmo_commit": OLMO_COMMIT, "official_config_sha256": OFFICIAL_CONFIG_SHA256,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=1))
    return 0 if not errors else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    stream = json.loads((args.training_dir / "stream_manifest.json").read_text())
    reader = StreamReader(args.training_dir, stream)
    v = reader.at_position(args.position)
    print(v[: args.n].tolist())
    return 0


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def make_custom_config(official: dict, data_paths: list[str], run_name: str, wandb_group: str) -> tuple[dict, dict]:
    custom = copy.deepcopy(official)
    custom["run_name"] = run_name
    custom["wandb"] = {"name": "${run_name}", "project": "olmo2-membership-injection", "group": wandb_group}
    custom["save_folder"] = "./runs/${run_name}"
    # Unsharded every 1000 mirrors the official ingredient runs exactly (the
    # published per-ingredient ladders are step 1000, 2000, ..., 23000, 23852),
    # so every kept checkpoint has an official counterpart. Sharded/ephemeral
    # saves are denser purely for crash recovery and leave no end-state files.
    custom["save_interval"] = 500
    custom["save_interval_ephemeral"] = 100
    custom["save_num_checkpoints_to_keep"] = 3
    custom["save_interval_unsharded"] = 1000
    custom["data"]["paths"] = data_paths
    return custom, assert_allowed_diff(official, custom)


def assert_allowed_diff(official: dict, custom: dict) -> dict:
    """Every difference between the two configs must be an allowed key."""
    before, after = flatten(official), flatten(custom)
    diffs = {k: {"official": before.get(k), "custom": after.get(k)} for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}
    unexpected = set(diffs) - ALLOWED_CONFIG_CHANGES
    if unexpected:
        raise ValueError(f"unexpected config changes: {sorted(unexpected)}")
    if "data.paths" not in diffs:
        raise ValueError("data.paths unchanged")
    return diffs


def config_diff_errors(official_path: Path, custom_path: Path, training_dir: Path, stream: dict) -> list[str]:
    import yaml
    errs = []
    if sha256_file(official_path) != OFFICIAL_CONFIG_SHA256:
        errs.append("official config hash differs from the pinned one")
    official = yaml.safe_load(official_path.read_text())
    custom = yaml.safe_load(custom_path.read_text())
    before, after = flatten(official), flatten(custom)
    diffs = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if diffs - ALLOWED_CONFIG_CHANGES:
        errs.append(f"custom config changes protected keys: {sorted(diffs - ALLOWED_CONFIG_CHANGES)}")
    expected = [str((training_dir / "data" / sh["path"]).resolve()) for sh in stream["shards"]]
    if custom.get("data", {}).get("paths") != expected:
        errs.append("custom config data.paths do not match the stream shards")
    if int(custom.get("stop_at", -1)) != int(official.get("stop_at", -2)):
        errs.append("stop_at changed")
    return errs


def cmd_make_config(args: argparse.Namespace) -> int:
    import yaml
    if sha256_file(args.official_config) != OFFICIAL_CONFIG_SHA256:
        raise SystemExit("official config hash differs from the pinned one")
    official = yaml.safe_load(args.official_config.read_text())
    stream = json.loads((args.training_dir / "stream_manifest.json").read_text())
    paths = [str((args.training_dir / "data" / sh["path"]).resolve()) for sh in stream["shards"]]
    custom, diffs = make_custom_config(official, paths, args.run_name, args.wandb_group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(custom, sort_keys=False, width=1000))
    atomic_json(args.deviations, {"official_config_sha256": OFFICIAL_CONFIG_SHA256, "custom_config_sha256": sha256_file(args.output),
                                  "allowed_keys": sorted(ALLOWED_CONFIG_CHANGES), "differences": diffs})
    print(f"wrote {args.output}; {len(diffs)} allowed differences: {sorted(diffs)}")
    return 0


# ---------------------------------------------------------------------------
# evaluation manifest
# ---------------------------------------------------------------------------
def cmd_make_eval_manifest(args: argparse.Namespace) -> int:
    I = args.injection_dir
    man = rd_csv(I / "manifest.csv.gz")
    null = rd_csv(I / "null_pool.csv.gz")
    rows = []
    for r in man:
        rel = os.path.relpath(r["tokens_path"], I) if os.path.isabs(r["tokens_path"]) else r["tokens_path"]
        nb = os.path.getsize(r["inject_text_path"]) if r.get("inject_text_path") and os.path.exists(r["inject_text_path"]) else ""
        rows.append({"doc_id": r["doc_id"], "cohort": r["cohort"], "stratum": r["stratum"], "arm": r["arm"],
                     "pair_id": r["pair_id"], "split": r["split"], "r": r["r"], "group": r["group"],
                     "tokens_relpath": rel, "n_tokens": r["n_tokens_inject"], "n_bytes": nb})
    for r in null:
        nb = os.path.getsize(r["text_path"]) if r.get("text_path") and os.path.exists(r["text_path"]) else ""
        rows.append({"doc_id": r["doc_id"], "cohort": r["cohort"], "stratum": "", "arm": "null", "pair_id": "",
                     "split": r["split"], "r": "0", "group": r["group"],
                     "tokens_relpath": f"null_tokens/{safe(r['doc_id'])}.npy", "n_tokens": "", "n_bytes": nb})
    missing = [x["doc_id"] for x in rows if not (I / x["tokens_relpath"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} token files missing, e.g. {missing[:3]} (run plan first)")
    with gzip.open(args.output, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {args.output}: {len(rows):,} documents ({Counter(x['arm'] for x in rows)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--injection-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="bundle dir (default: the injection dir itself)")
    p.add_argument("--target-updates", type=int, default=DEFAULT_TARGET_UPDATES)
    p.add_argument("--trim", type=float, default=0.0, help="positions outside [trim, 1-trim] of the run are never used")
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B", help="HF tokenizer for the null pool (== dolma2)")
    p.set_defaults(fn=lambda a: cmd_plan(argparse.Namespace(**{**vars(a), "out_dir": a.out_dir or a.injection_dir})))
    b = sub.add_parser("build")
    b.add_argument("--bundle", type=Path, required=True)
    b.add_argument("--official-data-manifest", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True, help="training dir: data/ + stream_manifest.json")
    b.add_argument("--output-shards", type=int, default=DEFAULT_OUTPUT_SHARDS)
    b.set_defaults(fn=cmd_build)
    v = sub.add_parser("verify")
    v.add_argument("--bundle", type=Path, required=True)
    v.add_argument("--training-dir", type=Path, required=True)
    v.add_argument("--official-data-manifest", type=Path, required=True)
    v.add_argument("--official-config", type=Path, default=None)
    v.add_argument("--custom-config", type=Path, default=None)
    v.add_argument("--background-samples", type=int, default=20_000)
    v.add_argument("--skip-hashes", action="store_true")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--output", type=Path, required=True)
    v.set_defaults(fn=cmd_verify)
    c = sub.add_parser("make-config")
    c.add_argument("--official-config", type=Path, required=True)
    c.add_argument("--training-dir", type=Path, required=True)
    c.add_argument("--run-name", default="OLMo2-1B-stage2-injection-v2")
    c.add_argument("--wandb-group", default="injection-v2-20260925")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--deviations", type=Path, required=True)
    c.set_defaults(fn=cmd_make_config)
    e = sub.add_parser("make-eval-manifest")
    e.add_argument("--injection-dir", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.set_defaults(fn=cmd_make_eval_manifest)
    s = sub.add_parser("simulate")
    s.add_argument("--training-dir", type=Path, required=True)
    s.add_argument("--position", type=int, required=True)
    s.add_argument("--n", type=int, default=16)
    s.set_defaults(fn=cmd_simulate)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
