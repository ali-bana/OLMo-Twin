"""End-to-end test of the v2 stream mechanics on a synthetic official corpus.

Builds a tiny 'official' stage-2 corpus (3 memmap shards), a tiny injection
set (documents shorter and longer than one chunk, one document with r = 3),
plans, builds the stream, and checks that reading the built shards the way
OLMo's loader does yields, at every training position, either the official
chunk the seed-42 run would have consumed there or the intended injected
chunk. Also checks placement invariants and the config guard.
"""
import gzip
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # training/midtrain
import build_injection_stream_v2 as bs  # noqa: E402

SEQ = bs.SEQUENCE_LENGTH


def _write_official(tmp: Path, chunks_per_shard=(700, 500, 900), seed=1):
    rng = np.random.default_rng(seed)
    root = tmp / "official"
    root.mkdir()
    entries = []
    for i, n in enumerate(chunks_per_shard):
        arr = rng.integers(0, bs.VOCAB_SIZE, size=n * SEQ + 7, dtype=np.uint32)  # 7 stray tokens: not a full chunk
        p = root / f"part-{i}.npy"
        arr.astype("<u4").tofile(p)
        entries.append({"index": i, "local_path": str(p), "status": "complete", "bytes": p.stat().st_size, "tokens": int(arr.size)})
    manifest = {"olmo_commit": bs.OLMO_COMMIT, "official_config_sha256": bs.OFFICIAL_CONFIG_SHA256, "entries": entries}
    mp = tmp / "official_manifest.json"
    mp.write_text(json.dumps(manifest))
    return mp, sum(chunks_per_shard)


def _write_injection(tmp: Path, rng):
    I = tmp / "inj"
    (I / "tokens" / "M").mkdir(parents=True)
    docs = {"a": 100, "b": SEQ + 5, "c": 3 * SEQ - 1, "d": 2 * SEQ}   # tokens incl. EOS
    man, copies = [], []
    rs = {"a": 1, "b": 3, "c": 1, "d": 1}
    for k, (d, n) in enumerate(docs.items()):
        ids = rng.integers(0, bs.VOCAB_SIZE - 1, size=n, dtype=np.int32); ids[-1] = bs.EOS_TOKEN_ID
        tp = I / "tokens" / "M" / f"{d}.npy"
        np.save(tp, ids)
        for arm in ("inject", "holdout"):
            did = d if arm == "inject" else d + "_h"
            man.append({"doc_id": did, "cohort": "M", "stratum": "s", "arm": arm, "pair_id": f"M-{k}", "split": "dev",
                        "r": rs[d], "burst": 0, "group": "postcutoff", "tokens_path": str(tp), "n_tokens_inject": n,
                        "n_tokens": n, "text_path": "", "inject_text_path": ""})
        for c in range(rs[d]):
            copies.append({"doc_id": d, "cohort": "M", "copy": c, "decile": c % 10, "position_frac": round(0.05 + 0.3 * c + 0.01 * k, 4), "burst": 0})
    # deliberate collision: two copies of b at nearly the same fraction and 'd' overlapping 'c'
    copies[1]["position_frac"] = copies[0]["position_frac"]
    for name, rows in (("manifest", man), ("copies", copies)):
        with gzip.open(I / f"{name}.csv.gz", "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with gzip.open(I / "null_pool.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id", "cohort", "text_path"]); w.writeheader()
    return I, docs, rs


def _plan(I: Path, target_updates: int):
    # bypass the HF tokenizer (no null docs) by calling the placement pieces directly
    man = bs.rd_csv(I / "manifest.csv.gz"); copies = bs.rd_csv(I / "copies.csv.gz")
    inject = {r["doc_id"]: r for r in man if r["arm"] == "inject"}
    n_chunks = {d: math.ceil(int(r["n_tokens_inject"]) / SEQ) for d, r in inject.items()}
    placed = bs.allocate_positions(copies, n_chunks, target_updates * bs.GLOBAL_BATCH_SIZE)
    order = bs.train_order(target_updates * bs.GLOBAL_BATCH_SIZE)
    (I / "stream_plan").mkdir(exist_ok=True)
    rows = []
    for c in placed:
        r = inject[c["doc_id"]]; ids = np.load(r["tokens_path"])
        pos = c["positions"]
        rows.append({"doc_id": c["doc_id"], "copy": int(c["copy"]), "cohort": "M", "n_tokens_with_eos": int(len(ids)),
                     "n_chunks": n_chunks[c["doc_id"]], "training_positions": pos, "physical_slots": [int(order[p]) for p in pos],
                     "first_update": pos[0] // 512 + 1, "last_update": pos[-1] // 512 + 1,
                     "tokens_relpath": str(Path(r["tokens_path"]).relative_to(I)),
                     "tokens_sha256": __import__("hashlib").sha256(ids.astype("<u4").tobytes()).hexdigest()})
    with gzip.open(I / "stream_plan" / "placement.jsonl.gz", "wt") as fh:
        for x in rows: fh.write(json.dumps(x) + "\n")
    (I / "stream_plan" / "placement_summary.json").write_text(json.dumps({"target_updates": target_updates, "target_chunks": target_updates * 512}))
    return rows


def test_allocation_invariants():
    copies = [{"doc_id": "x", "copy": 0, "position_frac": 0.5, "burst": 0, "decile": 5},
              {"doc_id": "x", "copy": 1, "position_frac": 0.5, "burst": 0, "decile": 5},   # same batch -> must shift a whole batch
              {"doc_id": "y", "copy": 0, "position_frac": 0.5, "burst": 0, "decile": 5}]
    placed = bs.allocate_positions(copies, {"x": 2, "y": 3}, 512 * 20)
    pos = [p for c in placed for p in c["positions"]]
    assert len(pos) == len(set(pos)) == 7
    xs = [c for c in placed if c["doc_id"] == "x"]
    assert {p // 512 for p in xs[0]["positions"]}.isdisjoint({p // 512 for p in xs[1]["positions"]})
    assert all(c["positions"] == list(range(c["positions"][0], c["positions"][0] + len(c["positions"]))) for c in placed)


def test_stream_reproduces_official_order_and_injections(tmp_path):
    rng = np.random.default_rng(0)
    mp, n_official = _write_official(tmp_path)
    I, docs, rs = _write_injection(tmp_path, rng)
    target_updates = 3                              # 1,536 chunks of 2,100 official
    rows = _plan(I, target_updates)
    out = tmp_path / "training"
    args = __import__("argparse").Namespace(bundle=I, official_data_manifest=mp, output_dir=out, output_shards=4)
    assert bs.cmd_build(args) == 0
    stream = json.loads((out / "stream_manifest.json").read_text())
    assert stream["target_tokens"] == target_updates * 512 * SEQ
    reader = bs.StreamReader(out, stream)
    entries, n = bs.official_inventory(mp)
    assert n == n_official
    ends = [e["chunk_end"] for e in entries]
    baseline = bs.train_order(n)[: target_updates * 512]
    injected = {}
    for row in rows:
        ids = np.load(I / row["tokens_relpath"])
        bg = bs.read_official_chunk(entries, ends, int(baseline[row["training_positions"][-1]]))
        mat = bs.injected_chunks_for(row, ids, bg)
        for k, p in enumerate(row["training_positions"]):
            injected[p] = mat[k]
    # every position: official chunk or the intended injected chunk
    for p in range(target_updates * 512):
        got = reader.at_position(p)
        want = injected[p] if p in injected else bs.read_official_chunk(entries, ends, int(baseline[p]))
        assert np.array_equal(got, want), f"position {p} differs"
    # copies == r and the stream's accounting
    assert sum(1 for r in rows if r["doc_id"] == "b") == 3
    assert stream["injected_chunks"] == sum(r["n_chunks"] for r in rows) == len(injected)
    # a document's chunks are contiguous in training order and the last chunk ends with EOS then background
    row = next(r for r in rows if r["doc_id"] == "b")
    last = reader.at_position(row["training_positions"][-1])
    tail = row["n_tokens_with_eos"] - (row["n_chunks"] - 1) * SEQ
    assert last[tail - 1] == bs.EOS_TOKEN_ID
    assert np.array_equal(last[tail:], bs.read_official_chunk(entries, ends, int(baseline[row["training_positions"][-1]]))[: SEQ - tail])
    # verify subcommand agrees (hashes + all injected chunks + sampled background)
    v = __import__("argparse").Namespace(bundle=I, training_dir=out, official_data_manifest=mp, official_config=None,
                                         custom_config=None, background_samples=200, skip_hashes=False, seed=0,
                                         output=tmp_path / "verify.json")
    assert bs.cmd_verify(v) == 0
    # rebuild is idempotent (existing shards verified, not rewritten)
    assert bs.cmd_build(args) == 0


def test_config_guard_rejects_protected_changes():
    official = {"run_name": "x", "seed": 42, "stop_at": 10, "data": {"paths": ["a"], "pad_direction": "right"}, "optimizer": {"learning_rate": 1e-4}}
    custom, diffs = bs.make_custom_config(official, ["b", "c"], "run", "grp")
    assert set(diffs) <= bs.ALLOWED_CONFIG_CHANGES and "data.paths" in diffs
    tampered = json.loads(json.dumps(custom)); tampered["optimizer"]["learning_rate"] = 2e-4
    with pytest.raises(ValueError):
        bs.assert_allowed_diff(official, tampered)
    unchanged = json.loads(json.dumps(official))
    with pytest.raises(ValueError):
        bs.assert_allowed_diff(official, unchanged)       # data.paths must differ
