#!/usr/bin/env python3
"""Adversarial near-duplicate audit of candidate OLMo 2 non-members.

The exact-match search (``olmo2_midtrain_membership.py``) guarantees detection
of a shared 44-consecutive-word run. That leaves a gap: a paper could be
present as an edited, reformatted, OCR-mangled, translated, or paraphrased copy
and still be reported absent. This tool closes that gap and produces
human-reviewable evidence for every surviving candidate.

Two detectors, deliberately chosen to fail in different ways:

1. ``scan``   exhaustive 8-word shingles with NO sampling. Because the audit
              set is ~18x smaller than the full candidate pool, every shingle
              fits in memory, so the guarantee tightens from 44 shared words
              to 8. Catches edits, reformatting, OCR noise, and partial reuse
              -- the fuzzy-duplicate class. Records, per candidate, the most
              similar corpus documents with containment ratio and the actual
              matching passage.

2. ``embed``  semantic search: candidate title+abstract against the opening of
              every pes2o document, exact brute-force cosine on GPU (no ANN,
              so no recall loss). Catches genuine paraphrase and translation,
              which are structurally invisible to any n-gram method. The
              candidate side uses title+abstract because that is what survives
              rewording -- and because the RedPajama cleaner strips it, so the
              n-gram detectors never see it.

Every detector writes its matches to disk for manual inspection rather than
collapsing them to a verdict. ``report`` merges them into one review file
ordered most-suspicious-first.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "olmo2_midtrain_membership",
    str(Path(__file__).with_name("olmo2_midtrain_membership.py")),
)
mm = importlib.util.module_from_spec(_spec)
sys.modules["olmo2_midtrain_membership"] = mm
_spec.loader.exec_module(mm)

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

log = mm.log
atomic_json = mm.atomic_json
normalize = mm.normalize

DEFAULT_SHINGLE = 8
EXCERPT_WORDS = 45


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def read_candidates(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_norm_text(path: str) -> bytes:
    return zstd.ZstdDecompressor().decompress(Path(path).read_bytes())


def word_offsets(data: bytes) -> np.ndarray:
    """Byte offset of the start of each word in space-normalized text."""
    if not data:
        return np.zeros(0, dtype=np.int64)
    spaces = np.frombuffer(data, dtype=np.uint8) == 32
    starts = np.flatnonzero(np.concatenate(([True], spaces))[: len(data)])
    # positions right after a space, plus position 0
    idx = np.flatnonzero(spaces) + 1
    return np.concatenate(([0], idx[idx < len(data)])).astype(np.int64)


def excerpt_at(data: bytes, word_index: int, words: int = EXCERPT_WORDS) -> str:
    offs = word_offsets(data)
    if not len(offs):
        return ""
    lo = max(0, word_index - words // 3)
    hi = min(len(offs) - 1, lo + words)
    start = int(offs[lo])
    end = int(offs[hi]) if hi < len(offs) else len(data)
    return data[start:end].decode("ascii", "replace")




# --------------------------------------------------------------------------
# fingerprints: exhaustive shingles, low-information ones removed
# --------------------------------------------------------------------------
#
# At 8-word sensitivity, normalization creates spurious long runs from
# enumerations: Wikipedia citation markers and subfigure labels "(a) (b) (c)"
# become "a b c d e f g h", which matches any paper containing a similar list.
# Real prose spans far more characters for the same word count, so a minimum
# byte span cleanly separates content from enumeration without touching
# ordinary text.

# Chosen from measured spans, not guessed: an 8-word enumeration spans ~15
# chars, short prose ~29, typical prose ~38, technical prose ~66. A 20-char
# floor removes only 7.2% of shingles -- the least distinctive ones -- while
# eliminating the enumeration artifact. A 40-char floor discarded 68% of
# shingles including ordinary prose, which would have caused false negatives.
MIN_SHINGLE_CHARS = 20


@mm.njit(cache=False)
def _shingles_with_spans_nb(data, k):  # pragma: no cover - compiled
    """Every k-word shingle hash and the character span it covers.

    The span of shingle i is end(word i+k-1) - start(word i), INCLUDING the
    last word. The previous implementation measured to the START of the last
    word, so every span was short by that word's length and the filter removed
    more real shingles than the documented calibration implied.

    No filtering happens here. Returning all shingles and filtering in numpy
    afterwards is what makes the candidate and corpus sides identical by
    construction rather than by convention.
    """
    n = len(data)
    words = np.empty(n // 2 + 2, dtype=np.uint64)
    starts = np.empty(n // 2 + 2, dtype=np.int64)
    ends = np.empty(n // 2 + 2, dtype=np.int64)
    count = 0
    h = np.uint64(0xCBF29CE484222325)
    prime = np.uint64(0x100000001B3)
    in_word = False
    wstart = 0
    for i in range(n):
        c = data[i]
        if c == 32:
            if in_word:
                words[count] = h
                starts[count] = wstart
                ends[count] = i
                count += 1
                h = np.uint64(0xCBF29CE484222325)
                in_word = False
        else:
            if not in_word:
                wstart = i
                in_word = True
            h = (h ^ np.uint64(c)) * prime
    if in_word:
        words[count] = h
        starts[count] = wstart
        ends[count] = n
        count += 1
    if count < k:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
    total = count - k + 1
    out = np.empty(total, dtype=np.uint64)
    spans = np.empty(total, dtype=np.int64)
    base = np.uint64(mm.HASH_BASE)
    inv = np.uint64(mm.INV_BASE)
    value = np.uint64(0)
    power = np.uint64(1)
    for i in range(k):
        value += words[i] * power
        power *= base
    top = power * inv
    for i in range(total):
        if i > 0:
            value = (value - words[i - 1]) * inv
            value += words[i + k - 1] * top
        out[i] = value
        spans[i] = ends[i + k - 1] - starts[i]
    return out, spans


def shingles_with_spans(data: bytes, k: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else data
    return _shingles_with_spans_nb(arr, k)


def shingles_filtered(
    data: bytes, k: int, min_chars: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kept hashes, their original shingle positions, kept-mask over ALL
    positions). This is the ONLY entry point both the fingerprint and scan
    sides use, so they cannot disagree on which shingles exist."""
    hashes, spans = shingles_with_spans(data, k)
    if not len(hashes):
        return hashes, np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    if min_chars <= 0:
        kept = np.ones(len(hashes), dtype=bool)
    else:
        kept = spans >= min_chars
    return hashes[kept], np.flatnonzero(kept), kept


def longest_run(
    positions: np.ndarray, kept: np.ndarray, k: int
) -> tuple[int, int, int]:
    """Longest contiguous run of matched shingle positions.

    Returns (run_shingles, run_words, start_position). Two matched positions
    p1 < p2 belong to the same run when every position strictly between them
    was REMOVED by the span filter -- a filtered shingle carries no evidence
    either way, and breaking the run there would chop a maths paper's reused
    section into fragments below any threshold. A kept-but-unmatched position
    between them ends the run.

    run_words is the number of words the run covers: last - first + k.
    """
    if len(positions) == 0:
        return 0, 0, -1
    pos = np.sort(np.asarray(positions, dtype=np.int64))
    # kept_cum[i] = number of kept positions in [0, i)
    kept_cum = np.concatenate(([0], np.cumsum(kept.astype(np.int64))))
    best_len = 1
    best_first = int(pos[0])
    best_last = int(pos[0])
    run_first = int(pos[0])
    run_n = 1
    for i in range(1, len(pos)):
        p1, p2 = int(pos[i - 1]), int(pos[i])
        # kept positions strictly inside (p1, p2)
        inside = kept_cum[p2] - kept_cum[p1 + 1] if p2 > p1 + 1 else 0
        if p2 == p1 + 1 or inside == 0:
            run_n += 1
        else:
            run_first = p2
            run_n = 1
        if run_n > best_len or (run_n == best_len and p2 - run_first > best_last - best_first):
            best_len = run_n
            best_first = run_first
            best_last = p2
    return best_len, best_last - best_first + k, best_first


def match_record(
    data: bytes, k: int, min_chars: int, fps: np.ndarray, owners: np.ndarray,
    bitmap: np.ndarray, mask: np.uint64, floor: int,
) -> dict[int, dict]:
    """Match one normalized corpus record against the candidate table.

    Returns {candidate_index: {"shingles": distinct shared shingles,
    "run_shingles": ..., "run_words": ..., "word": run start position}} for
    every candidate sharing at least `floor` distinct shingles.

    Positions are preserved all the way through -- the previous code ran
    np.unique before lookup, which sorted the hashes and made the recorded
    'word' offset an index into the sorted array, i.e. meaningless.
    """
    hashes, positions, kept = shingles_filtered(data, k, min_chars)
    if not len(hashes):
        return {}
    keep = np.empty(len(hashes), dtype=np.int64)
    nkeep = _bitmap_survivors(np.ascontiguousarray(hashes), bitmap, mask, keep)
    if not nkeep:
        return {}
    surv = keep[:nkeep]
    cand = hashes[surv]
    left = np.searchsorted(fps, cand, side="left")
    right = np.searchsorted(fps, cand, side="right")
    real = np.flatnonzero(right > left)
    if not len(real):
        return {}
    per_hashes: dict[int, set] = {}
    per_pos: dict[int, list] = {}
    for j in real:
        h = int(cand[j])
        p = int(positions[surv[j]])
        for owner in owners[left[j]:right[j]]:
            o = int(owner)
            s = per_hashes.get(o)
            if s is None:
                per_hashes[o] = {h}
                per_pos[o] = [p]
            else:
                s.add(h)
                per_pos[o].append(p)
    out: dict[int, dict] = {}
    for o, hs in per_hashes.items():
        n_distinct = len(hs)
        if n_distinct < floor:
            continue
        run_sh, run_words, start = longest_run(np.asarray(per_pos[o]), kept, k)
        out[o] = {
            "shingles": n_distinct,
            "run_shingles": run_sh,
            "run_words": run_words,
            "word": start,
        }
    return out


# --------------------------------------------------------------------------
# fail-open guards and provenance
# --------------------------------------------------------------------------


def require_shards(shards: list, corpus: Path, components) -> list:
    """An empty shard list is an error, never an empty result. The stage-1
    dclm scan once discovered zero shards because of a suffix spelling and
    reported 'no matches' across 6.6 TiB."""
    if not shards:
        raise SystemExit(
            f"discovered ZERO shards under {corpus} for components={components}. "
            "Refusing to run: a scan over nothing reports 'no matches', which is "
            "indistinguishable from a clean result."
        )
    return shards


def classify_records(lines) -> dict:
    """Count how many corpus lines were actually usable. Used by tests and by
    the scan's own accounting, so a schema change (text under another key)
    shows up as missing_text == rows instead of as a clean scan."""
    stats = {"rows": 0, "parse_failed": 0, "missing_text": 0, "usable": 0}
    for raw in lines:
        stats["rows"] += 1
        try:
            rec = json.loads(raw)
        except Exception:
            stats["parse_failed"] += 1
            continue
        if not isinstance(rec.get("text"), str):
            stats["missing_text"] += 1
            continue
        stats["usable"] += 1
    return stats


SUSPECT_UNUSABLE_FRACTION = 0.5


def shard_is_suspect(stats: dict) -> bool:
    rows = stats.get("rows", 0)
    if rows == 0:
        return False
    unusable = stats.get("parse_failed", 0) + stats.get("missing_text", 0)
    return unusable / rows >= SUSPECT_UNUSABLE_FRACTION


def file_digest(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def array_digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()


def code_version() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
    except Exception:
        return "unknown"


def check_provenance(meta: dict, *, candidates_sha256: str, fingerprints_sha256: str) -> None:
    """Refuse to combine artifacts that were not produced from each other."""
    want_c = meta.get("candidates_sha256")
    want_f = meta.get("fingerprints_sha256")
    if want_c and want_c != candidates_sha256:
        raise SystemExit(
            "candidate manifest does not match the one the fingerprints were built from\n"
            f"  fingerprints built from: {want_c}\n  manifest given now:      {candidates_sha256}"
        )
    if want_f and want_f != fingerprints_sha256:
        raise SystemExit(
            "fingerprint table on disk does not match the digest recorded at build time\n"
            f"  recorded: {want_f}\n  on disk:  {fingerprints_sha256}"
        )


def _fp_chunk(task):
    chunk_id, items, out_dir, k, min_chars = task
    dctx = zstd.ZstdDecompressor()
    fps, ows = [], []
    failed = 0
    for index, text_path in items:
        try:
            data = dctx.decompress(Path(text_path).read_bytes())
        except Exception:
            failed += 1
            continue
        # Same function the scan uses on the corpus side -- see shingles_filtered.
        sel, _, _ = shingles_filtered(data, k, min_chars)
        if not len(sel):
            failed += 1
            continue
        sel = np.unique(sel)
        fps.append(sel)
        ows.append(np.full(len(sel), index, dtype=np.uint32))
    if not fps:
        return {"chunk": chunk_id, "count": 0, "failed": failed, "docs": len(items)}
    fp = np.concatenate(fps); ow = np.concatenate(ows)
    np.save(Path(out_dir) / f"fp_{chunk_id:05d}.npy", fp)
    np.save(Path(out_dir) / f"ow_{chunk_id:05d}.npy", ow)
    return {"chunk": chunk_id, "count": int(len(fp)), "failed": failed, "docs": len(items)}


def cmd_fingerprints(args: argparse.Namespace) -> int:
    mm.set_progress_path(args.progress)
    rows = read_candidates(args.candidates)
    usable = [r for r in rows if r.get("parser_status") == "ok" and r.get("text_path")]
    log(f"candidates with cleaned text: {len(usable):,}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts = args.out_dir / "parts"
    if parts.exists():
        for stale in parts.glob("*.npy"):
            stale.unlink()
    parts.mkdir(parents=True, exist_ok=True)

    per = max(200, len(usable) // (args.workers * 8) + 1)
    tasks = [
        (cid, [(i, usable[i]["text_path"]) for i in range(st, min(st + per, len(usable)))],
         str(parts), args.window, args.min_shingle_chars)
        for cid, st in enumerate(range(0, len(usable), per))
    ]
    log(f"exhaustive {args.window}-gram, min span {args.min_shingle_chars} chars, "
        f"{len(tasks)} chunks / {args.workers} workers")
    started = time.time(); done = total = failed = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for res in pool.imap_unordered(_fp_chunk, tasks):
            done += res["docs"]; total += res["count"]; failed += res["failed"]
            mm.write_progress("audit_fingerprints", done, len(usable), started=started,
                              unit="papers", note=f"{total:,} shingles")
            log(f"  {mm.bar(done, len(usable))} {done:,}/{len(usable):,}  fp={total:,}")

    fp_all = np.empty(total, dtype=np.uint64); ow_all = np.empty(total, dtype=np.uint32)
    at = 0
    for path in sorted(parts.glob("fp_*.npy")):
        c = np.load(path); o = np.load(parts / path.name.replace("fp_", "ow_"))
        fp_all[at:at + len(c)] = c; ow_all[at:at + len(c)] = o; at += len(c)
    order = np.argsort(fp_all[:at], kind="stable")
    fp_all = fp_all[:at][order]; ow_all = ow_all[:at][order]
    np.save(args.out_dir / "fingerprints.npy", fp_all)
    np.save(args.out_dir / "owners.npy", ow_all)
    with gzip.open(args.out_dir / "index_docs.txt.gz", "wt", encoding="utf-8") as fh:
        for r in usable:
            fh.write(f"{r['doc_id']}\n")
    # Provenance: every downstream artifact records these and refuses to
    # combine with anything built from a different manifest or table.
    cand_digest = file_digest(args.candidates)
    fp_digest = array_digest(fp_all)
    atomic_json(args.out_dir / "fingerprint_meta.json", {
        "window": args.window, "winnow": 1, "unit": "words",
        # Honest statement of what is detectable: a shared run of >= k words
        # whose shingles each span >= min_shingle_chars. Shorter-span shingles
        # are absent on BOTH sides (symmetric filter), so they cannot match.
        "guaranteed_shared_words": args.window,
        "guarantee_note": (
            f"any shared run of >= {args.window} words is detected provided its "
            f"shingles span >= {args.min_shingle_chars} chars; the filter is applied "
            "identically to candidates and corpus"
        ),
        "min_shingle_chars": args.min_shingle_chars,
        "hash_base": mm.HASH_BASE, "documents": len(usable),
        "documents_without_fingerprints": failed,
        "fingerprints": int(len(fp_all)),
        "bytes_in_memory": int(fp_all.nbytes + ow_all.nbytes),
        "normalization": "NFKD+lower+alnum_fold+ws_collapse",
        "candidates_path": str(args.candidates),
        "candidates_sha256": cand_digest,
        "fingerprints_sha256": fp_digest,
        "code_version": code_version(),
        "note": "exhaustive (no winnowing); low-span shingles dropped on both "
                "sides to suppress enumeration artifacts",
    })
    for stale in parts.glob("*.npy"):
        stale.unlink()
    log(f"fingerprints={len(fp_all):,} ({(fp_all.nbytes+ow_all.nbytes)/2**30:.2f} GiB) "
        f"guarantee>={args.window} words, min span {args.min_shingle_chars} chars")
    return 0


# --------------------------------------------------------------------------
# bitmap prefilter
# --------------------------------------------------------------------------
#
# Exhaustive (unwinnowed) shingles mean every corpus 8-gram is looked up, not
# 6% of them. A searchsorted on a 3.2 GB sorted array costs ~28 cache misses,
# so 16x the lookups made the scan ~15x slower and unusable. A Bloom-style
# bitmap keyed on the low bits of the hash rejects the overwhelming majority
# of non-matching shingles in ONE cache miss; only survivors pay for the
# binary search. This is a pure accelerator -- it can produce false positives
# (which the binary search then discards) but never a false negative, so the
# 8-shared-word guarantee is preserved exactly.

BITMAP_BITS = 34  # 2 GiB; ~2.3% false-positive rate at 403M entries


@mm.njit(cache=False)
def _build_bitmap(hashes, bitmap, mask):  # pragma: no cover - compiled
    for i in range(len(hashes)):
        idx = hashes[i] & mask
        bitmap[idx >> np.uint64(6)] |= np.uint64(1) << (idx & np.uint64(63))


@mm.njit(cache=False)
def _bitmap_survivors(sel, bitmap, mask, out):  # pragma: no cover - compiled
    n = 0
    for i in range(len(sel)):
        idx = sel[i] & mask
        if (bitmap[idx >> np.uint64(6)] >> (idx & np.uint64(63))) & np.uint64(1):
            out[n] = i
            n += 1
    return n


def build_bitmap(fps: np.ndarray, bits: int = BITMAP_BITS) -> tuple[np.ndarray, int]:
    mask = np.uint64((1 << bits) - 1)
    bitmap = np.zeros(1 << (bits - 6), dtype=np.uint64)
    step = 1 << 25
    for start in range(0, len(fps), step):
        _build_bitmap(np.asarray(fps[start : start + step]), bitmap, mask)
    set_bits = int(np.unpackbits(bitmap.view(np.uint8)).sum())
    return bitmap, set_bits


# --------------------------------------------------------------------------
# scan: exhaustive 8-gram, with excerpt capture
# --------------------------------------------------------------------------

_G: dict = {}


def _init(state: dict) -> None:
    global _G
    _G = state


def _scan_shard(payload: dict) -> dict:
    g = _G
    fps: np.ndarray = g["fps"]
    owners: np.ndarray = g["owners"]
    k: int = g["shingle"]
    topk: int = g["topk"]
    floor: int = g["min_shingles"]
    out_dir = Path(g["out_dir"])
    path = payload["path"]
    component = payload["component"]

    marker = out_dir / "shards" / (payload["key"] + ".done.json")
    if marker.exists() and not g["force"]:
        return json.loads(marker.read_text())

    min_chars: int = g["min_chars"]
    # candidate index -> list of (distinct, run_shingles, run_words, ref, start, doc_bytes, excerpt)
    hits: dict[int, list] = {}
    rows = parse_failed = missing_text = too_short = 0
    started = time.time()

    for raw in mm.iter_lines(path):
        rows += 1
        try:
            record = mm.json_loads(raw) if hasattr(mm, "json_loads") else json.loads(raw)
        except Exception:
            parse_failed += 1
            continue
        text = record.get("text")
        if not isinstance(text, str):
            missing_text += 1
            continue
        data = normalize(text)
        if len(data) < 16:
            too_short += 1
            continue
        matched = match_record(
            data, k, min_chars, fps, owners, g["bitmap"], g["bitmask"], floor
        )
        if not matched:
            continue
        ref = mm._record_ref(record)
        for o, m in matched.items():
            # Excerpt at the START OF THE LONGEST RUN, which is what a human
            # needs to see; the old code excerpted at an index into a sorted
            # hash array, i.e. a random place in the record.
            snippet = (
                excerpt_at(data, m["word"]) if m["shingles"] >= g["excerpt_floor"] else ""
            )
            entry = hits.setdefault(o, [])
            entry.append((m["shingles"], m["run_shingles"], m["run_words"],
                          ref, m["word"], len(data), snippet))
            if len(entry) > topk * 4:
                entry.sort(key=lambda x: (-x[0], -x[2]))
                del entry[topk:]

    out = out_dir / "shards" / (payload["key"] + ".hits.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wt", encoding="utf-8") as handle:
        for o, entry in hits.items():
            entry.sort(key=lambda x: (-x[0], -x[2]))
            handle.write(
                json.dumps(
                    {
                        "doc": o,
                        "component": component,
                        "matches": [
                            {
                                "shingles": c,
                                "run_shingles": rs,
                                "run_words": rw,
                                "ref": r,
                                "word": p,
                                "doc_bytes": db,
                                "excerpt": ex,
                            }
                            for c, rs, rw, r, p, db, ex in entry[:topk]
                        ],
                    }
                )
                + "\n"
            )
    stats = {
        "component": component,
        "path": path,
        "rows": rows,
        "parse_failed": parse_failed,
        "missing_text": missing_text,
        "too_short": too_short,
        "usable": rows - parse_failed - missing_text - too_short,
        "candidates_hit": len(hits),
        "fingerprints_sha256": g["fingerprints_sha256"],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    stats["suspect"] = shard_is_suspect(stats)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(stats))
    return stats


def cmd_scan(args: argparse.Namespace) -> int:
    mm.set_progress_path(args.progress)
    rows = read_candidates(args.candidates)
    usable = [r for r in rows if r.get("parser_status") == "ok"]
    doc_ids = [r["doc_id"] for r in usable]
    log(f"audit candidates: {len(usable):,}")

    fps = np.load(args.fingerprint_dir / "fingerprints.npy", mmap_mode="r")
    owners = np.load(args.fingerprint_dir / "owners.npy", mmap_mode="r")
    meta = json.loads((args.fingerprint_dir / "fingerprint_meta.json").read_text())
    with gzip.open(args.fingerprint_dir / "index_docs.txt.gz", "rt") as fh:
        fp_docs = [line.strip() for line in fh if line.strip()]
    if fp_docs != doc_ids:
        raise SystemExit("fingerprint order does not match the candidate manifest")
    # Provenance: the manifest handed to us must be the one the fingerprints
    # were built from, and the table on disk must be the one that was built.
    cand_digest = file_digest(args.candidates)
    fp_digest = array_digest(np.asarray(fps))
    check_provenance(meta, candidates_sha256=cand_digest, fingerprints_sha256=fp_digest)
    min_chars = int(meta.get("min_shingle_chars", 0))
    log(
        f"exhaustive {meta['window']}-word shingles, winnow={meta['winnow']}, "
        f"span filter >= {min_chars} chars applied to BOTH sides; "
        f"{len(fps):,} entries ({(fps.nbytes + owners.nbytes) / 2**30:.2f} GiB); "
        f"provenance ok (candidates {cand_digest[:12]}, fingerprints {fp_digest[:12]})"
    )

    shards = mm.discover_shards(args.corpus, args.components)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    shards = require_shards(shards, args.corpus, args.components)
    total = sum(s.size for s in shards)
    log(f"shards={len(shards):,} compressed={total / 2**30:.1f} GiB workers={args.workers}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = [
        {"path": s.path, "component": s.component, "key": mm.shard_key(s)}
        for s in shards
    ]
    log("building bitmap prefilter ...")
    t0 = time.time()
    bitmap, set_bits = build_bitmap(fps)
    occupancy = set_bits / float(1 << BITMAP_BITS)
    log(
        f"bitmap {bitmap.nbytes / 2**30:.2f} GiB, {occupancy * 100:.2f}% occupied "
        f"-> ~{occupancy * 100:.2f}% of non-matching lookups fall through "
        f"({time.time() - t0:.0f}s)"
    )
    state = {
        "fps": fps,
        "owners": owners,
        "bitmap": bitmap,
        "bitmask": np.uint64((1 << BITMAP_BITS) - 1),
        "shingle": meta["window"],
        "min_chars": min_chars,
        "fingerprints_sha256": fp_digest,
        "topk": args.topk,
        "min_shingles": args.min_shingles,
        "out_dir": str(args.out_dir),
        "force": args.force,
        "excerpt_floor": args.excerpt_floor,
    }
    done = 0
    started = time.time()
    totals = {"rows": 0, "parse_failed": 0, "missing_text": 0, "too_short": 0, "usable": 0}
    suspect: list[str] = []
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init, initargs=(state,)) as pool:
        for i, st in enumerate(pool.imap_unordered(_scan_shard, payload, chunksize=1), 1):
            done += next((s.size for s in shards if s.path == st["path"]), 0)
            for key in totals:
                totals[key] += int(st.get(key, 0))
            if st.get("suspect"):
                suspect.append(st["path"])
                log(f"  SUSPECT shard (>= {SUSPECT_UNUSABLE_FRACTION:.0%} unusable rows): "
                    f"{os.path.basename(st['path'])}  rows={st['rows']} "
                    f"parse_failed={st.get('parse_failed')} missing_text={st.get('missing_text')}")
            mm.write_progress(
                "audit_scan", done, total, started=started, unit="bytes",
                note=f"{i}/{len(shards)} shards",
            )
            if i % 10 == 0 or i == len(shards):
                el = time.time() - started
                rate = done / max(el, 1e-9)
                log(
                    f"  {mm.bar(done, total)} {i}/{len(shards)}  "
                    f"{done / 2**30:.1f}/{total / 2**30:.1f} GiB  "
                    f"{rate / 2**20:.0f} MB/s  eta {(total - done) / max(rate, 1e-9) / 3600:.2f} h"
                )

    unusable = totals["parse_failed"] + totals["missing_text"]
    frac = unusable / max(totals["rows"], 1)
    summary = {
        "shards": len(shards),
        "compressed_bytes": total,
        "elapsed_seconds": round(time.time() - started, 1),
        "candidates": len(usable),
        "fingerprint_meta": meta,
        "min_shingles": args.min_shingles,
        "topk": args.topk,
        "records": totals,
        "unusable_fraction": round(frac, 6),
        "suspect_shards": suspect,
        "candidates_sha256": cand_digest,
        "fingerprints_sha256": fp_digest,
        "code_version": code_version(),
        "status": "ok" if not suspect else "suspect_shards",
    }
    if suspect:
        # Do not leave a normal-looking summary behind: downstream tools key on
        # scan_summary.json as the completion marker.
        atomic_json(args.out_dir / "scan_summary.FAILED.json", summary)
        raise SystemExit(
            f"{len(suspect)} of {len(shards)} shards were mostly unusable "
            f"(overall unusable fraction {frac:.2%}). A schema change or a broken "
            "decoder would look exactly like this AND like a clean scan. Refusing to "
            "write scan_summary.json; details in scan_summary.FAILED.json"
        )
    if frac > 0.01:
        log(f"  WARNING: {frac:.2%} of corpus rows were unparseable or had no text field")
    atomic_json(args.out_dir / "scan_summary.json", summary)
    log(f"audit scan complete in {(time.time() - started) / 3600:.2f} h; "
        f"rows={totals['rows']:,} usable={totals['usable']:,} unusable={frac:.3%}")
    return 0


# --------------------------------------------------------------------------
# embed: semantic search over pes2o
# --------------------------------------------------------------------------


def cmd_embed(args: argparse.Namespace) -> int:
    from sentence_transformers import SentenceTransformer
    import torch

    rows = [r for r in read_candidates(args.candidates) if r.get("parser_status") == "ok"]
    log(f"candidates: {len(rows):,}")
    model = SentenceTransformer(args.model, device="cuda")
    dim = model.get_sentence_embedding_dimension()
    log(f"model={args.model} dim={dim}")

    # Candidate side: title + abstract. This is what survives rewording, and
    # the cleaner strips it, so the n-gram detectors never see it.
    #
    # Wikipedia rows carry no abstract, which would reduce their query to a
    # bare title and make the search meaningless for them. Fall back to the
    # document's own opening, which is the closest analogue of an abstract --
    # the lead section. The stored text is normalized (lowercased, punctuation
    # folded); all-MiniLM-L6-v2 is uncased, so the casing difference against
    # raw corpus openings costs little.
    _dctx = zstd.ZstdDecompressor() if zstd is not None else None

    def _query_for(r: dict) -> str:
        title = (r.get("title") or "").strip()
        abstract = (r.get("abstract") or "").strip()
        if abstract:
            return f"{title}. {abstract}"
        path = r.get("text_path")
        if path and _dctx is not None:
            try:
                head = _dctx.decompress(Path(path).read_bytes())[:1200]
                return f"{title}. {head.decode('utf-8', 'replace')}"
            except Exception:
                pass
        return f"{title}."

    queries = [_query_for(r) for r in rows]
    q = model.encode(
        queries, batch_size=args.batch, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).to(torch.float16)
    log(f"query embeddings: {tuple(q.shape)}")

    shards = mm.discover_shards(args.corpus, args.components)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    log(f"embedding openings of {len(shards)} shards ({sum(s.size for s in shards)/2**30:.1f} GiB)")

    best_score = torch.full((len(rows), args.topk), -1.0, dtype=torch.float16, device="cuda")
    best_ref: list[list] = [[] for _ in rows]
    started = time.time()
    processed = 0

    for si, shard in enumerate(shards, 1):
        texts: list[str] = []
        refs: list[str] = []
        for raw in mm.iter_lines(shard.path):
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            t = rec.get("text")
            if not isinstance(t, str) or len(t) < 40:
                continue
            # Openings are the comparable unit: abstract-only pes2o records ARE
            # title+abstract, and full-text records begin with the title.
            texts.append(t[: args.opening_chars])
            refs.append(str(rec.get("id", ""))[:80])
            if len(texts) >= args.chunk:
                _embed_chunk(model, texts, refs, q, best_score, best_ref, args, torch)
                processed += len(texts)
                texts, refs = [], []
        if texts:
            _embed_chunk(model, texts, refs, q, best_score, best_ref, args, torch)
            processed += len(texts)
        el = time.time() - started
        log(
            f"  {mm.bar(si, len(shards))} shard {si}/{len(shards)}  "
            f"{processed:,} docs  {processed / max(el, 1e-9):.0f} docs/s  "
            f"eta {(len(shards) - si) * el / max(si, 1) / 3600:.2f} h"
        )
        mm.write_progress("audit_embed", si, len(shards), started=started,
                          unit="shards", note=f"{processed:,} docs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "embed_matches.jsonl.gz"
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            ms = sorted(best_ref[i], key=lambda x: -x[0])[: args.topk]
            fh.write(json.dumps({
                "doc_id": r["doc_id"], "month": r["month"],
                "matches": [{"cosine": round(float(s), 4), "ref": ref, "opening": op}
                            for s, ref, op in ms],
            }) + "\n")
    atomic_json(args.out_dir / "embed_summary.json", {
        "model": args.model, "dim": dim, "candidates": len(rows),
        "docs_embedded": processed, "opening_chars": args.opening_chars,
        "topk": args.topk, "elapsed_seconds": round(time.time() - started, 1),
        "components": sorted({s.component for s in shards}),
        "search": "exact brute-force cosine on GPU (no ANN, no recall loss)",
    })
    log(f"wrote {out}  ({processed:,} docs embedded)")
    return 0


def _embed_chunk(model, texts, refs, q, best_score, best_ref, args, torch):
    emb = model.encode(
        texts, batch_size=args.batch, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).to(torch.float16)
    sim = q @ emb.T                      # [n_candidates, n_docs]
    vals, idx = torch.topk(sim, k=min(args.topk, sim.shape[1]), dim=1)
    vals = vals.cpu().numpy()
    idx = idx.cpu().numpy()
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            s = float(vals[i, j])
            if s < args.min_cosine:
                continue
            best_ref[i].append((s, refs[idx[i, j]], texts[idx[i, j]][:200]))
        if len(best_ref[i]) > args.topk * 6:
            best_ref[i] = sorted(best_ref[i], key=lambda x: -x[0])[: args.topk]
    del emb, sim


# --------------------------------------------------------------------------
# report: the manual-review artifact
# --------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    """Rank every candidate by how much of it appears in the training corpus.

    Containment (shared distinct 8-grams / the paper's own total) is the
    discriminator, not raw match count: a paper and a Wikipedia article that
    cite the same book share 25-46 shingles through their bibliographies alone,
    which is <1% containment, while a genuine copy sits far higher.
    """
    import json as _json

    rows = [r for r in read_candidates(args.candidates) if r.get("parser_status") == "ok"]
    counts = np.load(args.shingle_counts) if args.shingle_counts else None
    density = np.load(args.lexical_density) if args.lexical_density else None

    best: dict[int, tuple] = {}
    for path in sorted((args.scan_dir / "shards").glob("*.hits.jsonl")):
        component = path.name.split("__")[0]
        with path.open() as fh:
            for line in fh:
                try:
                    e = _json.loads(line)
                except Exception:
                    continue
                for m in e.get("matches", []):
                    cur = best.get(e["doc"])
                    if cur is None or m["shingles"] > cur[0]:
                        best[e["doc"]] = (
                            m["shingles"], component, m.get("ref", ""),
                            m.get("excerpt", ""), m.get("doc_bytes", 0),
                        )

    embed: dict[str, list] = {}
    if args.embed_matches and Path(args.embed_matches).exists():
        with gzip.open(args.embed_matches, "rt") as fh:
            for line in fh:
                e = _json.loads(line)
                embed[e["doc_id"]] = e.get("matches", [])

    out_rows = []
    for i, r in enumerate(rows):
        total = int(counts[i]) if counts is not None else 0
        shared, comp, ref, exc, dbytes = best.get(i, (0, "", "", "", 0))
        # Cap: pre-dedupe scans could count corpus positions, exceeding the
        # paper's own shingle total for degenerate near-empty papers.
        capped = min(shared, total) if total else shared
        containment = (capped / total) if total else 0.0
        em = embed.get(r["doc_id"], [])
        dens = float(density[i]) if density is not None else -1.0
        # Papers whose cleaned text is LaTeX preamble, \input skeletons, or
        # pure math notation produce generic symbol shingles that collide with
        # the synthetic math components by chance. They dominate the top of a
        # naive containment ranking while being meaningless as matches -- and
        # they are poor MIA subjects regardless of membership, since an attack
        # would be scoring macros rather than prose.
        degenerate = dens >= 0 and (dens < args.min_density or total < 500)
        out_rows.append({
            "doc_id": r["doc_id"],
            "month": r["month"],
            "primary_category": r.get("primary_category", ""),
            "paper_shingles": total,
            "testable": "no" if total < args.min_testable_shingles else "yes",
            "shared_shingles": capped,
            "containment": round(containment, 5),
            "lexical_density": round(dens, 3) if dens >= 0 else "",
            "degenerate_text": "yes" if degenerate else "no",
            "review_priority": (
                "skip_degenerate" if degenerate
                else "HIGH" if containment >= 0.20
                else "medium" if containment >= 0.05
                else "low"
            ),
            "match_component": comp,
            "match_ref": ref[:120],
            "top_cosine": round(em[0]["cosine"], 4) if em else "",
            "cosine_ref": (em[0]["ref"][:80] if em else ""),
            "matching_passage": exc[:400].replace("\n", " "),
        })
    # Most suspicious first: high containment, then high semantic similarity.
    # Degenerate-text papers sort last: their high containment is an artifact,
    # and surfacing them first would bury the cases that actually need eyes.
    order = {"HIGH": 0, "medium": 1, "low": 2, "skip_degenerate": 3}
    out_rows.sort(
        key=lambda x: (order[x["review_priority"]], -x["containment"],
                       -(x["top_cosine"] or 0)),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "review_matches.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    bands = {"<1%": 0, "1-5%": 0, "5-20%": 0, "20-50%": 0, ">=50%": 0}
    for x in out_rows:
        c = x["containment"]
        k = ("<1%" if c < 0.01 else "1-5%" if c < 0.05 else "5-20%" if c < 0.20
             else "20-50%" if c < 0.50 else ">=50%")
        bands[k] += 1
    untestable = sum(1 for x in out_rows if x["testable"] == "no")
    prio = {}
    for x in out_rows:
        prio[x["review_priority"]] = prio.get(x["review_priority"], 0) + 1
    hi_cos = sum(1 for x in out_rows
                 if isinstance(x["top_cosine"], float)
                 and x["top_cosine"] >= args.cosine_flag and x["containment"] < 0.05)
    atomic_json(args.out_dir / "review_summary.json", {
        "candidates": len(out_rows),
        "with_any_match": sum(1 for x in out_rows if x["shared_shingles"] > 0),
        "containment_bands": bands,
        "untestable_below_shingles": {"threshold": args.min_testable_shingles,
                                      "count": untestable},
        "review_priority": prio,
        "degenerate_text_excluded": sum(
            1 for x in out_rows if x["degenerate_text"] == "yes"),
        "paraphrase_suspects_high_cosine_low_overlap": {
            "cosine_threshold": args.cosine_flag, "count": hi_cos},
        "guarantee": "shared 8-word run spanning >=20 chars, exhaustive (no sampling)",
    })
    log(f"wrote {path}")
    log(f"containment bands: {bands}")
    log(f"untestable (<{args.min_testable_shingles} shingles): {untestable:,}")
    log(f"review priority: {prio}")
    log(f"paraphrase suspects (cosine>={args.cosine_flag}, containment<5%): {hi_cos:,}")
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="exhaustive 8-gram audit scan")
    s.add_argument("--candidates", type=Path, required=True)
    s.add_argument("--fingerprint-dir", type=Path, required=True)
    s.add_argument("--corpus", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--components", nargs="*", default=None)
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--limit-shards", type=int, default=0)
    s.add_argument("--topk", type=int, default=5)
    s.add_argument("--min-shingles", type=int, default=3,
                   help="ignore corpus docs sharing fewer than this many 8-grams")
    s.add_argument("--excerpt-floor", type=int, default=10,
                   help="capture the matching passage above this many shared 8-grams")
    s.add_argument("--progress", type=Path, default=None)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_scan)

    f = sub.add_parser("fingerprints", help="exhaustive shingles, span-filtered")
    f.add_argument("--candidates", type=Path, required=True)
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--window", type=int, default=DEFAULT_SHINGLE)
    f.add_argument("--min-shingle-chars", type=int, default=MIN_SHINGLE_CHARS)
    f.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    f.add_argument("--progress", type=Path, default=None)
    f.set_defaults(func=cmd_fingerprints)

    e = sub.add_parser("embed", help="semantic search over pes2o openings")
    e.add_argument("--candidates", type=Path, required=True)
    e.add_argument("--corpus", type=Path, required=True)
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--components", nargs="*", default=["pes2o"])
    e.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    e.add_argument("--batch", type=int, default=1024)
    e.add_argument("--chunk", type=int, default=200_000,
                   help="docs per GPU similarity block")
    e.add_argument("--opening-chars", type=int, default=400)
    e.add_argument("--topk", type=int, default=5)
    e.add_argument("--min-cosine", type=float, default=0.55)
    e.add_argument("--limit-shards", type=int, default=0)
    e.add_argument("--progress", type=Path, default=None)
    e.set_defaults(func=cmd_embed)

    r = sub.add_parser("report", help="build the manual-review artifact")
    r.add_argument("--candidates", type=Path, required=True)
    r.add_argument("--scan-dir", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--shingle-counts", type=Path, default=None)
    r.add_argument("--embed-matches", type=Path, default=None)
    r.add_argument("--min-testable-shingles", type=int, default=100)
    r.add_argument("--lexical-density", type=Path, default=None)
    r.add_argument("--min-density", type=float, default=0.35,
                   help="below this fraction of tokens >=4 chars the cleaned "
                        "text is notation, not prose")
    r.add_argument("--cosine-flag", type=float, default=0.85)
    r.set_defaults(func=cmd_report)

    return p



def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
