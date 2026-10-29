#!/usr/bin/env python3
"""Reliably test arXiv papers for presence in the OLMo 2 mid-training mixture.

Answers one question per candidate paper: does any part of it appear in the
downloaded ``dolmino-mix-1124`` (Stage 2) corpus? The design goal is a low
false-negative rate, so every stage is recall-first and precision is recovered
afterwards by verification.

Three tiers run in a single corpus pass, because reading 1.5 TB is the
dominant cost and re-reading it per tier would be wasteful:

1. ``id``    exact arXiv identifiers extracted from record ids, URLs, and body
             text, then intersected with the candidate set. Cheap and
             memory-flat: the corpus is mined for id-shaped tokens rather than
             searched for 574k patterns.
2. ``meta``  normalized title/abstract hashes matched against pes2o's
             structured metadata, plus an Aho-Corasick sweep for candidate
             titles inside unstructured text.
3. ``text``  winnowed fingerprints over ``k``-word shingles of the cleaned
             paper body (default k=13, winnow window 32). Winnowing is a local
             algorithm, so any shared run of ``k + winnow - 1`` = 44
             consecutive words is *guaranteed* to produce a shared
             fingerprint: a miss at that length is conclusive, not
             probabilistic. Word shingles rather than byte windows because
             there are ~6 bytes per word, which makes the pass ~6x cheaper at
             the same span, and because n-gram overlap is the formulation the
             contamination literature uses.

All comparisons happen on aggressively normalized text (NFKD, lowercased,
non-alphanumerics folded to separators, whitespace collapsed) so that
punctuation, casing, math markup, and layout differences between
LaTeX-cleaned arXiv text and web/OCR-derived corpus text cannot cause a miss.
Non-ASCII characters become separators rather than being deleted, so
"word-dash-word" cannot collapse into a single token and hide a match.

Memory is bounded deliberately: the fingerprint table and the automaton are
built in the parent process and inherited read-only through ``fork``, so peak
usage does not scale with worker count.

Commands:

  prepare       collect candidates for a month range and clean their sources
  fingerprints  build the winnowed fingerprint table from cleaned texts
  scan          one tiered pass over the Stage 2 corpus
  verify        adjudicate hits into per-paper verdicts
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

try:
    import ahocorasick
except ImportError:  # pragma: no cover
    ahocorasick = None

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap(args[0]) if args and callable(args[0]) else wrap


# Same scheme as our earlier OLMo 3 corpus search (not part of this
# repository), so hashes stay comparable.
HASH_BASE = 0x100000001B3
MASK64 = (1 << 64) - 1
INV_BASE = pow(HASH_BASE, -1, 1 << 64)
U64 = np.uint64

DEFAULT_SHINGLE_WORDS = 13
DEFAULT_WINNOW_SHINGLES = 32
# NOTE: stage-1 dclm ships as ".jsonl.zstd" (four-letter suffix) while dolmino
# uses ".jsonl.zst". Missing the former made discover_shards return an EMPTY
# list, which a scan reports as "no matches" -- a silent false negative over
# the whole corpus rather than an error.
SUPPORTED_SUFFIXES = (
    ".jsonl.zst", ".json.zst", ".jsonl.zstd", ".json.zstd",
    ".jsonl.gz", ".json.gz", ".jsonl",
)

# New-style (2007-04 onward) and legacy arXiv identifiers. Kept separately for
# testing, but the scan uses the combined pattern so the raw record is walked
# once instead of twice -- that second pass was ~25% of total scan time.
ARXIV_NEW_RE = re.compile(rb"(?<![0-9a-zA-Z.])(\d{4}\.\d{4,5})(?:v\d+)?(?![0-9])")
ARXIV_OLD_RE = re.compile(
    rb"(?<![0-9a-zA-Z/])([a-z][a-z\-\.]{2,20}/\d{7})(?:v\d+)?(?![0-9])"
)
ARXIV_ANY_RE = re.compile(
    rb"(?<![0-9a-zA-Z.])(\d{4}\.\d{4,5})(?:v\d+)?(?![0-9])"
    rb"|(?<![0-9a-zA-Z/])([a-z][a-z\-\.]{2,20}/\d{7})(?:v\d+)?(?![0-9])"
)

# Cheap literal gate. The identifier regex costs ~27 MiB/s because of its
# lookbehinds and because DCLM records carry long numeric classifier arrays it
# backtracks through; a memchr-speed literal test skips it for the vast
# majority of records. Requiring "arxiv" context is also *more* precise: a bare
# "2301.11760" in running text is an ordinary decimal number, not evidence.
# Full-text presence is tier 3's job, so no membership recall is lost here.
ARXIV_HINT_RE = re.compile(rb"arxiv", re.IGNORECASE)

_KEEP = np.zeros(256, dtype=np.uint8)
for _c in range(48, 58):  # 0-9
    _KEEP[_c] = _c
for _c in range(97, 123):  # a-z
    _KEEP[_c] = _c
_SPACE = 32

# Combining marks left behind by NFKD. These must be *removed* (so that "ö"
# folds to "o"), whereas every other non-ASCII character must become a space
# (so that "word–word" does not collapse into "wordword" and hide a real
# match). Getting this backwards causes silent false negatives.
_COMBINING_RE = re.compile(
    "[̀-ͯ҃-҉֑-ֽً-ٟัิ-ฺ"
    "᪰-᫿᷀-᷿⃐-⃰︠-︯]"
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


_PROGRESS_PATH: Path | None = None
_PROGRESS_LAST = 0.0


def set_progress_path(path: Path | None) -> None:
    global _PROGRESS_PATH
    _PROGRESS_PATH = path


def write_progress(
    phase: str,
    done: float,
    total: float,
    *,
    started: float,
    unit: str = "",
    note: str = "",
    force: bool = False,
) -> None:
    """Publish a machine-readable progress record for external monitoring.

    A long unattended run is otherwise blind: the log lines scroll past and
    there is no way to answer "how far along is it" without re-deriving it.
    Throttled to one write per second unless forced.
    """
    global _PROGRESS_LAST
    if _PROGRESS_PATH is None:
        return
    now = time.time()
    if not force and now - _PROGRESS_LAST < 1.0:
        return
    _PROGRESS_LAST = now
    elapsed = max(now - started, 1e-9)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 and total > done else 0.0
    payload = {
        "phase": phase,
        "done": round(done, 3),
        "total": round(total, 3),
        "unit": unit,
        "percent": round(100.0 * done / total, 2) if total else 0.0,
        "rate_per_second": round(rate, 3),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(remaining, 1),
        "note": note,
        "updated_unix": int(now),
    }
    tmp = _PROGRESS_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.replace(tmp, _PROGRESS_PATH)
    except OSError:  # pragma: no cover - progress must never break the run
        pass


def bar(done: float, total: float, width: int = 34) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = int(width * min(done / total, 1.0))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------


@njit(cache=True)
def _normalize_nb(raw):  # pragma: no cover - compiled
    """Lowercase, fold non-alphanumerics to separators, and collapse runs.

    Fused into one pass over the bytes. The numpy version needed five passes
    and five allocations per document, which showed up as ~30% of scan time.
    """
    n = len(raw)
    out = np.empty(n, dtype=np.uint8)
    w = 0
    pending = False
    for i in range(n):
        c = raw[i]
        if 97 <= c <= 122 or 48 <= c <= 57:
            pass
        elif 65 <= c <= 90:
            c = c + 32
        else:
            pending = w > 0
            continue
        if pending:
            out[w] = 32
            w += 1
            pending = False
        out[w] = c
        w += 1
    return out[:w]


def normalize(text: str) -> bytes:
    """Fold to lowercase alphanumeric words separated by single spaces.

    Recall-first: punctuation, math markup, hyphenation, and layout differ
    between LaTeX-cleaned arXiv text and web/OCR text, and none of those
    differences should hide a real copy.
    """
    if not text:
        return b""
    try:
        # Fast path: pure ASCII needs no decomposition, and bytes.lower() is a
        # C-level operation. This covers most corpus documents.
        raw = text.encode("ascii").lower()
    except UnicodeEncodeError:
        folded = _COMBINING_RE.sub("", unicodedata.normalize("NFKD", text).lower())
        # "replace" (not "ignore") so surviving non-ASCII becomes "?", which
        # the byte map below turns into a space rather than deleting it.
        raw = folded.encode("ascii", "replace")
    if not raw:
        return b""
    if HAS_NUMBA:
        return _normalize_nb(np.frombuffer(raw, dtype=np.uint8)).tobytes()
    arr = np.frombuffer(raw, dtype=np.uint8)
    mapped = _KEEP[np.where((arr >= 65) & (arr <= 90), arr + 32, arr)]
    mapped = np.where(mapped == 0, _SPACE, mapped)
    return b" ".join(mapped.tobytes().split())


def norm_hash(text: str) -> int:
    data = normalize(text)
    if not data:
        return 0
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


# --------------------------------------------------------------------------
# rolling hashes and winnowing
# --------------------------------------------------------------------------


@njit(cache=True)
def _window_hashes_nb(data, width):  # pragma: no cover - compiled
    n = len(data) - width + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    out = np.empty(n, dtype=np.uint64)
    base = np.uint64(HASH_BASE)
    inv = np.uint64(INV_BASE)
    value = np.uint64(0)
    power = np.uint64(1)
    for i in range(width):
        value += np.uint64(data[i]) * power
        power *= base
    top = power * inv  # base ** (width - 1)
    out[0] = value
    for i in range(1, n):
        value = (value - np.uint64(data[i - 1])) * inv
        value += np.uint64(data[i + width - 1]) * top
        out[i] = value
    return out


@njit(cache=True)
def _shingle_hashes_nb(data, k):  # pragma: no cover - compiled
    """Rolling hashes over k-word shingles of space-separated normalized text.

    Word-level shingles cost ~6x less than sliding a byte window one byte at a
    time (there are ~6 bytes per word), while a shingle of k words still spans
    the same number of characters. This is also the n-gram-overlap formulation
    used by the contamination literature.
    """
    n = len(data)
    # Per-word FNV-1a hashes.
    words = np.empty(n // 2 + 2, dtype=np.uint64)
    count = 0
    h = np.uint64(0xCBF29CE484222325)
    prime = np.uint64(0x100000001B3)
    in_word = False
    for i in range(n):
        c = data[i]
        if c == 32:
            if in_word:
                words[count] = h
                count += 1
                h = np.uint64(0xCBF29CE484222325)
                in_word = False
        else:
            h = (h ^ np.uint64(c)) * prime
            in_word = True
    if in_word:
        words[count] = h
        count += 1
    if count < k:
        return np.empty(0, dtype=np.uint64)
    # Rolling polynomial over k consecutive word hashes.
    out = np.empty(count - k + 1, dtype=np.uint64)
    base = np.uint64(HASH_BASE)
    inv = np.uint64(INV_BASE)
    value = np.uint64(0)
    power = np.uint64(1)
    for i in range(k):
        value += words[i] * power
        power *= base
    top = power * inv
    out[0] = value
    for i in range(1, count - k + 1):
        value = (value - words[i - 1]) * inv
        value += words[i + k - 1] * top
        out[i] = value
    return out


def shingle_hashes(data: bytes, k: int) -> np.ndarray:
    if not data:
        return np.empty(0, dtype=np.uint64)
    arr = np.frombuffer(data, dtype=np.uint8)
    if HAS_NUMBA:
        return _shingle_hashes_nb(arr, k)
    parts = data.split(b" ")
    if len(parts) < k:
        return np.empty(0, dtype=np.uint64)
    wh = np.array(
        [
            int.from_bytes(hashlib.blake2b(p, digest_size=8).digest(), "little")
            for p in parts
        ],
        dtype=U64,
    )
    count = len(wh) - k + 1
    powers = np.ones(k, dtype=U64)
    if k > 1:
        np.multiply.accumulate(np.full(k - 1, HASH_BASE, dtype=U64), out=powers[1:])
    return np.array(
        [np.sum(wh[i : i + k] * powers) for i in range(count)], dtype=U64
    )


def window_hashes(data: bytes, width: int) -> np.ndarray:
    if len(data) < width:
        return np.empty(0, dtype=np.uint64)
    arr = np.frombuffer(data, dtype=np.uint8)
    if HAS_NUMBA:
        return _window_hashes_nb(arr, width)
    size = len(arr)
    powers = np.ones(size, dtype=U64)
    if size > 1:
        np.multiply.accumulate(np.full(size - 1, HASH_BASE, dtype=U64), out=powers[1:])
    prefix = np.zeros(size + 1, dtype=U64)
    np.cumsum(arr.astype(U64) * powers, out=prefix[1:])
    count = size - width + 1
    inverse = np.ones(count, dtype=U64)
    if count > 1:
        np.multiply.accumulate(np.full(count - 1, INV_BASE, dtype=U64), out=inverse[1:])
    return (prefix[width:] - prefix[:count]) * inverse


@njit(cache=True)
def _winnow_nb(hashes, winnow):  # pragma: no cover - compiled
    """Sliding-window minimum via a monotonic deque: O(n), not O(n * winnow).

    The naive nested scan costs 256x more work per byte and made a full corpus
    pass take tens of hours. Ties resolve to the rightmost minimum, matching
    the reference winnowing formulation.
    """
    n = len(hashes)
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    if n <= winnow:
        best = 0
        for i in range(1, n):
            if hashes[i] <= hashes[best]:
                best = i
        return hashes[best : best + 1].copy()
    out = np.empty(n, dtype=np.uint64)
    dq = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0  # deque occupies dq[head:tail]
    count = 0
    prev = -1
    for i in range(n):
        while tail > head and hashes[dq[tail - 1]] >= hashes[i]:
            tail -= 1
        dq[tail] = i
        tail += 1
        while dq[head] <= i - winnow:
            head += 1
        if i >= winnow - 1:
            best = dq[head]
            if best != prev:
                out[count] = hashes[best]
                count += 1
                prev = best
    return out[:count].copy()


def winnow(hashes: np.ndarray, size: int) -> np.ndarray:
    """Select the minimum hash in every window of ``size`` consecutive hashes.

    This is the local-algorithm property from Schleimer et al.: any shared run
    of at least ``size`` consecutive positions must select the same fingerprint
    on both sides, which is what turns a miss into a conclusive answer.
    """
    if len(hashes) == 0:
        return hashes
    if HAS_NUMBA:
        return _winnow_nb(hashes, size)
    picks = []
    prev = -1
    for start in range(0, max(1, len(hashes) - size + 1)):
        seg = hashes[start : start + size]
        best = start + int(np.argmin(seg))
        if best != prev:
            picks.append(hashes[best])
            prev = best
    return np.asarray(picks, dtype=np.uint64)


# --------------------------------------------------------------------------
# degenerate-shingle filtering
#
# PDF-derived text carries artefacts that LaTeX source never did: LaTeXiT and
# Inkscape embed `<latexit sha1_base64="...">` markers in figures, and mangled
# glyph runs decode to long strings of single characters. Normalization turns
# those into word streams like "l a t e x i t s h a 1 b a s e 6 4" or
# "q q q q q q q ...". A 13-word shingle of single characters spans ~26 bytes
# of pure toolchain noise, so any two papers built with the same tool appear to
# share text. Measured: one pes2o record was claimed by 387 unrelated
# candidates at exactly 79 fingerprints each, another by 88 at 1,525 each.
#
# The filter drops a fingerprint whose shingle spans fewer than `min_chars`
# characters. It is applied AFTER winnowing and only to candidate documents,
# so the Schleimer local-algorithm guarantee is untouched: the winnow selection
# is computed on the unmodified shingle stream exactly as the corpus side
# computes it, and we merely decline to look up fingerprints that carry no
# information. A genuine shared passage still yields its non-degenerate
# fingerprints. Dropping is therefore conservative in the safe direction --
# it can only remove matches that rest ENTIRELY on junk.
# --------------------------------------------------------------------------

MIN_SHINGLE_CHARS_DEFAULT = 0  # 0 = off, preserving previously validated runs


def _word_bounds_py(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    starts, ends = [], []
    at = 0
    for part in data.split(b" "):
        if part:
            starts.append(at)
            ends.append(at + len(part))
        at += len(part) + 1
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
    )


def _winnow_picks_py(hashes: np.ndarray, size: int) -> np.ndarray:
    """Indices selected by winnowing -- mirrors :func:`winnow` exactly."""
    n = len(hashes)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n <= size:
        best = 0
        for i in range(1, n):
            if hashes[i] <= hashes[best]:
                best = i
        return np.asarray([best], dtype=np.int64)
    picks = []
    prev = -1
    for start in range(0, n - size + 1):
        seg = hashes[start : start + size]
        off = len(seg) - 1 - int(np.argmin(seg[::-1]))  # rightmost minimum
        best = start + off
        if best != prev:
            picks.append(best)
            prev = best
    return np.asarray(picks, dtype=np.int64)


def shingle_spans(data: bytes, k: int, n_shingles: int) -> np.ndarray:
    """Character span of each k-word shingle."""
    starts, ends = _word_bounds_py(data)
    if len(starts) < k or n_shingles <= 0:
        return np.empty(0, dtype=np.int64)
    n = min(n_shingles, len(starts) - k + 1)
    return ends[k - 1 : k - 1 + n] - starts[:n]


def fingerprint_text_filtered(
    data: bytes, width: int, size: int, min_chars: int
) -> np.ndarray:
    """Winnowed fingerprints with degenerate shingles removed post-selection."""
    hashes = shingle_hashes(data, width)
    if not len(hashes) or min_chars <= 0:
        return winnow(hashes, size)
    picks = _winnow_picks_py(hashes, size)
    if not len(picks):
        return np.empty(0, dtype=np.uint64)
    spans = shingle_spans(data, width, len(hashes))
    if not len(spans):
        return hashes[picks]
    picks = picks[picks < len(spans)]
    keep = picks[spans[picks] >= min_chars]
    return hashes[keep]


def fingerprint_text(data: bytes, width: int, size: int) -> np.ndarray:
    """Winnowed fingerprints of ``data``, keyed on k-word shingles.

    ``width`` is the shingle length in words and ``size`` the winnow window in
    shingles, so a shared run of ``width + size - 1`` consecutive words is
    guaranteed to yield a shared fingerprint.
    """
    return winnow(shingle_hashes(data, width), size)


# --------------------------------------------------------------------------
# corpus shard iteration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Shard:
    component: str
    path: str
    size: int


def discover_shards(root: Path, components: list[str] | None) -> list[Shard]:
    data_dir = root / "data" if (root / "data").is_dir() else root
    shards: list[Shard] = []
    for comp_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if components and comp_dir.name not in components:
            continue
        for path in sorted(comp_dir.rglob("*")):
            if path.is_file() and str(path).endswith(SUPPORTED_SUFFIXES):
                shards.append(
                    Shard(comp_dir.name, str(path), path.stat().st_size)
                )
    return shards


def iter_lines(path: str) -> Iterator[bytes]:
    if path.endswith((".zst", ".zstd")):
        if zstd is None:
            raise SystemExit("zstandard is required for .zst shards")
        with open(path, "rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                for line in io.BufferedReader(reader, 1 << 22):
                    if line.strip():
                        yield line
    elif path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            for line in handle:
                if line.strip():
                    yield line
    else:
        with open(path, "rb") as handle:
            for line in handle:
                if line.strip():
                    yield line


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def yymm_of(arxiv_id: str) -> str | None:
    base = arxiv_id.split("v")[0]
    if "." in base:
        head = base.split(".")[0]
        return head if len(head) == 4 and head.isdigit() else None
    if "/" in base:
        digits = base.split("/")[1]
        return digits[:4] if len(digits) >= 4 and digits[:4].isdigit() else None
    return None


def month_key(yymm: str) -> str:
    yy, mm = int(yymm[:2]), yymm[2:]
    year = 1900 + yy if yy >= 91 else 2000 + yy
    return f"{year:04d}-{mm}"


def iter_abstract_records(root: Path) -> Iterator[dict]:
    for path in sorted(root.rglob("papers.jsonl")):
        with path.open("rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def _clean_one(task: tuple[str, str, str]) -> dict:
    doc_id, src_path, out_path = task
    global _CLEANER
    try:
        text, status = _CLEANER.clean_arxiv_source_file(Path(src_path))
    except Exception as exc:  # pragma: no cover - defensive
        return {"doc_id": doc_id, "parser_status": f"error_{type(exc).__name__}"}
    if not text:
        return {"doc_id": doc_id, "parser_status": status}
    data = normalize(text)
    if not data:
        return {"doc_id": doc_id, "parser_status": "empty_after_normalize"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = zstd.ZstdCompressor(level=6).compress(data)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as handle:
        handle.write(payload)
    os.replace(tmp, out_path)
    return {
        "doc_id": doc_id,
        "parser_status": "ok",
        "norm_bytes": len(data),
        "norm_sha256": hashlib.sha256(data).hexdigest(),
        "text_path": out_path,
    }


_CLEANER = None


def _init_cleaner(cleaner_path: str) -> None:
    global _CLEANER
    import importlib.util

    spec = importlib.util.spec_from_file_location("_rpj_cleaner", cleaner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CLEANER = module


def cmd_prepare(args: argparse.Namespace) -> int:
    set_progress_path(getattr(args, "progress", None))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    text_dir = out / "texts"

    log(f"scanning abstract records under {args.abstract_dir}")
    seen: set[str] = set()
    rows: list[dict] = []
    for record in iter_abstract_records(args.abstract_dir):
        raw_id = record.get("arxiv_id")
        if not isinstance(raw_id, str):
            continue
        base = raw_id.split("v")[0]
        yymm = yymm_of(raw_id)
        if not yymm:
            continue
        key = month_key(yymm)
        if not (args.start_month <= key <= args.end_month):
            continue
        if base in seen:
            continue
        seen.add(base)
        rows.append(
            {
                "doc_id": base,
                "yymm": yymm,
                "month": key,
                "title": (record.get("title") or "").strip(),
                "abstract": (record.get("abstract") or "").strip(),
                "primary_category": record.get("primary_category") or "",
            }
        )
    rows.sort(key=lambda r: r["doc_id"])
    log(f"candidates in [{args.start_month}, {args.end_month}]: {len(rows):,}")
    if args.limit:
        rows = rows[: args.limit]
        log(f"limited to {len(rows):,}")

    tasks = []
    for row in rows:
        src = args.src_root / row["yymm"] / f"{row['doc_id']}.gz"
        dest = text_dir / row["yymm"] / f"{row['doc_id'].replace('/', '_')}.zst"
        row["source_path"] = str(src)
        row["text_path"] = str(dest)
        if dest.exists() and not args.force:
            row["parser_status"] = "cached"
            continue
        if not src.is_file():
            row["parser_status"] = "missing_source"
            continue
        tasks.append((row["doc_id"], str(src), str(dest)))

    log(f"cleaning {len(tasks):,} sources with {args.workers} workers")
    results: dict[str, dict] = {}
    if tasks:
        started = time.time()
        ctx = mp.get_context("fork")
        with ctx.Pool(
            args.workers, initializer=_init_cleaner, initargs=(str(args.cleaner),)
        ) as pool:
            for index, res in enumerate(
                pool.imap_unordered(_clean_one, tasks, chunksize=32), 1
            ):
                results[res["doc_id"]] = res
                write_progress(
                    "prepare", index, len(tasks), started=started, unit="papers",
                    note=args.start_month,
                )
                if index % 5000 == 0 or index == len(tasks):
                    rate = index / max(time.time() - started, 1e-9)
                    eta = (len(tasks) - index) / max(rate, 1e-9)
                    log(
                        f"  {bar(index, len(tasks))} {index:,}/{len(tasks):,}"
                        f"  {rate:.0f}/s  eta {eta / 60:.1f} min"
                    )
        log(f"cleaning done in {time.time() - started:.0f}s")

    for row in rows:
        res = results.get(row["doc_id"])
        if res:
            row.update({k: v for k, v in res.items() if k != "doc_id"})
        elif row.get("parser_status") == "cached":
            path = Path(row["text_path"])
            data = zstd.ZstdDecompressor().decompress(path.read_bytes())
            row["parser_status"] = "ok"
            row["norm_bytes"] = len(data)
            row["norm_sha256"] = hashlib.sha256(data).hexdigest()

    fields = [
        "doc_id",
        "yymm",
        "month",
        "primary_category",
        "title",
        "abstract",
        "source_path",
        "text_path",
        "parser_status",
        "norm_bytes",
        "norm_sha256",
    ]
    manifest = out / "candidates.csv.gz"
    with gzip.open(manifest, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.get("parser_status", "unknown")] = (
            status_counts.get(row.get("parser_status", "unknown"), 0) + 1
        )
    atomic_json(
        out / "prepare_summary.json",
        {
            "start_month": args.start_month,
            "end_month": args.end_month,
            "candidates": len(rows),
            "parser_status": status_counts,
            "abstract_dir": str(args.abstract_dir),
            "src_root": str(args.src_root),
        },
    )
    log(f"wrote {manifest}")
    log(f"parser status: {status_counts}")
    return 0


# --------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------


def read_candidates(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fp_chunk(task: tuple[int, list, str, int, int, int]) -> dict:
    chunk_id, items, out_dir, width, size, min_chars = task
    dctx = zstd.ZstdDecompressor()
    fps: list[np.ndarray] = []
    ows: list[np.ndarray] = []
    failed = 0
    for index, text_path in items:
        try:
            data = dctx.decompress(Path(text_path).read_bytes())
        except Exception:
            failed += 1
            continue
        sel = fingerprint_text_filtered(data, width, size, min_chars)
        if not len(sel):
            failed += 1
            continue
        sel = np.unique(sel)
        fps.append(sel)
        ows.append(np.full(len(sel), index, dtype=np.uint32))
    if not fps:
        return {"chunk": chunk_id, "count": 0, "failed": failed, "docs": len(items)}
    fp = np.concatenate(fps)
    ow = np.concatenate(ows)
    np.save(Path(out_dir) / f"fp_{chunk_id:05d}.npy", fp)
    np.save(Path(out_dir) / f"ow_{chunk_id:05d}.npy", ow)
    return {
        "chunk": chunk_id,
        "count": int(len(fp)),
        "failed": failed,
        "docs": len(items),
    }


def cmd_fingerprints(args: argparse.Namespace) -> int:
    set_progress_path(getattr(args, "progress", None))
    rows = read_candidates(args.candidates)
    usable = [r for r in rows if r.get("parser_status") == "ok" and r.get("text_path")]
    log(f"candidates={len(rows):,}  with cleaned text={len(usable):,}")
    if not usable:
        raise SystemExit("no cleaned candidate texts to fingerprint")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts = args.out_dir / "parts"
    if parts.exists():
        for stale in parts.glob("*.npy"):
            stale.unlink()
    parts.mkdir(parents=True, exist_ok=True)

    workers = max(1, getattr(args, "workers", 1))
    per_chunk = max(200, len(usable) // (workers * 8) + 1)
    tasks = []
    for chunk_id, start in enumerate(range(0, len(usable), per_chunk)):
        items = [
            (i, usable[i]["text_path"])
            for i in range(start, min(start + per_chunk, len(usable)))
        ]
        tasks.append((chunk_id, items, str(parts), args.window, args.winnow,
                      int(getattr(args, 'min_shingle_chars', 0))))

    log(f"fingerprinting {len(usable):,} docs in {len(tasks)} chunks / {workers} workers")
    started = time.time()
    done_docs = 0
    total = 0
    failed = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(_fp_chunk, tasks):
            done_docs += res["docs"]
            total += res["count"]
            failed += res["failed"]
            write_progress(
                "fingerprints", done_docs, len(usable), started=started,
                unit="papers", note=f"{total:,} fingerprints",
            )
            rate = done_docs / max(time.time() - started, 1e-9)
            log(
                f"  {bar(done_docs, len(usable))} {done_docs:,}/{len(usable):,}"
                f"  {rate:.0f}/s  fp={total:,}"
                f"  eta {(len(usable) - done_docs) / max(rate, 1e-9) / 60:.1f} min"
            )

    log(f"concatenating {total:,} fingerprints")
    fp_all = np.empty(total, dtype=np.uint64)
    ow_all = np.empty(total, dtype=np.uint32)
    at = 0
    for path in sorted(parts.glob("fp_*.npy")):
        chunk = np.load(path)
        own = np.load(parts / path.name.replace("fp_", "ow_"))
        fp_all[at : at + len(chunk)] = chunk
        ow_all[at : at + len(chunk)] = own
        at += len(chunk)
    fp_all = fp_all[:at]
    ow_all = ow_all[:at]
    order = np.argsort(fp_all, kind="stable")
    fp_all = fp_all[order]
    ow_all = ow_all[order]

    np.save(args.out_dir / "fingerprints.npy", fp_all)
    np.save(args.out_dir / "owners.npy", ow_all)
    with gzip.open(args.out_dir / "index_docs.txt.gz", "wt", encoding="utf-8") as fh:
        for row in usable:
            fh.write(f"{row['doc_id']}\n")
    atomic_json(
        args.out_dir / "fingerprint_meta.json",
        {
            "window": args.window,
            "winnow": args.winnow,
            "unit": "words",
            "guaranteed_shared_words": args.window + args.winnow - 1,
            "hash_base": HASH_BASE,
            "documents": len(usable),
            "documents_without_fingerprints": failed,
            "fingerprints": int(len(fp_all)),
            "bytes_in_memory": int(fp_all.nbytes + ow_all.nbytes),
            "normalization": "NFKD+lower+alnum_fold+ws_collapse",
        },
    )
    for stale in parts.glob("*.npy"):
        stale.unlink()
    write_progress(
        "fingerprints", len(usable), len(usable), started=started,
        unit="papers", note="complete", force=True,
    )
    log(
        f"fingerprints={len(fp_all):,} for {len(usable):,} docs  "
        f"({(fp_all.nbytes + ow_all.nbytes) / 2**30:.2f} GiB)  "
        f"guarantee>={args.window + args.winnow - 1} words  "
        f"no_fp_docs={failed:,}"
    )
    return 0


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------

_G: dict = {}


def _init_scan(state: dict) -> None:
    global _G
    _G = state


def _record_ref(record: dict) -> str:
    for key in ("id", "_id"):
        if isinstance(record.get(key), str):
            return record[key][:200]
    meta = record.get("metadata")
    if isinstance(meta, dict):
        for key in ("provenance", "url", "warc_url"):
            if isinstance(meta.get(key), str):
                return meta[key][:200]
    return ""


def _scan_shard(shard_value: dict) -> dict:
    shard = Shard(**shard_value)
    g = _G
    id_set: dict[str, int] = g["id_set"]
    title_hash: dict[int, int] = g["title_hash"]
    abstract_hash: dict[int, int] = g["abstract_hash"]
    automaton = g["automaton"]
    fps: np.ndarray = g["fps"]
    owners: np.ndarray = g["owners"]
    window: int = g["window"]
    winnow_size: int = g["winnow"]
    out_dir = Path(g["out_dir"])
    cap = g["max_hits_per_doc"]

    marker = out_dir / "shards" / (shard_key(shard) + ".done.json")
    if marker.exists() and not g["force"]:
        return json.loads(marker.read_text())

    hits: dict[tuple[int, str], dict] = {}
    rows = 0
    text_bytes = 0
    started = time.time()

    def add(idx: int, kind: str, ref: str, extra: dict | None = None) -> None:
        key = (idx, kind)
        entry = hits.get(key)
        if entry is None:
            entry = {"doc": idx, "kind": kind, "n": 0, "refs": [], "best": 0}
            hits[key] = entry
        entry["n"] += 1
        item = {"ref": ref}
        if extra:
            item.update(extra)
        count = int(item.get("fingerprints", 0))
        if count > entry["best"]:
            entry["best"] = count
        if len(entry["refs"]) < cap:
            entry["refs"].append(item)
        elif count:
            # Keep the STRONGEST evidence, not the first seen. Retaining the
            # first N would let incidental one-fingerprint matches fill the cap
            # and evict the corpus document that actually contains the paper --
            # a false negative manufactured by the cap itself.
            weakest = min(
                range(len(entry["refs"])),
                key=lambda i: int(entry["refs"][i].get("fingerprints", 0)),
            )
            if count > int(entry["refs"][weakest].get("fingerprints", 0)):
                entry["refs"][weakest] = item

    for raw in iter_lines(shard.path):
        rows += 1
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        text = record.get("text")
        if not isinstance(text, str):
            continue
        ref = _record_ref(record)

        # --- tier 1: arXiv identifiers, mined from the record itself ---
        # Run against the raw record (not the normalized text) because
        # normalization folds the "." in "2301.11760" into a space. Gated on a
        # literal "arxiv" so the expensive pattern runs only where an
        # identifier could plausibly be referenced.
        if ARXIV_HINT_RE.search(raw):
            for match in ARXIV_ANY_RE.finditer(raw):
                token = (match.group(1) or match.group(2)).decode("ascii", "ignore")
                idx = id_set.get(token)
                if idx is not None:
                    add(idx, "id", ref, {"token": token})

        # --- tier 2a: pes2o structured metadata ---
        meta = record.get("metadata")
        if isinstance(meta, dict):
            mt = meta.get("title")
            if isinstance(mt, str) and mt:
                idx = title_hash.get(norm_hash(mt))
                if idx is not None:
                    add(idx, "meta_title", ref)
            ma = meta.get("abstract")
            if isinstance(ma, str) and ma:
                idx = abstract_hash.get(norm_hash(ma))
                if idx is not None:
                    add(idx, "meta_abstract", ref)

        data = normalize(text)
        if not data:
            continue
        text_bytes += len(data)

        # --- tier 2b: candidate titles inside unstructured text ---
        if automaton is not None:
            padded = b" " + data + b" "
            try:
                sdata = padded.decode("ascii")
            except UnicodeDecodeError:  # pragma: no cover
                sdata = padded.decode("ascii", "ignore")
            for _, idx in automaton.iter(sdata):
                add(idx, "title_in_text", ref)

        # --- tier 3: winnowed fingerprints ---
        if len(fps):
            sel = fingerprint_text(data, window, winnow_size)
            if len(sel):
                left = np.searchsorted(fps, sel, side="left")
                right = np.searchsorted(fps, sel, side="right")
                found = right > left
                if found.any():
                    per_doc: dict[int, int] = {}
                    for lo, hi in zip(left[found], right[found]):
                        for owner in owners[lo:hi]:
                            per_doc[int(owner)] = per_doc.get(int(owner), 0) + 1
                    for idx, count in per_doc.items():
                        add(idx, "text", ref, {"fingerprints": count})

    out_path = out_dir / "shards" / (shard_key(shard) + ".hits.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wt", encoding="utf-8") as handle:
        for entry in hits.values():
            handle.write(json.dumps(entry) + "\n")
    stats = {
        "component": shard.component,
        "path": shard.path,
        "rows": rows,
        "normalized_bytes": text_bytes,
        "hit_entries": len(hits),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(stats))
    return stats


def shard_key(shard: Shard) -> str:
    digest = hashlib.sha256(shard.path.encode()).hexdigest()[:12]
    stem = Path(shard.path).name.replace(".", "_")
    return f"{shard.component}__{stem}__{digest}"


def cmd_scan(args: argparse.Namespace) -> int:
    set_progress_path(getattr(args, "progress", None))
    rows = read_candidates(args.candidates)
    usable = [r for r in rows if r.get("parser_status") == "ok"]
    doc_ids = [r["doc_id"] for r in usable]
    index_of = {d: i for i, d in enumerate(doc_ids)}

    fp_dir = args.fingerprint_dir
    fps = np.load(fp_dir / "fingerprints.npy", mmap_mode="r")
    owners = np.load(fp_dir / "owners.npy", mmap_mode="r")
    meta = json.loads((fp_dir / "fingerprint_meta.json").read_text())
    with gzip.open(fp_dir / "index_docs.txt.gz", "rt") as fh:
        fp_docs = [line.strip() for line in fh if line.strip()]
    if fp_docs != doc_ids:
        raise SystemExit(
            "fingerprint document order does not match the candidate manifest; "
            "rebuild fingerprints from this manifest"
        )

    id_set = {d: i for i, d in enumerate(doc_ids)}
    title_hash: dict[int, int] = {}
    abstract_hash: dict[int, int] = {}
    automaton = None
    if ahocorasick is not None and not args.no_title_automaton:
        automaton = ahocorasick.Automaton()
    added = 0
    for i, row in enumerate(usable):
        title_norm = normalize(row.get("title", ""))
        if len(title_norm) >= args.min_title_bytes:
            title_hash.setdefault(
                int.from_bytes(
                    hashlib.blake2b(title_norm, digest_size=8).digest(), "big"
                ),
                i,
            )
            if automaton is not None:
                automaton.add_word(f" {title_norm.decode('ascii')} ", i)
                added += 1
        abstract_norm = normalize(row.get("abstract", ""))
        if len(abstract_norm) >= args.min_abstract_bytes:
            abstract_hash.setdefault(
                int.from_bytes(
                    hashlib.blake2b(abstract_norm, digest_size=8).digest(), "big"
                ),
                i,
            )
    if automaton is not None and added:
        automaton.make_automaton()
        log(f"title automaton: {added:,} titles")
    else:
        automaton = None

    shards = discover_shards(args.corpus, args.components)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    total_bytes = sum(s.size for s in shards)
    log(
        f"docs={len(doc_ids):,}  fingerprints={len(fps):,}  "
        f"({(fps.nbytes + owners.nbytes) / 2**30:.2f} GiB)  "
        f"guarantee>={meta['guaranteed_shared_words']} words"
    )
    log(
        f"shards={len(shards):,}  compressed={total_bytes / 2**30:.1f} GiB  "
        f"workers={args.workers}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "id_set": id_set,
        "title_hash": title_hash,
        "abstract_hash": abstract_hash,
        "automaton": automaton,
        "fps": fps,
        "owners": owners,
        "window": meta["window"],
        "winnow": meta["winnow"],
        "out_dir": str(args.out_dir),
        "max_hits_per_doc": args.max_refs_per_hit,
        "force": args.force,
    }

    done_bytes = 0
    started = time.time()
    ctx = mp.get_context("fork")
    payload = [s.__dict__ for s in shards]
    with ctx.Pool(args.workers, initializer=_init_scan, initargs=(state,)) as pool:
        for i, stats in enumerate(
            pool.imap_unordered(_scan_shard, payload, chunksize=1), 1
        ):
            done_bytes += next(
                (s.size for s in shards if s.path == stats["path"]), 0
            )
            write_progress(
                "scan", done_bytes, total_bytes, started=started, unit="bytes",
                note=f"{i}/{len(shards)} shards",
            )
            if i % 10 == 0 or i == len(shards):
                el = time.time() - started
                rate = done_bytes / max(el, 1e-9) / 2**20
                remain = (total_bytes - done_bytes) / max(
                    done_bytes / max(el, 1e-9), 1e-9
                )
                log(
                    f"  {bar(done_bytes, total_bytes)} {i}/{len(shards)} shards"
                    f"  {done_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB"
                    f"  {rate:.0f} MB/s  eta {remain / 3600:.2f} h"
                )
    atomic_json(
        args.out_dir / "scan_summary.json",
        {
            "shards": len(shards),
            "compressed_bytes": total_bytes,
            "elapsed_seconds": round(time.time() - started, 1),
            "documents": len(doc_ids),
            "fingerprint_meta": meta,
            "components": sorted({s.component for s in shards}),
        },
    )
    log(f"scan complete in {(time.time() - started) / 60:.1f} min")
    return 0


# --------------------------------------------------------------------------
# verify / report
# --------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    rows = read_candidates(args.candidates)
    usable = [r for r in rows if r.get("parser_status") == "ok"]
    doc_ids = [r["doc_id"] for r in usable]

    agg: dict[int, dict] = {}
    files = sorted((args.scan_dir / "shards").glob("*.hits.jsonl"))
    log(f"merging {len(files):,} shard hit files")
    for path in files:
        with path.open("rt", encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                idx = entry["doc"]
                acc = agg.setdefault(
                    idx,
                    {
                        "kinds": {},
                        "best_doc_fp": 0,
                        "docs_with_text": 0,
                        "total_fp": 0,
                        "top_ref": "",
                    },
                )
                kind = entry["kind"]
                acc["kinds"][kind] = acc["kinds"].get(kind, 0) + entry["n"]
                if kind == "text":
                    acc["docs_with_text"] += entry["n"]
                    for item in entry.get("refs", []):
                        fp = int(item.get("fingerprints", 0))
                        acc["total_fp"] += fp
                        # Strength is distinct fingerprints shared with ONE
                        # corpus document. Summing across documents would turn
                        # a single boilerplate phrase seen in a thousand
                        # records into an apparent thousand-fingerprint match.
                        if fp > acc["best_doc_fp"]:
                            acc["best_doc_fp"] = fp
                            acc["top_ref"] = item.get("ref", "")

    out_rows = []
    verdict_counts: dict[str, int] = {}
    for i, row in enumerate(usable):
        acc = agg.get(i)
        if not acc:
            verdict, detail = "no_evidence", ""
        else:
            kinds = acc["kinds"]
            best = acc["best_doc_fp"]
            has_title = "meta_title" in kinds
            has_abstract = "meta_abstract" in kinds
            # Two materially different kinds of membership, kept separate on
            # purpose. pes2o records are title+abstract only (measured: median
            # 183 words, zero words beyond title+abstract), while the
            # RedPajama cleaner strips everything before the first section --
            # so candidate fingerprints cover the BODY and exclude the
            # abstract. A pes2o match therefore proves the model saw the
            # abstract, not the body. An attack that scores body text must not
            # treat those two as the same condition.
            strong_text = best >= args.strong_text_fingerprints
            probable_text = best >= args.probable_text_fingerprints
            meta_exact = has_title and has_abstract
            if meta_exact and strong_text:
                verdict = "member_abstract_and_fulltext"
            elif meta_exact:
                verdict = "member_abstract_only"
            elif strong_text:
                verdict = "member_fulltext"
            elif has_title or has_abstract:
                verdict = "member_metadata_partial"
            elif probable_text:
                verdict = "probable_fulltext"
            elif best >= args.min_text_fingerprints:
                verdict = "partial_text_overlap"
            elif best >= 1:
                verdict = "weak_text_overlap"
            elif "title_in_text" in kinds:
                verdict = "title_only"
            elif "id" in kinds:
                verdict = "citation_only"
            else:
                verdict = "weak_other"
            detail = json.dumps(
                {
                    "kinds": kinds,
                    "best_single_doc_fingerprints": best,
                    "corpus_docs_with_text_overlap": acc["docs_with_text"],
                    "summed_fingerprints": acc["total_fp"],
                    "top_ref": acc["top_ref"][:120],
                }
            )
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        out_rows.append(
            {
                "doc_id": row["doc_id"],
                "month": row["month"],
                "primary_category": row.get("primary_category", ""),
                "norm_bytes": row.get("norm_bytes", ""),
                "verdict": verdict,
                "evidence": detail,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "verdicts.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "doc_id",
                "month",
                "primary_category",
                "norm_bytes",
                "verdict",
                "evidence",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    atomic_json(
        args.out_dir / "verify_summary.json",
        {
            "documents": len(out_rows),
            "verdicts": verdict_counts,
            "min_text_fingerprints": args.min_text_fingerprints,
        },
    )
    log(f"wrote {path}")
    for k, v in sorted(verdict_counts.items(), key=lambda kv: -kv[1]):
        log(f"  {k}: {v:,}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Summarize verdicts and measure tier-3 recall against known positives.

    The positive control costs nothing extra. A candidate whose normalized
    title AND abstract both match a pes2o record's structured metadata is
    certainly present in Stage 2. Asking how many of those the independent
    text tier also found gives a measured recall for the fingerprint path,
    instead of assuming it.
    """
    rows = read_candidates(args.candidates)
    verdicts = {}
    with gzip.open(args.verdicts, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            verdicts[row["doc_id"]] = row

    by_year: dict[str, dict[str, int]] = {}
    parser_by_year: dict[str, dict[str, int]] = {}
    kinds_of: dict[str, dict] = {}
    for row in rows:
        year = row["month"][:4]
        parser_by_year.setdefault(year, {})
        status = row.get("parser_status", "unknown")
        parser_by_year[year][status] = parser_by_year[year].get(status, 0) + 1
        v = verdicts.get(row["doc_id"])
        if not v:
            continue
        by_year.setdefault(year, {})
        by_year[year][v["verdict"]] = by_year[year].get(v["verdict"], 0) + 1
        try:
            kinds_of[row["doc_id"]] = json.loads(v["evidence"] or "{}")
        except json.JSONDecodeError:
            kinds_of[row["doc_id"]] = {}

    # Positive control: title AND abstract both matched pes2o metadata.
    known = [
        d
        for d, ev in kinds_of.items()
        if "meta_title" in (ev.get("kinds") or {})
        and "meta_abstract" in (ev.get("kinds") or {})
    ]
    def fp_of(doc: str) -> int:
        ev = kinds_of.get(doc) or {}
        # verify writes best_single_doc_fingerprints; an older field name here
        # silently reported 0% recall, so accept both and prefer the current.
        return int(
            ev.get("best_single_doc_fingerprints")
            or ev.get("text_fingerprints")
            or 0
        )

    detected_by_text = [d for d in known if fp_of(d) >= 1]
    strongly = [d for d in known if fp_of(d) >= args.strong_text_fingerprints]
    control = {
        "known_positives": len(known),
        "found_by_text_tier": len(detected_by_text),
        "found_at_threshold": len(strongly),
        "text_tier_recall": (
            round(len(detected_by_text) / len(known), 4) if known else None
        ),
        "text_tier_recall_at_threshold": (
            round(len(strongly) / len(known), 4) if known else None
        ),
        "min_text_fingerprints": args.min_text_fingerprints,
        "strong_text_fingerprints": args.strong_text_fingerprints,
    }

    lines = ["# OLMo 2 Mid-Training Membership Search", ""]
    lines.append(f"- Candidates: {len(rows):,}")
    lines.append(f"- Adjudicated: {len(verdicts):,}")
    lines.append("")
    lines.append("## Verdicts by year")
    lines.append("")
    all_verdicts = sorted({v for d in by_year.values() for v in d})
    lines.append("| year | " + " | ".join(all_verdicts) + " |")
    lines.append("|---|" + "---|" * len(all_verdicts))
    for year in sorted(by_year):
        cells = [f"{by_year[year].get(v, 0):,}" for v in all_verdicts]
        lines.append(f"| {year} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Positive control (measured, not assumed)")
    lines.append("")
    if known:
        lines.append(
            f"{len(known):,} candidates matched a pes2o record on BOTH normalized "
            f"title and abstract, so they are certainly in Stage 2. Of those, the "
            f"independent text tier found {len(detected_by_text):,} "
            f"({100 * len(detected_by_text) / len(known):.2f}%) with at least one "
            f"fingerprint and {len(strongly):,} "
            f"({100 * len(strongly) / len(known):.2f}%) at the full-text "
            f"threshold of {args.strong_text_fingerprints}."
        )
        miss = len(known) - len(detected_by_text)
        if miss:
            lines.append("")
            lines.append(
                f"The {miss:,} known positives the text tier did not find "
                f"({100 * miss / len(known):.2f}%) must NOT be read as a "
                f"false-negative rate for the fingerprint method. This control "
                f"set is defined by having both title and abstract metadata, "
                f"and in dolmino only the abstract-only pes2o shard carries "
                f"those fields (median ~177 words); the full-text shards "
                f"(median ~3,720 words) have no title/abstract metadata at "
                f"all. So these records are abstract-only by construction, "
                f"while candidate fingerprints cover the paper BODY -- the "
                f"RedPajama cleaner strips everything before the first "
                f"section, abstract included. The two sides share almost no "
                f"text by design. The "
                f"{len(detected_by_text):,} that DID match must additionally "
                f"appear in a full-text shard."
            )
            lines.append("")
            lines.append(
                "A separate cross-tier check does bound the method's "
                "reliability: on pass A, whether a paper's title also appeared "
                "in the same corpus record rose monotonically with shared "
                "fingerprints -- 0.1% at 1, 0.4% at 2-9, 30.1% at 10-49, "
                "90.4% at 50-499, 94.0% at 500+. That is why the full-text "
                "threshold is 50 and why 10-49 is labelled probable rather "
                "than confirmed."
            )
    else:
        lines.append(
            "No candidate matched pes2o on both title and abstract, so tier-3 "
            "recall could not be measured from this run."
        )
    lines.append("")
    lines.append("## Coverage limits")
    lines.append("")
    lines.append("| year | " + " | ".join(
        sorted({s for d in parser_by_year.values() for s in d})) + " |")
    statuses = sorted({s for d in parser_by_year.values() for s in d})
    lines.append("|---|" + "---|" * len(statuses))
    for year in sorted(parser_by_year):
        cells = [f"{parser_by_year[year].get(s, 0):,}" for s in statuses]
        lines.append(f"| {year} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Papers without a cleanable local LaTeX source were never fingerprinted. "
        "They are NOT nonmembers -- they are untested, and must be excluded from "
        "any nonmember pool rather than counted as absent."
    )
    lines.append("")
    lines.append(
        "A `no_evidence` verdict means no exact evidence under these tests. It "
        "does not exclude a paraphrase, a translation, or an OCR-mangled copy."
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(
        args.out_dir / "report_summary.json",
        {
            "candidates": len(rows),
            "adjudicated": len(verdicts),
            "verdicts_by_year": by_year,
            "parser_status_by_year": parser_by_year,
            "positive_control": control,
        },
    )
    log(f"wrote {args.out_dir / 'report.md'}")
    log(f"positive control: {json.dumps(control)}")
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="collect candidates and clean sources")
    p.add_argument("--abstract-dir", type=Path, required=True)
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--start-month", required=True)
    p.add_argument("--end-month", required=True)
    p.add_argument(
        "--cleaner",
        type=Path,
        default=Path(__file__).with_name("redpajama_arxiv_cleaner_standalone.py"),
    )
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--progress", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("fingerprints", help="build the winnowed fingerprint table")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--progress", type=Path, default=None)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--window", "--shingle-words", dest="window", type=int,
                   default=DEFAULT_SHINGLE_WORDS,
                   help="shingle length in words (default 13)")
    p.add_argument("--winnow", "--winnow-shingles", dest="winnow", type=int,
                   default=DEFAULT_WINNOW_SHINGLES,
                   help="winnow window in shingles (default 32)")
    p.add_argument("--min-shingle-chars", type=int,
                   default=MIN_SHINGLE_CHARS_DEFAULT,
                   help="drop fingerprints whose shingle spans fewer than this "
                        "many characters; suppresses PDF-toolchain artefacts "
                        "such as latexit base64 markers and glyph runs that "
                        "normalize to single-character word streams "
                        "(0 = off, the pre-PDF behaviour)")
    p.set_defaults(func=cmd_fingerprints)

    p = sub.add_parser("scan", help="one tiered pass over the corpus")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--fingerprint-dir", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--components", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit-shards", type=int, default=0)
    p.add_argument("--max-refs-per-hit", type=int, default=8)
    p.add_argument("--min-title-bytes", type=int, default=24)
    p.add_argument("--min-abstract-bytes", type=int, default=64)
    p.add_argument("--no-title-automaton", action="store_true")
    p.add_argument("--progress", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("report", help="summarize verdicts and measure recall")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--verdicts", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-text-fingerprints", type=int, default=2)
    p.add_argument("--strong-text-fingerprints", type=int, default=50)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("verify", help="adjudicate hits into verdicts")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--scan-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-text-fingerprints", type=int, default=2,
                   help="distinct fingerprints in ONE corpus doc for partial overlap")
    # Calibrated, not guessed. Cross-tier corroboration on pass A (whether the
    # paper's title also appears in the same corpus record) rises monotonically
    # with fingerprint count: 0.1% at 1 fp, 0.4% at 2-9, 30.1% at 10-49, 90.4%
    # at 50-499, 94.0% at 500+. So >=50 is where a match is reliably the same
    # paper, and 10-49 is a genuinely ambiguous band that deserves its own
    # label rather than being forced either way.
    p.add_argument("--strong-text-fingerprints", type=int, default=50,
                   help="fingerprints in ONE corpus doc to call it a full-text "
                        "member (>=50: 90%% corroborated by an independent tier)")
    p.add_argument("--probable-text-fingerprints", type=int, default=10,
                   help="fingerprints for the ambiguous band (10-49: ~30%% "
                        "corroborated; needs adjudication)")
    p.set_defaults(func=cmd_verify)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
