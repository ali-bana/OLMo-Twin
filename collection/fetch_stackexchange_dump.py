#!/usr/bin/env python3
"""Range-parallel, resumable downloader for the archive.org StackExchange dump.

archive.org throttles each connection to ~20 KB/s but permits several in
parallel (measured: 1 connection 22 KB/s, 8 connections 140 KB/s aggregate),
so each file is fetched as fixed-size byte ranges on a thread pool. Completed
ranges are recorded in <file>.ranges.json every 30 s, so an interrupted run
resumes without refetching. The /download/ URL redirects to a storage node;
it is resolved once per file with a HEAD request.

Files are the per-site archives, e.g. `math.stackexchange.com.7z` (Posts.xml
and the other tables inside) or, for Stack Overflow, the split
`stackoverflow.com-Posts.7z`.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

UA = os.environ.get("OLMO_TWIN_USER_AGENT", "OLMo-Twin/1.0 (research code)")
BASE = "https://archive.org/download/stackexchange/"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_range(url: str, path: Path, start: int, end: int, done: set, lock: threading.Lock,
                tries: int = 8) -> int:
    want = end - start + 1
    for attempt in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"},
                             stream=True, timeout=(30, 180))
            if r.status_code not in (200, 206):
                raise IOError(f"http {r.status_code}")
            buf = bytearray()
            for chunk in r.iter_content(1 << 16):
                buf.extend(chunk)
            if len(buf) != want:
                raise IOError(f"short read {len(buf)} != {want}")
            with lock:
                with open(path, "r+b") as fh:
                    fh.seek(start)
                    fh.write(buf)
                done.add(start)
            return want
        except Exception:
            time.sleep(min(60, 3 * (attempt + 1)))
    raise IOError(f"range {start}-{end} of {path.name} failed after {tries} tries")


def download(name: str, out_dir: Path, conns: int, chunk: int) -> None:
    path = out_dir / name
    meta = out_dir / (name + ".ranges.json")
    head = requests.head(BASE + name, headers={"User-Agent": UA}, allow_redirects=True, timeout=60)
    head.raise_for_status()
    url, size = head.url, int(head.headers["Content-Length"])
    done: set = set(json.loads(meta.read_text())) if meta.exists() else set()
    if not path.exists() or path.stat().st_size != size:
        with open(path, "wb") as fh:
            fh.truncate(size)
        done = set()
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk) if s not in done]
    log(f"{name}: {size / 2**30:.2f} GiB, {len(ranges)} ranges to fetch "
        f"({len(done)} already done), {conns} connections")
    lock = threading.Lock()
    got, t0, last = 0, time.time(), time.time()

    def save() -> None:
        meta.write_text(json.dumps(sorted(done)))

    with ThreadPoolExecutor(conns) as ex:
        futs = [ex.submit(fetch_range, url, path, s, e, done, lock) for s, e in ranges]
        for f in futs:
            got += f.result()
            if time.time() - last > 30:
                last = time.time()
                save()
                rate = got / max(time.time() - t0, 1e-9)
                have = min(len(done) * chunk, size)
                log(f"  {name}: {have / 2**30:.2f}/{size / 2**30:.2f} GiB  "
                    f"{rate / 2**20:.2f} MB/s  eta {(size - have) / max(rate, 1) / 3600:.1f} h")
    meta.unlink(missing_ok=True)
    log(f"{name}: complete ({size:,} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--conns", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8 << 20)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.files:
        download(name, a.out_dir, a.conns, a.chunk)
    log("ALL FILES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
