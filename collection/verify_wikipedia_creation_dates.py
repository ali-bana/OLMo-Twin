#!/usr/bin/env python3
"""Re-label the Wikipedia pools using each page's TRUE first-revision date.

Why this exists
---------------
The first pool build derived "created after the cutoff" from the page-creation
log, matching log entries to pages by TITLE. That is wrong. A `create` log entry
fires whenever anything is created at a title -- a redirect left behind by a page
move, a recreation after deletion -- while the underlying article may be decades
old. Measured consequence: page 201575 "White-tailed eagle", first revision
2003-03-26, was labelled as created 2026-03, and it is present verbatim in the
2023 corpus snapshot. Roughly half of group A showed heavy overlap with the
training corpus for exactly this reason.

A page's identity is its page_id, and its age is the timestamp of its first
revision. That is what this script fetches, one page per API call (the API
refuses rvdir/rvlimit for multi-page queries), and it rewrites the group labels:

    A  first_revision >= cutoff                      genuinely new
    B  first_revision <  cutoff <= last_revision     pre-existing, edited after
    X  first_revision <  cutoff, no edit after       neither; dropped

Pages whose first revision cannot be established are dropped rather than
guessed: an unverified date is not a basis for a non-membership claim.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = os.environ.get("OLMO_TWIN_USER_AGENT", "OLMo-Twin/1.0 (research code)")
CUTOFF = "2025-04-17"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


BACKOFF = (2, 4, 8, 16, 32, 60)


def first_revision(session: requests.Session, page_id: str,
                   attempts: int = len(BACKOFF)) -> tuple[str, str | None, str | None]:
    """(page_id, ISO timestamp of the earliest revision or None, error kind).

    error kind is None on success, 'missing' when the API knows no such page
    (deleted), or the exception class name / HTTP status after all retries.
    The first version gave up after 4 quick tries and recorded nothing, so
    2,869 pages -- 97% of them live and retrievable -- were dropped as
    'unknown'.
    """
    last_err = "unknown"
    for attempt in range(attempts):
        try:
            r = session.get(
                API,
                params={
                    "action": "query", "prop": "revisions", "pageids": page_id,
                    "rvdir": "newer", "rvlimit": 1, "rvprop": "timestamp",
                    "format": "json", "formatversion": "2",
                },
                timeout=60,
            )
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"http_{r.status_code}"
                raise IOError(last_err)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing"):
                return page_id, None, "missing"
            revs = pages[0].get("revisions") or []
            if not revs:
                return page_id, None, "no_revisions"
            return page_id, revs[0]["timestamp"], None
        except Exception as exc:
            last_err = last_err if last_err.startswith("http_") else type(exc).__name__
            if attempt + 1 < attempts:
                time.sleep(BACKOFF[attempt])
    return page_id, None, last_err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiki-pool", type=Path, required=True)
    ap.add_argument("--page-meta", type=Path, required=True,
                    help="page_meta.tsv.gz: page_id, title, last_revision")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--cutoff", default=CUTOFF)
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(gzip.open(args.wiki_pool, "rt", newline="")))
    log(f"pool rows: {len(rows):,}")

    last_rev: dict[str, str] = {}
    with gzip.open(args.page_meta, "rt") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                last_rev[p[0]] = p[2]

    page_ids = [r["doc_id"].split(":", 1)[1] for r in rows]
    log(f"fetching first revision for {len(page_ids):,} pages "
        f"({args.concurrency} concurrent)")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Persistent cache: every answer (and every final failure) is appended as
    # it arrives, so a rerun only fetches what is still unknown.
    cache_path = args.out.parent / "first_revisions_cache.jsonl"
    firsts: dict[str, str | None] = {}
    errors: dict[str, str] = {}
    if cache_path.exists():
        with cache_path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("ts"):
                    firsts[rec["page_id"]] = rec["ts"]
                elif rec.get("error") == "missing":
                    firsts[rec["page_id"]] = None
                    errors[rec["page_id"]] = "missing"
        log(f"cache: {len(firsts):,} pages already resolved from {cache_path}")
    todo = [p for p in page_ids if p not in firsts]
    log(f"fetching first revision for {len(todo):,} pages ({args.concurrency} concurrent)")

    def run_pass(ids: list[str], workers: int, label: str) -> list[str]:
        failed: list[str] = []
        started = time.time()
        with cache_path.open("a") as cache, ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (pid, ts, err) in enumerate(
                pool.map(lambda p: first_revision(session, p), ids), 1
            ):
                if err is None or err == "missing":
                    firsts[pid] = ts
                    if err:
                        errors[pid] = err
                    cache.write(json.dumps({"page_id": pid, "ts": ts, "error": err}) + "\n")
                else:
                    errors[pid] = err
                    failed.append(pid)
                if i % 1000 == 0:
                    rate = i / max(time.time() - started, 1e-9)
                    log(f"  [{label}] {i:,}/{len(ids):,}  {rate:.0f}/s  "
                        f"eta {(len(ids)-i)/max(rate,1e-9)/60:.0f} min  transient failures so far {len(failed):,}")
        return failed

    failed = run_pass(todo, args.concurrency, "pass 1")
    if failed:
        # Transient failures were 97% of the original drops; retry them slowly
        # and serially before giving up on any of them.
        log(f"pass 2: retrying {len(failed):,} transient failures serially")
        time.sleep(30)
        failed = run_pass(failed, 1, "pass 2")
    if failed:
        log(f"  {len(failed):,} pages still unresolved after two passes; error kinds: "
            f"{dict(Counter(errors[p] for p in failed).most_common(5))}")

    out_rows, counts = [], Counter()
    for r in rows:
        pid = r["doc_id"].split(":", 1)[1]
        first = firsts.get(pid)
        last = last_rev.get(pid, "")
        if not first:
            # 'missing' = the API has no such page (deleted); anything else is
            # a transient failure that survived two passes. Both are excluded
            # -- an unverified date cannot support a non-membership claim --
            # but they are counted separately so a rerun's yield is visible.
            kind = errors.get(pid, "unknown")
            counts["dropped_page_missing" if kind == "missing" else "dropped_unresolved_transient"] += 1
            continue
        if first[:10] >= args.cutoff:
            grp, month = "A", first[:7]
        elif last[:10] >= args.cutoff:
            grp, month = "B", last[:7]
        else:
            counts["dropped_no_edit_after_cutoff"] += 1
            continue
        counts[f"group_{grp}"] += 1
        if r["group"] != grp:
            counts[f"relabelled_{r['group']}_to_{grp}"] += 1
        out_rows.append({
            **r, "doc_id": f"wiki{grp}:{pid}", "group": grp, "month": month,
            "first_revision": first, "last_revision": last,
        })

    cols = list(rows[0].keys()) + ["first_revision", "last_revision"]
    with gzip.open(args.out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    log("")
    for k in sorted(counts):
        log(f"  {k:<38} {counts[k]:>7,}")
    log(f"\nkept {len(out_rows):,} of {len(rows):,}")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
