#!/usr/bin/env python3
"""Post-cutoff StackExchange (question, accepted answer) pairs via the public API.

Why the API. The archive.org `stackexchange` dump ends at 2024-03-31: Stack
Exchange stopped publishing open dumps in 2024, so no public dump contains a
post created after the model's release (2025-04-17). The API does, with exact
creation dates, and its `body` HTML is byte-identical to the dump's `Body`
(verified on 4/4 posts present in both), so stackexchange_dolma_text renders
it exactly as Dolma's records are rendered.

Quota. Anonymous: 300 requests/day per IP, 100 items each. A site costs
~ceil(N/100) question pages + ~ceil(N/100) answer batches. Every response's
`backoff` is honoured and the run stops when quota_remaining < 3.

Non-membership by construction requires BOTH posts to postdate the cutoff --
a pre-cutoff question is already training data paired with its older
answers. Output schema matches build_stackexchange_pool.py so the combined
manifest treats both identically.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
import zstandard as zstd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import olmo2_midtrain_membership as mm  # noqa: E402
import stackexchange_dolma_text as sdt  # noqa: E402

API = "https://api.stackexchange.com/2.3"
UA = os.environ.get("OLMO_TWIN_USER_AGENT", "OLMo-Twin/1.0 (research code)")
CUTOFF = "2025-04-17"
CUTOFF_EPOCH = int(datetime(2025, 4, 17, tzinfo=timezone.utc).timestamp())
MIN_WORDS = 50


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Client:
    def __init__(self, key: str | None):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.key = key
        self.quota = None
        self.calls = 0

    def get(self, path: str, **params) -> dict:
        if self.key:
            params["key"] = self.key
        for attempt in range(5):
            r = self.s.get(f"{API}{path}", params=params, timeout=60)
            self.calls += 1
            if r.status_code == 400 and "throttle" in r.text.lower():
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            self.quota = d.get("quota_remaining")
            if d.get("backoff"):
                time.sleep(int(d["backoff"]) + 1)
            return d
        raise IOError(f"API {path} failed after retries")


def month_of(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m")


def fetch_site(c: Client, site: str, max_questions: int, min_quota: int) -> list[dict]:
    """Post-cutoff questions with an accepted answer, and those answers."""
    qs: dict[int, dict] = {}
    # The API returns 400 for page > 25 (an anonymous paging cap). Paging is
    # therefore cursor-based: when page 25 is reached, fromdate advances to
    # the newest question seen and paging restarts.
    fromdate, page, pages = CUTOFF_EPOCH, 1, 0
    while len(qs) < max_questions:
        if c.quota is not None and c.quota < min_quota:
            log(f"  quota nearly exhausted ({c.quota}); stopping question pages")
            break
        d = c.get("/questions", site=site, fromdate=fromdate, sort="creation",
                  order="asc", pagesize=100, page=page, filter="withbody")
        pages += 1
        items = d.get("items", [])
        for q in items:
            if q.get("accepted_answer_id") and q.get("creation_date", 0) >= CUTOFF_EPOCH:
                qs[q["question_id"]] = q
        if not d.get("has_more") or not items:
            break
        if page >= 25:
            fromdate = max(q["creation_date"] for q in items) + 1
            page = 1
        else:
            page += 1
    log(f"  {site}: {len(qs):,} post-cutoff questions with accepted answers ({pages} pages)")
    aids = [q["accepted_answer_id"] for q in qs.values()]
    answers: dict[int, dict] = {}
    for i in range(0, len(aids), 100):
        if c.quota is not None and c.quota < min_quota:
            log(f"  quota nearly exhausted ({c.quota}); stopping answer batches")
            break
        d = c.get("/answers/" + ";".join(map(str, aids[i:i + 100])), site=site,
                  pagesize=100, filter="withbody")
        for a in d.get("items", []):
            answers[a["answer_id"]] = a
    pairs = []
    for q in qs.values():
        a = answers.get(q["accepted_answer_id"])
        if a is None or a.get("creation_date", 0) < CUTOFF_EPOCH:
            continue
        pairs.append({"q": q, "a": a})
    log(f"  {site}: {len(pairs):,} pairs with both posts after {CUTOFF}  (quota left {c.quota})")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", required=True, help="API site names, e.g. superuser askubuntu")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-questions", type=int, default=600, help="per site")
    ap.add_argument("--per-month", type=int, default=200, help="pairs sampled per (site, month)")
    ap.add_argument("--key", default=None, help="SE API key (raises quota to 10,000/day)")
    ap.add_argument("--min-quota", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    texts_dir = args.out_dir / "texts"
    texts_dir.mkdir(exist_ok=True)
    c = Client(args.key)
    rng = random.Random(args.seed)
    cctx = zstd.ZstdCompressor(level=6)
    cols = ["doc_id", "forum", "month", "q_id", "a_id", "q_created", "a_created",
            "a_score", "accepted", "n_words", "norm_bytes", "text_path",
            "raw_text_path", "parser_status"]
    # Incremental output: each site's rows are appended as soon as they are
    # rendered, so a failure on one site (the first run died on an HTTP 400
    # and lost 616 finished pairs) costs only that site. Sites already in the
    # file are skipped, so a rerun resumes.
    out = args.out_dir / "se_pool.csv.gz"
    done_forums: set[str] = set()
    if out.exists():
        for r in csv.DictReader(gzip.open(out, "rt", newline="")):
            done_forums.add(r["forum"])
        log(f"resuming: forums already in {out.name}: {sorted(done_forums)}")
    rows: list[dict] = []
    for site in args.sites:
        forum = f"{site}_stackexchange_com" if "." not in site and site not in (
            "superuser", "serverfault", "askubuntu", "stackoverflow", "mathoverflow") else site
        forum = {"superuser": "superuser_com", "serverfault": "serverfault_com",
                 "askubuntu": "askubuntu_com", "stackoverflow": "stackoverflow_com",
                 "mathoverflow": "mathoverflow_net"}.get(site, forum)
        if forum in done_forums:
            log(f"{site}: already fetched, skipping")
            continue
        log(f"{site} -> forum {forum}")
        try:
            pairs = fetch_site(c, site, args.max_questions, args.min_quota)
        except Exception as exc:
            log(f"  {site}: FAILED ({type(exc).__name__}: {str(exc)[:120]}); continuing with next site")
            continue
        site_rows: list[dict] = []
        by_month: dict[str, list] = defaultdict(list)
        for p in pairs:
            by_month[month_of(p["q"]["creation_date"])].append(p)
        kept = short = 0
        for month in sorted(by_month):
            pop = sorted(by_month[month], key=lambda x: (x["q"]["question_id"], x["a"]["answer_id"]))
            for item in rng.sample(pop, min(args.per_month, len(pop))):
                q, a = item["q"], item["a"]
                text = sdt.record_text(html.unescape(q.get("title", "")), q["body"], a["body"])
                nw = len(text.split())
                if nw < MIN_WORDS:
                    short += 1
                    continue
                norm = mm.normalize(text)
                doc_id = f"{forum}-{q['question_id']}-{a['answer_id']}"
                raw_path = texts_dir / f"{doc_id}.txt"
                zst_path = texts_dir / f"{doc_id}.zst"
                raw_path.write_text(text, encoding="utf-8")
                zst_path.write_bytes(cctx.compress(norm))
                site_rows.append({
                    "doc_id": f"se:{doc_id}", "forum": forum, "month": month,
                    "q_id": q["question_id"], "a_id": a["answer_id"],
                    "q_created": datetime.fromtimestamp(q["creation_date"], timezone.utc).isoformat(),
                    "a_created": datetime.fromtimestamp(a["creation_date"], timezone.utc).isoformat(),
                    "a_score": a.get("score", 0), "accepted": 1, "n_words": nw,
                    "norm_bytes": len(norm), "text_path": str(zst_path),
                    "raw_text_path": str(raw_path), "parser_status": "ok",
                })
                kept += 1
        log(f"  {site}: kept {kept:,} (short {short})")
        # append this site now; a later failure cannot lose it
        new_file = not out.exists()
        with gzip.open(out, "at", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if new_file:
                w.writeheader()
            for r in site_rows:
                w.writerow(r)
        rows.extend(site_rows)
        if c.quota is not None and c.quota < args.min_quota:
            log("quota exhausted; stopping")
            break
    all_rows = list(csv.DictReader(gzip.open(out, "rt", newline=""))) if out.exists() else []
    per = Counter((r["forum"], r["month"]) for r in all_rows)
    log(f"\nTOTAL {len(all_rows):,} pairs from {len({r['forum'] for r in all_rows})} sites "
        f"({len(rows):,} new this run); API calls {c.calls}, quota left {c.quota}")
    for f in sorted({k[0] for k in per}):
        log(f"  {f}: " + " ".join(f"{m[-5:]}:{per[(f, m)]}" for m in sorted(k[1] for k in per if k[0] == f)))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
