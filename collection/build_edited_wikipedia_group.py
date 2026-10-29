#!/usr/bin/env python3
"""Edited-Wikipedia group: pages whose TRAINED version is known exactly.

Group B of the Wikipedia pool (pages that existed before the cutoff and were
edited after it) is the positive control for the scan: 86% are detected as
members. This script turns the detected pages into an analysis group of their
own -- near-duplicate members -- by recovering the exact text OLMo 2 trained
on and aligning it with the current (edited) text.

Why it is possible: both `olmo-mix-1124/wiki` and `dolmino-mix-1124/wiki`
are the same two shards (byte-identical; checked by `ancestors`), one record
per page keyed by the Wikipedia page id, carrying the trained revision id.

Subcommands
  ancestors   stream the wiki component once, keep the records whose id is a
              group-B (or group-A, as a check) page id and whose url host is
              en.wikipedia.org; write ancestors.jsonl.gz. With --evidence, a
              page whose scan match was a wiki record under a DIFFERENT id
              (page moved/merged) takes that record as its ancestor
              (`via` = "scan_ref" instead of "page_id"), but only when the
              scan excluded the page on that match (span or count).
  align       per page: word-level shared/novel mask from 8-word shingle
              coverage (the same unit the scan uses), lifted to OLMo tokens
              through tokenizer offsets; strata by novel-token fraction;
              masks/<doc_id>.npz + edited_group.csv.gz + summary.json

Alignment rule. Words are `\\w+` runs of the raw text; each is folded with the
scan's `normalize` (lower, NFKD, alnum). An edited word is SHARED if every
folded sub-word lies inside some 8-gram that also occurs in the ancestor,
otherwise NOVEL. A token is NOVEL if it overlaps a novel word; SHARED if it
overlaps only shared words; a token touching no word (whitespace,
punctuation) inherits the flag of the next word (previous at the end). The
trailing EOS token is marked 2. `token_ids` is stored alongside the mask so it
can be checked against the scorer's arrays before use.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import olmo2_midtrain_membership as mm  # noqa: E402

K = 8
STRATA = [(0.0, 0.10, "lt10"), (0.10, 0.25, "10-25"), (0.25, 0.50, "25-50"),
          (0.50, 0.75, "50-75"), (0.75, 1.01, "ge75")]
WORD_RE = re.compile(r"\w+", re.UNICODE)
ID_RE = re.compile(rb'"id":\s*"(\d+)"')
EOS_MASK = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stratum(frac: float) -> str:
    for lo, hi, name in STRATA:
        if lo <= frac < hi:
            return name
    return STRATA[-1][2]


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------
def fold_words(text: str) -> tuple[list[tuple[int, int]], list[str], list[int]]:
    """Raw word spans, the flattened folded sub-word sequence, and for every
    folded sub-word the index of the raw word it came from."""
    spans, folded, owner = [], [], []
    for m in WORD_RE.finditer(text):
        parts = mm.normalize(m.group()).decode("ascii").split()
        if not parts:
            continue
        spans.append((m.start(), m.end()))
        wi = len(spans) - 1
        for p in parts:
            folded.append(p)
            owner.append(wi)
    return spans, folded, owner


def shingle_set(words: list[str], k: int = K) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def coverage(words: list[str], other: set[tuple[str, ...]], k: int = K) -> np.ndarray:
    """cov[i] = True iff folded word i lies inside an k-gram shared with `other`."""
    n = len(words)
    cov = np.zeros(n, dtype=bool)
    for i in range(n - k + 1):
        if tuple(words[i:i + k]) in other:
            cov[i:i + k] = True
    return cov


def longest_true_run(a: np.ndarray) -> int:
    best = cur = 0
    for v in a:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def word_masks(edited: str, ancestor: str) -> dict:
    e_spans, e_words, e_owner = fold_words(edited)
    a_spans, a_words, _ = fold_words(ancestor)
    a_set, e_set = shingle_set(a_words), shingle_set(e_words)
    e_cov = coverage(e_words, a_set)
    a_cov = coverage(a_words, e_set)
    # raw word shared iff all its folded sub-words are covered
    shared = np.ones(len(e_spans), dtype=bool)
    for cov, wi in zip(e_cov, e_owner):
        if not cov:
            shared[wi] = False
    return {
        "spans": e_spans, "word_shared": shared,
        "n_words": len(e_spans), "n_words_novel": int((~shared).sum()),
        "n_folded": len(e_words), "n_ancestor_words": len(a_spans),
        "frac_ancestor_retained": float(a_cov.mean()) if len(a_cov) else 0.0,
        "longest_shared_run_words": int(longest_true_run(e_cov)),
    }


def token_mask(offsets: list[tuple[int, int]], spans: list[tuple[int, int]],
               word_shared: np.ndarray) -> np.ndarray:
    """Lift the word mask to tokens (see module docstring). Returns int8:
    0 shared, 1 novel, for the text tokens only (EOS is appended by caller)."""
    n_tok = len(offsets)
    mask = np.full(n_tok, -1, dtype=np.int8)
    starts = np.array([s for s, _ in spans], dtype=np.int64)
    ends = np.array([e for _, e in spans], dtype=np.int64)
    for ti, (ts, te) in enumerate(offsets):
        if te <= ts:
            continue
        # words overlapping [ts, te): starts < te and ends > ts
        lo = np.searchsorted(ends, ts, side="right")      # first word with end > ts
        hi = np.searchsorted(starts, te, side="left")     # first word with start >= te
        if hi > lo:
            mask[ti] = 1 if (~word_shared[lo:hi]).any() else 0
    # tokens touching no word inherit the next assigned flag (previous at end)
    nxt = -1
    for ti in range(n_tok - 1, -1, -1):
        if mask[ti] >= 0:
            nxt = mask[ti]
        elif nxt >= 0:
            mask[ti] = nxt
    prev = 0
    for ti in range(n_tok):
        if mask[ti] >= 0:
            prev = mask[ti]
        else:
            mask[ti] = prev
    return mask


# --------------------------------------------------------------------------
# ancestors
# --------------------------------------------------------------------------
def cmd_ancestors(args: argparse.Namespace) -> int:
    pool = list(csv.DictReader(gzip.open(args.wiki_labels, "rt", newline="")))
    group: dict[str, str] = {}  # page id -> group
    for r in pool:
        group[r["doc_id"].split(":", 1)[1]] = r["group"]
    # record id -> [(page_id, via)]; a page's own id first, scan refs second
    want: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pid in group:
        want[pid].append((pid, "page_id"))
    n_alias = 0
    if args.evidence:
        for r in csv.DictReader(gzip.open(args.evidence, "rt", newline="")):
            if not r["doc_id"].startswith("wiki"):
                continue
            pid = r["doc_id"].split(":", 1)[1]
            if pid not in group:      # unverified-date pages carry no label
                continue
            # only a match the scan itself excluded on is evidence of an
            # ancestor; a weak best-match (shared boilerplate) is not
            strong = {x for x in r.get("reasons", "").split("|")} & {"span", "count"}
            if not strong:
                continue
            for comp_col, ref_col in (("run_component", "run_ref"), ("shingles_component", "shingles_ref")):
                if r[comp_col].endswith("__wiki") and r[ref_col] and r[ref_col] != pid:
                    if (pid, "scan_ref") not in want[r[ref_col]]:
                        want[r[ref_col]].append((pid, "scan_ref"))
                        n_alias += 1
    log(f"{len(group):,} page ids from {args.wiki_labels.name} ({Counter(group.values())}); "
        f"{n_alias} alias record ids from scan refs")
    shards = sorted(p for p in args.corpus_wiki.rglob("*") if p.is_file() and p.suffix in (".gz", ".zst", ".zstd", ".jsonl", ".json"))
    if not shards:
        raise SystemExit(f"no shards under {args.corpus_wiki}")
    if args.check_identical_to:
        import filecmp
        other = sorted(p for p in args.check_identical_to.rglob("*") if p.is_file())
        same = len(other) == len(shards) and all(
            filecmp.cmp(a, b, shallow=False) for a, b in zip(shards, other))
        log(f"identical to {args.check_identical_to}: {same}")
        if not same:
            raise SystemExit("stage-1 and stage-2 wiki components differ; scan both")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    n_rows = n_dup = 0
    out_path = args.out_dir / "ancestors.jsonl.gz"
    with gzip.open(out_path, "wt") as out:
        for sh in shards:
            log(f"streaming {sh.name}")
            for line in mm.iter_lines(str(sh)):
                n_rows += 1
                m = ID_RE.search(line[:200])
                if not m:
                    continue
                rid = m.group(1).decode()
                if rid not in want:
                    continue
                d = json.loads(line)
                host = urlparse(d.get("metadata", {}).get("url", "")).netloc
                if host != "en.wikipedia.org":
                    continue
                for pid, via in want[rid]:
                    prev = found.get(pid)
                    if prev is not None:
                        if via == "page_id" and prev["via"] == "page_id":
                            n_dup += 1      # two corpus records with the same own id
                            continue
                        if via == "scan_ref":
                            continue        # own-id record, or an earlier alias, already there
                        # via == "page_id" replaces an alias found earlier
                    rec = {"page_id": pid, "group": group[pid], "ancestor_record_id": rid, "via": via,
                           "revid": d.get("metadata", {}).get("revid"),
                           "added": d.get("added"), "length": d.get("metadata", {}).get("length"),
                           "provenance": d.get("metadata", {}).get("provenance"), "text": d["text"]}
                    found[pid] = rec
        for rec in found.values():
            out.write(json.dumps(rec) + "\n")
    by_group = Counter(r["group"] for r in found.values())
    by_via = Counter(r["via"] for r in found.values())
    summary = {"corpus_rows": n_rows, "pages_wanted": len(group), "wanted_by_group": dict(Counter(group.values())),
               "alias_record_ids": n_alias, "ancestors_found": len(found), "found_by_group": dict(by_group),
               "found_by_via": dict(by_via), "duplicate_ids_skipped": n_dup, "shards": [str(s) for s in shards]}
    (args.out_dir / "ancestors_summary.json").write_text(json.dumps(summary, indent=1))
    log(f"rows {n_rows:,}; ancestors found {len(found):,} {dict(by_group)} via {dict(by_via)}; "
        f"missing {len(group) - len(found):,}; dup ids {n_dup}")
    own_a = [p for p, r in found.items() if r["group"] == "A" and r["via"] == "page_id"]
    if own_a:
        log(f"WARNING: {len(own_a)} group-A (post-cutoff creation) pages have a corpus record under "
            f"their OWN id -> label or page-id problem; listed in ancestors_groupA.txt")
        (args.out_dir / "ancestors_groupA.txt").write_text("\n".join(own_a) + "\n")
    alias_a = sum(1 for r in found.values() if r["group"] == "A" and r["via"] == "scan_ref")
    if alias_a:
        log(f"note: {alias_a} group-A pages take a strongly matched wiki record under another id as "
            f"ancestor (text that predates the page: split/recreated articles); they are scan-excluded anyway")
    log(f"wrote {out_path}")
    return 0


# --------------------------------------------------------------------------
# align
# --------------------------------------------------------------------------
def cmd_align(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.is_fast, "offsets need a fast tokenizer"
    pool = {r["doc_id"].split(":", 1)[1]: r for r in
            csv.DictReader(gzip.open(args.wiki_labels, "rt", newline=""))}
    evidence = {}
    if args.evidence:
        for r in csv.DictReader(gzip.open(args.evidence, "rt", newline="")):
            if r["doc_id"].startswith("wiki"):
                evidence[r["doc_id"].split(":", 1)[1]] = r
    masks_dir = args.out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for line in gzip.open(args.out_dir / "ancestors.jsonl.gz", "rt"):
        anc = json.loads(line)
        pid = anc["page_id"]
        if anc["group"] != "B":
            continue
        meta = pool[pid]
        edited = Path(meta["raw_text_path"]).read_text(encoding="utf-8")
        w = word_masks(edited, anc["text"])
        enc = tok(edited, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"] + [tok.eos_token_id]
        tm = token_mask(enc["offset_mapping"], w["spans"], w["word_shared"])
        tm = np.concatenate([tm, np.array([EOS_MASK], dtype=np.int8)])
        n_text_tok = len(tm) - 1
        frac_tok = float((tm[:-1] == 1).mean()) if n_text_tok else 0.0
        frac_word = w["n_words_novel"] / w["n_words"] if w["n_words"] else 0.0
        doc_id = f"wikiB:{pid}"
        safe = doc_id.replace(":", "_")       # same naming as the scorer's tokens/<doc>.npz
        np.savez_compressed(masks_dir / f"{safe}.npz", token_ids=np.asarray(ids, dtype=np.int32),
                            mask=tm)
        ev = evidence.get(pid, {})
        rows.append({
            "doc_id": doc_id, "page_id": pid, "title": meta["title"],
            "first_revision": meta.get("first_revision", ""), "last_revision": meta.get("last_revision", ""),
            "ancestor_record_id": anc["ancestor_record_id"], "ancestor_via": anc["via"],
            "ancestor_revid": anc["revid"], "ancestor_added": anc["added"],
            "n_words": w["n_words"], "n_words_novel": w["n_words_novel"], "frac_novel_words": round(frac_word, 4),
            "n_tokens": len(ids), "n_tokens_novel": int((tm == 1).sum()), "frac_novel_tokens": round(frac_tok, 4),
            "n_ancestor_words": w["n_ancestor_words"], "frac_ancestor_retained": round(w["frac_ancestor_retained"], 4),
            "longest_shared_run_words": w["longest_shared_run_words"],
            "stratum": stratum(frac_tok),
            "scan_excluded": ev.get("excluded", ""), "scan_reasons": ev.get("reasons", ""),
            "ngram_detected": int(bool(set(ev.get("reasons", "").split("|")) & {"span", "count", "semantic"})),
            "scan_best_run_words": ev.get("best_run_words", ""), "scan_best_shingles": ev.get("best_shingles", ""),
            "scan_run_component": ev.get("run_component", ""),
            "raw_text_path": meta["raw_text_path"], "mask_path": str(masks_dir / f"{safe}.npz"),
        })
        n += 1
        if n % 1000 == 0:
            log(f"  {n:,} aligned")
    out = args.out_dir / "edited_group.csv.gz"
    with gzip.open(out, "wt", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    strata = Counter(r["stratum"] for r in rows)
    by_excl = defaultdict(Counter)
    for r in rows:
        by_excl[r["stratum"]]["ngram_detected" if r["ngram_detected"] else "ngram_missed"] += 1
    fr = np.array([r["frac_novel_tokens"] for r in rows])
    summary = {
        "pages": len(rows), "k": K, "model": args.model,
        "strata_by_novel_token_fraction": {name: strata.get(name, 0) for _, _, name in STRATA},
        "strata_x_ngram_detection": {k: dict(v) for k, v in by_excl.items()},
        "identical_text": sum(1 for r in rows if r["n_words_novel"] == 0),
        "frac_novel_tokens_quantiles": {q: float(np.quantile(fr, q / 100)) for q in (10, 25, 50, 75, 90)},
        "ngram_detected": sum(1 for r in rows if r["ngram_detected"]),
    }
    (args.out_dir / "align_summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary, indent=1))
    log(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ancestors")
    a.add_argument("--wiki-labels", type=Path, required=True, help="wiki_pools_verified.csv.gz")
    a.add_argument("--corpus-wiki", type=Path, required=True, help="…/dolmino-mix-1124/data/wiki")
    a.add_argument("--check-identical-to", type=Path, default=None, help="…/olmo-mix-1124/data/wiki")
    a.add_argument("--evidence", type=Path, default=None,
                   help="stageG/candidates_all_evidence.csv.gz: take wiki scan refs as alias ancestor ids")
    a.add_argument("--out-dir", type=Path, required=True)
    a.set_defaults(fn=cmd_ancestors)
    b = sub.add_parser("align")
    b.add_argument("--wiki-labels", type=Path, required=True)
    b.add_argument("--evidence", type=Path, default=None, help="stageG/candidates_all_evidence.csv.gz")
    b.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    b.add_argument("--out-dir", type=Path, required=True)
    b.set_defaults(fn=cmd_align)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
