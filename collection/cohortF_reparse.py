#!/usr/bin/env python3
"""Cohort F: produce the two missing cells of the form x arm 2x2.

Cohort F's arm assignment is perfectly confounded with surface form (every
injected document is `redpajama_latex`, every held-out partner is
`pes2o_style_pdf`), so the control model alone separates the arms. Fixing it
needs the other form of each paper, parsed by the SAME converters that built
the injection bundle:

  set A  PDF form of the 278 injected papers      pes2o_style_pdf
  set B  LaTeX form of the 278 held-out papers    redpajama_latex
  set C  PDF form of 20 held-out papers whose text is ALREADY in the bundle
         -- the pipeline-identity control, diffed byte-for-byte

Nothing in the converters is touched. This script calls exactly the functions
`build_postcutoff_arxiv_pool.py rawtext` called when the bundle was built:

  PDF    pes2o_style_pdf_parser.grobid_fulltext(pdf, server)
         pes2o_style_pdf_parser.tei_to_pes2o_text(tei, unigram_lm,
                                                  fallback_abstract=<metadata abstract>)
  LaTeX  redpajama_arxiv_cleaner_standalone.clean_arxiv_source_file(src)

and takes every input (PDF path, source archive, metadata abstract) from the
same `pool_parsed_merged.csv.gz` row the original pass used.

Two controls are reported, not one:
  * set C vs the shipped bundle text (`texts_all/<id>.txt`) -- the
    pipeline-identity control;
  * every freshly parsed document vs the file the ORIGINAL pass left on disk
    (`postcutoff/raw/{pdf,latex}/<yymm>/<id>.txt`) -- 556 documents instead of
    20, since that pass wrote both forms for every paper.

Failures are reported per id, never silently dropped.

The arXiv ids of the three sets are in ``cohortF_reparse_ids/`` next to this
script (the default of --request-dir).

Note on the recorded converter hashes: versions() writes the first 16 hex
digits of the SHA-256 of pes2o_style_pdf_parser.py,
redpajama_arxiv_cleaner_standalone.py and build_postcutoff_arxiv_pool.py into
REPARSE_REPORT.json. The copies in this repository differ from the files used
for the original re-parse only in docstrings and, in
build_postcutoff_arxiv_pool.py, in the HTTP User-Agent line (no conversion
code changed), so these prefixes differ from the ones in the original
REPARSE_REPORT.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pes2o_style_pdf_parser as pp  # noqa: E402
import redpajama_arxiv_cleaner_standalone as rp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

W = paths.COLLECTION
SETS = {
    "A": ("ids_A_pdf_of_injected.txt", "pdf_of_injected", "pdf"),
    "B": ("ids_B_latex_of_holdout.txt", "latex_of_holdout", "latex"),
    "C": ("ids_C_pipeline_check.txt", "pipeline_check", "pdf"),
}
_LM = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pdf_init(lm_path: str | None) -> None:
    global _LM
    _LM = json.loads(Path(lm_path).read_text()) if lm_path else None


def _pdf_one(task):
    """Byte-for-byte the body of build_postcutoff_arxiv_pool._raw_pdf_one,
    minus the normalize() check (done here as a diagnostic instead)."""
    doc_id, pdf_path, out_txt, server, meta_abstract = task
    # GROBID answers 503 when its worker pool is saturated, and
    # grobid_fulltext() reports that as None exactly like a real failure.
    # Retrying the UNCHANGED function is a call-site concern, not a converter
    # change: the request and the parsing are identical, only transient
    # service pressure is absorbed.
    tei = None
    for attempt in range(4):
        try:
            tei = pp.grobid_fulltext(Path(pdf_path), server)
        except Exception as exc:
            return {"doc_id": doc_id, "status": f"grobid_{type(exc).__name__}"}
        if tei:
            break
        time.sleep(2 * (attempt + 1))
    if not tei:
        return {"doc_id": doc_id, "status": "grobid_empty"}
    doc = pp.tei_to_pes2o_text(tei, _LM, fallback_abstract=meta_abstract)
    text = (doc or {}).get("text", "")
    if not text:
        return {"doc_id": doc_id, "status": "tei_empty"}
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(out_txt).write_text(text, encoding="utf-8")
    return {"doc_id": doc_id, "status": "ok", "bytes": len(text.encode()),
            "abstract_source": (doc or {}).get("abstract_source", "")}


def _latex_one(task):
    """Byte-for-byte the body of build_postcutoff_arxiv_pool._raw_latex_one."""
    doc_id, src_path, out_txt = task
    try:
        text, status = rp.clean_arxiv_source_file(Path(src_path))
    except Exception as exc:
        return {"doc_id": doc_id, "status": f"error_{type(exc).__name__}"}
    if not text:
        return {"doc_id": doc_id, "status": status}
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(out_txt).write_text(text, encoding="utf-8")
    return {"doc_id": doc_id, "status": "ok", "bytes": len(text.encode())}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def versions(server: str) -> dict:
    out = {"pes2o_style_pdf_parser.py": sha(HERE / "pes2o_style_pdf_parser.py")[:16],
           "redpajama_arxiv_cleaner_standalone.py": sha(HERE / "redpajama_arxiv_cleaner_standalone.py")[:16],
           "build_postcutoff_arxiv_pool.py": sha(HERE / "build_postcutoff_arxiv_pool.py")[:16]}
    try:
        import requests
        out["grobid_version"] = requests.get(f"{server}/api/version", timeout=10).text.strip()
    except Exception as exc:
        out["grobid_version"] = f"unavailable ({type(exc).__name__})"
    try:
        out["grobid_image"] = subprocess.run(
            ["docker", "inspect", "-f", "{{.Config.Image}}", "grobid"],
            capture_output=True, text=True, timeout=20).stdout.strip() or "n/a"
    except Exception:
        out["grobid_image"] = "n/a"
    out["grobid_request"] = ("processFulltextDocument, consolidateHeader=0, "
                             "consolidateCitations=0, segmentSentences=0")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request-dir", type=Path,
                    default=paths.code_dir("collection") / "cohortF_reparse_ids")
    ap.add_argument("--pool", type=Path, default=W / "postcutoff" / "pool_parsed_merged.csv.gz")
    ap.add_argument("--bundle-texts", type=Path, default=paths.BUNDLE / "texts_all")
    ap.add_argument("--unigram-lm", type=Path, default=paths.COLLECTION / "pes2o_unigram_lm.json")
    ap.add_argument("--server", default=pp.DEFAULT_GROBID)
    ap.add_argument("--out-dir", type=Path, default=paths.COHORTF_REPARSE)
    ap.add_argument("--grobid-workers", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    csv.field_size_limit(sys.maxsize)

    pool = {r["doc_id"]: r for r in csv.DictReader(gzip.open(args.pool, "rt", newline=""))}
    report: dict = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "converters": versions(args.server), "sets": {}}
    log("converters: " + json.dumps(report["converters"], indent=1))

    ctx = mp.get_context("fork")
    for key, (idfile, subdir, form) in SETS.items():
        ids = [l.strip() for l in (args.request_dir / idfile).read_text().split("\n") if l.strip()]
        out = args.out_dir / subdir
        out.mkdir(parents=True, exist_ok=True)
        missing_input, tasks = [], []
        for i in ids:
            r = pool.get(i)
            if not r:
                missing_input.append((i, "not in the parsed pool")); continue
            dest = str(out / f"{i}.txt")
            if form == "pdf":
                src = r.get("pdf_path", "")
                if not src or not os.path.exists(src):
                    missing_input.append((i, "PDF missing on disk")); continue
                tasks.append((i, src, dest, args.server, r.get("abstract", "")))
            else:
                src = r.get("src_path", "")
                if not src or not os.path.exists(src):
                    missing_input.append((i, "LaTeX source missing on disk")); continue
                tasks.append((i, src, dest))
        log(f"set {key} ({subdir}, {form}): {len(ids)} ids, {len(tasks)} to parse, "
            f"{len(missing_input)} without input")
        results = []
        if tasks and form == "pdf":
            with ctx.Pool(args.grobid_workers, initializer=_pdf_init,
                          initargs=(str(args.unigram_lm),)) as p:
                for n, res in enumerate(p.imap_unordered(_pdf_one, tasks, chunksize=1), 1):
                    results.append(res)
                    if n % 50 == 0:
                        log(f"  {n}/{len(tasks)}")
        elif tasks:
            with ctx.Pool(args.workers) as p:
                for n, res in enumerate(p.imap_unordered(_latex_one, tasks, chunksize=4), 1):
                    results.append(res)
                    if n % 50 == 0:
                        log(f"  {n}/{len(tasks)}")

        ok = [r for r in results if r["status"] == "ok"]
        failed = [(r["doc_id"], r["status"]) for r in results if r["status"] != "ok"]
        # control 1: against the file the ORIGINAL pass left on disk
        col = "pdf_raw_path" if form == "pdf" else "latex_raw_path"
        same_orig = diff_orig = no_orig = 0
        differing = []
        for r in ok:
            orig = pool[r["doc_id"]].get(col, "")
            if not orig or not os.path.exists(orig):
                no_orig += 1; continue
            if Path(orig).read_bytes() == (out / f"{r['doc_id']}.txt").read_bytes():
                same_orig += 1
            else:
                diff_orig += 1; differing.append(r["doc_id"])
        entry = {"ids": len(ids), "parsed_ok": len(ok), "failed": failed,
                 "missing_input": missing_input, "form": form,
                 "vs_original_pass": {"identical": same_orig, "differing": diff_orig,
                                      "no_original": no_orig, "differing_ids": differing[:20]}}
        # control 2 (set C only): against the text shipped in the bundle
        if key == "C":
            same_b = diff_b = miss_b = 0
            dlist = []
            for r in ok:
                b = args.bundle_texts / f"{r['doc_id']}.txt"
                if not b.exists():
                    miss_b += 1; continue
                if b.read_bytes() == (out / f"{r['doc_id']}.txt").read_bytes():
                    same_b += 1
                else:
                    diff_b += 1; dlist.append(r["doc_id"])
            entry["vs_shipped_bundle"] = {"identical": same_b, "differing": diff_b,
                                          "not_in_bundle": miss_b, "differing_ids": dlist}
        report["sets"][key] = entry
        log(f"set {key}: parsed ok {len(ok)}, failed {len(failed)}; "
            f"identical to the original pass {same_orig}/{same_orig + diff_orig}"
            + (f"; identical to the shipped bundle {entry['vs_shipped_bundle']['identical']}/"
               f"{entry['vs_shipped_bundle']['identical'] + entry['vs_shipped_bundle']['differing']}"
               if key == "C" else ""))

    (args.out_dir / "REPARSE_REPORT.json").write_text(json.dumps(report, indent=1))
    log(f"wrote {args.out_dir}/REPARSE_REPORT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
