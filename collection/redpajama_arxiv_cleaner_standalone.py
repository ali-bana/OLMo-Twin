#!/usr/bin/env python3
"""Standalone reproduced RedPajama v1 arXiv cleaner.

This is the exact cleaning path that produced the arXiv training text in
OLMo 2 (``olmo-mix-1124/arxiv``) and the OLMo 3 ``rpj-proofpile-arxiv``
component, ported verbatim into a dependency-free script (stdlib only).
Adapted from ``togethercomputer/RedPajama-Data`` @ ``rp_v1``,
``data_prep/arxiv/arxiv_cleaner.py`` (Apache License 2.0; see NOTICE).

Validation: byte-for-byte reproduction of the training text was confirmed for
419/419 sampled OLMo 3 direct members and 200/200 sampled
OLMo 2 rows, provided the source revision matches the RedPajama snapshot
(February 2023) — a local mirror holding a newer revision reproduces that
newer revision instead.

Deliberately preserved RedPajama quirks (the datasets were built with them):
  - the "paragraph" alternative in the section regex is missing a trailing
    "|", so a lone ``\\paragraph{...}`` does NOT count as a section start;
  - inline comment removal ``[^\\\\]%.+$`` also deletes the character
    preceding the "%";
  - macro names/values pass through ``unicode-escape`` before ``re.sub``, so
    a macro with a non-ASCII value makes ``re.sub`` raise and the whole paper
    is dropped (status ``clean_failed``);
  - any UTF-8 decode error in any ``.tex`` member drops the paper
    (status ``load_failed``);
  - files cleaned to "" still contribute a "\\n" separator when a project has
    several ``.tex`` files.

Input is a raw arXiv e-print package: a gzipped tarball of a LaTeX project or
a gzipped single TeX file (the ``<id>.gz`` layout of the bulk mirror).
PDF-only papers have no ``.gz`` and were never in the dataset.

Usage:
  # clean one source package to stdout
  python redpajama_arxiv_cleaner_standalone.py 2301.04567.gz

  # locate by arXiv ID under a bulk mirror and write to a file
  python redpajama_arxiv_cleaner_standalone.py \
      --arxiv-id 2301.04567 --src-root /path/to/arxiv/src --out paper.txt

  # batch: clean many packages into a directory, JSON status per paper
  python redpajama_arxiv_cleaner_standalone.py *.gz --out-dir cleaned/ --json

Exit code 0 when every input cleaned OK, 1 otherwise. Statuses:
``ok``, ``missing_source``, ``load_failed``, ``clean_failed``, ``empty_text``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

LOAD_FAILED = "load_failed"      # unreadable archive / non-UTF-8 -> dropped
CLEAN_FAILED = "clean_failed"    # clean_tex_files raised -> dropped
EMPTY_TEXT = "empty_text"        # cleaned to "" -> dropped


# ---------------------------------------------------------------------------
# Locate the raw source exactly like RedPajama saw it
# ---------------------------------------------------------------------------


def _strip_version(arxiv_id: str) -> str:
    """'2301.04567v3' -> '2301.04567'."""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _src_location(arxiv_id: str) -> tuple[str, str] | None:
    """Return (month_dir, file_stem) in the bulk-src layout, or None.

    New-style: '2301.04567'  -> ('2301', '2301.04567')
    Old-style: 'hep-ph/9905312' -> ('9905', 'hep-ph9905312')
    """
    clean = _strip_version(arxiv_id)
    if re.match(r"^\d{4}\.\d{4,5}$", clean):
        return clean[:4], clean
    m = re.match(r"^([a-z\-]+(?:\.[A-Z]+)?)/?(\d{7,})$", clean, re.IGNORECASE)
    if m:
        cat = re.sub(r"[./]", "", m.group(1))
        num = m.group(2)
        return num[:4], cat + num
    return None


def find_src_file(arxiv_id: str, src_root: Path) -> Path | None:
    """Find the per-paper source archive below ``src_root/<yymm>/``.

    RedPajama only ever saw ``*.gz`` members of the bulk tars, so PDF-only
    papers (``<id>.pdf``) are treated as missing.
    """
    loc = _src_location(arxiv_id)
    if loc is None:
        return None
    month, stem = loc
    month_dir = src_root / month
    exact = month_dir / f"{stem}.gz"
    if exact.is_file():
        return exact
    if not month_dir.is_dir():
        return None
    for path in sorted(month_dir.glob(f"{stem}*")):
        if path.is_file() and path.suffix.lower() == ".gz":
            return path
    return None


def load_tex_project(src_file: Path) -> list[str] | None:
    """Verbatim port of RedPajama's ``_tex_proj_loader`` (minus the mtime).

    Returns the list of TeX file contents in archive order, or None when
    RedPajama would have logged "failed to process" (paper absent from the
    dataset).
    """
    files_and_content: list[str] = []
    try:
        # if it is a tarball (possibly gzipped), take the .tex members
        with tarfile.open(src_file) as sub_tf:
            for member in sub_tf.getmembers():
                if member.name.endswith(".tex"):
                    file_content = sub_tf.extractfile(member).read()
                    try:
                        file_content = file_content.decode("utf-8")
                    except UnicodeDecodeError:
                        return None
                    files_and_content.append(file_content)
    except tarfile.ReadError:
        # otherwise try opening it as a gzipped single file
        try:
            with gzip.open(src_file, "rb") as gz:
                file_content = gz.read()
        except Exception:
            return None
        try:
            file_content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            return None
        files_and_content.append(file_content)
    except Exception:
        return None
    return files_and_content


# ---------------------------------------------------------------------------
# The RedPajama v1 arXiv cleaner, ported verbatim
# ---------------------------------------------------------------------------


def _build_non_arg_macros_dict(file_content: str) -> dict[str, str]:
    """Extract no-argument ``\\newcommand`` / ``\\def`` macros (verbatim)."""
    # regex for extracting \newcommand macros without arguments
    non_arg_nc_reg = re.compile(
        pattern=r'\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$',
        flags=re.MULTILINE,
    )
    # regex for extracting \def macros without arguments
    non_arg_def_reg = re.compile(
        pattern=r'\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$',
        flags=re.MULTILINE,
    )
    macros = {}
    for reg in [non_arg_nc_reg, non_arg_def_reg]:
        for match in reg.finditer(file_content):
            # convert the macro name and value to a raw string usable in re.sub
            macro_name = match.group(1).encode("unicode-escape").decode("utf-8")
            macro_val = match.group(2).encode("unicode-escape").decode("utf-8")
            macros[macro_name] = macro_val
    return macros


def _clean_tex_file(file_content: str, arg_macros: dict, non_arg_macros: dict) -> str:
    """Clean a single TeX file (verbatim port, quirks included)."""
    # find the first occurrence of a \section-like header and replace
    # everything before it with an empty string
    pattern = r"^(.*?)("
    pattern += r"\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
    pattern += r"\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
    pattern += r")"

    # if no section-like header is found, return an empty string
    if not re.search(pattern, file_content, flags=re.DOTALL):
        return ""

    # replace everything with the second group of the match (i.e. everything
    # after and including the section header)
    file_content = re.sub(
        pattern=pattern,
        repl=r"\2",
        string=file_content,
        flags=re.DOTALL,
    )

    # remove all line comments
    file_content = re.sub(
        pattern=r"(?m)^%.*\n?",
        repl=r"",
        string=file_content,
        flags=re.MULTILINE,
    )

    # remove all in-line comments within a line
    file_content = re.sub(
        # pattern matches a "%" that is not preceded by a backslash
        pattern=r"[^\\]%.+$",
        repl=r"",
        string=file_content,
        flags=re.MULTILINE,
    )

    # find the first occurrence of either \appendix or \bibliography and
    # replace everything after it with an empty string
    pattern = r"("
    pattern += r"\\appendix|"
    pattern += r"\\begin\{references\}|"
    pattern += r"\\begin\{REFERENCES\}|"
    pattern += r"\\begin\{thebibliography\}|"
    pattern += r"\\bibliography\{.*\}"
    pattern += r").*$"

    file_content = re.sub(
        pattern=pattern,
        repl=r"",
        string=file_content,
        flags=re.DOTALL,
    )

    # inline-expand all non-arg macros
    for macro_name, macro_value in non_arg_macros.items():
        file_content = re.sub(
            # grouped to make sure the macro is not part of a longer
            # alphanumeric word
            pattern=r"(" + macro_name + r")" + r"([^a-zA-Z0-9])",
            # replace the macro with its value and add back the character
            # that was matched after the macro
            repl=macro_value + r"\2",
            string=file_content,
        )

    # inline-expand all macros that use args
    # (never implemented in RedPajama either)
    for macro_name, macro_value in arg_macros.items():
        pass

    return file_content


def clean_tex_files(tex_files: list[str]) -> str:
    """Clean a whole TeX project (verbatim port of RedPajama)."""
    if len(tex_files) == 0:
        return ""

    # macros are collected from ALL files of the project (comments included,
    # since this runs on the raw file contents) ...
    non_arg_macros = {}
    for file_content in tex_files:
        non_arg_macros.update(_build_non_arg_macros_dict(file_content))

    # macros that take arguments were never supported
    arg_macros: dict[str, str] = {}

    # ... then every file is cleaned and the results are joined with "\n"
    # (files cleaned to "" still contribute a separator!)
    cleaned_latex_file_str = "\n".join(
        _clean_tex_file(
            file_content=file_content,
            arg_macros=arg_macros,
            non_arg_macros=non_arg_macros,
        )
        for file_content in tex_files
    )

    return cleaned_latex_file_str


def clean_arxiv_source_file(src_file: Path) -> tuple[str | None, str]:
    """Raw archive -> (training_text, status); text is None unless 'ok'."""
    tex_files = load_tex_project(src_file)
    if tex_files is None:
        return None, LOAD_FAILED
    try:
        cleaned = clean_tex_files(tex_files)
    except Exception:
        # create_record was wrapped in try/except -> paper dropped
        return None, CLEAN_FAILED
    if len(cleaned) == 0:
        return None, EMPTY_TEXT
    return cleaned, "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduced RedPajama v1 arXiv cleaner (OLMo training text)",
    )
    parser.add_argument(
        "sources", nargs="*", type=Path,
        help="raw arXiv source packages (<id>.gz)",
    )
    parser.add_argument(
        "--arxiv-id", action="append", default=[],
        help="locate this ID under --src-root instead of giving a path "
        "(repeatable)",
    )
    parser.add_argument(
        "--src-root", type=Path,
        help="bulk mirror root with <yymm>/<id>.gz layout (for --arxiv-id)",
    )
    parser.add_argument("--out", type=Path, help="output file (single input only)")
    parser.add_argument(
        "--out-dir", type=Path,
        help="write one <stem>.txt per input into this directory",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print one JSON status object per input instead of plain text",
    )
    args = parser.parse_args()

    jobs: list[tuple[str, Path | None]] = [(str(p), p) for p in args.sources]
    for arxiv_id in args.arxiv_id:
        if args.src_root is None:
            parser.error("--arxiv-id requires --src-root")
        jobs.append((arxiv_id, find_src_file(arxiv_id, args.src_root)))
    if not jobs:
        parser.error("give source package paths and/or --arxiv-id")
    if args.out and len(jobs) > 1:
        parser.error("--out only works with a single input; use --out-dir")

    failures = 0
    for label, src in jobs:
        if src is None or not src.is_file():
            text, status = None, "missing_source"
        else:
            text, status = clean_arxiv_source_file(src)
        if status != "ok":
            failures += 1
        result = {
            "input": label,
            "source_path": str(src) if src else "",
            "status": status,
            "cleaned_chars": len(text) if text else 0,
            "cleaned_sha256": hashlib.sha256(text.encode()).hexdigest() if text else "",
        }
        destination = None
        if text is not None:
            if args.out_dir:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(label).stem if src else label.replace("/", "_")
                destination = args.out_dir / f"{stem}.txt"
            elif args.out:
                destination = args.out
            if destination is not None:
                destination.write_text(text, encoding="utf-8")
                result["output"] = str(destination)
        if args.json:
            print(json.dumps(result, sort_keys=True))
        elif text is not None and destination is None:
            sys.stdout.write(text)
        else:
            print(f"{label}: {status}"
                  + (f" -> {destination} ({result['cleaned_chars']} chars)"
                     if destination else ""),
                  file=sys.stderr if status != "ok" else sys.stdout)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
