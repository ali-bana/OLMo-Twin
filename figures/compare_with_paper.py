#!/usr/bin/env python3
"""Compare regenerated figures and macros with the paper's.

Usage:
    python figures/compare_with_paper.py --generated DIR --paper DIR [--dpi 150] [--verbose]

Both directories have the layout figures/*.pdf and tables/numbers.tex: the
output directory of figures/make_all.sh (default paper/ under the data root)
and the source directory of the paper.

Figures: every PDF found in either figures/ directory is rendered at --dpi
(default 150) in grey scale with pypdfium2 and compared pixel by pixel. Each
figure is reported as identical, as the share of pixels that differ, as a
size mismatch (different page size in pixels or a different number of pages),
or as missing on one side.

Macros: both numbers.tex files are parsed (one \\newcommand or \\renewcommand
per line) and the values are compared after normalization: \\num{...} is
unwrapped, \\xspace is dropped, and thousands separators ({,} and ,), thin
spaces (\\,), empty groups and spaces are removed, so that the generator's
macro file and a copy reformatted for siunitx compare equal when the numbers
agree. The macros that are common, differ, or are defined on one side only
are reported; --verbose lists every differing and one-sided macro.

The script only reads files. It needs numpy and pypdfium2.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

MACRO = re.compile(r"\\(?:re)?newcommand\{?\\([A-Za-z]+)\}?\{(.*)\}\s*(%.*)?$")


# ------------------------------------------------------------------ figures
def render(path: Path, dpi: int) -> list:
    """Every page of a PDF as a grey-scale array."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        return [np.asarray(doc[i].render(scale=dpi / 72).to_pil().convert("L")) for i in range(len(doc))]
    finally:
        doc.close()


def compare_figures(gen_dir: Path, paper_dir: Path, dpi: int) -> dict:
    names = sorted({p.name for p in gen_dir.glob("*.pdf")} | {p.name for p in paper_dir.glob("*.pdf")})
    counts = {"identical": 0, "differ": 0, "size": 0, "missing": 0}
    for name in names:
        g, p = gen_dir / name, paper_dir / name
        if not g.exists() or not p.exists():
            side = "generated" if not g.exists() else "paper"
            print(f"  {name:28s} missing from the {side} figures")
            counts["missing"] += 1
            continue
        a, b = render(g, dpi), render(p, dpi)
        if len(a) != len(b):
            print(f"  {name:28s} page count {len(a)} vs {len(b)}")
            counts["size"] += 1
            continue
        if any(x.shape != y.shape for x, y in zip(a, b)):
            shapes = ", ".join(f"{x.shape[1]}x{x.shape[0]} vs {y.shape[1]}x{y.shape[0]} px" for x, y in zip(a, b))
            print(f"  {name:28s} size mismatch ({shapes})")
            counts["size"] += 1
            continue
        share = float(np.mean(np.concatenate([(x != y).ravel() for x, y in zip(a, b)])))
        if share == 0:
            print(f"  {name:28s} identical")
            counts["identical"] += 1
        else:
            print(f"  {name:28s} {100 * share:.2f}% of pixels differ")
            counts["differ"] += 1
    return counts


# ------------------------------------------------------------------- macros
def parse_macros(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("%"):
            continue
        m = MACRO.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def normalize(value: str) -> str:
    v = value.strip().replace("\\xspace", "")
    v = re.sub(r"\\num\{([^{}]*)\}", r"\1", v)
    return v.replace("{,}", "").replace(",", "").replace("\\,", "").replace(" ", "").replace("{}", "")


def compare_macros(gen_file: Path, paper_file: Path, verbose: bool) -> dict:
    g, p = parse_macros(gen_file), parse_macros(paper_file)
    common = sorted(set(g) & set(p))
    differ = [k for k in common if normalize(g[k]) != normalize(p[k])]
    gen_only = sorted(set(g) - set(p))
    paper_only = sorted(set(p) - set(g))
    print(f"  paper defines {len(p)}, generated defines {len(g)}; common {len(common)}, "
          f"identical {len(common) - len(differ)}, differ {len(differ)}; "
          f"paper only {len(paper_only)}, generated only {len(gen_only)}")
    if verbose:
        for k in differ:
            print(f"    differ      \\{k:28s} paper={normalize(p[k])}  generated={normalize(g[k])}")
        for k in paper_only:
            print(f"    paper only  \\{k:28s} {normalize(p[k])}")
        for k in gen_only:
            print(f"    gen. only   \\{k:28s} {normalize(g[k])}")
    return dict(common=len(common), differ=len(differ), paper_only=len(paper_only), generated_only=len(gen_only))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare regenerated figures and numbers.tex with the paper's.")
    ap.add_argument("--generated", type=Path, required=True,
                    help="output directory of the generators (contains figures/ and tables/)")
    ap.add_argument("--paper", type=Path, required=True,
                    help="source directory of the paper (contains figures/ and tables/)")
    ap.add_argument("--dpi", type=int, default=150, help="rendering resolution (default: %(default)s)")
    ap.add_argument("--verbose", "-v", action="store_true", help="list every differing and one-sided macro")
    args = ap.parse_args(argv)

    print(f"figures ({args.dpi} dpi): {args.generated / 'figures'} vs {args.paper / 'figures'}")
    fc = compare_figures(args.generated / "figures", args.paper / "figures", args.dpi)
    print(f"  {fc['identical']} identical, {fc['differ']} differ, {fc['size']} size mismatch, "
          f"{fc['missing']} missing on one side")

    gnum, pnum = args.generated / "tables" / "numbers.tex", args.paper / "tables" / "numbers.tex"
    print(f"macros: {gnum} vs {pnum}")
    if gnum.exists() and pnum.exists():
        compare_macros(gnum, pnum, args.verbose)
    else:
        print(f"  skipped: {gnum if not gnum.exists() else pnum} not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
