#!/usr/bin/env python3
"""The testbed schematic for the paper (figures/design.pdf).

A drawing, not a data figure: the official data stream, the treated stream
with injected slots, the matched pairs and the coin flip, the checkpoint
ladders, and the post-training stages (SFT, DPO, two RLVR stages). Drawn at IEEE
single-column width. Reads no data.

Usage:
    python figures/paper_design_figure.py [--out DIR]

writes DIR/design.pdf; DIR defaults to paper/figures under the data root
(olmo_twin.paths), the directory the other two figure scripts write their
figures to by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root (olmo_twin)
from olmo_twin import paths  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = paths.PAPER_OUT / "figures" / "design.pdf"   # --out rebinds the directory
COL = 3.40

C_BG = "#d9d9d9"       # official chunks
C_INJ = "#2a78d6"      # injected slots
C_HOLD = "#eb6834"     # held-out partner
C_CTRL = "#777777"
C_TXT = "#222222"


def strip(ax, y, label, injected_at=(), n=40, h=0.42):
    for i in range(n):
        col = C_INJ if i in injected_at else C_BG
        ax.add_patch(Rectangle((i * 0.25, y), 0.24, h, facecolor=col, edgecolor="white", lw=0.3))
    ax.text(-0.15, y + h / 2, label, ha="right", va="center", fontsize=6.6, color=C_TXT)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
                         "pdf.fonttype": 42, "font.size": 6.5})
    fig, ax = plt.subplots(figsize=(COL, 3.15))
    ax.set_xlim(-2.3, 10.9); ax.set_ylim(-0.25, 6.45); ax.axis("off")

    # --- pairs box (top left) and its explanation (top right)
    bx, by, bw, bh = -2.25, 3.95, 3.75, 2.35
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.04", facecolor="white", edgecolor="#999", lw=0.6))
    ax.text(bx + bw / 2, by + bh - 0.28, "7,814 matched pairs", ha="center", va="center", fontsize=5.9, fontweight="bold")
    ly = by + bh - 0.68
    ax.add_patch(Rectangle((bx + 0.15, ly - 0.1), 0.22, 0.2, facecolor=C_INJ, edgecolor="none"))
    ax.text(bx + 0.45, ly, "injected", ha="left", va="center", fontsize=5.2, color=C_INJ)
    ax.add_patch(Rectangle((bx + 2.05, ly - 0.1), 0.22, 0.2, facecolor=C_HOLD, edgecolor="none"))
    ax.text(bx + 2.35, ly, "held out", ha="left", va="center", fontsize=5.2, color=C_HOLD)
    for k, yy in enumerate((5.2, 4.83, 4.46)):
        left_inj = k != 1
        ax.add_patch(Rectangle((-1.85, yy - 0.13), 0.9, 0.26, facecolor=C_INJ if left_inj else C_HOLD, edgecolor="none"))
        ax.add_patch(Rectangle((-0.45, yy - 0.13), 0.9, 0.26, facecolor=C_HOLD if left_inj else C_INJ, edgecolor="none"))
        ax.text(-0.7, yy, "\u2194", ha="center", va="center", fontsize=5.5, color="#555")
    tx = 1.8
    for yy, line in ((5.75, "pairs matched on length quartile and"),
                     (5.45, "pre-treatment bits/byte; one fair coin"),
                     (5.15, "per pair decides which member is injected"),
                     (4.7, "every document verified absent from all"),
                     (4.4, "13 training components (8.3 TiB scanned)")):
        ax.text(tx, yy, line, ha="left", va="center", fontsize=5.8, color="#444")

    # --- streams
    strip(ax, 2.95, "AI2 seed-42 run\n(the control)", injected_at=(), n=40)
    strip(ax, 2.1, "treated run", injected_at=(6, 7, 19, 31, 32), n=40)
    ax.text(10.15, 3.16, "12,212,224 chunks,", ha="left", va="center", fontsize=5.6, color="#555")
    ax.text(10.15, 2.92, "byte-identical", ha="left", va="center", fontsize=5.6, color="#555")
    ax.text(10.15, 2.31, "24,081 slots", ha="left", va="center", fontsize=5.6, color=C_INJ)
    ax.text(10.15, 2.07, "replaced (0.197%)", ha="left", va="center", fontsize=5.6, color=C_INJ)
    for i in range(0, 40, 4):
        for yb in (2.95, 2.1):
            ax.plot([i * 0.25 + 0.12] * 2, [yb - 0.17, yb - 0.04], color=C_TXT, lw=0.5)
    ax.text(5.4, 1.73, "checkpoints every 1,000 updates", ha="center", va="center", fontsize=5.5, color="#555")
    ax.text(5.4, 1.49, "23,852 updates, 50B tokens, linear anneal to zero",
            ha="center", va="center", fontsize=5.5, color="#555")

    # --- post-training and scoring row
    bw, step = 2.75, 3.35
    boxes = [(-2.2, "SFT", "both twins,\nsame recipe"),
             (-2.2 + step, "DPO", "both twins,\nsame recipe"),
             (-2.2 + 2 * step, "RLVR", "two stages,\nboth twins"),
             (-2.2 + 3 * step, "score", "step-matched\nmodel pairs")]
    for x0, head, body in boxes:
        ax.add_patch(FancyBboxPatch((x0, 0.1), bw, 1.05, boxstyle="round,pad=0.03", facecolor="#f3f1ee", edgecolor="#999", lw=0.5))
        ax.text(x0 + bw / 2, 0.93, head, ha="center", va="center", fontsize=6.0, fontweight="bold")
        ax.text(x0 + bw / 2, 0.46, body, ha="center", va="center", fontsize=5.3, color="#333")
    for x0, _, _ in boxes[:-1]:
        ax.annotate("", xy=(x0 + step - 0.03, 0.63), xytext=(x0 + bw + 0.03, 0.63), arrowprops=dict(arrowstyle="->", lw=0.6, color="#666"))
    ax.annotate("", xy=(-1.0, 1.18), xytext=(-1.0, 1.72), arrowprops=dict(arrowstyle="->", lw=0.6, color="#666"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draw the design schematic (design.pdf).")
    ap.add_argument("--out", type=Path, default=OUT.parent,
                    help="directory that receives design.pdf (default: %(default)s, "
                         "i.e. paper/figures under the data root)")
    OUT = ap.parse_args().out / "design.pdf"
    main()
