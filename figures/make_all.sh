#!/usr/bin/env bash
# Regenerate the paper's tables, macro file and figures, in the order that
# reproduces the published versions (see figures/README.md):
#
#   1. figures/paper_figures.py       tables/*.tex, tables/numbers.tex and
#                                     analysis/paper_numbers.json (plus its own
#                                     versions of the 11 data figures; its
#                                     attacks_auc.pdf is the published one)
#   2. figures/published_figures.py   the other 10 data figures exactly as
#                                     published; overwrites those of step 1
#   3. figures/paper_design_figure.py figures/design.pdf
#
# Usage:
#   figures/make_all.sh [--out DIR]
#
# DIR receives figures/ and tables/; the default is paper/ under the data root
# ($OLMO_TWIN_DATA/paper, or <repository>/data/paper when OLMO_TWIN_DATA is not
# set). The analysis outputs under the data root must exist (analysis/README.md);
# on a fresh data root run analysis/run_analysis.sh instead, which calls
# figures/paper_figures.py twice (--first-pass, then the full run).
# CPU only; use the analysis environment (env/analysis/requirements.txt).
# Set PYTHON to choose the interpreter (default: python3).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
eval "$(PYTHONPATH="$REPO" "$PYTHON" -m olmo_twin.paths --shell)"

OUT="$OLMO_TWIN_PAPER"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --out=*) OUT="${1#--out=}"; shift ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (usage: figures/make_all.sh [--out DIR])" >&2; exit 2 ;;
  esac
done

export MPLBACKEND=Agg
echo "data root: $OLMO_TWIN_DATA"
echo "output:    $OUT"

echo "== 1/3 figures/paper_figures.py"
"$PYTHON" "$REPO/figures/paper_figures.py" --out "$OUT"
echo "== 2/3 figures/published_figures.py"
"$PYTHON" "$REPO/figures/published_figures.py" --out "$OUT"
echo "== 3/3 figures/paper_design_figure.py"
"$PYTHON" "$REPO/figures/paper_design_figure.py" --out "$OUT/figures"

echo "done: $OUT/figures (12 PDFs), $OUT/tables, $OLMO_TWIN_ANALYSIS/paper_numbers.json"
