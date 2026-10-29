#!/usr/bin/env bash
# Run the analysis of the paper in order, on a CPU: from the score files of the
# scored models to the analysis tables, the paper's tables, macros and figures,
# Table 4 on all pairs and the post-training/RLVR results. analysis/README.md
# lists what every step reads and writes.
#
# Usage:
#   bash analysis/run_analysis.sh
#
# Environment:
#   OLMO_TWIN_DATA  data root (default: <repository>/data), see olmo_twin/paths.py
#   PYTHON          interpreter of the analysis environment (default: python3)
#   WORKERS         processes of the parallel steps (default: 32; results do not depend on it)
#   SKIP            steps to leave out, separated by spaces or commas, e.g. SKIP="attack_table rlvr"
#   FORCE=1         run the steps that need token ids or document texts even if these look absent
#   PAPER_TABLES    optional directory with the paper's numbers.tex and calibrated.tex, used by the
#                   Table 4 step for its cross-checks and interval macros
#                   (default: the generator's own output, $OLMO_TWIN_PAPER/tables)
#   PAPER_DIR       optional paper sources (tables/numbers.tex, tables/stages.tex) for the
#                   reproduction check of analysis/rlvr/finalize.py
#
# Steps, in order:
#   attack_table token_neff token_cache ladder calibrated first_pass screening collection
#   sensitivity length cohortF_audit cohortF_check full_pass published_figures table4 rlvr
#
# Every step overwrites its outputs under the data root in place.
set -euo pipefail

# The script takes no arguments (it is configured through the environment
# variables above); -h/--help prints this header instead of running.
case "${1:-}" in
    "") ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "usage: bash analysis/run_analysis.sh (no arguments; see --help for the environment variables)" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
eval "$(PYTHONPATH="$REPO" "$PYTHON" -m olmo_twin.paths --shell)"
export OLMO_TWIN_DATA            # the Python scripts resolve the same data root
export MPLBACKEND="${MPLBACKEND:-Agg}"
WORKERS="${WORKERS:-32}"
FORCE="${FORCE:-0}"
SKIP_LIST=" ${SKIP:-} "
SKIP_LIST="${SKIP_LIST//,/ }"
A="$REPO/analysis"
FIG="$REPO/figures"

echo "repository: $REPO"
echo "data root:  $OLMO_TWIN_DATA"

# step NAME: announce NAME and return 0, or return 1 if NAME is listed in SKIP
step() {
    if [[ "$SKIP_LIST" == *" $1 "* ]]; then
        echo "--- skip $1 (SKIP)"
        return 1
    fi
    echo "=== $1  $(date -u +%H:%M:%SZ)"
}

# guard NAME REASON TEST...: return 0 if TEST succeeds or FORCE=1, else report and return 1
guard() {
    local name="$1" reason="$2"
    shift 2
    if "$@"; then
        return 0
    fi
    if [[ "$FORCE" == 1 ]]; then
        echo "--- $name: $reason; running anyway (FORCE=1)"
        return 0
    fi
    echo "--- skip $name: $reason; the existing outputs are kept (FORCE=1 runs it anyway)"
    return 1
}

# has_token_ids DIR: the per-token files DIR/tokens/*.npz still contain token ids
has_token_ids() {
    "$PYTHON" - "$1" <<'PY'
import os
import sys

import numpy as np

tokens = os.path.join(sys.argv[1], "tokens")
try:
    first = next(e.path for e in os.scandir(tokens) if e.name.endswith(".npz"))
except (FileNotFoundError, StopIteration):
    sys.exit(1)
with np.load(first) as z:
    sys.exit(0 if "token_ids" in z.files else 1)
PY
}

has_texts() { [[ -d "$OLMO_TWIN_BUNDLE/texts_all" ]]; }
ext_ids() { has_token_ids "$OLMO_TWIN_RUN/scores_ext/ext-injected"; }
endpoint_ids() { has_token_ids "$OLMO_TWIN_RUN/scores/injected-step23852"; }
attack_inputs() { ext_ids && has_texts; }
cohortF_texts() { has_texts && [[ -d "$OLMO_TWIN_COHORTF_REPARSE" ]]; }
cohortF_tokens() { [[ -d "$OLMO_TWIN_BUNDLE/tokens/F" && -d "$OLMO_TWIN_RUN/cohortF_2x2/tokens" ]] && ext_ids; }

# ---------------------------------------------------------------- per-document tables
if step attack_table && guard attack_table "needs per-token files with token ids and the document texts (bundle/texts_all)" attack_inputs; then
    "$PYTHON" "$A/build_injection_v2_attack_table.py" \
        --evaluation-manifest "$OLMO_TWIN_BUNDLE/evaluation_manifest.csv.gz" \
        --design-manifest "$OLMO_TWIN_BUNDLE/manifest.csv.gz" \
        --placement "$OLMO_TWIN_BUNDLE/stream_plan/placement.jsonl.gz" \
        --texts-index "$OLMO_TWIN_BUNDLE/texts_index.csv.gz" \
        --bundle "$OLMO_TWIN_BUNDLE" \
        --scores "injected=$OLMO_TWIN_RUN/scores_ext/ext-injected" \
        --scores "seed42=$OLMO_TWIN_RUN/scores_ext/ext-seed42" \
        --scores "stage1=$OLMO_TWIN_RUN/scores_ext/ext-stage1-pretreatment" \
        --scores "sibling=$OLMO_TWIN_RUN/scores/sibling-seed42069-step23852" \
        --out "$OLMO_TWIN_ANALYSIS/per_document_attacks_prelim.csv.gz" --workers "$WORKERS"
fi

if step token_neff && guard token_neff "needs per-token files with token ids" endpoint_ids; then
    NWORKERS="$WORKERS" "$PYTHON" "$A/compute_token_neff.py"
fi

if step token_cache; then
    WORKERS="$WORKERS" "$PYTHON" "$A/precompute_token_cache.py"
fi

if step ladder; then
    "$PYTHON" "$A/build_ladder_table.py"
fi

if step calibrated; then
    if ext_ids; then
        "$PYTHON" "$A/calibrated_attacks.py" --workers "$WORKERS" --recompute
    elif [[ -f "$OLMO_TWIN_ANALYSIS/cache/doc_stats.npz" ]]; then
        echo "--- calibrated: per-token files without token ids; using analysis/cache/doc_stats.npz"
        "$PYTHON" "$A/calibrated_attacks.py" --workers "$WORKERS"
    elif guard calibrated "needs per-token files with token ids or analysis/cache/doc_stats.npz" false; then
        "$PYTHON" "$A/calibrated_attacks.py" --workers "$WORKERS" --recompute
    fi
fi

# ---------------------------------------------------------------- first pass of the generator
if step first_pass; then
    "$PYTHON" "$FIG/paper_figures.py" --out "$OLMO_TWIN_PAPER" --first-pass
fi

# ---------------------------------------------------------------- analyses read by the generator
if step screening; then
    "$PYTHON" -u "$A/screening_analysis.py"
fi
if step collection; then
    "$PYTHON" -u "$A/collection_by_domain.py"
fi
if step sensitivity; then
    "$PYTHON" -u "$A/exclusion_sensitivity.py"
fi
if step length; then
    "$PYTHON" -u "$A/length_truncation_analysis.py"
fi
if step cohortF_audit && guard cohortF_audit "needs the document texts (bundle/texts_all) and cohortF_reparse/" cohortF_texts; then
    "$PYTHON" -u "$A/audit_cohortF_data.py"
fi
if step cohortF_check && guard cohortF_check "needs token arrays (bundle/tokens/F, run/cohortF_2x2/tokens) and per-token files with token ids" cohortF_tokens; then
    "$PYTHON" -u "$A/cohortF_2x2_check.py"
fi

# ---------------------------------------------------------------- full run of the generator
if step full_pass; then
    "$PYTHON" "$FIG/paper_figures.py" --out "$OLMO_TWIN_PAPER"
fi
if step published_figures; then
    "$PYTHON" "$FIG/published_figures.py" --out "$OLMO_TWIN_PAPER"
    "$PYTHON" "$FIG/paper_design_figure.py" --out "$OLMO_TWIN_PAPER/figures"
fi

# ---------------------------------------------------------------- Table 4 on all NAT1 pairs
if step table4; then
    T4="$OLMO_TWIN_ANALYSIS/table4_all_pairs"
    if [[ -n "${PAPER_TABLES:-}" ]]; then
        "$PYTHON" -u "$A/recompute_table4_all_pairs.py" --out "$T4" --paper-tables "$PAPER_TABLES"
        "$PYTHON" -u "$A/format_table4.py" --in "$T4" --numbers "$PAPER_TABLES/numbers.tex"
    else
        "$PYTHON" -u "$A/recompute_table4_all_pairs.py" --out "$T4"
        "$PYTHON" -u "$A/format_table4.py" --in "$T4" --numbers "$OLMO_TWIN_PAPER/tables/numbers.tex"
    fi
fi

# ---------------------------------------------------------------- post-training and RLVR
if step rlvr; then
    "$PYTHON" -u "$A/rlvr/recompute.py" "$WORKERS"
    "$PYTHON" -u "$A/rlvr/rl_minus_dpo.py"
    if [[ -n "${PAPER_DIR:-}" ]]; then
        "$PYTHON" -u "$A/rlvr/finalize.py" "$OLMO_TWIN_RLVR" --paper-dir "$PAPER_DIR"
    else
        "$PYTHON" -u "$A/rlvr/finalize.py" "$OLMO_TWIN_RLVR"
    fi
fi

echo "done: analysis outputs in $OLMO_TWIN_ANALYSIS, tables and figures in $OLMO_TWIN_PAPER, RLVR results in $OLMO_TWIN_RLVR"
