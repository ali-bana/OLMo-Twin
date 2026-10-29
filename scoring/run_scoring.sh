#!/usr/bin/env bash
# Score the evaluation set under one model listed in scoring/models.tsv.
#
#   scoring/run_scoring.sh LABEL [RANK WORLD]   score every WORLD-th document of the evaluation
#                                               manifest, starting at document RANK (default 0 1);
#                                               with WORLD = 1 the merge pass follows directly
#   scoring/run_scoring.sh LABEL merge          merge the per-rank files into scores.csv.gz
#   scoring/run_scoring.sh --list               print the labels
#
# One process uses one GPU. To split one model over N GPUs or jobs, start
# `run_scoring.sh LABEL r N` for r = 0, ..., N-1 (as N processes with different
# CUDA_VISIBLE_DEVICES, or as N jobs of whatever scheduler you use), then run
# `run_scoring.sh LABEL merge` once all of them have finished. A rank that was
# killed continues where it stopped when it is started again with the same
# RANK and WORLD. The paper's models were scored with WORLD = 8, except the
# checkpoint ladders and the cohort-F re-parse (WORLD = 1); the documents are
# scored independently, so the split does not change any score.
#
# Environment:
#   OLMO_TWIN_DATA  data root (default <repository>/data); the paths in models.tsv are relative to it
#   PYTHON          Python interpreter (default python3). The OLMo-native scorers need the `olmo`
#                   package of OLMo at commit 090253d, the Hugging Face scorer `transformers`
#                   (see env/README.md).
#   OLMO_REPO       optional OLMo checkout; prepended to PYTHONPATH for the OLMo-native scorers
#   MODEL_PATH      optional model directory that replaces the models.tsv entry, e.g. when a glob
#                   matches more than one RLVR run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
TABLE="$HERE/models.tsv"
PY="${PYTHON:-python3}"

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit "${1:-2}"
}

# Data rows of models.tsv (no comments, no header).
table_rows() { awk -F'\t' '!/^#/ && NF > 0 && $1 != "label"' "$TABLE"; }

[[ $# -ge 1 ]] || usage
case "$1" in
    -h|--help) usage 0 ;;
    --list) table_rows | awk -F'\t' '{printf "%-30s %-9s %s\n", $1, $2, $4}'; exit 0 ;;
esac

LABEL="$1"
MERGE=0
RANK=0
WORLD=1
if [[ "${2:-}" == merge ]]; then
    [[ $# -eq 2 ]] || usage
    MERGE=1
else
    [[ $# -le 3 ]] || usage
    RANK="${2:-0}"
    WORLD="${3:-1}"
    if ! [[ "$RANK" =~ ^[0-9]+$ && "$WORLD" =~ ^[0-9]+$ ]] || (( WORLD < 1 || RANK >= WORLD )); then
        echo "error: RANK and WORLD must be integers with 0 <= RANK < WORLD (got '$RANK' '$WORLD')" >&2
        exit 2
    fi
fi

# Data layout (OLMO_TWIN_DATA, OLMO_TWIN_BUNDLE, ...), resolved by olmo_twin/paths.py.
LAYOUT="$(PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m olmo_twin.paths --shell)" || {
    echo "error: could not run '$PY -m olmo_twin.paths --shell'" >&2
    exit 1
}
eval "$LAYOUT"

ROW="$(table_rows | awk -F'\t' -v l="$LABEL" '$1 == l')"
if [[ -z "$ROW" ]]; then
    echo "error: label '$LABEL' is not in $TABLE (run with --list to see the labels)" >&2
    exit 2
fi
if [[ "$(printf '%s\n' "$ROW" | wc -l)" -ne 1 ]]; then
    echo "error: label '$LABEL' appears more than once in $TABLE" >&2
    exit 2
fi
IFS=$'\t' read -r _ SCORER MODEL_REL OUT_REL BUNDLE_REL EXPECT_SEED EXTRA <<< "$ROW"

case "$SCORER" in
    olmo)     SCRIPT="$HERE/score_injection_v2.py";          MODEL_FLAG=--checkpoint ;;
    olmo_ext) SCRIPT="$HERE/score_injection_v2_extended.py"; MODEL_FLAG=--checkpoint ;;
    hf)       SCRIPT="$HERE/score_injection_v2_hf.py";       MODEL_FLAG=--model ;;
    *) echo "error: unknown scorer '$SCORER' for '$LABEL' in $TABLE" >&2; exit 2 ;;
esac

if [[ "$BUNDLE_REL" == "-" ]]; then
    BUNDLE_DIR="$OLMO_TWIN_BUNDLE"
else
    BUNDLE_DIR="$OLMO_TWIN_DATA/$BUNDLE_REL"
fi
MANIFEST="$BUNDLE_DIR/evaluation_manifest.csv.gz"
OUT="$OLMO_TWIN_DATA/$OUT_REL"
EXTRA_ARGS=()
if [[ "$EXTRA" != "-" ]]; then
    read -r -a EXTRA_ARGS <<< "$EXTRA"
fi

# Model directory: MODEL_PATH, else the models.tsv entry (a glob that must match one directory).
if [[ -n "${MODEL_PATH:-}" ]]; then
    MODEL="$MODEL_PATH"
else
    shopt -s nullglob
    MATCHES=( "$OLMO_TWIN_DATA"/$MODEL_REL )
    shopt -u nullglob
    if (( ${#MATCHES[@]} == 1 )); then
        MODEL="${MATCHES[0]}"
    elif (( ${#MATCHES[@]} == 0 )); then
        MODEL="$OLMO_TWIN_DATA/$MODEL_REL"
    else
        echo "error: $OLMO_TWIN_DATA/$MODEL_REL matches ${#MATCHES[@]} directories:" >&2
        printf '  %s\n' "${MATCHES[@]}" >&2
        echo "set MODEL_PATH to the one to score" >&2
        exit 2
    fi
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "error: no evaluation manifest at $MANIFEST" >&2
    exit 1
fi

# Number of CSV records (not lines) in a gzipped CSV file, header excluded.
count_rows() {
    "$PY" -c 'import csv, gzip, sys
csv.field_size_limit(sys.maxsize)
with gzip.open(sys.argv[1], "rt", newline="") as fh:
    print(sum(1 for _ in csv.DictReader(fh)))' "$1"
}

EXPECTED="$(count_rows "$MANIFEST")"
echo "[$(date -u +%FT%TZ)] $LABEL: $SCORER scorer, $EXPECTED documents"
echo "  model:    $MODEL"
echo "  manifest: $MANIFEST"
echo "  output:   $OUT"

# A merged file counts as complete when it has one row per document of the
# evaluation manifest (every document gets a row, including those skipped as
# too long). A shorter one is reported and removed; the per-rank files are kept,
# so scoring resumes from them and the merge is redone.
if [[ -f "$OUT/scores.csv.gz" ]]; then
    HAVE="$(count_rows "$OUT/scores.csv.gz")"
    if (( HAVE >= EXPECTED )); then
        echo "already merged with $HAVE rows; nothing to do"
        exit 0
    fi
    echo "existing $OUT/scores.csv.gz has only $HAVE of $EXPECTED rows; removing it and resuming" >&2
    rm -f "$OUT/scores.csv.gz"
fi

ARGS=(--evaluation-manifest "$MANIFEST" --bundle "$BUNDLE_DIR" "$MODEL_FLAG" "$MODEL"
      --revision "$LABEL" --out-dir "$OUT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

merge() {
    "$PY" "$SCRIPT" "${ARGS[@]}" --merge
    local n
    n="$(count_rows "$OUT/scores.csv.gz")"
    echo "[$(date -u +%FT%TZ)] $LABEL merged: $n rows"
    if (( n < EXPECTED )); then
        echo "error: only $n of $EXPECTED rows; a rank has not finished. Removing scores.csv.gz;" >&2
        echo "       start the unfinished ranks again, then merge." >&2
        rm -f "$OUT/scores.csv.gz"
        exit 1
    fi
    if (( n != EXPECTED )); then
        echo "note: $n rows for $EXPECTED documents (duplicate rows?)" >&2
    fi
}

if (( MERGE )); then
    merge
    exit 0
fi

# Checks before loading the model.
if [[ "$SCORER" == hf ]]; then
    [[ -f "$MODEL/config.json" ]] || { echo "error: no Hugging Face model at $MODEL" >&2; exit 1; }
else
    [[ -f "$MODEL/model.pt" && -f "$MODEL/config.yaml" ]] || {
        echo "error: no OLMo checkpoint (model.pt, config.yaml) at $MODEL" >&2; exit 1; }
    # Scoring a checkpoint as the wrong arm would invert the comparison, so the
    # seed declared in its config.yaml is checked rather than trusting the path.
    SEED="$(grep -m1 '^seed:' "$MODEL/config.yaml" | awk '{print $2}' || true)"
    RUN_NAME="$(grep -m1 '^run_name:' "$MODEL/config.yaml" | awk '{print $2}' || true)"
    echo "  config:   seed ${SEED:-?}, run_name ${RUN_NAME:-?}"
    if [[ "$EXPECT_SEED" != "-" && "$SEED" != "$EXPECT_SEED" ]]; then
        echo "error: $MODEL declares seed '$SEED' but '$LABEL' expects seed '$EXPECT_SEED'" >&2
        exit 1
    fi
    if [[ -n "${OLMO_REPO:-}" ]]; then
        export PYTHONPATH="$OLMO_REPO${PYTHONPATH:+:$PYTHONPATH}"
    fi
fi

mkdir -p "$OUT"
"$PY" "$SCRIPT" "${ARGS[@]}" --rank "$RANK" --world-size "$WORLD"
echo "[$(date -u +%FT%TZ)] $LABEL rank $RANK/$WORLD done"
if (( WORLD == 1 )); then
    merge
fi
