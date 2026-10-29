#!/usr/bin/env bash
# Download the checkpoint ladder of one of AI2's OLMo 2 1B (0425) stage-2 runs
# from olmo-checkpoints.org: config.yaml and model.pt of steps 1000, 2000, ...,
# 23000 and of the final step 23852 (24 checkpoints, about 6 GB each).
#
# Usage: bash training/midtrain/download_ai2_stage2_checkpoints.sh ingredient3|ingredient1
#   ingredient3  seed 42, the control: the run our injected run replicates,
#                -> $OLMO_TWIN_DATA/assets/checkpoints/official_stage2_ingredient3_seed42
#   ingredient1  seed 42069, the retrained reference
#                -> $OLMO_TWIN_DATA/assets/checkpoints/official_stage2_ingredient1
# AI2's ingredient numbers are not in seed order: ingredient1 = seed 42069,
# ingredient2 = seed 666, ingredient3 = seed 42.
#
# Environment: DEST overrides the target directory. Restartable: complete files
# are kept, partial downloads resume (wget -c into <file>.part, which is renamed
# only after wget succeeds). Aborts unless every config.yaml declares the
# expected seed.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"
: "${OLMO_TWIN_ASSETS:?python3 -m olmo_twin.paths --shell failed}"

INGREDIENT="${1:-}"
case "$INGREDIENT" in
    ingredient3) SEED=42;    DEST="${DEST:-$OLMO_TWIN_ASSETS/checkpoints/official_stage2_ingredient3_seed42}" ;;
    ingredient1) SEED=42069; DEST="${DEST:-$OLMO_TWIN_ASSETS/checkpoints/official_stage2_ingredient1}" ;;
    *) echo "usage: $0 ingredient3|ingredient1" >&2; exit 2 ;;
esac
BASE="https://olmo-checkpoints.org/ai2-llm/peteish1/stage2/olmo2-1b-0425_stage2_${INGREDIENT}"
steps=$(seq 1000 1000 23000; echo 23852)
for s in $steps; do
    d="$DEST/step${s}-unsharded"; mkdir -p "$d"
    for f in config.yaml model.pt; do
        [[ -s "$d/$f" ]] && continue
        # only promote a completed file: wget -c exits nonzero on truncation
        wget -q -c -O "$d/$f.part" "$BASE/step${s}-unsharded/$f"
        mv "$d/$f.part" "$d/$f"
    done
    seed=$(grep -m1 '^seed:' "$d/config.yaml" | awk '{print $2}')
    [[ "$seed" == "$SEED" ]] || { echo "ABORT: step$s has seed $seed, expected $SEED" >&2; exit 2; }
    echo "step${s} ok (seed $SEED) $(date -u +%H:%M:%S)"
done
(cd "$DEST" && find . -name model.pt -printf '%s %p\n' | sort -k2V > SIZES.txt)
touch "$DEST/.download_complete"
echo ALL-DONE
