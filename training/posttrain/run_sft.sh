#!/usr/bin/env bash
# SFT of one arm: open-instruct finetune.py on one node with 8 GPUs
# (accelerate + DeepSpeed ZeRO-3), flags from sft_args.sh. We used 8
# A100-SXM4-80GB GPUs; 2 epochs, 13,466 updates.
#
# Usage:
#   OPEN_INSTRUCT_DIR=/path/to/open-instruct ARM=baseline|injected [PORT=29500] \
#     bash training/posttrain/run_sft.sh
#
#   ARM=baseline  the control, from AI2's seed-42 stage-2 endpoint
#                 ($OLMO_TWIN_DATA/assets/hf_models/OLMo-2-0425-1B-seed42-hf)
#   ARM=injected  the treated model, from our mid-training endpoint
#                 ($OLMO_TWIN_DATA/assets/hf_models/olmo2-1b-injection-v2-final-hf)
#   (both converted with convert_to_hf.sh)
#
# Output: $OLMO_TWIN_DATA/posttrain/runs/sft_<ARM>/olmo2_1b_sft_<ARM>__1__0
# (final model and step_<N> checkpoints every 1000 updates).
#
# Environment:
#   OPEN_INSTRUCT_DIR  (required) open-instruct checkout at OPEN_INSTRUCT_COMMIT
#                      (versions.env) with open-instruct.patch applied
#   PORT               main process port of accelerate (default 29500)
# The run tokenizes the dataset into the local cache if it is not there yet;
# on GPU nodes without internet access run cache_datasets.sh sft first and
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1.
set -euo pipefail
# No arguments (configured through the environment variables above);
# -h/--help prints this header instead of running.
case "${1:-}" in
    "") ;;
    -h|--help) awk 'NR > 1 && /^#/ { f = 1; sub(/^# ?/, ""); print; next } f { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "usage: bash training/posttrain/run_sft.sh (no arguments; see --help)" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"
: "${OLMO_TWIN_DATA:?python3 -m olmo_twin.paths --shell failed}"
source "$SCRIPT_DIR/versions.env"
ARM="${ARM:?set ARM=baseline or ARM=injected}"
OI="${OPEN_INSTRUCT_DIR:?set OPEN_INSTRUCT_DIR to the patched open-instruct checkout}"
OI="$(cd -- "$OI" && pwd)"
case "$ARM" in
    # Official stage-2 seed-42 endpoint = AI2's "ingredient3" (their numbering
    # is not seed order: ingredient1=seed42069, 2=seed666, 3=seed42). Our
    # injected run replicates seed 42, so seed 42 is the only valid twin.
    baseline) MODEL_DIR="$OLMO_TWIN_ASSETS/hf_models/OLMo-2-0425-1B-seed42-hf" ;;
    injected) MODEL_DIR="$OLMO_TWIN_ASSETS/hf_models/olmo2-1b-injection-v2-final-hf" ;;
    *) echo "ERROR: ARM must be baseline|injected" >&2; exit 2 ;;
esac
OUT="$OLMO_TWIN_POSTTRAIN/runs/sft_${ARM}"

# --- fail-closed preflight -------------------------------------------------
[[ "$(git -C "$OI" rev-parse HEAD)" == "$OPEN_INSTRUCT_COMMIT" ]] \
    || { echo "ERROR: open-instruct not at pinned commit $OPEN_INSTRUCT_COMMIT" >&2; exit 2; }
grep -q "OI_RUN_TIMESTAMP" "$OI/open_instruct/finetune.py" \
    && grep -q "keep_last_n_checkpoints=args.keep_last_n_checkpoints" "$OI/open_instruct/finetune.py" \
    || { echo "ERROR: open-instruct.patch is not applied in $OI" >&2; exit 2; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "ERROR: missing model $MODEL_DIR" >&2; exit 2; }
[[ -e "$OUT/.complete" ]] && { echo "ERROR: $OUT already complete" >&2; exit 2; }
mkdir -p "$OUT"

export HF_HOME="$OLMO_TWIN_ASSETS/hf_cache_posttrain"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
# The patched finetune.py names the run directory with this value instead of
# each rank's own launch time, so all ranks (and a restart) use one directory.
export OI_RUN_TIMESTAMP=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# NLTK_DATA (punkt), if set, is passed on unchanged; the post-training image
# sets it (env/posttrain/Dockerfile).
export PYTHONPATH="$OI${PYTHONPATH:+:$PYTHONPATH}"

source "$SCRIPT_DIR/sft_args.sh"
cd "$OI"
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --main_process_port "${PORT:-29500}" \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    "${sft_args[@]}"

touch "$OUT/.complete"
echo "=== SFT $ARM DONE $(date -u) ==="
