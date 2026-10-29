#!/usr/bin/env bash
# DPO of one arm: open-instruct dpo_tune_cache.py on one node with 8 GPUs
# (accelerate + DeepSpeed ZeRO-2), flags from dpo_args.sh, starting from this
# arm's SFT model, so each arm runs mid-training -> SFT -> DPO with identical
# recipes. We used 8 A100-SXM4-80GB GPUs; 1 epoch, 2,905 updates.
#
# Usage:
#   OPEN_INSTRUCT_DIR=/path/to/open-instruct ARM=baseline|injected [PORT=29500] \
#     bash training/posttrain/run_dpo.sh
#
# Input:  $OLMO_TWIN_DATA/posttrain/runs/sft_<ARM>/olmo2_1b_sft_<ARM>__1__0 (run_sft.sh)
# Output: $OLMO_TWIN_DATA/posttrain/runs/dpo_<ARM>/olmo2_1b_dpo_<ARM>__111__0
# (final model and step_<N> checkpoints every 500 updates).
#
# The ZeRO-2 configuration that dpo used is not part of open-instruct at the
# pinned commit; ds_configs/stage2_no_offloading_accelerate.conf (a copy of
# AI2's configs/ds_configs/stage2_accelerate.conf at open-instruct commit
# fc7748e) is copied into the checkout under that name if it is missing.
#
# Environment: as run_sft.sh (OPEN_INSTRUCT_DIR required, PORT). On GPU nodes
# without internet access run cache_datasets.sh dpo (same ARM) first and
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1.
set -euo pipefail
# No arguments (configured through the environment variables above);
# -h/--help prints this header instead of running.
case "${1:-}" in
    "") ;;
    -h|--help) awk 'NR > 1 && /^#/ { f = 1; sub(/^# ?/, ""); print; next } f { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "usage: bash training/posttrain/run_dpo.sh (no arguments; see --help)" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"
: "${OLMO_TWIN_DATA:?python3 -m olmo_twin.paths --shell failed}"
source "$SCRIPT_DIR/versions.env"
ARM="${ARM:?set ARM=baseline or ARM=injected}"
case "$ARM" in baseline|injected) ;; *) echo "ERROR: ARM must be baseline|injected" >&2; exit 2 ;; esac
OI="${OPEN_INSTRUCT_DIR:?set OPEN_INSTRUCT_DIR to the patched open-instruct checkout}"
OI="$(cd -- "$OI" && pwd)"
MODEL_DIR="$OLMO_TWIN_POSTTRAIN/runs/sft_${ARM}/olmo2_1b_sft_${ARM}__1__0"
OUT="$OLMO_TWIN_POSTTRAIN/runs/dpo_${ARM}"
DS_CONFIG=configs/ds_configs/stage2_no_offloading_accelerate.conf

# --- fail-closed preflight -------------------------------------------------
[[ "$(git -C "$OI" rev-parse HEAD)" == "$OPEN_INSTRUCT_COMMIT" ]] \
    || { echo "ERROR: open-instruct not at pinned commit $OPEN_INSTRUCT_COMMIT" >&2; exit 2; }
grep -q "OI_RUN_TIMESTAMP" "$OI/open_instruct/dpo_tune_cache.py" \
    && grep -q "keep_last_n_checkpoints=args.keep_last_n_checkpoints" "$OI/open_instruct/dpo_tune_cache.py" \
    || { echo "ERROR: open-instruct.patch is not applied in $OI" >&2; exit 2; }
[[ -f "$MODEL_DIR/pytorch_model.bin" ]] || { echo "ERROR: no SFT model for arm $ARM at $MODEL_DIR" >&2; exit 2; }
if [[ ! -f "$OI/$DS_CONFIG" ]]; then
    cp "$SCRIPT_DIR/ds_configs/stage2_no_offloading_accelerate.conf" "$OI/$DS_CONFIG"
fi
cmp -s "$SCRIPT_DIR/ds_configs/stage2_no_offloading_accelerate.conf" "$OI/$DS_CONFIG" \
    || { echo "ERROR: $OI/$DS_CONFIG differs from training/posttrain/ds_configs/" >&2; exit 2; }
[[ -e "$OUT/.complete" ]] && { echo "ERROR: $OUT already complete" >&2; exit 2; }
mkdir -p "$OUT"
echo "arm=$ARM  from=$MODEL_DIR  out=$OUT"

export HF_HOME="$OLMO_TWIN_ASSETS/hf_cache_posttrain"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OI_RUN_TIMESTAMP=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$OI${PYTHONPATH:+:$PYTHONPATH}"

source "$SCRIPT_DIR/dpo_args.sh"
cd "$OI"
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --main_process_port "${PORT:-29500}" \
    --use_deepspeed \
    --deepspeed_config_file "$DS_CONFIG" \
    --deepspeed_multinode_launcher standard \
    "${dpo_args[@]}"

touch "$OUT/.complete"
echo "=== DPO $ARM DONE $(date -u) ==="
