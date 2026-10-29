#!/usr/bin/env bash
# Tokenize a post-training dataset into open-instruct's local dataset cache
# ($OLMO_TWIN_DATA/posttrain/dataset_cache) without training
# (--cache_dataset_only). The cache is keyed by a hash of the dataset and
# tokenizer configuration, including the tokenizer path, so this script passes
# exactly the arguments of the training run (sourced from *_args.sh) with the
# same ARM and MODEL_DIR. A training run that can reach the Hugging Face Hub
# builds the same cache itself; this step is for setups whose GPU nodes have
# no internet access (run it on a machine that has, sharing the data root).
#
# Usage:
#   OPEN_INSTRUCT_DIR=/path/to/open-instruct [ARM=baseline|injected] [CPU_ONLY=1] \
#     bash training/posttrain/cache_datasets.sh sft|dpo|rlvr
#
#   sft   tulu-3-sft-olmo-2-mixture-0225. finetune.py takes the tokenizer from
#         the start model (MODEL_DIR), so the cache is specific to the arm:
#         ARM (default baseline) selects the arm's converted start model, as in
#         run_sft.sh.
#   dpo   olmo-2-0425-1b-preference-mix. dpo_args.sh passes
#         --tokenizer_name_or_path "$MODEL_DIR", the arm's SFT model, so DPO
#         caching must follow SFT of that arm; ARM defaults to baseline.
#   rlvr  both RLVR datasets (stages 1 and 2); the tokenizer is the 7B-DPO
#         tokenizer directory of versions_rl.env, so one cache serves both arms
#         (ARM=cache only names the run).
#
# Environment:
#   OPEN_INSTRUCT_DIR  (required) open-instruct checkout at OPEN_INSTRUCT_COMMIT
#                      (versions.env) with open-instruct.patch applied
#   CPU_ONLY           1: put cpu_shims/ on PYTHONPATH (hosts without a GPU driver)
#   PYTHON             interpreter of the post-training environment (default python3)
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"
: "${OLMO_TWIN_DATA:?python3 -m olmo_twin.paths --shell failed}"
source "$SCRIPT_DIR/versions.env"
WHAT="${1:?usage: cache_datasets.sh sft|dpo|rlvr}"
OI="${OPEN_INSTRUCT_DIR:?set OPEN_INSTRUCT_DIR to the patched open-instruct checkout}"
OI="$(cd -- "$OI" && pwd)"
PYTHON="${PYTHON:-python3}"
[[ "$(git -C "$OI" rev-parse HEAD)" == "$OPEN_INSTRUCT_COMMIT" ]] \
    || { echo "ERROR: open-instruct not at pinned commit $OPEN_INSTRUCT_COMMIT" >&2; exit 2; }

export HF_HOME="$OLMO_TWIN_ASSETS/hf_cache_posttrain"
export WANDB_MODE=disabled
pythonpath="$OI"
[[ "${CPU_ONLY:-0}" == 1 ]] && pythonpath="$pythonpath:$SCRIPT_DIR/cpu_shims"
export PYTHONPATH="$pythonpath${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OLMO_TWIN_POSTTRAIN/dataset_cache"
cd "$OI"

case "$WHAT" in
    sft)
        ARM="${ARM:-baseline}"
        case "$ARM" in
            baseline) MODEL_DIR="$OLMO_TWIN_ASSETS/hf_models/OLMo-2-0425-1B-seed42-hf" ;;
            injected) MODEL_DIR="$OLMO_TWIN_ASSETS/hf_models/olmo2-1b-injection-v2-final-hf" ;;
            *) echo "ERROR: ARM must be baseline|injected" >&2; exit 2 ;;
        esac
        OUT="$OLMO_TWIN_POSTTRAIN/runs/sft_${ARM}"
        source "$SCRIPT_DIR/sft_args.sh"
        # Tracking flags trigger wandb.login() even in cache-only mode, and they
        # do not enter the dataset config hash: strip them for the cache run.
        filtered=(); skip=0
        for a in "${sft_args[@]}"; do
            if (( skip )); then skip=0; continue; fi
            case "$a" in
                --with_tracking) ;;
                --report_to) skip=1 ;;
                *) filtered+=("$a") ;;
            esac
        done
        sft_args=("${filtered[@]}")
        "$PYTHON" "${sft_args[@]}" --cache_dataset_only
        ;;
    dpo)
        ARM="${ARM:-baseline}"
        MODEL_DIR="$OLMO_TWIN_POSTTRAIN/runs/sft_${ARM}/olmo2_1b_sft_${ARM}__1__0"
        [[ -f "$MODEL_DIR/tokenizer_config.json" ]] \
            || { echo "ERROR: no SFT model for arm $ARM at $MODEL_DIR (run run_sft.sh first)" >&2; exit 2; }
        OUT="$OLMO_TWIN_POSTTRAIN/runs/dpo_${ARM}"
        source "$SCRIPT_DIR/dpo_args.sh"
        "$PYTHON" "${dpo_args[@]}" --cache_dataset_only
        ;;
    rlvr)
        source "$SCRIPT_DIR/versions_rl.env"
        export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
        JOBTMP="$(mktemp -d "${TMPDIR:-/tmp}/olmo_twin_rlcache.XXXXXX")"
        trap 'rm -rf "$JOBTMP"' EXIT
        export XDG_CACHE_HOME="$JOBTMP/xdg"
        ARM=cache
        MODEL_DIR="$OLMO_TWIN_POSTTRAIN/runs/dpo_baseline/olmo2_1b_dpo_baseline__111__0"
        OUT="$JOBTMP/out"
        for STAGE in 1 2; do
            source "$SCRIPT_DIR/rl_args.sh"
            echo "=== caching stage $STAGE ($RL_DATA)"
            "$PYTHON" "${rl_args[@]}" --cache_dataset_only
        done
        ;;
    *) echo "usage: cache_datasets.sh sft|dpo|rlvr" >&2; exit 2 ;;
esac
echo "dataset cache ready:"; du -sh "$OLMO_TWIN_POSTTRAIN/dataset_cache"/* 2>/dev/null | tail -5 || true
