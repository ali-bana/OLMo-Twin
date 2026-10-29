#!/usr/bin/env bash
# Launch the injected mid-training run: AI2's published OLMo 2 1B stage-2
# configuration, with data.paths pointing at the injected stream, run by the
# unmodified OLMo trainer at OLMO_COMMIT (versions.env) from AI2's stage-1
# endpoint. Run this script once on every node (plain torchrun, no scheduler).
#
# Usage, on each node:
#   OLMO_REPO=/path/to/OLMo NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<host of rank 0> \
#   MASTER_PORT=29500 bash training/midtrain/train.sh
#   bash training/midtrain/train.sh --preflight-only     # checks only, then exit
#
# Environment:
#   OLMO_REPO      (required) OLMo checkout at OLMO_COMMIT; torchrun runs
#                  scripts/train.py from it. If several nodes write checkpoints
#                  to one shared network file system, consider applying
#                  olmo-shared-fs-checkpoint.patch (see training/README.md).
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT  (required) torchrun rendezvous;
#                  set them yourself or from your scheduler
#   GPUS_PER_NODE  processes per node (default 8)
#   RUN_NAME       default: the run name saved with our run
#   ENABLE_WANDB   1 (default): log to Weights & Biases in offline mode
#                  (WANDB_MODE=offline unless set), as our run did; OLMo writes
#                  the log under the save folder (checks/check_lr_schedule.py
#                  reads it). 0: pass --wandb=null.
#   OLMO_SHARED_FS 1 (default): only global rank 0 creates and renames
#                  checkpoint directories (upstream OLMo, olmo/torch_util.py);
#                  needed when all nodes save to one file system, as ours did.
#                  OLMo enables this for any non-empty value; 0 unsets it.
#   SKIP_PREFLIGHT 1: skip the checks below
#   PYTHON         interpreter for the preflight verification (default python3)
#
# Preflight (on NODE_RANK 0 only, before torchrun starts there; the other
# nodes wait at the rendezvous): the OLMo checkout is at OLMO_COMMIT, AI2's
# configuration file and the placement file have their pinned SHA-256, the
# four stage-1 checkpoint files match their SHA256SUMS, and
# build_injection_stream_v2.py verify re-checks the stream (sizes, all
# injected chunks, the same 5,000 background positions each time, the
# configuration diff). Our run repeated these checks before each launch.
#
# Our run used 2 nodes x 8 GPUs (16 processes, 32 sequences per GPU per
# update as eight micro-batches of four) instead of AI2's 256 GPUs; the global
# batch (512 x 4096 tokens) and the data order do not depend on the number of
# GPUs. It ran as five launches; the last four resumed from the run's own
# checkpoints (--try_load_latest_save=true restores model, optimizer,
# data-loader and random-number-generator state). The first two launches
# still had ephemeral checkpoints; --save_interval_ephemeral=null was added
# from the third launch on and is set here from the start (a command-line
# override, so the configuration file and its hash are unchanged).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"
: "${OLMO_TWIN_DATA:?python3 -m olmo_twin.paths --shell failed}"
source "$SCRIPT_DIR/versions.env"

PREFLIGHT_ONLY=0
case "${1:-}" in
    "") ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *) echo "usage: $0 [--preflight-only]" >&2; exit 2 ;;
esac

OLMO_REPO="${OLMO_REPO:?set OLMO_REPO to an OLMo checkout at commit $OLMO_COMMIT}"
OLMO_REPO="$(cd -- "$OLMO_REPO" && pwd)"
PYTHON="${PYTHON:-python3}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CONFIG="$OLMO_TWIN_STREAM/config/olmo2-1b-injection-v2.yaml"
CHECKPOINT_ROOT="$OLMO_TWIN_ASSETS/checkpoints/official_olmo_stage1_4001B"
# OLMo restores an unsharded checkpoint only from a path ending in
# "-unsharded"; download_stage1_checkpoint.sh creates this alias.
LOAD_PATH="${CHECKPOINT_ROOT}-unsharded"
SAVE_FOLDER="$OLMO_TWIN_RUN/checkpoints"
# The name our run saved in checkpoints/config.yaml; a different name shows up
# as a difference in training/checks/compare_configs.py.
RUN_NAME="${RUN_NAME:-olmo2-1b-injection-v2-olmo2-1b-injection-v2}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"

preflight() {
    for path in "$OLMO_REPO/$OFFICIAL_CONFIG_RELATIVE" "$CONFIG" "$LOAD_PATH" "$CHECKPOINT_ROOT/SHA256SUMS" \
        "$OLMO_TWIN_BUNDLE/stream_plan/placement.jsonl.gz" "$OLMO_TWIN_STREAM/stream_manifest.json"; do
        [[ -e "$path" ]] || { printf 'ERROR: required path is missing: %s\n' "$path" >&2; exit 2; }
    done
    [[ "$(git -C "$OLMO_REPO" rev-parse HEAD)" == "$OLMO_COMMIT" ]] \
        || { echo "ERROR: $OLMO_REPO is not at commit $OLMO_COMMIT" >&2; exit 2; }
    printf '%s  %s\n' "$OFFICIAL_CONFIG_SHA256" "$OLMO_REPO/$OFFICIAL_CONFIG_RELATIVE" | sha256sum --check --strict
    printf '%s  %s\n' "$PLACEMENT_SHA256" "$OLMO_TWIN_BUNDLE/stream_plan/placement.jsonl.gz" | sha256sum --check --strict
    (cd "$CHECKPOINT_ROOT" && sha256sum --check --strict SHA256SUMS)
    mkdir -p "$OLMO_TWIN_RUN"
    "$PYTHON" "$SCRIPT_DIR/build_injection_stream_v2.py" verify \
        --bundle "$OLMO_TWIN_BUNDLE" \
        --training-dir "$OLMO_TWIN_STREAM" \
        --official-data-manifest "$OLMO_TWIN_ASSETS/configs/stage2_tokenized_manifest.json" \
        --official-config "$OLMO_REPO/$OFFICIAL_CONFIG_RELATIVE" \
        --custom-config "$CONFIG" \
        --background-samples 5000 --skip-hashes \
        --output "$OLMO_TWIN_RUN/bundle_preflight-$(date +%s).json"
}

if (( PREFLIGHT_ONLY )); then
    preflight
    echo "preflight ok"
    exit 0
fi

NNODES="${NNODES:?set NNODES (number of nodes)}"
NODE_RANK="${NODE_RANK:?set NODE_RANK (0 .. NNODES-1)}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR (host of node rank 0)}"
MASTER_PORT="${MASTER_PORT:?set MASTER_PORT (a free TCP port on that host)}"

if [[ "${SKIP_PREFLIGHT:-0}" != 1 && "$NODE_RANK" == 0 ]]; then
    preflight
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
OLMO_SHARED_FS="${OLMO_SHARED_FS:-1}"
if [[ "$OLMO_SHARED_FS" == 0 ]]; then unset OLMO_SHARED_FS; else export OLMO_SHARED_FS; fi

train_args=(
    "$CONFIG"
    "--load_path=$LOAD_PATH"
    --try_load_latest_save=true
    --save_overwrite=true
    "--evaluators=[]"
)
if [[ "$ENABLE_WANDB" == 1 ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
else
    train_args+=(--wandb=null)
fi
train_args+=(
    "--run_name=$RUN_NAME"
    "--save_folder=$SAVE_FOLDER"
    --save_interval_ephemeral=null
)
mkdir -p "$SAVE_FOLDER"
cd "$OLMO_REPO"
exec torchrun \
    "--nnodes=$NNODES" \
    "--nproc-per-node=$GPUS_PER_NODE" \
    "--node-rank=$NODE_RANK" \
    "--master-addr=$MASTER_ADDR" \
    "--master-port=$MASTER_PORT" \
    scripts/train.py "${train_args[@]}"
