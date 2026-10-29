#!/usr/bin/env bash
# One RLVR stage of one arm: open-instruct's GRPO script
# (grpo_vllm_thread_ray_gtrl.py, unpatched) on a Ray cluster of two nodes with
# 8 GPUs each, flags from rl_args.sh (AI2's original 1B commands). Ray places
# AI2's layout itself: learners on 4 + 8 GPUs and one vLLM engine with tensor
# parallelism 4 on the remaining 4 GPUs. 2,604 steps per stage.
#
# Usage:
#   first node:   OPEN_INSTRUCT_DIR=... RAY_PORT=6379 bash training/posttrain/run_rlvr.sh head STAGE ARM
#   second node:  OPEN_INSTRUCT_DIR=... RAY_HEAD_IP=<address of the first node> RAY_PORT=6379 \
#                   bash training/posttrain/run_rlvr.sh worker
#   STAGE  1 (RLVR-GSM-MATH-IF-Mixed-Constraints, from the arm's DPO model) or
#          2 (RLVR-MATH, from the arm's final RLVR1 model)
#   ARM    baseline | injected
#
# The head starts Ray, waits until the worker has joined (16 GPUs), runs the
# GRPO driver and stops Ray; the worker's `ray start --block` then returns.
# Output: $OLMO_TWIN_DATA/posttrain/runs/rlvr<STAGE>_<ARM>/olmo2_1b_rlvr<STAGE>_<ARM>__1__<t>
# (final model) and ..._checkpoints/step_<N> (every 200 steps). <t> is the
# launch time: the GRPO script is not patched, so its run directories carry a
# timestamp; stage 2 starts from the newest saved RLVR1 run directory.
#
# Environment:
#   OPEN_INSTRUCT_DIR  (required, both nodes) open-instruct checkout at
#                      OPEN_INSTRUCT_COMMIT (versions.env), with the vLLM/Ray
#                      environment of env/posttrain-rlvr/
#   RAY_PORT           (required, both nodes) port of the Ray head
#   RAY_HEAD_IP        (required on the worker) address of the head node
#   RAY_NODE_IP        optional --node-ip-address for this node's Ray daemon
#   PYTHON             interpreter (default python3)
#   TMPDIR             if set, the compiler and vLLM caches (XDG, Triton, vLLM,
#                      TorchInductor) go to a fresh directory under it
# On GPU nodes without internet access run cache_datasets.sh rlvr first and
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 on both
# nodes.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"
: "${OLMO_TWIN_DATA:?python3 -m olmo_twin.paths --shell failed}"
source "$SCRIPT_DIR/versions.env"; source "$SCRIPT_DIR/versions_rl.env"
ROLE="${1:?usage: run_rlvr.sh head STAGE ARM | run_rlvr.sh worker}"
OI="${OPEN_INSTRUCT_DIR:?set OPEN_INSTRUCT_DIR to the open-instruct checkout}"
OI="$(cd -- "$OI" && pwd)"
RAY_PORT="${RAY_PORT:?set RAY_PORT (port of the Ray head)}"
PYTHON="${PYTHON:-python3}"
NCPU="$(nproc)"
node_ip=()
[[ -n "${RAY_NODE_IP:-}" ]] && node_ip=(--node-ip-address="$RAY_NODE_IP")

export HF_HOME="$OLMO_TWIN_ASSETS/hf_cache_posttrain"
export WANDB_MODE=disabled VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 RAY_USAGE_STATS_ENABLED=0
export TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
if [[ -n "${TMPDIR:-}" ]]; then
    JOBTMP="$(mktemp -d "$TMPDIR/olmo_twin_rlvr.XXXXXX")"
    export XDG_CACHE_HOME="$JOBTMP/xdg" TRITON_CACHE_DIR="$JOBTMP/triton" \
           VLLM_CACHE_ROOT="$JOBTMP/vllm" TORCHINDUCTOR_CACHE_DIR="$JOBTMP/inductor"
fi
export PYTHONPATH="$OI${PYTHONPATH:+:$PYTHONPATH}"
cd "$OI"

if [[ "$ROLE" == worker ]]; then
    RAY_HEAD_IP="${RAY_HEAD_IP:?set RAY_HEAD_IP (address of the head node)}"
    echo "[$(hostname)] ray worker -> $RAY_HEAD_IP:$RAY_PORT"
    for _ in $(seq 1 120); do (exec 3<>"/dev/tcp/$RAY_HEAD_IP/$RAY_PORT") 2>/dev/null && break; sleep 2; done
    # Returns when the head stops Ray after the driver has finished.
    ray start --address="$RAY_HEAD_IP:$RAY_PORT" ${node_ip[@]+"${node_ip[@]}"} \
        --num-gpus=8 --num-cpus="$NCPU" --disable-usage-stats --block || true
    exit 0
fi
[[ "$ROLE" == head ]] || { echo "usage: run_rlvr.sh head STAGE ARM | run_rlvr.sh worker" >&2; exit 2; }

STAGE="${2:?usage: run_rlvr.sh head STAGE ARM}"; ARM="${3:?usage: run_rlvr.sh head STAGE ARM}"
case "$ARM" in baseline|injected) ;; *) echo "ERROR: ARM must be baseline|injected" >&2; exit 2 ;; esac
# Chain within each arm: DPO -> RLVR1 -> RLVR2, identical recipes.
case "$STAGE" in
    1) MODEL_DIR="$OLMO_TWIN_POSTTRAIN/runs/dpo_${ARM}/olmo2_1b_dpo_${ARM}__111__0" ;;
    # the final RLVR1 model: the run dir (not ..._checkpoints) that holds a
    # saved model; the newest one if a stage was ever re-run.
    2) MODEL_DIR=$(ls -d "$OLMO_TWIN_POSTTRAIN/runs/rlvr1_${ARM}"/olmo2_1b_rlvr1_${ARM}__1__* 2>/dev/null \
                   | grep -v '_checkpoints$' | while read -r d; do [[ -f "$d/config.json" ]] && echo "$d"; done | tail -1 || true) ;;
    *) echo "ERROR: STAGE must be 1|2" >&2; exit 2 ;;
esac
OUT="$OLMO_TWIN_POSTTRAIN/runs/rlvr${STAGE}_${ARM}"

# --- fail-closed preflight -------------------------------------------------
[[ "$(git -C "$OI" rev-parse HEAD)" == "$OPEN_INSTRUCT_COMMIT" ]] \
    || { echo "ERROR: open-instruct not at pinned commit $OPEN_INSTRUCT_COMMIT" >&2; exit 2; }
[[ -n "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]] \
    || { echo "ERROR: no start model for stage $STAGE arm $ARM (${MODEL_DIR:-none})" >&2; exit 2; }
[[ -f "$OLMO_TWIN_DATA/$RL_TOKENIZER_DIR/tokenizer.json" ]] \
    || { echo "ERROR: missing tokenizer $OLMO_TWIN_DATA/$RL_TOKENIZER_DIR" >&2; exit 2; }
if [[ "$STAGE" == 2 && ! -e "$OLMO_TWIN_POSTTRAIN/runs/rlvr1_${ARM}/.complete" ]]; then
    echo "ERROR: RLVR1 for $ARM not complete" >&2; exit 2
fi
[[ -e "$OUT/.complete" ]] && { echo "ERROR: $OUT already complete" >&2; exit 2; }
mkdir -p "$OUT"; rm -f "$OUT/.driver_exit"
export MODEL_DIR OUT
echo "stage=$STAGE arm=$ARM from=$MODEL_DIR out=$OUT"; date -u

echo "[$(hostname)] ray head on port $RAY_PORT"
ray start --head ${node_ip[@]+"${node_ip[@]}"} --port="$RAY_PORT" \
    --num-gpus=8 --num-cpus="$NCPU" --include-dashboard=false --disable-usage-stats --block &
HEAD_PID=$!
if [[ -n "${RAY_NODE_IP:-}" ]]; then export RAY_ADDRESS="$RAY_NODE_IP:$RAY_PORT"; else export RAY_ADDRESS=auto; fi
n=0
for _ in $(seq 1 120); do
    n=$("$PYTHON" -c "import ray; ray.init(address='$RAY_ADDRESS', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null | tail -1 || true)
    [[ "$n" == 16 ]] && break
    sleep 5
done
echo "[$(hostname)] ray cluster GPUs: ${n:-0}"
if [[ "$n" != 16 ]]; then
    echo "ERROR: worker never joined" >&2; echo 3 > "$OUT/.driver_exit"
    ray stop --force >/dev/null 2>&1 || true; kill "$HEAD_PID" 2>/dev/null || true; exit 3
fi

source "$SCRIPT_DIR/rl_args.sh"
echo "[$(hostname)] driver: stage $STAGE arm $ARM from $MODEL_DIR -> $OUT"
rc=0
"$PYTHON" "${rl_args[@]}" || rc=$?
echo "$rc" > "$OUT/.driver_exit"
echo "[$(hostname)] driver exit $rc"
ray stop --force >/dev/null 2>&1 || true
kill "$HEAD_PID" 2>/dev/null || true; wait "$HEAD_PID" 2>/dev/null || true
if [[ "$rc" == 0 ]] && ls "$OUT"/olmo2_1b_rlvr${STAGE}_${ARM}__1__*[0-9]/config.json >/dev/null 2>&1; then
    touch "$OUT/.complete"
    echo "=== RLVR$STAGE $ARM DONE $(date -u) ==="
else
    echo "=== RLVR$STAGE $ARM FAILED (driver exit $rc) $(date -u) ===" >&2
    exit 1
fi
