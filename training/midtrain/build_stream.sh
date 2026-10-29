#!/usr/bin/env bash
# Materialize the injected training stream (64 shards, about 200 GB) from AI2's
# stage-2 token files and the frozen design bundle, write the training
# configuration and verify the result. CPU and I/O only; the build is
# resumable per shard. Never run two of these at once on the same data root.
#
# Usage:
#   OLMO_REPO=/path/to/OLMo bash training/midtrain/build_stream.sh
#
# Environment:
#   OLMO_REPO  (required) OLMo checkout at OLMO_COMMIT (versions.env); the
#              published configuration is read from it
#   PYTHON     Python interpreter with numpy and pyyaml (default python3)
#
# Inputs (data root, see olmo_twin/paths.py):
#   bundle/   the frozen design, including stream_plan/placement.jsonl.gz and
#             the token files listed in SHA256SUMS
#   assets/configs/stage2_tokenized_manifest.json and the 576 token files it
#             lists (prepare_stage2_assets.py plan/download)
# Outputs:
#   stream/data/train-*-of-00064.npy, stream/stream_manifest.json
#   stream/config/olmo2-1b-injection-v2.yaml (AI2's file with only data.paths,
#             run identity and checkpoint keys changed; asserted)
#   stream/config/config_deviations.json, stream/bundle_verification.json
set -euo pipefail
# No arguments (configured through the environment variables above);
# -h/--help prints this header instead of running.
case "${1:-}" in
    "") ;;
    -h|--help) awk 'NR > 1 && /^#/ { f = 1; sub(/^# ?/, ""); print; next } f { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "usage: bash training/midtrain/build_stream.sh (no arguments; see --help)" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"
: "${OLMO_TWIN_DATA:?python3 -m olmo_twin.paths --shell failed}"
source "$SCRIPT_DIR/versions.env"
OLMO_REPO="${OLMO_REPO:?set OLMO_REPO to an OLMo checkout at commit $OLMO_COMMIT}"
PYTHON="${PYTHON:-python3}"
BUNDLE="$OLMO_TWIN_BUNDLE"
TRAIN="$OLMO_TWIN_STREAM"
OFFICIAL_CONFIG="$OLMO_REPO/$OFFICIAL_CONFIG_RELATIVE"
OFFICIAL_MANIFEST="$OLMO_TWIN_ASSETS/configs/stage2_tokenized_manifest.json"
CONFIG="$TRAIN/config/olmo2-1b-injection-v2.yaml"
TOOL="$SCRIPT_DIR/build_injection_stream_v2.py"
run(){ "$PYTHON" "$TOOL" "$@"; }

echo "=== $(date -u) bundle checksums"
(cd "$BUNDLE" && sha256sum --check --quiet SHA256SUMS)
printf '%s  %s\n' "$PLACEMENT_SHA256" "$BUNDLE/stream_plan/placement.jsonl.gz" | sha256sum --check --strict
[[ "$(git -C "$OLMO_REPO" rev-parse HEAD)" == "$OLMO_COMMIT" ]] \
    || { echo "ERROR: $OLMO_REPO is not at commit $OLMO_COMMIT" >&2; exit 2; }
printf '%s  %s\n' "$OFFICIAL_CONFIG_SHA256" "$OFFICIAL_CONFIG" | sha256sum --check --strict
mkdir -p "$TRAIN/config"
echo "=== $(date -u) build"
run build --bundle "$BUNDLE" --official-data-manifest "$OFFICIAL_MANIFEST" \
    --output-dir "$TRAIN" --output-shards 64
echo "=== $(date -u) make-config"
run make-config --official-config "$OFFICIAL_CONFIG" --training-dir "$TRAIN" \
    --run-name OLMo2-1B-stage2-injection-v2 --wandb-group injection-v2-20260925 \
    --output "$CONFIG" --deviations "$TRAIN/config/config_deviations.json"
echo "=== $(date -u) verify (full hashes, 20k background samples, every injected chunk)"
run verify --bundle "$BUNDLE" --training-dir "$TRAIN" \
    --official-data-manifest "$OFFICIAL_MANIFEST" \
    --official-config "$OFFICIAL_CONFIG" --custom-config "$CONFIG" \
    --output "$TRAIN/bundle_verification.json"
du -sh "$TRAIN/data"
echo "=== $(date -u) DONE"
