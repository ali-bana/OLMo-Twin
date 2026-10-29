#!/usr/bin/env bash
# Convert an OLMo-native unsharded checkpoint to Hugging Face format with
# OLMo's own converter, then give it the released tokenizer files of
# allenai/OLMo-2-0425-1B, so that both arms and AI2's released models share one
# tokenizer byte for byte (chat template included).
#
# Usage:
#   OLMO_REPO=/path/to/OLMo [CPU_ONLY=1] bash training/posttrain/convert_to_hf.sh <olmo-checkpoint-dir> <hf-output-dir>
#
# The two start points of post-training (paths relative to the data root):
#   control: assets/checkpoints/official_stage2_ingredient3_seed42/step23852-unsharded
#            -> assets/hf_models/OLMo-2-0425-1B-seed42-hf
#   treated: run/checkpoints/step23852-unsharded
#            -> assets/hf_models/olmo2-1b-injection-v2-final-hf
#
# Environment:
#   OLMO_REPO  (required) OLMo checkout at the commit of training/midtrain/versions.env;
#              provides scripts/convert_olmo2_to_hf.py and the dolma2 tokenizer
#   CPU_ONLY   1: put cpu_shims/ first on PYTHONPATH (for hosts without a GPU
#              driver, where deepspeed's triton import would crash)
#   PYTHON     interpreter with torch and transformers (default python3; we
#              used the post-training image, env/posttrain/)
# Needs the tokenizer files of allenai/OLMo-2-0425-1B in
# $OLMO_TWIN_DATA/assets/hf_models/OLMo-2-0425-1B (see training/README.md).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"
: "${OLMO_TWIN_ASSETS:?python3 -m olmo_twin.paths --shell failed}"
SRC="${1:?usage: convert_to_hf.sh <olmo-ckpt-dir> <hf-out-dir>}"
DST="${2:?usage: convert_to_hf.sh <olmo-ckpt-dir> <hf-out-dir>}"
OLMO_REPO="${OLMO_REPO:?set OLMO_REPO to an OLMo checkout}"
PYTHON="${PYTHON:-python3}"
TOKENIZER_DIR="$OLMO_TWIN_ASSETS/hf_models/OLMo-2-0425-1B"
[[ -f "$SRC/model.pt" && -f "$SRC/config.yaml" ]] || { echo "ERROR: $SRC lacks model.pt/config.yaml" >&2; exit 2; }
[[ -e "$DST" ]] && { echo "ERROR: $DST already exists" >&2; exit 2; }
if [[ "${CPU_ONLY:-0}" == 1 ]]; then
    export PYTHONPATH="$SCRIPT_DIR/cpu_shims${PYTHONPATH:+:$PYTHONPATH}"
fi
"$PYTHON" "$OLMO_REPO/scripts/convert_olmo2_to_hf.py" \
    --input_dir "$SRC" --output_dir "$DST" \
    --tokenizer_json_path "$OLMO_REPO/olmo_data/tokenizers/allenai_dolma2.json" \
    --no_tmp_cleanup
# The converter's own removal of its temporary directory can fail on a network
# file system ("Directory not empty") after the weights are written; it is
# skipped (--no_tmp_cleanup) and retried here.
for _ in 1 2 3; do rm -rf "$DST/tmp" 2>/dev/null && break; sleep 5; done
# The converter writes its own tokenizer; overwrite it with the released files.
for f in tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json vocab.json merges.txt; do
    [[ -f "$TOKENIZER_DIR/$f" ]] && cp "$TOKENIZER_DIR/$f" "$DST/$f"
done
echo "converted: $DST"; ls -la "$DST"
