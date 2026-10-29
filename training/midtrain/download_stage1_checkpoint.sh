#!/usr/bin/env bash
set -euo pipefail

# Download the original OLMo-trainer checkpoint used by the published 1B
# stage-2 config. The Hugging Face checkpoint contains model weights only; the
# official continuation also restores AdamW state from optim.pt.
#
# Usage: bash training/midtrain/download_stage1_checkpoint.sh
#   ROOT     target directory (default: $OLMO_TWIN_DATA/assets/checkpoints/
#            official_olmo_stage1_4001B, see olmo_twin/paths.py)
#   BASE_URL source (default: AI2's published stage-1 endpoint, step 1907359)
# About 17.8 GB; restartable. Every file is checked against its pinned size
# and SHA-256.

# No arguments (configured through the environment variables above);
# -h/--help prints this header instead of running.
case "${1:-}" in
    "") ;;
    -h|--help) awk 'NR > 1 && /^#/ { f = 1; sub(/^# ?/, ""); print; next } f { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "usage: bash training/midtrain/download_stage1_checkpoint.sh (no arguments; see --help)" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
eval "$(PYTHONPATH="$REPO" python3 -m olmo_twin.paths --shell)"
: "${OLMO_TWIN_ASSETS:?python3 -m olmo_twin.paths --shell failed}"
ROOT="${ROOT:-$OLMO_TWIN_ASSETS/checkpoints/official_olmo_stage1_4001B}"
BASE_URL="${BASE_URL:-https://olmo-checkpoints.org/ai2-llm/peteish1/step1907359-unsharded}"
LOG="${LOG:-$ROOT/download.log}"

mkdir -p "$ROOT"
exec > >(tee -a "$LOG") 2>&1

download() {
  local name="$1"
  local expected_bytes="$2"
  local expected_sha256="$3"
  local target="$ROOT/$name"
  local partial="$target.part"

  if [[ -f "$target" ]] && [[ "$(stat -c '%s' "$target")" == "$expected_bytes" ]]; then
    if printf '%s  %s\n' "$expected_sha256" "$target" | sha256sum --check --strict >/dev/null; then
      printf 'already verified: %s (%s bytes)\n' "$name" "$expected_bytes"
      return
    fi
    printf 'existing file failed its pinned SHA-256; downloading again: %s\n' "$name"
    rm -f -- "$target"
  fi

  if [[ -f "$partial" ]] && [[ "$(stat -c '%s' "$partial")" == "$expected_bytes" ]]; then
    if printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum --check --strict >/dev/null; then
      mv -- "$partial" "$target"
      printf 'completed verified partial: %s (%s bytes)\n' "$name" "$expected_bytes"
      return
    fi
    printf 'discarding full-size partial with the wrong SHA-256: %s\n' "$name"
    rm -f -- "$partial"
  elif [[ -f "$partial" ]] && (( $(stat -c '%s' "$partial") > expected_bytes )); then
    printf 'discarding oversized partial: %s\n' "$name"
    rm -f -- "$partial"
  fi

  printf 'downloading: %s\n' "$name"
  curl --fail --location --retry 12 --retry-all-errors \
    --connect-timeout 30 --continue-at - \
    "$BASE_URL/$name" --output "$partial"

  local actual_bytes
  actual_bytes="$(stat -c '%s' "$partial")"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    printf 'ERROR: %s has %s bytes; expected %s\n' \
      "$name" "$actual_bytes" "$expected_bytes" >&2
    exit 1
  fi
  if ! printf '%s  %s\n' "$expected_sha256" "$partial" \
      | sha256sum --check --strict >/dev/null; then
    printf 'ERROR: %s failed its pinned SHA-256.\n' "$name" >&2
    exit 1
  fi
  mv -- "$partial" "$target"
}

download config.yaml 163724 \
  ec6f24df31443597a5a281b09b03d46aba7f21f942d572bdff8d64fff18d21a3
download train.pt 15500 \
  7f5b2e840fa256f589dab6ae206e667e1ef4891d9c26ba1d4db364d6bdf45939
download model.pt 5939703038 \
  7b5a229bde2dc24a48fc269411e27c2bf3020446483a1e99a611d6985633b80b
download optim.pt 11879442406 \
  c5baee99495d0af7596cb58b69fd2ef108282bcf6348f02b8958ef9c6e41772a

cat > "$ROOT/SHA256SUMS" <<'EOF'
ec6f24df31443597a5a281b09b03d46aba7f21f942d572bdff8d64fff18d21a3  config.yaml
7f5b2e840fa256f589dab6ae206e667e1ef4891d9c26ba1d4db364d6bdf45939  train.pt
7b5a229bde2dc24a48fc269411e27c2bf3020446483a1e99a611d6985633b80b  model.pt
c5baee99495d0af7596cb58b69fd2ef108282bcf6348f02b8958ef9c6e41772a  optim.pt
EOF
(cd "$ROOT" && sha256sum --check --strict SHA256SUMS)
cat > "$ROOT/checkpoint_identity.json" <<EOF
{
  "checkpoint_url": "$BASE_URL/",
  "config_sha256": "ec6f24df31443597a5a281b09b03d46aba7f21f942d572bdff8d64fff18d21a3",
  "model_bytes": 5939703038,
  "model_sha256": "7b5a229bde2dc24a48fc269411e27c2bf3020446483a1e99a611d6985633b80b",
  "optimizer_bytes": 11879442406,
  "optimizer_sha256": "c5baee99495d0af7596cb58b69fd2ef108282bcf6348f02b8958ef9c6e41772a",
  "role": "official OLMo 2 1B stage-1 endpoint loaded by stage 2",
  "source_step": 1907359,
  "source_tokens": 4000021741568,
  "train_state_sha256": "7f5b2e840fa256f589dab6ae206e667e1ef4891d9c26ba1d4db364d6bdf45939"
}
EOF
touch "$ROOT/.download_complete"
# OLMo's trainer restores an unsharded checkpoint (model, optimizer and trainer
# state) only when the load path ends in "-unsharded" (Trainer.restore_checkpoint
# in olmo/train.py); otherwise it tries a sharded checkpoint. train.sh therefore
# loads through this alias, which prepare_stage2_assets.py verify
# (--checkpoint-alias) also checks.
ALIAS="${ROOT%/}-unsharded"
if [[ -e "$ALIAS" && ! -L "$ALIAS" ]]; then
  printf 'ERROR: %s exists and is not a symbolic link\n' "$ALIAS" >&2
  exit 1
fi
ln -sfn "$(basename -- "${ROOT%/}")" "$ALIAS"
printf 'official stage-1 checkpoint complete: %s (load alias: %s)\n' "$ROOT" "$ALIAS"
