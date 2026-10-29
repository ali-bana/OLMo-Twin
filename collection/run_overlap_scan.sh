#!/usr/bin/env bash
# Exhaustive membership audit of the post-cutoff injection candidates against
# EVERY component of both OLMo 2 training stages.
#
# Method: 8-word shingles, every shingle kept (no winnowing), bitmap-prefiltered
# exact lookup. This catches any shared run of >=8 words. The winnowed scan used
# earlier only guarantees runs of >=44 words, which is why it is not used here --
# the whole point is to be unable to miss anything.
#
# Components are ordered smallest-first so a configuration error surfaces in
# minutes rather than after a day of dclm. dolmino's pes2o is byte-identical to
# stage-1's, but it is scanned anyway: 20 minutes is not worth resting the
# result on an unverified assumption.
#
# Usage:
#   CORPUS_ROOT=/path/to/olmo2-corpora bash collection/run_overlap_scan.sh
#
# Environment:
#   CORPUS_ROOT  (required) directory holding olmo-mix-1124/ and
#                dolmino-mix-1124/, each with data/<component>/ as downloaded
#                from the Hugging Face Hub
#   WORK         working directory (default $OLMO_TWIN_COLLECTION/audit_v2); it
#                must hold combined.csv.gz (build_combined_candidates.py) and
#                fp/ (olmo2_nonmember_audit.py fingerprints); one
#                scan_<stage>__<component>/ directory is written per target
#   PYTHON       Python interpreter (default python3)
#   WORKERS      scan worker processes (default 11)
#   TARGETS, EXCLUDE_TARGETS  see below
set -uo pipefail

# No arguments (configured through the environment variables above);
# -h/--help prints this header instead of running.
case "${1:-}" in
    "") ;;
    -h|--help) awk 'NR > 1 && /^#/ { f = 1; sub(/^# ?/, ""); print; next } f { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "usage: bash collection/run_overlap_scan.sh (no arguments; see --help)" >&2; exit 2 ;;
esac
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
eval "$(PYTHONPATH=$REPO python3 -m olmo_twin.paths --shell)"
WORK=${WORK:-$OLMO_TWIN_COLLECTION/audit_v2}
PY=${PYTHON:-python3}
: "${CORPUS_ROOT:?set CORPUS_ROOT to the directory holding olmo-mix-1124/ and dolmino-mix-1124/}"
CORP=$CORPUS_ROOT
WORKERS=${WORKERS:-11}

mkdir -p "$WORK"
say(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$WORK/run.log"; }

# stage:component. Dolmino (the mid-training run being modified) first and
# smallest first within each stage; the two dclm crawls last because they are
# 95% of the bytes and, being Common Crawl through 2022-12, can hold
# post-cutoff documents only via text reuse.
#
#   TARGETS="a:b c:d"   overrides the list
#   EXCLUDE_TARGETS="olmo-mix-1124:dclm"   drops entries (e.g. to run the 6.6 TiB
#                     stage-1 crawl separately, alongside training)
DEFAULT_TARGETS=(
  "dolmino-mix-1124:wiki"
  "dolmino-mix-1124:stackexchange"
  "dolmino-mix-1124:math"
  "dolmino-mix-1124:flan"
  "olmo-mix-1124:wiki"
  "olmo-mix-1124:algebraic-stack"
  "olmo-mix-1124:open-web-math"
  "olmo-mix-1124:arxiv"
  "olmo-mix-1124:starcoder"
  "dolmino-mix-1124:pes2o"
  "olmo-mix-1124:pes2o"
  "dolmino-mix-1124:dclm"
  "olmo-mix-1124:dclm"
)
if [ -n "${TARGETS:-}" ]; then
  read -r -a TARGETS <<< "$TARGETS"
else
  TARGETS=()
  for t in "${DEFAULT_TARGETS[@]}"; do
    skip=0
    for x in ${EXCLUDE_TARGETS:-}; do [ "$t" = "$x" ] && skip=1; done
    [ $skip = 0 ] && TARGETS+=("$t")
  done
fi

say "=== START comprehensive audit: ${#TARGETS[@]} stage/component targets ==="
failed=()
for t in "${TARGETS[@]}"; do
  stage="${t%%:*}"; comp="${t##*:}"
  out="$WORK/scan_${stage}__${comp}"
  # cmd_scan writes scan_summary.json only on a clean pass (a run with
  # suspect shards writes scan_summary.FAILED.json instead), so its presence
  # is a reliable completion marker and a restart resumes instead of redoing
  # a day.
  if [ -f "$out/scan_summary.json" ]; then
    say "  skip $stage/$comp (already complete)"
    continue
  fi
  say "  --> $stage / $comp"
  if $PY "$REPO/collection/olmo2_nonmember_audit.py" scan \
      --candidates "$WORK/combined.csv.gz" \
      --fingerprint-dir "$WORK/fp" \
      --corpus "$CORP/$stage" --components "$comp" \
      --out-dir "$out" --workers "$WORKERS" \
      >>"$WORK/scan.log" 2>&1; then
    say "      done $stage/$comp"
  else
    say "      FAILED $stage/$comp  (see $WORK/scan.log)"
    failed+=("$stage/$comp")
  fi
done

# Never announce success after a failure. The previous version printed
# "ALL COMPONENTS COMPLETE" unconditionally, so a log could read FAILED on one
# line and COMPLETE on the next.
if [ "${#failed[@]}" -gt 0 ]; then
  say "=== FINISHED WITH ${#failed[@]} FAILED TARGET(S): ${failed[*]} ==="
  exit 1
fi
say "=== ALL ${#TARGETS[@]} COMPONENTS COMPLETE ==="
