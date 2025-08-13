#!/usr/bin/env bash
# Usage: bash scripts/new_exp.sh myexp
set -euo pipefail
EXP=${1:-test}
STAMP=$(date +%Y%m%d-%H%M%S)
BASE="runs/$EXP/$STAMP"
mkdir -p "$BASE/models" "$BASE/updates" "$BASE/simulation"
echo "Created $BASE"
