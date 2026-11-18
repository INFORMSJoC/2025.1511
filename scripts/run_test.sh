#!/usr/bin/env bash
# Evaluate HCEMOR on one fold (closed-set + open-set).
#
# The evaluation tensors of dataset2 are not shipped with the repository (see
# README, section "Data availability"); point HCEMOR_EMB_ROOT at them and set
# HCEMOR_DATASET=dataset2. HCEMOR_FOLD selects the fold id (default 4).
#
# Usage: HCEMOR_CKPT=<path.pt> scripts/run_test.sh [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
export HCEMOR_DATA_ROOT="${HCEMOR_DATA_ROOT:-$PWD/data}"
export HCEMOR_EMB_ROOT="${HCEMOR_EMB_ROOT:-$HCEMOR_DATA_ROOT/embeddings}"
export HCEMOR_CKPT="${HCEMOR_CKPT:-$HCEMOR_DATA_ROOT/model/fold5/train_V6_410.pt}"
test -f "$HCEMOR_CKPT" || { echo "checkpoint not found: $HCEMOR_CKPT"; exit 1; }
python src/hcemor/test6_4.py "$@"
