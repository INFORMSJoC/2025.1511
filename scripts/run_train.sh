#!/usr/bin/env bash
# Train HCEMOR (DCANet + MCLUFI) on all five folds and ten seeds.
#
# Note: --batch_size must equal the number of rows in the background-semantics
# tensor data/support/top16.npy (16). The background tensor is batch-coupled in
# the forward pass, so a different batch size raises a shape error.
#
# Usage: scripts/run_train.sh [extra args passed to train_DCD_v6_4.py]
set -euo pipefail
cd "$(dirname "$0")/.."
export HCEMOR_DATA_ROOT="${HCEMOR_DATA_ROOT:-$PWD/data}"
python src/hcemor/train_DCD_v6_4.py --batch_size 16 "$@"
