#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_test.sh"
bash "$SCRIPT_DIR/test_intermediate_ops_bitwidth.sh"
bash "$SCRIPT_DIR/test_sigmoid_bitwidth_configs.sh"
bash "$SCRIPT_DIR/test_weight_activation_bitwidth.sh"
