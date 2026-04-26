#!/usr/bin/env bash
# End-to-end smoke test gate for Phase 5.
#
# Runs:
#   1. pytest (the four test files: data, loss, forward, inference)
#   2. The smoke training + evaluation script
# Both must pass on CPU in well under 10 minutes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo
echo "==> [1/2] pytest"
"$PYTHON" -m pytest tests/ -v
echo

echo "==> [2/2] end-to-end training + inference smoke"
"$PYTHON" scripts/smoke_test.py
