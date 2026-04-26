#!/usr/bin/env bash
# Smoke gate for entropy_filtered/.
#
#   1. pytest baseline/ + entropy_filtered/ (combined unit tests)
#   2. End-to-end filtered training + evaluation in entropy_filtered/scripts/smoke_test.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo
echo "==> [1a/2] pytest baseline/"
( cd baseline && "$PYTHON" -m pytest tests/ -v )
echo
echo "==> [1b/2] pytest entropy_filtered/"
( cd entropy_filtered && "$PYTHON" -m pytest tests/ -v --rootdir=. )
echo

echo "==> [2/2] entropy-filtered training + inference smoke"
"$PYTHON" entropy_filtered/scripts/smoke_test.py
