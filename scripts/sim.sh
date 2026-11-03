#!/bin/bash
export PATH="$HOME/oss-cad-suite/bin:$PATH"
cd "$(dirname "$0")/.."
${PYTHON:-python3} tb/run.py "$1" 2>&1 \
  | grep -E "Error|error|%Warning|PASS|FAIL|assert|TESTS=|INFO.*(rows|link|sampler|sampled|candidate)" | head -40
