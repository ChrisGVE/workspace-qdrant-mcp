#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if command -v runghc &>/dev/null; then
  runghc -isrc src/Main.hs
elif command -v stack &>/dev/null; then
  stack runghc -- -isrc src/Main.hs 2>/dev/null
else
  echo "Error: No Haskell toolchain found (need runghc or stack)" >&2
  exit 1
fi
