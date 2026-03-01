#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mix compile --no-deps-check >/dev/null 2>/dev/null
mix run -e "Bookshelf.main()" --no-compile --no-deps-check
