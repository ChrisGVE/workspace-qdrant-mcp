#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Try dune first, fall back to direct ocamlopt compilation
if command -v dune &>/dev/null; then
  dune exec bin/main.exe 2>/dev/null
else
  mkdir -p _build
  # Compile modules in dependency order
  ocamlopt -I _build -c -o _build/models.cmx lib/models.ml
  ocamlopt -I _build -c -o _build/storage.cmx lib/storage.ml
  ocamlopt -I _build -c -o _build/utils.cmx lib/utils.ml
  ocamlopt -I _build -c -o _build/main.cmx bin/main.ml
  ocamlopt -I _build -o _build/main \
    _build/models.cmx _build/storage.cmx _build/utils.cmx _build/main.cmx
  ./_build/main
fi
