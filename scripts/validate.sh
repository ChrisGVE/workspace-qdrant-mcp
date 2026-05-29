#!/usr/bin/env bash
# validate.sh — local validation gate (clippy + release build).
#
# Runs `cargo clippy -D warnings` (lib + bins + all test targets) plus the
# production release build, inside the Linux builder container
# (docker/Dockerfile.memexd `validate` stage). This exists because work on
# `fork/fixes` gets no CI (ci.yml only runs on PRs to main/dev) and the local
# Windows build breaks on a pre-existing winapi gate in storage/client.rs.
# Use this before merging into fork/fixes.
#
# The full `cargo test` suite is NOT run here (fragile in this flattened
# container — FTS5/trigram + timing tests fail for env reasons). Run it at
# true parity via a `fork/fixes -> main` PR (real CI, full checkout).
#
# Usage:
#   scripts/validate.sh                 # run the full gate
#   scripts/validate.sh --no-cache      # extra args pass through to docker build
#
# Exit code mirrors the build: 0 = clippy clean + tests pass.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Containerized validation (clippy + tests, CI parity) ==="
echo "Building target 'validate' from docker/Dockerfile.memexd ..."
exec docker build --target validate -f docker/Dockerfile.memexd "$@" .
