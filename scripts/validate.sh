#!/usr/bin/env bash
# validate.sh — local validation gate (clippy + release build).
#
# Runs `cargo clippy -D warnings` (lib + bins + all test targets) plus the
# production release build, inside the Linux builder container
# (docker/Dockerfile.memexd `validate` stage). Work lands on `dev`/`main`,
# which DO get CI (ci.yml runs on push/PR to main/dev). This gate is a FAST
# LOCAL pre-check before pushing, plus a workaround for the local Windows
# build, which breaks on a pre-existing winapi gate in storage/client.rs.
# Run it before pushing `dev` or promoting `dev -> main`.
#
# The full `cargo test` suite is NOT run here (fragile in this flattened
# container — FTS5/trigram + timing tests fail for env reasons). CI runs it at
# true parity on the `dev`/`main` push or the `dev -> main` PR (full checkout).
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
