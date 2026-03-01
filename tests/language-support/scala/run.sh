#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Scala 3 runner (scala-cli based). Filter JVM warnings and build noise from stderr/stdout.
scala run . 2>/dev/null | grep -v '^WARNING:' | grep -v '^Compiling project' | grep -v '^\[' | grep -v '^Downloading ' | grep -v '^Downloaded ' | grep -v '^Failed to download'
