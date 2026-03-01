#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
odin run . 2>/dev/null || odin run .
