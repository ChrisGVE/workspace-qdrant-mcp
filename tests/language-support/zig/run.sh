#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
zig build run 2>/dev/null || zig build run
