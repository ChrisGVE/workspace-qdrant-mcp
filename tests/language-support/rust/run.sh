#!/bin/bash
cd "$(dirname "$0")"
cargo run --quiet 2>/dev/null || cargo run
