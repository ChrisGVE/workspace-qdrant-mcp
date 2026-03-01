#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
if command -v lua5.4 >/dev/null 2>&1; then
    lua5.4 main.lua
else
    lua main.lua
fi
