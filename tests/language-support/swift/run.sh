#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
swift run 2>&1 | grep -v "^\[" | grep -v "^Build" | grep -v "^Compil" | grep -v "^Link" | grep -v "^Apply" | grep -v "^Emit" | grep -v "^Write" | grep -v "^Planning"
