#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p obj
gnatmake -o bookshelf -D obj src/main.adb -aIsrc
