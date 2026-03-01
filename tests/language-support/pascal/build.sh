#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p obj
fpc -obookshelf -FUobj -Fusrc src/main.pas >&2
# FPC places binary next to source file; move if needed
if [ -f src/bookshelf ] && [ ! -f bookshelf ]; then
    mv src/bookshelf .
fi
