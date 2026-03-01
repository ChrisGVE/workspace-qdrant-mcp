#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ebin
erlc -o ebin src/*.erl
erl -noshell -pa ebin -eval 'main:main()' -s init stop
