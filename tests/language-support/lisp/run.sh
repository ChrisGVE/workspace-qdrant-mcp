#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
if command -v sbcl >/dev/null 2>&1; then
    sbcl --noinform --non-interactive \
         --load src/models.lisp \
         --load src/storage.lisp \
         --load src/utils.lisp \
         --load src/main.lisp
elif command -v clisp >/dev/null 2>&1; then
    clisp -norc -q -q \
         -i src/models.lisp \
         -i src/storage.lisp \
         -i src/utils.lisp \
         -i src/main.lisp \
         -x '(quit)' 2>/dev/null
else
    echo "Error: No Common Lisp implementation found (sbcl or clisp required)" >&2
    exit 1
fi
