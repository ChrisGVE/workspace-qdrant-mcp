#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Prefer Homebrew OpenJDK, fall back to system javac
if [ -x /usr/local/opt/openjdk/bin/javac ]; then
    JAVAC=/usr/local/opt/openjdk/bin/javac
elif command -v javac &>/dev/null && javac -version &>/dev/null; then
    JAVAC=javac
else
    echo "Error: javac not found" >&2
    exit 1
fi

mkdir -p out
$JAVAC -d out src/bookshelf/*.java
