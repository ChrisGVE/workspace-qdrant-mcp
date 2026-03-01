#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Prefer Homebrew OpenJDK, fall back to system java
if [ -x /usr/local/opt/openjdk/bin/java ]; then
    JAVA=/usr/local/opt/openjdk/bin/java
elif command -v java &>/dev/null && java -version &>/dev/null; then
    JAVA=java
else
    echo "Error: java not found" >&2
    exit 1
fi

./build.sh && $JAVA -cp out bookshelf.Main
