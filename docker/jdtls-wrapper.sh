#!/bin/sh
# Eclipse JDT Language Server (Java LSP) launcher for the memexd daemon.
#
# Invoked by the daemon (uid 1000 `memexd`), which sets CWD to the project root
# and speaks LSP over stdio. Two things the naive `java -jar … -configuration
# /opt/jdtls/config_linux` invocation got wrong, both of which left Java
# permanently red on the dashboard:
#
#   1. Java version. The current JDT bundles require Java 21
#      (Require-Capability osgi.ee=JavaSE;version=21); the image's old
#      default-jre-headless was Java 17, so the bundles never resolved. We now
#      ship a Temurin 21 JRE at /opt/java21 and call it explicitly.
#
#   2. Writable areas. Eclipse writes OSGi state into the *configuration* area
#      and per-workspace state into the *data* area. The shipped
#      /opt/jdtls/config_linux is root-owned (read-only for `memexd`), so copy
#      it once into a per-user writable cache; derive a stable, unique -data dir
#      from the project root (CWD) so concurrent jdtls instances for different
#      projects don't collide.
set -e

JAR="$(ls /opt/jdtls/plugins/org.eclipse.equinox.launcher_*.jar 2>/dev/null | head -1)"
if [ -z "$JAR" ]; then
  echo "jdtls: equinox launcher JAR not found under /opt/jdtls/plugins" >&2
  exit 1
fi

CACHE="${HOME:-/tmp}/.cache/jdtls"

# Writable copy of the shipped configuration area (one-time per container).
CONFIG="$CACHE/config"
if [ ! -e "$CONFIG/.wqm-initialized" ]; then
  rm -rf "$CONFIG"
  mkdir -p "$CONFIG"
  cp -a /opt/jdtls/config_linux/. "$CONFIG/"
  touch "$CONFIG/.wqm-initialized"
fi

# Per-workspace data dir, keyed by the project root so parallel projects don't
# share (and lock) the same Eclipse workspace.
DATA="$CACHE/ws/$(printf '%s' "$PWD" | md5sum | cut -c1-16)"
mkdir -p "$DATA"

exec /opt/java21/bin/java \
    -Declipse.application=org.eclipse.jdt.ls.core.id1 \
    -Dosgi.bundles.defaultStartLevel=4 \
    -Declipse.product=org.eclipse.jdt.ls.core.product \
    -Xmx1G \
    --add-modules=ALL-SYSTEM \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    -jar "$JAR" \
    -configuration "$CONFIG" \
    -data "$DATA" \
    "$@"
