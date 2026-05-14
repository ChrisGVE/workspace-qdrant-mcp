#!/bin/bash
# memexd container entrypoint — path-abstraction stale-override defense.
#
# Design source: docs/specs/16-path-abstraction.md §9.1 (three-layer check).
#
# Layered checks before exec'ing memexd:
#
#   Layer 1 — Hash check.
#     Read `# wqm-config-hash: <hex>` from the bind-mounted override
#     /etc/docker-compose-wqm.override.yaml and compare against the hash
#     freshly computed from /etc/wqm/config.yaml's `mounts:` section.
#     Mismatch ⇒ hard abort.
#
#   Layer 2 — Mount-present validation.
#     For each `(host, container)` entry in config.yaml, verify the
#     container path is a directory inside this container. A missing
#     `volumes:` line in the override produces a stat failure here.
#     Mismatch ⇒ hard abort.
#
#   Layer 3 — Spurious-mount warning (non-fatal).
#     Read /proc/self/mountinfo, list bind mounts under expected
#     mount-map prefixes that are NOT in the config, and warn. User-added
#     debug mounts are legitimate, so this layer never aborts.
#
# When all checks pass, the script execs memexd with the CMD args.
#
# Environment overrides (primarily for tests / debugging):
#   WQM_OVERRIDE_PATH        path of bind-mounted override file
#                              default: /etc/docker-compose-wqm.override.yaml
#   WQM_CONFIG_PATH          path of bind-mounted config file
#                              default: /etc/wqm/config.yaml
#   WQM_MOUNTINFO_PATH       path of mountinfo (override for tests)
#                              default: /proc/self/mountinfo
#   WQM_MEMEXD_BIN           path of memexd binary
#                              default: /usr/local/bin/memexd
#   WQM_ENTRYPOINT_SKIP_EXEC if non-empty, skip the final `exec` and
#                            print the planned argv instead (test hook)

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────
# Defaults (env-overridable for tests).
# ────────────────────────────────────────────────────────────────────────
: "${WQM_OVERRIDE_PATH:=/etc/docker-compose-wqm.override.yaml}"
: "${WQM_CONFIG_PATH:=/etc/wqm/config.yaml}"
: "${WQM_MOUNTINFO_PATH:=/proc/self/mountinfo}"
: "${WQM_MEMEXD_BIN:=/usr/local/bin/memexd}"

readonly HASH_HEADER_PREFIX='# wqm-config-hash:'

# Exit codes — kept stable for integration tests.
readonly EXIT_OK=0
readonly EXIT_HASH_MISMATCH=10
readonly EXIT_MOUNT_MISSING=11
readonly EXIT_USAGE=64
readonly EXIT_CONFIG_INVALID=78

# ────────────────────────────────────────────────────────────────────────
# Logging helpers — all diagnostics go to stderr to keep stdout free for
# the daemon's own structured logs after exec.
# ────────────────────────────────────────────────────────────────────────
log_info() { printf '[entrypoint] %s\n' "$*" >&2; }
log_warn() { printf '[entrypoint] WARNING: %s\n' "$*" >&2; }
log_error() { printf '[entrypoint] ERROR: %s\n' "$*" >&2; }

# ────────────────────────────────────────────────────────────────────────
# Layer implementations are introduced in subsequent commits.
# ────────────────────────────────────────────────────────────────────────

main() {
	log_info "memexd entrypoint starting"
	log_info "override: ${WQM_OVERRIDE_PATH}"
	log_info "config:   ${WQM_CONFIG_PATH}"

	# Subsequent commits hook Layer 1, 2, 3 here.

	if [ -n "${WQM_ENTRYPOINT_SKIP_EXEC:-}" ]; then
		log_info "WQM_ENTRYPOINT_SKIP_EXEC set — would exec: ${WQM_MEMEXD_BIN} $*"
		return ${EXIT_OK}
	fi

	log_info "exec ${WQM_MEMEXD_BIN} $*"
	exec "${WQM_MEMEXD_BIN}" "$@"
}

main "$@"
