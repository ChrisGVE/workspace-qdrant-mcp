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
# Hash extraction — read the `# wqm-config-hash: <hex>` line from the
# override file. Mirrors the Rust extract_hash_header() (cli/src/commands/
# docker/generate_compose.rs). Tolerant of leading blanks; takes the FIRST
# matching line.
#
# Stdin: ignored.  Args: override file path.  Stdout: hex hash or empty.
# ────────────────────────────────────────────────────────────────────────
extract_hash_header() {
	local override_path="$1"
	if [ ! -f "${override_path}" ]; then
		return 0
	fi
	# `awk` walks lines until it finds one whose leading non-space matches
	# the prefix, then prints the remainder trimmed.
	awk -v prefix="${HASH_HEADER_PREFIX}" '
        {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            if (index(line, prefix) == 1) {
                rest = substr(line, length(prefix) + 1)
                sub(/^[[:space:]]+/, "", rest)
                sub(/[[:space:]]+$/, "", rest)
                print rest
                exit
            }
        }
    ' "${override_path}"
}

# ────────────────────────────────────────────────────────────────────────
# Config mount-section helpers — parsing and hash computation.
#
# We delegate to Python + PyYAML for two reasons:
#
#   1. PyYAML's safe_dump(default_flow_style=False, sort_keys=False) emits
#      bytes byte-for-byte identical to serde_yaml_ng::to_string of the
#      equivalent Vec<YamlMountEntry> across all common-case inputs (and
#      the edge cases probed during T11 design: empty list, paths with
#      `:`, `#`, trailing whitespace, tilde). The hash invariant of spec
#      §9.1 is therefore preserved without reimplementing serde_yaml_ng's
#      quoting state machine in shell.
#
#   2. PyYAML's safe_load handles the full config.yaml grammar (anchors,
#      flow vs block, comments) which awk/grep cannot.
#
# The Python interpreter is added to Dockerfile.memexd's runtime stage in
# the same PR as this entrypoint (subtask 11.13).
# ────────────────────────────────────────────────────────────────────────

# Print each (host TAB container) pair from config.yaml on its own line.
# Empty output ⇒ no mounts (still valid).
config_mount_pairs() {
	local config_path="$1"
	python3 - "${config_path}" <<'PY'
import sys, yaml
path = sys.argv[1]
try:
    with open(path, 'r', encoding='utf-8') as f:
        doc = yaml.safe_load(f) or {}
except (OSError, yaml.YAMLError) as e:
    sys.stderr.write(f"config parse error: {e}\n")
    sys.exit(78)
if not isinstance(doc, dict):
    sys.stderr.write("config.yaml top level must be a mapping\n")
    sys.exit(78)
mounts = doc.get('mounts') or []
if not isinstance(mounts, list):
    sys.stderr.write("config.yaml `mounts` must be a list\n")
    sys.exit(78)
for i, m in enumerate(mounts):
    if not isinstance(m, dict):
        sys.stderr.write(f"mounts[{i}] is not a mapping\n")
        sys.exit(78)
    host = m.get('host')
    container = m.get('container')
    if not isinstance(host, str) or not isinstance(container, str):
        sys.stderr.write(f"mounts[{i}] missing host/container string fields\n")
        sys.exit(78)
    # Tab-delimited: tabs are not allowed in canonical paths.
    sys.stdout.write(f"{host}\t{container}\n")
PY
}

# Compute the spec-compliant SHA-256 over the canonical YAML
# serialisation of `config.yaml`'s mounts section.  Matches
# wqm-common::paths::mount_section_hash byte-for-byte.
compute_config_hash() {
	local config_path="$1"
	python3 - "${config_path}" <<'PY'
import hashlib, sys, yaml
path = sys.argv[1]
try:
    with open(path, 'r', encoding='utf-8') as f:
        doc = yaml.safe_load(f) or {}
except (OSError, yaml.YAMLError) as e:
    sys.stderr.write(f"config parse error: {e}\n")
    sys.exit(78)
mounts = doc.get('mounts') or [] if isinstance(doc, dict) else []
# Re-serialise in the form serde_yaml_ng emits for Vec<YamlMountEntry>.
# Empty list ⇒ "[]\n"; non-empty ⇒ block style, keys in declaration order.
normalised = [{'host': m.get('host', ''), 'container': m.get('container', '')} for m in mounts]
ser = yaml.safe_dump(normalised, default_flow_style=False, sort_keys=False)
sys.stdout.write(hashlib.sha256(ser.encode('utf-8')).hexdigest())
PY
}

# ────────────────────────────────────────────────────────────────────────
# Layer 1 — Hash check
#
# Aborts with EXIT_HASH_MISMATCH when override hash header is absent or
# disagrees with the live config hash.  This is the dominant failure mode
# the spec defends against: user edits config.yaml, forgets to re-run
# `wqm docker generate-compose`, container starts with stale mount set.
# ────────────────────────────────────────────────────────────────────────
layer1_hash_check() {
	log_info "layer 1: verifying override hash against live config"

	if [ ! -f "${WQM_OVERRIDE_PATH}" ]; then
		log_error "override file not found: ${WQM_OVERRIDE_PATH}"
		log_error "  generate it on the host with: wqm docker generate-compose"
		return ${EXIT_HASH_MISMATCH}
	fi
	if [ ! -f "${WQM_CONFIG_PATH}" ]; then
		log_error "config file not found: ${WQM_CONFIG_PATH}"
		log_error "  ensure the host bind-mounts config.yaml → ${WQM_CONFIG_PATH}"
		return ${EXIT_CONFIG_INVALID}
	fi

	local recorded
	recorded="$(extract_hash_header "${WQM_OVERRIDE_PATH}")"
	if [ -z "${recorded}" ]; then
		log_error "override ${WQM_OVERRIDE_PATH} is missing the '${HASH_HEADER_PREFIX} <hex>' header"
		log_error "  regenerate it: wqm docker generate-compose"
		return ${EXIT_HASH_MISMATCH}
	fi

	local live
	if ! live="$(compute_config_hash "${WQM_CONFIG_PATH}")"; then
		log_error "failed to compute hash from ${WQM_CONFIG_PATH}"
		return ${EXIT_CONFIG_INVALID}
	fi

	if [ "${recorded}" != "${live}" ]; then
		log_error "docker-compose.override.yaml is stale (hash mismatch)"
		log_error "  override recorded: ${recorded}"
		log_error "  live config hash:  ${live}"
		log_error "  Config file changed but override not regenerated."
		log_error "  Fix: run 'wqm docker generate-compose' on the host, then restart the container."
		return ${EXIT_HASH_MISMATCH}
	fi

	log_info "layer 1: ok (hash ${recorded:0:12}…)"
	return ${EXIT_OK}
}

# ────────────────────────────────────────────────────────────────────────
# Main orchestration — layer hooks added in subsequent commits.
# ────────────────────────────────────────────────────────────────────────
main() {
	log_info "memexd entrypoint starting"
	log_info "override: ${WQM_OVERRIDE_PATH}"
	log_info "config:   ${WQM_CONFIG_PATH}"

	layer1_hash_check || exit $?
	# Layer 2 + Layer 3 are hooked in subsequent commits.

	if [ -n "${WQM_ENTRYPOINT_SKIP_EXEC:-}" ]; then
		log_info "WQM_ENTRYPOINT_SKIP_EXEC set — would exec: ${WQM_MEMEXD_BIN} $*"
		return ${EXIT_OK}
	fi

	log_info "exec ${WQM_MEMEXD_BIN} $*"
	exec "${WQM_MEMEXD_BIN}" "$@"
}

main "$@"
