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

	# An absent override means the deployment is not using the
	# generate-compose flow — it hand-wires its volumes directly (the
	# reference compose binds identical host/container paths itself). There
	# is no generated override to go stale, so there is nothing for layer 1
	# to verify: skip it. The hash check exists solely to catch a *stale*
	# override (config edited, generate-compose not re-run), which can only
	# happen when an override is actually mounted. Layer 2 still validates
	# that every config-declared mount is present, so hand-wired deployments
	# remain covered.
	if [ ! -f "${WQM_OVERRIDE_PATH}" ]; then
		log_info "layer 1: skipped (no override mounted at ${WQM_OVERRIDE_PATH})"
		return ${EXIT_OK}
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
# Mount-present validation — for each config mount entry, the container
# directory must exist as a directory inside the container. A missing
# `volumes:` line in the override file leaves the container path absent.
# We treat both "non-existent" and "exists but not a directory" as failure
# modes; the spec only requires the directory case but checking the type
# defends against a host-side file being accidentally mounted.
# ────────────────────────────────────────────────────────────────────────
mount_present() {
	local container_path="$1"
	[ -d "${container_path}" ]
}

# ────────────────────────────────────────────────────────────────────────
# Layer 2 — Mount-present validation
#
# Iterates over (host, container) pairs from config.yaml and aborts with
# EXIT_MOUNT_MISSING on the first container path that is not a directory.
# Reports the container path of the failing entry; the host path is
# included as context only (per §9.1.1 the host path is opaque inside
# the container and is informational).
# ────────────────────────────────────────────────────────────────────────
layer2_mount_present() {
	log_info "layer 2: checking each config mount is present in container"

	local pairs
	if ! pairs="$(config_mount_pairs "${WQM_CONFIG_PATH}")"; then
		log_error "failed to parse mounts section of ${WQM_CONFIG_PATH}"
		return ${EXIT_CONFIG_INVALID}
	fi

	if [ -z "${pairs}" ]; then
		log_info "layer 2: ok (no mounts declared in config.yaml)"
		return ${EXIT_OK}
	fi

	local checked=0
	local host container
	# Read pairs line-by-line; tab-delimited.
	while IFS=$'\t' read -r host container; do
		if [ -z "${host}" ] || [ -z "${container}" ]; then
			continue
		fi
		if ! mount_present "${container}"; then
			log_error "Required mount missing: ${container}"
			log_error "  declared in ${WQM_CONFIG_PATH} as: host=${host} container=${container}"
			log_error "  The docker-compose.override.yaml is missing the corresponding volumes: line."
			log_error "  Fix: run 'wqm docker generate-compose' on the host, then restart the container."
			return ${EXIT_MOUNT_MISSING}
		fi
		checked=$((checked + 1))
	done <<<"${pairs}"

	log_info "layer 2: ok (${checked} mount(s) verified present)"
	return ${EXIT_OK}
}

# ────────────────────────────────────────────────────────────────────────
# /proc/self/mountinfo parser — emits mount points one per line.
#
# The mountinfo line format (man proc(5)) has 11+ space-separated fields:
#
#   36 35 98:0 /mnt1 /mnt1 rw,noatime master:1 - ext3 /dev/root rw,...
#   │  │  │    │     │     │          │   │    │    │         │
#   1  2  3    4     5     6          7  -|fs| 9    10        11
#
# Field 5 is the mount point as the kernel renders it (octal-escaped
# spaces, tabs, newlines, backslashes — \040 \011 \012 \134). The
# remainder before `-` is optional tag list, then `-`, fstype, source,
# super-opts. Bind mounts surface as a non-`/` value in field 4 (the
# root within the source filesystem), but for the spurious-mount check
# we only care about mount points the user could plausibly conflate
# with a config-declared volume, so we emit every mount point and let
# the caller filter against the config list and a denylist of system
# pseudofs locations.
#
# Stdin: ignored.  Args: mountinfo path.  Stdout: one path per line.
#
# Note: detect_spurious_mounts() below re-implements this parser inside
# Python because the python3 stdin-script (`python3 -`) and the heredoc
# script body conflict when this function's stdout is piped in. The shell
# version is kept as a standalone helper for ad-hoc diagnostics and to
# satisfy spec §9.1 layer-3 contract that a mountinfo parser exists.
# ────────────────────────────────────────────────────────────────────────
parse_mountinfo() {
	local mountinfo_path="$1"
	if [ ! -r "${mountinfo_path}" ]; then
		# No mountinfo (e.g., running outside Linux) — emit nothing.
		return 0
	fi
	# Field 5 is the mount point. Decode kernel octal escapes for space,
	# tab, newline, backslash; those are the only ones the kernel emits.
	awk '
        {
            mp = $5
            # Replace octal escapes the kernel uses for special chars.
            gsub(/\\040/, " ", mp)
            gsub(/\\011/, "\t", mp)
            gsub(/\\012/, "\n", mp)
            gsub(/\\134/, "\\", mp)
            print mp
        }
    ' "${mountinfo_path}"
}

# ────────────────────────────────────────────────────────────────────────
# Detect spurious bind mounts — mount points present in mountinfo that
# are NOT declared as containers in config.yaml AND are not under
# well-known system pseudo-filesystems (/proc, /sys, /dev, /run, the
# memexd runtime bind targets covered by spec §9.2: /etc/wqm,
# /var/lib/wqm, /qdrant/storage, /etc/docker-compose-wqm.override.yaml).
#
# Args: none. Globals: WQM_CONFIG_PATH, WQM_MOUNTINFO_PATH.
# Stdout: spurious mount points, one per line.
# Exit:   0 always (best-effort; missing inputs → empty output).
# ────────────────────────────────────────────────────────────────────────
detect_spurious_mounts() {
	local pairs
	if [ ! -r "${WQM_MOUNTINFO_PATH}" ]; then
		return 0
	fi
	if ! pairs="$(config_mount_pairs "${WQM_CONFIG_PATH}" 2>/dev/null)"; then
		pairs=""
	fi

	# Build the list of expected container mount points: declared containers
	# plus the spec §9.2 runtime bind targets plus the override file path.
	local expected
	expected=$(
		if [ -n "${pairs}" ]; then
			printf '%s\n' "${pairs}" | awk -F'\t' 'NF==2 {print $2}'
		fi
		printf '%s\n' \
			"/etc/wqm" \
			"/etc/wqm/config.yaml" \
			"/var/lib/wqm" \
			"/qdrant/storage" \
			"${WQM_OVERRIDE_PATH}"
	)

	# Python filter: open mountinfo, parse field 5 with octal unescaping,
	# drop anything in `expected` or under a denied system pseudofs prefix.
	# Mountinfo path goes via argv[1], expected list (newline-delimited)
	# via WQM_EXPECTED env so we sidestep the python3 `-` stdin-vs-heredoc
	# conflict.
	WQM_EXPECTED="${expected}" python3 - "${WQM_MOUNTINFO_PATH}" <<'PY'
import os, re, sys
mountinfo_path = sys.argv[1]
expected = set(
    line for line in os.environ.get("WQM_EXPECTED", "").splitlines() if line
)
denied = (
    "/proc", "/sys", "/dev", "/run",
    "/etc/hostname", "/etc/hosts", "/etc/resolv.conf", "/etc/mtab",
)


def unescape(p: str) -> str:
    # Kernel mountinfo escapes space, tab, newline, backslash as octal.
    return re.sub(
        r"\\(0[0-9]{2}|1[0-3][0-7])",
        lambda m: chr(int(m.group(1), 8)),
        p,
    )


def under_denied(p: str) -> bool:
    if p == "/":
        return True
    for d in denied:
        if p == d or p.startswith(d + "/"):
            return True
    return False


try:
    with open(mountinfo_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split(" ")
            if len(fields) < 5:
                continue
            mp = unescape(fields[4])
            if not mp or under_denied(mp) or mp in expected:
                continue
            sys.stdout.write(mp + "\n")
except OSError:
    pass
PY
}

# ────────────────────────────────────────────────────────────────────────
# Layer 3 — Spurious-mount warning (best-effort, non-fatal)
#
# Reports any bind mount in /proc/self/mountinfo that we cannot attribute
# to a config entry or a spec §9.2 runtime bind. Does not abort: user-added
# scratch mounts for debugging are a legitimate use-case (§9.1).
# ────────────────────────────────────────────────────────────────────────
layer3_spurious_warning() {
	log_info "layer 3: scanning for unexpected bind mounts"

	if [ ! -r "${WQM_MOUNTINFO_PATH}" ]; then
		log_info "layer 3: skipped (${WQM_MOUNTINFO_PATH} not readable)"
		return ${EXIT_OK}
	fi

	local spurious
	spurious="$(detect_spurious_mounts || true)"

	if [ -z "${spurious}" ]; then
		log_info "layer 3: ok (no unexpected bind mounts)"
		return ${EXIT_OK}
	fi

	local count=0
	while IFS= read -r mp; do
		if [ -z "${mp}" ]; then
			continue
		fi
		log_warn "Unexpected bind mount detected: ${mp}"
		log_warn "  Not present in config.yaml mounts section."
		log_warn "  This may be intentional (e.g., debugging), but verify it's expected."
		count=$((count + 1))
	done <<<"${spurious}"

	log_info "layer 3: ${count} unexpected mount(s) reported (non-fatal)"
	return ${EXIT_OK}
}

# ────────────────────────────────────────────────────────────────────────
# Main orchestration — run all three layers, then exec memexd.
# ────────────────────────────────────────────────────────────────────────
main() {
	log_info "memexd entrypoint starting"
	log_info "override: ${WQM_OVERRIDE_PATH}"
	log_info "config:   ${WQM_CONFIG_PATH}"

	layer1_hash_check || exit $?
	layer2_mount_present || exit $?
	layer3_spurious_warning || exit $?

	if [ -n "${WQM_ENTRYPOINT_SKIP_EXEC:-}" ]; then
		log_info "WQM_ENTRYPOINT_SKIP_EXEC set — would exec: ${WQM_MEMEXD_BIN} $*"
		return ${EXIT_OK}
	fi

	log_info "exec ${WQM_MEMEXD_BIN} $*"
	exec "${WQM_MEMEXD_BIN}" "$@"
}

main "$@"
