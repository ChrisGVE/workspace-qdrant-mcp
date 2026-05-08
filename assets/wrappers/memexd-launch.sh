#!/usr/bin/env sh
# memexd-launch — launch the memexd daemon with optional secret env file.
#
# The daemon reads its API key for remote embedding providers from an env
# var named by `embedding.api_key_env_var` (default `OPENAI_API_KEY`). On
# macOS, launchd does not inherit interactive shell environments, so this
# wrapper exposes three opt-in mechanisms for getting the key into the
# daemon's environment:
#
#   (1) Env file — sourced if present, must be mode 0600 or 0400. Path:
#       $WQM_ENV_FILE  >  $XDG_CONFIG_HOME/memexd/env  >  $HOME/.config/memexd/env
#   (2) Inherited environment — anything already in the wrapper's env is
#       passed through. This covers `launchctl setenv KEY value` and any
#       upstream `EnvironmentVariables` block in the LaunchAgent plist.
#   (3) Container env — when the daemon runs in Docker, this wrapper is
#       irrelevant; the container receives env vars via env_file or the
#       `environment:` block in docker-compose.
#
# Resolution order: the env file is sourced first (if present), then the
# inherited environment fills in any missing vars. Existing env vars are
# overridden by env-file values (env-file wins) so an explicit launchctl
# setenv can be cleared by removing the entry from the file.

set -eu

memexd_bin="${MEMEXD_BIN:-${HOME}/.local/bin/memexd}"

candidate_env_file() {
	if [ -n "${WQM_ENV_FILE:-}" ]; then
		printf '%s' "$WQM_ENV_FILE"
		return
	fi
	if [ -n "${XDG_CONFIG_HOME:-}" ] && [ -f "${XDG_CONFIG_HOME}/memexd/env" ]; then
		printf '%s' "${XDG_CONFIG_HOME}/memexd/env"
		return
	fi
	if [ -f "${HOME}/.config/memexd/env" ]; then
		printf '%s' "${HOME}/.config/memexd/env"
	fi
}

env_file="$(candidate_env_file || true)"
if [ -n "${env_file:-}" ] && [ -f "$env_file" ]; then
	# `ls -l` is the only mode display identical across GNU and BSD
	# userlands (stat -c / stat -f differ). Match the rwx string directly.
	mode_str="$(ls -l "$env_file" 2>/dev/null | awk 'NR==1{print $1}')"
	case "$mode_str" in
	-rw------- | -r--------)
		set -a
		# shellcheck disable=SC1090
		. "$env_file"
		set +a
		;;
	*)
		echo "memexd-launch: refusing to source $env_file — mode $mode_str (must be 0600 or 0400)" >&2
		exit 78
		;;
	esac
fi

exec "$memexd_bin" "$@"
