//! Shared user-facing messages for `wqm service` commands.
//!
//! Keeps the exact wording of docker hints, remote-daemon errors, and
//! "both sources running" warnings in a single place so the five handlers
//! stay consistent and so test assertions can reference the canonical text.
//!
//! All helpers are tiny wrappers around `println!` / `eprintln!`. Tests that
//! need to verify wording use the `msg_*` pure formatters instead of calling
//! the side-effecting printers, so captured-output harnesses are unnecessary.

// ─── Pure formatters (used by handlers AND tests) ────────────────────────

/// Warning shown when both a local PID and a docker container are running
/// and we have decided to act on the local process.
pub fn msg_warn_both() -> String {
    "Warning: memexd running in both local and docker; controlling local.".to_string()
}

/// Stdout hint shown in `DockerOnly` mode, telling the user which
/// docker-compose verb to use instead.
///
/// `verb` is expected to be one of: `up`, `down`, `restart`.
pub fn msg_info_docker(verb: &str) -> String {
    format!("memexd running in Docker. Use 'docker compose {verb}' to manage.")
}

/// Error shown when the user invokes a local lifecycle operation against a
/// `RemoteOnly` daemon we are not allowed to mutate.
///
/// `op` is a short verb like `start`, `stop`, `restart`.
pub fn msg_err_remote(op: &str, addr: &str) -> String {
    format!("cannot {op} remote memexd at {addr}")
}

/// Error shown when a stop-style operation runs against `None`.
pub fn msg_err_nothing_running() -> String {
    "nothing running: no local, docker, or remote memexd detected.".to_string()
}

// ─── Side-effecting helpers ──────────────────────────────────────────────

/// Print the "both local+docker" warning to stderr.
pub fn warn_both() {
    eprintln!("{}", msg_warn_both());
}

/// Print the docker-compose hint to stdout.
pub fn info_docker(verb: &str) {
    println!("{}", msg_info_docker(verb));
}

/// Print the remote-daemon error to stderr.
pub fn err_remote(op: &str, addr: &str) {
    eprintln!("{}", msg_err_remote(op, addr));
}

/// Print "nothing running" to stderr.
pub fn err_nothing_running() {
    eprintln!("{}", msg_err_nothing_running());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warn_both_text_stable() {
        assert_eq!(
            msg_warn_both(),
            "Warning: memexd running in both local and docker; controlling local."
        );
    }

    #[test]
    fn info_docker_uses_verb() {
        assert_eq!(
            msg_info_docker("up"),
            "memexd running in Docker. Use 'docker compose up' to manage."
        );
        assert_eq!(
            msg_info_docker("down"),
            "memexd running in Docker. Use 'docker compose down' to manage."
        );
        assert_eq!(
            msg_info_docker("restart"),
            "memexd running in Docker. Use 'docker compose restart' to manage."
        );
    }

    #[test]
    fn err_remote_interpolates_op_and_addr() {
        assert_eq!(
            msg_err_remote("start", "10.0.0.1:50051"),
            "cannot start remote memexd at 10.0.0.1:50051"
        );
        assert_eq!(
            msg_err_remote("stop", "127.0.0.1:50051"),
            "cannot stop remote memexd at 127.0.0.1:50051"
        );
    }

    #[test]
    fn err_nothing_running_text_stable() {
        assert_eq!(
            msg_err_nothing_running(),
            "nothing running: no local, docker, or remote memexd detected."
        );
    }
}
