//! Human-actionable rendering of daemon RPC failures (#109).
//!
//! When memexd is not running, tonic surfaces the raw transport message
//! `tcp connect error: …` (code `Unavailable`), which reads like a network
//! bug rather than the actual condition. Every user-visible error path in
//! the MCP server and CLI should render daemon `Status` errors through
//! [`status_user_message`] so "daemon down" is stated plainly with the
//! command that fixes it.

use tonic::{Code, Status};

/// Advice appended to every daemon-unreachable error message.
const START_HINT: &str = "is memexd running? Start it with `wqm service start`";

/// Render a daemon RPC [`Status`] as a user-facing message.
///
/// - `Unavailable` (daemon not running / connection refused) → a plain
///   "daemon not reachable" statement plus the start hint, with the raw
///   transport message kept in parentheses for diagnosis.
/// - Everything else → the status message as-is (matching the previous
///   behaviour of `status.message()`).
pub fn status_user_message(status: &Status) -> String {
    match status.code() {
        Code::Unavailable => format!("daemon not reachable ({}) — {START_HINT}", status.message()),
        _ => status.message().to_string(),
    }
}

/// True when `status` means the daemon could not be reached at all
/// (as opposed to the daemon rejecting or failing the request).
pub fn is_daemon_unreachable(status: &Status) -> bool {
    status.code() == Code::Unavailable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_maps_to_actionable_message() {
        let status = Status::unavailable("tcp connect error: Connection refused (os error 61)");
        let msg = status_user_message(&status);
        assert!(msg.starts_with("daemon not reachable"), "got: {msg}");
        assert!(msg.contains("tcp connect error"), "raw cause kept: {msg}");
        assert!(msg.contains("wqm service start"), "start hint: {msg}");
    }

    #[test]
    fn other_codes_pass_message_through() {
        let status = Status::invalid_argument("pattern must not be empty");
        assert_eq!(status_user_message(&status), "pattern must not be empty");
    }

    #[test]
    fn unreachable_detection_tracks_code() {
        assert!(is_daemon_unreachable(&Status::unavailable("x")));
        assert!(!is_daemon_unreachable(&Status::internal("x")));
    }
}
