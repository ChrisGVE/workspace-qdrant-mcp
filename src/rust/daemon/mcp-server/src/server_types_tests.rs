//! Tests for `server_types`.

use super::*;

// ── Constants ──────────────────────────────────────────────────────────────

#[test]
fn server_name_matches_typescript() {
    // TS: export const SERVER_NAME = 'workspace-qdrant-mcp'; (server-types.ts:14)
    assert_eq!(SERVER_NAME, "workspace-qdrant-mcp");
}

#[test]
fn server_version_base_matches_typescript() {
    // TS pattern: `0.1.0-beta1 (${BUILD_NUMBER})`
    assert_eq!(SERVER_VERSION_BASE, "0.1.0-beta1");
}

#[test]
fn server_version_string_format() {
    let v = server_version_string();
    assert!(
        v.starts_with("0.1.0-beta1 ("),
        "version string should start with '0.1.0-beta1 (', got: {v:?}"
    );
    assert!(
        v.ends_with(')'),
        "version string should end with ')', got: {v:?}"
    );
    // Build number is a non-empty 4-digit hex string
    let build = &v["0.1.0-beta1 (".len()..v.len() - 1];
    assert!(!build.is_empty(), "build number must not be empty");
    assert!(
        build
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c.is_ascii_uppercase()),
        "build number should be hex digits, got: {build:?}"
    );
}

#[test]
fn build_number_is_non_empty() {
    assert!(!BUILD_NUMBER.is_empty());
}

#[test]
fn default_http_host_matches_typescript() {
    // TS: export const DEFAULT_HTTP_HOST = '127.0.0.1'; (server-types.ts:106)
    assert_eq!(DEFAULT_HTTP_HOST, "127.0.0.1");
}

#[test]
fn default_http_port_matches_typescript() {
    // TS: export const DEFAULT_HTTP_PORT = 6335; (server-types.ts:107)
    assert_eq!(DEFAULT_HTTP_PORT, 6335_u16);
}

#[test]
fn default_http_path_matches_typescript() {
    // TS: export const DEFAULT_HTTP_PATH = '/mcp'; (server-types.ts:108)
    assert_eq!(DEFAULT_HTTP_PATH, "/mcp");
}

#[test]
fn heartbeat_interval_matches_typescript() {
    // TS: export const HEARTBEAT_INTERVAL_MS = 30 * 1000; (server-types.ts:11)
    assert_eq!(HEARTBEAT_INTERVAL_MS, 30_000_u64);
}

// ── ServerMode ─────────────────────────────────────────────────────────────

#[test]
fn server_mode_default_is_stdio() {
    assert_eq!(ServerMode::default(), ServerMode::Stdio);
}

#[test]
fn server_mode_variants_are_distinct() {
    assert_ne!(ServerMode::Stdio, ServerMode::Http);
}

#[test]
fn server_mode_clone_and_copy() {
    let m = ServerMode::Http;
    let m2 = m;
    assert_eq!(m, m2);
    let m3 = m.clone();
    assert_eq!(m, m3);
}

// ── HttpTransportOptions ───────────────────────────────────────────────────

#[test]
fn http_transport_options_default_uses_constants() {
    let opts = HttpTransportOptions::default();
    assert_eq!(opts.host, DEFAULT_HTTP_HOST);
    assert_eq!(opts.port, DEFAULT_HTTP_PORT);
    assert_eq!(opts.path, DEFAULT_HTTP_PATH);
    assert!(opts.tls.is_none());
}

#[test]
fn http_transport_options_with_tls() {
    let opts = HttpTransportOptions {
        host: "0.0.0.0".to_string(),
        port: 8443,
        path: "/mcp".to_string(),
        tls: Some(HttpTlsOptions {
            cert_path: PathBuf::from("/etc/ssl/cert.pem"),
            key_path: PathBuf::from("/etc/ssl/key.pem"),
            ca_path: None,
        }),
    };
    assert!(opts.tls.is_some());
    let tls = opts.tls.unwrap();
    assert_eq!(tls.cert_path, PathBuf::from("/etc/ssl/cert.pem"));
    assert!(tls.ca_path.is_none());
}

#[test]
fn http_tls_options_with_ca_path() {
    let tls = HttpTlsOptions {
        cert_path: PathBuf::from("/cert.pem"),
        key_path: PathBuf::from("/key.pem"),
        ca_path: Some(PathBuf::from("/ca.pem")),
    };
    assert_eq!(tls.ca_path, Some(PathBuf::from("/ca.pem")));
}

// ── SessionState ───────────────────────────────────────────────────────────

#[test]
fn session_state_new_has_unique_uuid() {
    let s1 = SessionState::new();
    let s2 = SessionState::new();
    assert_ne!(
        s1.session_id, s2.session_id,
        "each session must have a unique UUID"
    );
}

#[test]
fn session_state_default_fields() {
    let s = SessionState::new();
    assert!(s.project_id.is_none());
    assert!(s.project_path.is_none());
    assert!(s.watch_path.is_none());
    assert!(!s.is_worktree);
    assert!(!s.daemon_connected);
    assert!(!s.cleaned);
    assert!(s.current_branch.is_none());
}

#[test]
fn session_state_default_trait_matches_new() {
    let s1 = SessionState::default();
    let s2 = SessionState::new();
    // Not the same UUID (each call generates a new one), but all other
    // default fields must be identical.
    assert!(s1.project_id.is_none());
    assert!(!s1.daemon_connected);
    assert!(!s2.daemon_connected);
}

#[test]
fn daemon_connected_transitions() {
    let mut s = SessionState::new();
    assert!(!s.daemon_connected);
    s.set_daemon_connected(true);
    assert!(s.daemon_connected);
    s.set_daemon_connected(false);
    assert!(!s.daemon_connected);
}

#[test]
fn set_project_populates_fields() {
    let mut s = SessionState::new();
    assert!(!s.is_project_registered());
    s.set_project(
        "proj-001".to_string(),
        PathBuf::from("/home/user/proj"),
        None,
    );
    assert!(s.is_project_registered());
    assert_eq!(s.project_id.as_deref(), Some("proj-001"));
    assert_eq!(s.project_path, Some(PathBuf::from("/home/user/proj")));
    // No explicit watch_path → falls back to project_path
    assert_eq!(s.watch_path, Some(PathBuf::from("/home/user/proj")));
}

#[test]
fn set_project_with_explicit_watch_path() {
    let mut s = SessionState::new();
    s.set_project(
        "proj-002".to_string(),
        PathBuf::from("/home/user/proj"),
        Some(PathBuf::from("/realpath/proj")),
    );
    assert_eq!(s.watch_path, Some(PathBuf::from("/realpath/proj")));
}

#[test]
fn clear_project_resets_all_project_fields() {
    let mut s = SessionState::new();
    s.set_project("pid".to_string(), PathBuf::from("/p"), None);
    s.is_worktree = true;
    s.set_branch("main".to_string());
    s.clear_project();
    assert!(!s.is_project_registered());
    assert!(s.project_path.is_none());
    assert!(s.watch_path.is_none());
    assert!(!s.is_worktree);
    assert!(s.current_branch.is_none());
}

#[test]
fn set_branch_stores_branch() {
    let mut s = SessionState::new();
    s.set_branch("feat/my-branch".to_string());
    assert_eq!(s.current_branch.as_deref(), Some("feat/my-branch"));
}

#[test]
fn set_watch_path_overrides() {
    let mut s = SessionState::new();
    s.set_project("p".to_string(), PathBuf::from("/orig"), None);
    s.set_watch_path(PathBuf::from("/resolved"));
    assert_eq!(s.watch_path, Some(PathBuf::from("/resolved")));
}

#[test]
fn is_project_registered_false_when_no_project() {
    let s = SessionState::new();
    assert!(!s.is_project_registered());
}

#[test]
fn session_state_clone() {
    let mut s = SessionState::new();
    s.set_daemon_connected(true);
    s.set_project("p".to_string(), PathBuf::from("/p"), None);
    let s2 = s.clone();
    assert_eq!(s2.session_id, s.session_id);
    assert!(s2.daemon_connected);
    assert_eq!(s2.project_id, s.project_id);
}
