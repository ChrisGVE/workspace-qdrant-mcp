//! AC-T2: in stdio mode, stdout must carry only JSON-RPC frames — never log
//! lines.  This test runs the compiled `workspace-qdrant-mcp` binary (which currently
//! initialises logging in stdio mode and emits its startup banner) and asserts
//! that the banner went to stderr, leaving stdout free of log contamination.
//!
//! The full request/response purity test (invoke a tool, assert every stdout
//! line parses as a JSON-RPC frame) lands with the stdio transport in task 31;
//! this guards the logging-sink invariant for the current entrypoint.

use std::process::Command;

/// Path to the compiled binary under test, provided by Cargo to integration
/// tests via the `CARGO_BIN_EXE_<name>` environment variable.
fn binary_path() -> &'static str {
    env!("CARGO_BIN_EXE_workspace-qdrant-mcp")
}

#[test]
fn stdio_mode_stdout_has_no_log_contamination() {
    let output = Command::new(binary_path())
        .env("WQM_LOG_LEVEL", "info")
        // Ensure no inherited override forces a different sink.
        .env_remove("WQM_MCP_LOG_LEVEL")
        .output()
        .expect("failed to run workspace-qdrant-mcp binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Core guarantee: stdout is empty (no banner, no log line) in stdio mode.
    assert!(
        stdout.trim().is_empty(),
        "stdio-mode stdout must be free of log output, got: {stdout:?}"
    );

    // The startup banner must have been routed to stderr instead.
    assert!(
        stderr.contains("workspace-qdrant MCP server starting"),
        "startup banner should appear on stderr, got: {stderr:?}"
    );
}
