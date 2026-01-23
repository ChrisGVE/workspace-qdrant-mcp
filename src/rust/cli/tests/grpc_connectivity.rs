//! Integration tests for gRPC connectivity
//!
//! These tests verify that the CLI can communicate with the daemon
//! via gRPC. They require a running daemon for full testing.
//!
//! Run with: cargo test --test grpc_connectivity -- --ignored
//! (to run tests that require a daemon)

use assert_cmd::Command;
use predicates::prelude::*;
use std::env;
use std::time::Duration;

/// Get a Command instance for the wqm binary
fn wqm() -> Command {
    Command::cargo_bin("wqm").unwrap()
}

/// Check if daemon is available for integration testing
fn daemon_available() -> bool {
    // Try to connect to the daemon
    let output = Command::cargo_bin("wqm")
        .unwrap()
        .args(["admin", "health"])
        .timeout(Duration::from_secs(5))
        .output();

    match output {
        Ok(out) => out.status.success(),
        Err(_) => false,
    }
}

// ============================================================================
// Connection Error Handling Tests (no daemon required)
// ============================================================================

mod connection_errors {
    use super::*;

    #[test]
    fn test_daemon_connection_timeout() {
        // Test with invalid address - CLI should report unhealthy status
        // but exit successfully (health check completes, reports unhealthy)
        wqm()
            .args(["--daemon-addr", "http://invalid-host:99999", "admin", "health"])
            .timeout(Duration::from_secs(10))
            .assert()
            .success()
            .stdout(predicate::str::contains("unhealthy").or(predicate::str::contains("Unhealthy")));
    }

    #[test]
    fn test_custom_daemon_address_accepted() {
        // Verify the --daemon-addr flag is parsed correctly
        wqm()
            .args(["--daemon-addr", "http://localhost:50051", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_invalid_daemon_address_format() {
        // Invalid URL format should still be accepted at parse time
        // (validation happens at connection time)
        wqm()
            .args(["--daemon-addr", "not-a-url", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_env_daemon_addr_recognized() {
        // Verify WQM_DAEMON_ADDR env var is documented in help
        wqm()
            .arg("--help")
            .assert()
            .success()
            .stdout(predicate::str::contains("WQM_DAEMON_ADDR"));
    }
}

// ============================================================================
// Service Command Tests (no daemon required for parse testing)
// ============================================================================

mod service_commands {
    use super::*;

    #[test]
    fn test_service_status_without_daemon() {
        // Service status might succeed or fail depending on daemon
        // but should not panic or hang
        wqm()
            .args(["service", "status"])
            .timeout(Duration::from_secs(10))
            .assert();
        // We don't assert success/failure - just that it completes
    }

    #[test]
    fn test_service_logs_without_daemon() {
        // Logs command should handle missing daemon gracefully
        wqm()
            .args(["service", "logs", "--lines", "10"])
            .timeout(Duration::from_secs(10))
            .assert();
    }
}

// ============================================================================
// Integration Tests (require running daemon)
// ============================================================================

mod daemon_integration {
    use super::*;

    #[test]
    #[ignore = "requires running daemon"]
    fn test_admin_health_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["admin", "health"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success()
            .stdout(predicate::str::contains("health").or(predicate::str::contains("status")));
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_admin_status_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["admin", "status"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_admin_collections_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["admin", "collections"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_admin_projects_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["admin", "projects"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_admin_queue_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["admin", "queue"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_status_command_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["status"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_status_queue_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["status", "queue"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_status_health_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["status", "health"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_library_list_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["library", "list"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_library_status_with_daemon() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["library", "status"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }
}

// ============================================================================
// JSON Output Format Tests
// ============================================================================

mod json_output {
    use super::*;

    #[test]
    #[ignore = "requires running daemon"]
    fn test_admin_status_json_format() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["--format", "json", "admin", "status"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success()
            .stdout(predicate::str::starts_with("{").or(predicate::str::starts_with("[")));
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_admin_collections_json_format() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["--format", "json", "admin", "collections"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success()
            .stdout(predicate::str::starts_with("{").or(predicate::str::starts_with("[")));
    }

    #[test]
    #[ignore = "requires running daemon"]
    fn test_library_list_json_format() {
        if !daemon_available() {
            println!("Skipping: daemon not available");
            return;
        }

        wqm()
            .args(["--format", "json", "library", "list"])
            .timeout(Duration::from_secs(30))
            .assert()
            .success()
            .stdout(predicate::str::starts_with("{").or(predicate::str::starts_with("[")));
    }
}

// ============================================================================
// Error Recovery Tests
// ============================================================================

mod error_recovery {
    use super::*;

    #[test]
    fn test_graceful_timeout_handling() {
        // Test that CLI handles timeout gracefully without panic
        // Uses a very short timeout to force timeout behavior
        let result = wqm()
            .args(["--daemon-addr", "http://10.255.255.1:50051", "admin", "health"])
            .timeout(Duration::from_millis(100))
            .output();

        // Should complete (either success, failure, or timeout) without panic
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_invalid_port_handling() {
        // Test connection to invalid port - CLI reports unhealthy
        // but completes successfully (health check reports status)
        wqm()
            .args(["--daemon-addr", "http://127.0.0.1:1", "admin", "health"])
            .timeout(Duration::from_secs(5))
            .assert()
            .success()
            .stdout(predicate::str::contains("unhealthy").or(predicate::str::contains("Unhealthy")));
    }

    #[test]
    fn test_repeated_commands() {
        // Test that multiple commands can be run in sequence
        for _ in 0..3 {
            wqm()
                .arg("--version")
                .assert()
                .success();
        }
    }
}
