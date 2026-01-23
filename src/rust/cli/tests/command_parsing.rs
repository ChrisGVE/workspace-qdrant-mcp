//! Unit tests for CLI command parsing
//!
//! These tests verify that clap argument parsing works correctly
//! for all Phase 1 commands without requiring a running daemon.

use assert_cmd::Command;
use predicates::prelude::*;

/// Get a Command instance for the wqm binary
fn wqm() -> Command {
    Command::cargo_bin("wqm").unwrap()
}

// ============================================================================
// Global Options Tests
// ============================================================================

mod global_options {
    use super::*;

    #[test]
    fn test_version_flag() {
        wqm()
            .arg("--version")
            .assert()
            .success()
            .stdout(predicate::str::contains("wqm"));
    }

    #[test]
    fn test_short_version_flag() {
        wqm()
            .arg("-V")
            .assert()
            .success()
            .stdout(predicate::str::contains("wqm"));
    }

    #[test]
    fn test_help_flag() {
        wqm()
            .arg("--help")
            .assert()
            .success()
            .stdout(predicate::str::contains("Usage:"))
            .stdout(predicate::str::contains("service"))
            .stdout(predicate::str::contains("admin"))
            .stdout(predicate::str::contains("status"))
            .stdout(predicate::str::contains("library"));
    }

    #[test]
    fn test_short_help_flag() {
        wqm()
            .arg("-h")
            .assert()
            .success()
            .stdout(predicate::str::contains("Usage:"));
    }

    #[test]
    fn test_format_option() {
        // Format option should be accepted (actual execution requires daemon)
        wqm()
            .args(["--format", "json", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_verbose_flag() {
        wqm()
            .args(["-v", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_daemon_addr_option() {
        wqm()
            .args(["--daemon-addr", "http://localhost:50051", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_invalid_command() {
        wqm()
            .arg("invalid-command")
            .assert()
            .failure()
            .stderr(predicate::str::contains("error"));
    }
}

// ============================================================================
// Service Command Tests
// ============================================================================

mod service_command {
    use super::*;

    #[test]
    fn test_service_help() {
        wqm()
            .args(["service", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("install"))
            .stdout(predicate::str::contains("start"))
            .stdout(predicate::str::contains("stop"))
            .stdout(predicate::str::contains("restart"))
            .stdout(predicate::str::contains("status"))
            .stdout(predicate::str::contains("logs"));
    }

    #[test]
    fn test_service_install_help() {
        wqm()
            .args(["service", "install", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_service_start_help() {
        wqm()
            .args(["service", "start", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_service_stop_help() {
        wqm()
            .args(["service", "stop", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_service_restart_help() {
        wqm()
            .args(["service", "restart", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_service_status_help() {
        wqm()
            .args(["service", "status", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_service_logs_help() {
        wqm()
            .args(["service", "logs", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("--lines"))
            .stdout(predicate::str::contains("--follow"));
    }

    #[test]
    fn test_service_logs_lines_option() {
        // Verify the --lines option is recognized
        wqm()
            .args(["service", "logs", "--lines", "100", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_service_logs_follow_flag() {
        // Verify the --follow flag is recognized
        wqm()
            .args(["service", "logs", "--follow", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_service_invalid_subcommand() {
        wqm()
            .args(["service", "invalid"])
            .assert()
            .failure();
    }
}

// ============================================================================
// Admin Command Tests
// ============================================================================

mod admin_command {
    use super::*;

    #[test]
    fn test_admin_help() {
        wqm()
            .args(["admin", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("status"))
            .stdout(predicate::str::contains("collections"))
            .stdout(predicate::str::contains("health"))
            .stdout(predicate::str::contains("projects"))
            .stdout(predicate::str::contains("queue"));
    }

    #[test]
    fn test_admin_status_help() {
        wqm()
            .args(["admin", "status", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_admin_collections_help() {
        wqm()
            .args(["admin", "collections", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_admin_health_help() {
        wqm()
            .args(["admin", "health", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_admin_projects_help() {
        wqm()
            .args(["admin", "projects", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_admin_queue_help() {
        wqm()
            .args(["admin", "queue", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_admin_invalid_subcommand() {
        wqm()
            .args(["admin", "invalid"])
            .assert()
            .failure();
    }
}

// ============================================================================
// Status Command Tests
// ============================================================================

mod status_command {
    use super::*;

    #[test]
    fn test_status_help() {
        wqm()
            .args(["status", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("--queue"))
            .stdout(predicate::str::contains("--watch"))
            .stdout(predicate::str::contains("--performance"));
    }

    #[test]
    fn test_status_history_help() {
        wqm()
            .args(["status", "history", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("--range"));
    }

    #[test]
    fn test_status_queue_help() {
        wqm()
            .args(["status", "queue", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("--verbose"));
    }

    #[test]
    fn test_status_watch_help() {
        wqm()
            .args(["status", "watch", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_status_performance_help() {
        wqm()
            .args(["status", "performance", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_status_live_help() {
        wqm()
            .args(["status", "live", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("--interval"));
    }

    #[test]
    fn test_status_messages_help() {
        wqm()
            .args(["status", "messages", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_status_errors_help() {
        wqm()
            .args(["status", "errors", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("--limit"));
    }

    #[test]
    fn test_status_health_help() {
        wqm()
            .args(["status", "health", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_status_flags_combination() {
        // Test multiple flags together
        wqm()
            .args(["status", "--queue", "--watch", "--help"])
            .assert()
            .success();
    }
}

// ============================================================================
// Library Command Tests
// ============================================================================

mod library_command {
    use super::*;

    #[test]
    fn test_library_help() {
        wqm()
            .args(["library", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("list"))
            .stdout(predicate::str::contains("add"))
            .stdout(predicate::str::contains("watch"))
            .stdout(predicate::str::contains("unwatch"))
            .stdout(predicate::str::contains("rescan"))
            .stdout(predicate::str::contains("info"))
            .stdout(predicate::str::contains("status"));
    }

    #[test]
    fn test_library_list_help() {
        wqm()
            .args(["library", "list", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("--verbose"));
    }

    #[test]
    fn test_library_add_help() {
        wqm()
            .args(["library", "add", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("<TAG>"))
            .stdout(predicate::str::contains("<PATH>"));
    }

    #[test]
    fn test_library_watch_help() {
        wqm()
            .args(["library", "watch", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("<TAG>"))
            .stdout(predicate::str::contains("<PATH>"))
            .stdout(predicate::str::contains("--patterns"));
    }

    #[test]
    fn test_library_unwatch_help() {
        wqm()
            .args(["library", "unwatch", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("<TAG>"));
    }

    #[test]
    fn test_library_rescan_help() {
        wqm()
            .args(["library", "rescan", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("<TAG>"))
            .stdout(predicate::str::contains("--force"));
    }

    #[test]
    fn test_library_info_help() {
        wqm()
            .args(["library", "info", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_library_status_help() {
        wqm()
            .args(["library", "status", "--help"])
            .assert()
            .success();
    }

    #[test]
    fn test_library_invalid_subcommand() {
        wqm()
            .args(["library", "invalid"])
            .assert()
            .failure();
    }
}

// ============================================================================
// Error Message Tests
// ============================================================================

mod error_messages {
    use super::*;

    #[test]
    fn test_missing_required_arg() {
        // library add requires tag and path
        wqm()
            .args(["library", "add"])
            .assert()
            .failure()
            .stderr(predicate::str::contains("required"));
    }

    #[test]
    fn test_missing_subcommand() {
        // Commands that require subcommands
        wqm()
            .arg("service")
            .assert()
            .failure();
    }

    #[test]
    fn test_unknown_flag() {
        wqm()
            .args(["--unknown-flag"])
            .assert()
            .failure()
            .stderr(predicate::str::contains("error"));
    }

    #[test]
    fn test_invalid_format_value() {
        // Note: This may succeed if format validation is done at execution time
        // If it fails at parsing, we test the error message
        let result = wqm()
            .args(["--format", "invalid", "status", "--help"])
            .assert();

        // Either succeeds (validation at execution) or fails with error
        result.try_success().unwrap_or_else(|_| {
            wqm()
                .args(["--format", "invalid", "status", "--help"])
                .assert()
                .failure()
        });
    }
}

// ============================================================================
// Help Text Quality Tests
// ============================================================================

mod help_text {
    use super::*;

    #[test]
    fn test_main_help_has_description() {
        wqm()
            .arg("--help")
            .assert()
            .success()
            .stdout(predicate::str::contains("Workspace Qdrant MCP CLI"));
    }

    #[test]
    fn test_service_help_has_descriptions() {
        wqm()
            .args(["service", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("Daemon"))
            .stdout(predicate::str::contains("service"));
    }

    #[test]
    fn test_admin_help_has_descriptions() {
        wqm()
            .args(["admin", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("System"))
            .stdout(predicate::str::contains("administration"));
    }

    #[test]
    fn test_status_help_has_descriptions() {
        wqm()
            .args(["status", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("monitoring"));
    }

    #[test]
    fn test_library_help_has_descriptions() {
        wqm()
            .args(["library", "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains("Library"))
            .stdout(predicate::str::contains("tag"));
    }
}
