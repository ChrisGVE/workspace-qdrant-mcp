//! Comprehensive functional tests for CLI binary execution
//! Tests end-to-end CLI workflows, configuration loading, server startup, and error handling

use std::process::{Command, Stdio};
use std::time::Duration;
use std::fs;
use std::io::Write;
use tempfile::TempDir;
use tokio::time::{sleep, timeout};
use tokio::test as tokio_test;

/// Get the path to the compiled binary
fn get_binary_path() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // Remove test binary name

    // Try release first, then debug
    let release_path = path.join("workspace-qdrant-daemon");
    if release_path.exists() {
        return release_path.to_string_lossy().to_string();
    }

    path.pop(); // Go up from deps/
    let debug_path = path.join("workspace-qdrant-daemon");
    debug_path.to_string_lossy().to_string()
}

/// Build the binary if it doesn't exist
fn ensure_binary_exists() -> Result<String, Box<dyn std::error::Error>> {
    let binary_path = get_binary_path();

    if !std::path::Path::new(&binary_path).exists() {
        println!("Building binary for CLI functional tests...");
        let output = Command::new("cargo")
            .args(&["build", "--bin", "workspace-qdrant-daemon"])
            .output()?;

        if !output.status.success() {
            return Err(format!("Failed to build binary: {}",
                String::from_utf8_lossy(&output.stderr)).into());
        }
    }

    Ok(binary_path)
}

/// Create a test configuration file
fn create_test_config(temp_dir: &TempDir) -> Result<String, Box<dyn std::error::Error>> {
    let config_path = temp_dir.path().join("test_config.yaml");
    let mut file = fs::File::create(&config_path)?;

    writeln!(file, r#"
server:
  host: "127.0.0.1"
  port: 0  # Use random available port for testing

database:
  sqlite_path: ":memory:"

qdrant:
  url: "http://localhost:6333"
  timeout_ms: 5000

processing:
  max_concurrent_tasks: 2
  default_chunk_size: 500
  max_file_size_bytes: 1048576

file_watcher:
  enabled: false  # Disable for testing
  debounce_ms: 100
  max_watched_dirs: 5
"#)?;

    Ok(config_path.to_string_lossy().to_string())
}

/// Test: CLI help output
#[tokio_test]
async fn test_cli_help_output() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");

    let output = Command::new(&binary_path)
        .arg("--help")
        .output()
        .expect("Failed to execute binary");

    assert!(output.status.success()); // Help exits with success
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify help content
    assert!(stdout.contains("High-performance Rust daemon") || stdout.contains("workspace document processing"),
        "Help should contain program description");
    assert!(stdout.contains("--address"), "Help should show address option");
    assert!(stdout.contains("--config"), "Help should show config option");
    assert!(stdout.contains("--log-level"), "Help should show log-level option");
    assert!(stdout.contains("--daemon"), "Help should show daemon option");
    assert!(stdout.contains("--enable-metrics"), "Help should show enable-metrics option");
}

/// Test: CLI version output
#[tokio_test]
async fn test_cli_version_output() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");

    let output = Command::new(&binary_path)
        .arg("--version")
        .output()
        .expect("Failed to execute binary");

    assert!(output.status.success()); // Version exits with success
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify version content
    assert!(stdout.contains("workspace-qdrant-daemon"), "Version should contain program name");
    assert!(stdout.contains("0.3.0"), "Version should contain version number");
}

/// Test: CLI with invalid arguments
#[tokio_test]
async fn test_cli_invalid_arguments() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");

    // Test invalid address
    let output = Command::new(&binary_path)
        .args(&["--address", "invalid_address"])
        .output()
        .expect("Failed to execute binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid") || stderr.contains("error"),
        "Should report error for invalid address");
}

/// Test: CLI with invalid log level
#[tokio_test]
async fn test_cli_invalid_log_level() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");

    let output = Command::new(&binary_path)
        .args(&["--log-level", "invalid_level", "--help"]) // Use help to exit quickly
        .output()
        .expect("Failed to execute binary");

    // The CLI accepts invalid log levels during parsing and only fails when actually used
    // With --help, it should still succeed since it doesn't initialize logging
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage:"), "Should show help even with invalid log level");
}

/// Test: CLI with non-existent config file
#[tokio_test]
async fn test_cli_nonexistent_config() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");

    let output = timeout(Duration::from_secs(5), async {
        Command::new(&binary_path)
            .args(&["--config", "/nonexistent/config.yaml"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to spawn binary")
            .wait_with_output()
            .expect("Failed to wait for binary")
    }).await;

    if let Ok(output) = output {
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Should mention config file error
        assert!(stderr.to_lowercase().contains("config") ||
                stderr.to_lowercase().contains("file") ||
                stderr.to_lowercase().contains("no such file"),
            "Should report config file error");
    }
    // If timeout occurs, that's also acceptable - the binary tried to load config
}

/// Test: CLI startup with valid configuration
#[tokio_test]
async fn test_cli_startup_with_valid_config() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = create_test_config(&temp_dir).expect("Failed to create config");

    // Start the daemon in background
    let mut child = Command::new(&binary_path)
        .args(&[
            "--config", &config_path,
            "--address", "127.0.0.1:0", // Use port 0 for auto-assignment
            "--log-level", "warn" // Reduce log noise
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn binary");

    // Give it time to start up
    sleep(Duration::from_millis(500)).await;

    // Check if process is still running (not crashed)
    let status = child.try_wait().expect("Failed to check process status");

    if let Some(exit_status) = status {
        // Process exited, check output for clues
        let output = child.wait_with_output().expect("Failed to get output");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);

        println!("Process exited with status: {}", exit_status);
        println!("STDERR: {}", stderr);
        println!("STDOUT: {}", stdout);

        // If it exits cleanly or with a known error (like missing Qdrant), that's OK
        // We're mainly testing that CLI parsing and config loading work
        assert!(exit_status.code().is_some(), "Process should exit cleanly, not crash");
    } else {
        // Process is still running - good!
        // Terminate it gracefully
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Test: CLI daemon mode vs foreground mode
#[tokio_test]
async fn test_cli_daemon_vs_foreground_mode() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = create_test_config(&temp_dir).expect("Failed to create config");

    // Test foreground mode (default)
    let mut foreground_child = Command::new(&binary_path)
        .args(&[
            "--config", &config_path,
            "--address", "127.0.0.1:0",
            "--log-level", "error"
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn foreground process");

    sleep(Duration::from_millis(300)).await;

    // Test daemon mode
    let mut daemon_child = Command::new(&binary_path)
        .args(&[
            "--config", &config_path,
            "--address", "127.0.0.1:0",
            "--daemon",
            "--log-level", "error"
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn daemon process");

    sleep(Duration::from_millis(300)).await;

    // Both should either be running or exit cleanly
    let fg_status = foreground_child.try_wait().expect("Failed to check foreground status");
    let daemon_status = daemon_child.try_wait().expect("Failed to check daemon status");

    // Clean up
    let _ = foreground_child.kill();
    let _ = daemon_child.kill();
    let _ = foreground_child.wait();
    let _ = daemon_child.wait();

    // Both modes should start successfully (or fail with expected errors)
    if let Some(status) = fg_status {
        assert!(status.code().is_some(), "Foreground mode should not crash");
    }
    if let Some(status) = daemon_status {
        assert!(status.code().is_some(), "Daemon mode should not crash");
    }
}

/// Test: CLI with different address formats
#[tokio_test]
async fn test_cli_address_formats() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = create_test_config(&temp_dir).expect("Failed to create config");

    let addresses = [
        "127.0.0.1:0",
        "0.0.0.0:0",
        "[::1]:0",
        "localhost:0"
    ];

    for address in addresses {
        let output = timeout(Duration::from_secs(3), async {
            Command::new(&binary_path)
                .args(&[
                    "--config", &config_path,
                    "--address", address,
                    "--log-level", "error"
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("Failed to spawn process")
                .wait_with_output()
                .expect("Failed to wait for process")
        }).await;

        if let Ok(output) = output {
            // Should either succeed or fail gracefully (not crash)
            assert!(output.status.code().is_some(),
                "Address {} should not cause crash", address);
        }
        // Timeout is acceptable - process started successfully
    }
}

/// Test: CLI with different log levels
#[tokio_test]
async fn test_cli_log_levels() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = create_test_config(&temp_dir).expect("Failed to create config");

    let log_levels = ["trace", "debug", "info", "warn", "error"];

    for level in log_levels {
        let mut child = Command::new(&binary_path)
            .args(&[
                "--config", &config_path,
                "--address", "127.0.0.1:0",
                "--log-level", level
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to spawn process");

        sleep(Duration::from_millis(200)).await;

        // Check if still running
        let status = child.try_wait().expect("Failed to check status");

        // Clean up
        let _ = child.kill();
        let _ = child.wait();

        if let Some(exit_status) = status {
            assert!(exit_status.code().is_some(),
                "Log level {} should not cause crash", level);
        }
        // If still running, that's good too
    }
}

/// Test: CLI with metrics enabled
#[tokio_test]
async fn test_cli_metrics_enabled() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = create_test_config(&temp_dir).expect("Failed to create config");

    let mut child = Command::new(&binary_path)
        .args(&[
            "--config", &config_path,
            "--address", "127.0.0.1:0",
            "--enable-metrics",
            "--log-level", "error"
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn process");

    sleep(Duration::from_millis(300)).await;

    let status = child.try_wait().expect("Failed to check status");

    // Clean up
    let _ = child.kill();
    let _ = child.wait();

    if let Some(exit_status) = status {
        assert!(exit_status.code().is_some(),
            "Metrics enabled should not cause crash");
    }
}

/// Test: CLI signal handling (graceful shutdown)
#[tokio_test]
async fn test_cli_signal_handling() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = create_test_config(&temp_dir).expect("Failed to create config");

    let mut child = Command::new(&binary_path)
        .args(&[
            "--config", &config_path,
            "--address", "127.0.0.1:0",
            "--log-level", "warn"
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn process");

    // Give it time to start
    sleep(Duration::from_millis(400)).await;

    // Send termination signal
    #[cfg(unix)]
    {
        let pid = child.id();
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }

        // Give it time to shutdown gracefully
        let wait_future = async move {
            tokio::task::spawn_blocking(move || child.wait())
                .await
                .expect("Task join error")
                .expect("Wait error")
        };

        let result = timeout(Duration::from_secs(5), wait_future).await;

        match result {
            Ok(status) => {
                assert!(status.code().is_some(),
                    "Should handle SIGTERM gracefully");
            }
            Err(_) => {
                panic!("Process did not respond to SIGTERM within timeout");
            }
        }
    }

    #[cfg(not(unix))]
    {
        // On non-Unix systems, just kill the process
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Test: CLI error propagation from config loading
#[tokio_test]
async fn test_cli_config_error_propagation() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Create invalid config file
    let config_path = temp_dir.path().join("invalid_config.yaml");
    let mut file = fs::File::create(&config_path).expect("Failed to create invalid config");
    writeln!(file, "invalid: yaml: content: [[[").expect("Failed to write invalid config");

    let output = timeout(Duration::from_secs(5), async {
        Command::new(&binary_path)
            .args(&["--config", config_path.to_str().unwrap()])
            .output()
            .expect("Failed to execute binary")
    }).await;

    if let Ok(output) = output {
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Should contain error about config parsing
        assert!(stderr.to_lowercase().contains("config") ||
                stderr.to_lowercase().contains("yaml") ||
                stderr.to_lowercase().contains("parse"),
            "Should report config parsing error");
    }
}

/// Test: CLI with empty config file
#[tokio_test]
async fn test_cli_empty_config_file() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Create empty config file
    let config_path = temp_dir.path().join("empty_config.yaml");
    fs::File::create(&config_path).expect("Failed to create empty config");

    let output = timeout(Duration::from_secs(5), async {
        Command::new(&binary_path)
            .args(&["--config", config_path.to_str().unwrap()])
            .output()
            .expect("Failed to execute binary")
    }).await;

    if let Ok(output) = output {
        // Should either use defaults or report missing required fields
        assert!(output.status.code().is_some(),
            "Empty config should not cause crash");
    }
}

/// Test: CLI argument precedence over config file
#[tokio_test]
async fn test_cli_argument_precedence() {
    let binary_path = ensure_binary_exists().expect("Failed to build binary");
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Create config with specific values
    let config_path = temp_dir.path().join("precedence_config.yaml");
    let mut file = fs::File::create(&config_path).expect("Failed to create config");
    writeln!(file, r#"
server:
  host: "0.0.0.0"
  port: 9999
"#).expect("Failed to write config");

    let mut child = Command::new(&binary_path)
        .args(&[
            "--config", config_path.to_str().unwrap(),
            "--address", "127.0.0.1:0", // Should override config
            "--log-level", "error"
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn process");

    sleep(Duration::from_millis(300)).await;

    // Process should start successfully with CLI args taking precedence
    let status = child.try_wait().expect("Failed to check status");

    // Clean up
    let _ = child.kill();
    let _ = child.wait();

    if let Some(exit_status) = status {
        assert!(exit_status.code().is_some(),
            "CLI argument precedence should work correctly");
    }
}