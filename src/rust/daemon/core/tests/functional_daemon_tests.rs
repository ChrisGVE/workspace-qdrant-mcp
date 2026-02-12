//! Functional tests for the memexd daemon
//!
//! These tests start the actual `memexd` binary as a subprocess, interact via gRPC,
//! and verify end-to-end behavior. They fill the gap between unit tests and manual testing.
//!
//! Requirements:
//! - The `memexd` binary must be built before running these tests
//! - Docker must be running (for Qdrant via testcontainers)
//!
//! Run with:
//!   cargo build --package memexd
//!   cargo test --package workspace-qdrant-core --features functional_tests \
//!     --test functional_daemon_tests -- --test-threads=1 --nocapture

#![cfg(feature = "functional_tests")]

use std::collections::HashMap;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::sleep;
use tonic::transport::Channel;

// Import gRPC client stubs from the grpc crate
use workspace_qdrant_grpc::proto::{
    system_service_client::SystemServiceClient,
    project_service_client::ProjectServiceClient,
    RegisterProjectRequest,
};

/// Find the memexd binary in the build output directory
fn find_memexd_binary() -> PathBuf {
    // Try debug build first (most common during development)
    let debug_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/debug/memexd");

    if debug_path.exists() {
        return debug_path;
    }

    // Try release build
    let release_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/release/memexd");

    if release_path.exists() {
        return release_path;
    }

    // Try workspace-level target directory
    let workspace_debug = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3) // up to src/rust
        .unwrap()
        .join("target/debug/memexd");

    if workspace_debug.exists() {
        return workspace_debug;
    }

    panic!(
        "memexd binary not found. Build it first with: \
         cd src/rust/daemon && cargo build --package memexd\n\
         Searched: {:?}, {:?}, {:?}",
        debug_path, release_path, workspace_debug
    );
}

/// Find a free TCP port by binding to port 0
fn find_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind to a free port");
    listener.local_addr().unwrap().port()
}

/// Wait for a TCP port to become available (daemon has started)
async fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port))
            .await
            .is_ok()
        {
            return true;
        }
        sleep(Duration::from_millis(100)).await;
    }
    false
}

/// A running daemon subprocess with its configuration
struct DaemonProcess {
    child: Child,
    grpc_port: u16,
    _config_dir: TempDir,
    _db_dir: TempDir,
}

impl DaemonProcess {
    /// Start a new daemon instance with a fresh database
    async fn start() -> anyhow::Result<Self> {
        Self::start_with_env(HashMap::new()).await
    }

    /// Start a new daemon instance with additional environment variables
    async fn start_with_env(extra_env: HashMap<String, String>) -> anyhow::Result<Self> {
        let binary = find_memexd_binary();
        let grpc_port = find_free_port();
        let config_dir = TempDir::new()?;
        let db_dir = TempDir::new()?;
        let db_path = db_dir.path().join("state.db");
        let pid_file = config_dir.path().join("memexd.pid");

        let mut cmd = Command::new(&binary);
        cmd.arg("--foreground")
            .arg("--grpc-port")
            .arg(grpc_port.to_string())
            .arg("--pid-file")
            .arg(pid_file.to_str().unwrap())
            .arg("--log-level")
            .arg("debug")
            .env("WQM_DATABASE_PATH", db_path.to_str().unwrap())
            // Use a non-existent Qdrant URL to prevent real connections in basic tests
            // Tests that need Qdrant will override this
            .env("QDRANT_URL", "http://127.0.0.1:1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        for (key, value) in &extra_env {
            cmd.env(key, value);
        }

        let child = cmd.spawn().map_err(|e| {
            anyhow::anyhow!(
                "Failed to start memexd from {:?}: {}. Build it first.",
                binary,
                e
            )
        })?;

        let process = DaemonProcess {
            child,
            grpc_port,
            _config_dir: config_dir,
            _db_dir: db_dir,
        };

        // Wait for the daemon to start accepting gRPC connections
        if !wait_for_port(grpc_port, Duration::from_secs(30)).await {
            // If daemon didn't start, kill it and report error
            let mut failed = process;
            let _ = failed.child.kill();
            let _ = failed.child.wait();
            // Prevent Drop from double-killing
            std::mem::forget(failed);

            return Err(anyhow::anyhow!(
                "Daemon failed to start within 30 seconds on port {}",
                grpc_port,
            ));
        }

        Ok(process)
    }

    /// Get a gRPC channel connected to this daemon
    async fn grpc_channel(&self) -> anyhow::Result<Channel> {
        let endpoint = format!("http://127.0.0.1:{}", self.grpc_port);
        let channel = Channel::from_shared(endpoint)?
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .connect()
            .await?;
        Ok(channel)
    }

    /// Get a SystemService gRPC client
    async fn system_client(&self) -> anyhow::Result<SystemServiceClient<Channel>> {
        let channel = self.grpc_channel().await?;
        Ok(SystemServiceClient::new(channel))
    }

    /// Get a ProjectService gRPC client
    async fn project_client(&self) -> anyhow::Result<ProjectServiceClient<Channel>> {
        let channel = self.grpc_channel().await?;
        Ok(ProjectServiceClient::new(channel))
    }

    /// Send SIGTERM to the daemon for graceful shutdown
    #[cfg(unix)]
    fn send_sigterm(&self) {
        unsafe {
            libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM);
        }
    }

    /// Stop the daemon gracefully, with SIGKILL fallback
    async fn stop(mut self) -> std::process::ExitStatus {
        #[cfg(unix)]
        {
            self.send_sigterm();
            // Wait up to 10 seconds for graceful shutdown
            for _ in 0..100 {
                if let Ok(Some(status)) = self.child.try_wait() {
                    return status;
                }
                sleep(Duration::from_millis(100)).await;
            }
        }

        // Fallback: kill
        let _ = self.child.kill();
        self.child.wait().expect("Failed to wait for daemon process")
    }
}

impl Drop for DaemonProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// =============================================================================
// Test Scenarios
// =============================================================================

/// Test 1: Start daemon, send gRPC health check, verify response
#[tokio::test]
async fn test_daemon_health_check() {
    let daemon = DaemonProcess::start()
        .await
        .expect("Failed to start daemon");

    let mut client = daemon
        .system_client()
        .await
        .expect("Failed to create gRPC client");

    // Send health check
    let response = client
        .health(())
        .await
        .expect("Health check failed");

    let health = response.into_inner();

    // Verify health response
    // ServiceStatus::HEALTHY = 1
    assert_eq!(
        health.status, 1,
        "Expected SERVICE_STATUS_HEALTHY (1), got {}",
        health.status
    );
    assert!(health.timestamp.is_some(), "Health response should include timestamp");

    // Clean shutdown
    let status = daemon.stop().await;
    assert!(
        status.success(),
        "Daemon should exit cleanly, got: {:?}",
        status
    );
}

/// Test 2: Verify daemon reports system status
#[tokio::test]
async fn test_daemon_system_status() {
    let daemon = DaemonProcess::start()
        .await
        .expect("Failed to start daemon");

    let mut client = daemon
        .system_client()
        .await
        .expect("Failed to create gRPC client");

    // Get system status
    let response = client
        .get_status(())
        .await
        .expect("GetStatus failed");

    let status = response.into_inner();

    // Verify basic status fields
    assert!(
        status.uptime_since.is_some(),
        "System status should include uptime_since"
    );

    let exit_status = daemon.stop().await;
    assert!(exit_status.success());
}

/// Test 3: Verify queue stats are accessible
#[tokio::test]
async fn test_daemon_queue_stats() {
    let daemon = DaemonProcess::start()
        .await
        .expect("Failed to start daemon");

    let mut client = daemon
        .system_client()
        .await
        .expect("Failed to create gRPC client");

    // Get queue stats
    let response = client
        .get_queue_stats(())
        .await
        .expect("GetQueueStats failed");

    let stats = response.into_inner();

    // Fresh daemon should have no pending items
    assert_eq!(
        stats.pending_count, 0,
        "Fresh daemon should have 0 pending queue items"
    );
    assert_eq!(
        stats.in_progress_count, 0,
        "Fresh daemon should have 0 in-progress items"
    );

    let exit_status = daemon.stop().await;
    assert!(exit_status.success());
}

/// Test 4: Register a project via gRPC and verify it's tracked
#[tokio::test]
async fn test_project_registration() {
    let daemon = DaemonProcess::start()
        .await
        .expect("Failed to start daemon");

    let mut client = daemon
        .project_client()
        .await
        .expect("Failed to create gRPC client");

    // Create a temp directory as a "project"
    let project_dir = TempDir::new().expect("Failed to create temp dir");
    let project_path = project_dir.path().to_str().unwrap().to_string();

    // Register the project
    let response = client
        .register_project(RegisterProjectRequest {
            path: project_path.clone(),
            project_id: "abcdef012345".to_string(),
            name: Some("test-project".to_string()),
            git_remote: None,
            register_if_new: true,
        })
        .await
        .expect("RegisterProject failed");

    let result = response.into_inner();
    assert_eq!(result.project_id, "abcdef012345");
    assert!(result.is_active, "Newly registered project should be active");

    // Verify project appears in list
    let list_response = client
        .list_projects(workspace_qdrant_grpc::proto::ListProjectsRequest {
            priority_filter: None,
            active_only: true,
        })
        .await
        .expect("ListProjects failed");

    let projects = list_response.into_inner();
    assert!(
        projects.total_count >= 1,
        "Should have at least 1 registered project"
    );

    let found = projects
        .projects
        .iter()
        .any(|p| p.project_id == "abcdef012345");
    assert!(found, "Registered project should appear in list");

    let exit_status = daemon.stop().await;
    assert!(exit_status.success());
}

/// Test 5: Graceful shutdown via SIGTERM produces clean exit
#[tokio::test]
#[cfg(unix)]
async fn test_graceful_shutdown() {
    let daemon = DaemonProcess::start()
        .await
        .expect("Failed to start daemon");

    // Verify daemon is alive first
    let mut client = daemon
        .system_client()
        .await
        .expect("Failed to create gRPC client");

    let _ = client
        .health(())
        .await
        .expect("Health check before shutdown failed");

    // Send SIGTERM and wait for clean exit
    let status = daemon.stop().await;

    assert!(
        status.success(),
        "Daemon should exit with code 0 after SIGTERM, got: {:?}",
        status.code()
    );
}
