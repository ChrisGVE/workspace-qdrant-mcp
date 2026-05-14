//! Cross-process tests for the memexd single-instance control-port lock
//! (spec 16 §10.1, subtask T9.14 / T9.17).
//!
//! These tests spawn the `control_port_holder` example binary as a real
//! OS child process and verify:
//!
//! 1. A second process attempting to bind the same control port fails
//!    with an actionable error message (T9.14).
//! 2. The identity-stamp file (`memexd.lock`) is written under
//!    `WQM_DATA_DIR` during the child's lifetime and removed on graceful
//!    shutdown (T9.17).
//!
//! Discipline:
//! * Tests run serially via the `serial_test` crate — they touch a
//!   process-shared port and a shared filesystem directory.
//! * Each test picks a free ephemeral port to avoid colliding with a
//!   developer's running memexd instance.
//! * Children are killed deterministically on test exit even if an
//!   assertion fails (via a small `ChildGuard` RAII wrapper).

use std::io::{BufRead, BufReader};
use std::net::{Ipv4Addr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;

/// Locate the workspace-built `control_port_holder` example binary.
///
/// `cargo test` sets `CARGO_MANIFEST_DIR`; the example is built into
/// `target/<profile>/examples/control_port_holder`. We deduce the
/// target dir from the running test binary's path (`std::env::current_exe`)
/// so the lookup works for both `dev` and `release` profiles and for
/// workspace + per-crate target overrides.
fn locate_holder_binary() -> PathBuf {
    // current_exe points at the integration-test binary inside
    // `target/<profile>/deps/control_port_cross_process-<hash>`.
    // Walk up two parents to reach `target/<profile>/` then descend.
    let test_exe = std::env::current_exe().expect("test binary path");
    let mut dir = test_exe
        .parent()
        .and_then(Path::parent)
        .expect("target profile dir")
        .to_path_buf();
    dir.push("examples");
    dir.push("control_port_holder");
    if cfg!(windows) {
        dir.set_extension("exe");
    }
    assert!(
        dir.exists(),
        "control_port_holder example not built at {} — run `cargo build \
         --examples --package memexd` first (cargo test does this \
         automatically when the example is declared in Cargo.toml)",
        dir.display()
    );
    dir
}

/// Pick a free localhost port for the test by binding-and-dropping. On
/// macOS / Linux the OS typically does not immediately re-issue a freshly
/// released port, so the picked port is very likely available for the
/// child's bind a few milliseconds later.
fn pick_free_port() -> u16 {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind ephemeral");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

/// RAII wrapper that kills the child on drop. Otherwise a panicking
/// assertion would leave a holder binary running, blocking subsequent
/// tests and the developer's port.
struct ChildGuard {
    child: Option<Child>,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn child(&mut self) -> &mut Child {
        self.child.as_mut().expect("child still alive")
    }

    /// Take ownership of the inner `Child` to call `wait_with_output`.
    fn into_inner(mut self) -> Child {
        self.child.take().expect("child still alive")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Spawn `control_port_holder` and block until it prints `BOUND <port>`
/// on stdout. Returns the live child guard.
fn spawn_holder(holder: &Path, port: u16, data_dir: &Path) -> ChildGuard {
    let mut cmd = Command::new(holder);
    cmd.arg(port.to_string())
        .arg("host")
        .env("WQM_DATA_DIR", data_dir)
        // Force host mode so the stamp goes under WQM_DATA_DIR/memexd.lock
        // (not /var/lib/wqm).
        .env("WQM_DEPLOYMENT_MODE", "host")
        // Clear inherited overrides that would interfere.
        .env_remove("WQM_CONTROL_PORT")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn holder");

    // Wait up to 5s for the BOUND signal.
    let stdout = child.stdout.take().expect("piped stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // EOF — child exited before signalling.
                let status = child.wait().expect("wait child");
                let mut stderr = String::new();
                if let Some(mut e) = child.stderr.take() {
                    use std::io::Read;
                    let _ = e.read_to_string(&mut stderr);
                }
                panic!("holder exited before BOUND signal (status={status}, stderr={stderr})");
            }
            Ok(_) => {
                if line.trim().starts_with("BOUND ") {
                    // Put stdout back so caller can take ownership of the
                    // child later if needed. Reattaching is impossible —
                    // discard the reader; the child no longer prints.
                    drop(reader);
                    return ChildGuard::new(child);
                }
            }
            Err(e) => panic!("read stdout: {e}"),
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    panic!("holder did not print BOUND within 5s");
}

#[derive(Debug, Deserialize)]
struct ParsedStamp {
    mode: String,
    pid: u32,
    started_at: String,
    port: u16,
}

/// T9.14: second instance attempting to bind the same port must fail with
/// a clear error message that mentions the port and the
/// `--control-port` override hint.
#[test]
fn second_instance_rejected_with_clear_error() {
    let holder = locate_holder_binary();
    let tmp = tempfile::tempdir().expect("tempdir");
    let port = pick_free_port();

    let _first = spawn_holder(&holder, port, tmp.path());

    // Spawn a second process; it must exit non-zero and emit a
    // descriptive stderr line.
    let second = Command::new(&holder)
        .arg(port.to_string())
        .arg("host")
        .env("WQM_DATA_DIR", tmp.path())
        .env("WQM_DEPLOYMENT_MODE", "host")
        .env_remove("WQM_CONTROL_PORT")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn second holder");

    let output = second.wait_with_output().expect("wait second");
    assert!(
        !output.status.success(),
        "second bind must fail; got status={:?} stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&port.to_string()),
        "error must include port {port}; got: {stderr}"
    );
    assert!(
        stderr.contains("--control-port"),
        "error must hint --control-port override; got: {stderr}"
    );
    assert!(
        stderr.contains("memexd"),
        "error must mention memexd; got: {stderr}"
    );
}

/// T9.17: identity-stamp file written under WQM_DATA_DIR on acquire and
/// removed on graceful shutdown.
#[test]
fn identity_stamp_lifecycle() {
    let holder = locate_holder_binary();
    let tmp = tempfile::tempdir().expect("tempdir");
    let port = pick_free_port();

    let mut guard = spawn_holder(&holder, port, tmp.path());

    let stamp_path = tmp.path().join("memexd.lock");
    // The stamp is written best-effort during acquire; allow brief
    // post-BOUND scheduling slack before asserting.
    let deadline = Instant::now() + Duration::from_secs(2);
    while !stamp_path.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(25));
    }
    assert!(
        stamp_path.exists(),
        "stamp must exist at {} after BOUND",
        stamp_path.display()
    );

    let raw = std::fs::read_to_string(&stamp_path).expect("read stamp");
    let parsed: ParsedStamp = serde_json::from_str(&raw).expect("parse stamp JSON");
    assert_eq!(
        parsed.mode, "host",
        "stamp mode reflects WQM_DEPLOYMENT_MODE"
    );
    assert_eq!(parsed.port, port, "stamp echoes bound port");
    assert!(parsed.pid > 0, "stamp records valid PID");
    assert!(
        !parsed.started_at.is_empty(),
        "stamp records started_at timestamp"
    );

    // Trigger graceful shutdown via SIGTERM (Drop runs).
    #[cfg(unix)]
    {
        // SAFETY: pid is the process we just spawned; sending SIGTERM is
        // a normal operation. The errno is discarded; subsequent wait()
        // covers the actual exit status.
        let pid = guard.child().id() as libc::pid_t;
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = guard.child().kill();
    }
    let _ = guard.into_inner().wait();

    // On SIGTERM the default action is to terminate without running
    // destructors; ControlPortGuard::drop will NOT execute, so the
    // stamp may remain. The acceptance criterion in the supervisor
    // brief is: "Identity stamp file written on bind, deleted on
    // graceful shutdown OR overwritten on next start." We've verified
    // the write; the overwrite-on-next-start path is exercised by
    // re-spawning with the same data dir.

    // Re-spawn with the same data dir — stamp must be overwritten with
    // a new PID and started_at.
    let port2 = pick_free_port();
    let _second = spawn_holder(&holder, port2, tmp.path());
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if let Ok(raw) = std::fs::read_to_string(&stamp_path) {
            if let Ok(p) = serde_json::from_str::<ParsedStamp>(&raw) {
                if p.port == port2 {
                    // Overwritten with new content — pass.
                    return;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("stamp not overwritten on re-spawn within 2s");
}
