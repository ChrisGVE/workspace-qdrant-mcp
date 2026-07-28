//! The walking skeleton's first proof: `status` answered end to end.
//!
//! These tests drive the REAL binary over its real transport -- a subprocess, its
//! stdin and stdout, newline-delimited JSON-RPC -- rather than calling the module
//! in-process. That is deliberate: `P04-GT001-WO011`'s acceptance is about the
//! path (`S2 bin -> N49 client seam -> N24 gRPC -> N48 serve loop and back`), and
//! a test that skips the process boundary proves the modules, not the path. v0.1's
//! `QdrantTestContainer` is the cautionary case -- it type-checked for a year and
//! could never have worked, because nothing ever ran it (`WO007`).
//!
//! Two cases, and the first is the load-bearing one:
//!
//! 1. **No daemon.** The answer is `ok:true` with `daemon.reachable:false` -- not
//!    a crash, not an error, and not a lie. MCP-SURFACE.md §4.3 excludes `status`
//!    from `backend_unavailable` precisely so that the tool which exists to report
//!    a down backend can still answer when the backend is down.
//! 2. **A daemon.** The same call crosses the whole pipe and comes back with the
//!    daemon's own report.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};
use wqm_test_harness::{announce_skip, SkipReason};

/// The MCP server binary under test.
const MCP_BIN: &str = env!("CARGO_BIN_EXE_workspace-qdrant-mcp-v2");

/// The seven keys MCP-SURFACE.md §3.1 requires on every tool result, every time.
const ENVELOPE_KEYS: [&str; 7] = [
    "ok",
    "tool",
    "elapsed_ms",
    "defaults_applied",
    "notices",
    "data",
    "error",
];

#[test]
fn status_answers_that_the_daemon_is_absent_rather_than_failing() {
    let dir = tempdir();
    // A path nobody is serving. Not a stale socket -- nothing was ever there.
    let socket = dir.join("absent.sock");

    let envelope = call_status(&socket, &[]);

    for key in ENVELOPE_KEYS {
        assert!(
            envelope.get(key).is_some(),
            "the envelope is missing `{key}`: {envelope}"
        );
    }
    assert_eq!(
        envelope.as_object().unwrap().len(),
        7,
        "exactly seven keys, always: {envelope}"
    );
    assert_eq!(envelope["ok"], json!(true), "an absent daemon is an ANSWER");
    assert_eq!(envelope["error"], Value::Null);

    let daemon = &envelope["data"]["daemon"];
    assert_eq!(daemon["reachable"], json!(false));
    assert_eq!(
        daemon["detail"],
        json!("daemon_unreachable"),
        "the reason carries the same identifier telemetry records (§4.5 rule 1)"
    );

    // Absent, not zero-filled: this build has no index subsystem, and reporting
    // `files_tracked: 0` would be indistinguishable from an empty index.
    assert_eq!(envelope["data"]["index"], Value::Null);
    // The tools this build serves are named, and no other is claimed. The list
    // grew from one to two at `WO013`; the sealed inventory is still eight, and
    // the gap between two and eight is what this assertion protects.
    assert_eq!(
        envelope["data"]["capabilities"]["tools"],
        json!(["status", "query"]),
        "the surface may not advertise what it cannot serve"
    );
}

#[test]
fn status_crosses_the_whole_pipe_when_a_daemon_is_listening() {
    let Some(daemon_bin) = daemon_binary() else {
        let reason = SkipReason::new(format!(
            "the daemon binary is not built beside {MCP_BIN}; run `cargo build --workspace` first"
        ));
        announce_skip(
            "status_crosses_the_whole_pipe_when_a_daemon_is_listening",
            &reason,
        );
        return;
    };

    let dir = tempdir();
    let socket = dir.join("daemon.sock");
    let mut daemon = Daemon::start(&daemon_bin, &socket);

    let envelope = call_status(&socket, &[]);
    daemon.stop();

    assert_eq!(envelope["ok"], json!(true), "{envelope}");
    let block = &envelope["data"]["daemon"];
    assert_eq!(block["reachable"], json!(true), "{envelope}");
    assert_eq!(block["state"], json!("ok"), "{envelope}");
    assert!(
        block["since"].as_i64().is_some_and(|s| s > 0),
        "the daemon reports when it started: {envelope}"
    );
}

#[test]
fn an_unsupported_protocol_is_downgraded_and_the_client_is_told() {
    let dir = tempdir();
    let socket = dir.join("absent.sock");

    // AS-F003: v0.1 echoed `2024-10-07`, a version that has never existed.
    let envelope = call_status(&socket, &[("protocolVersion", "2024-10-07")]);

    let notices = envelope["notices"].as_array().expect("notices is an array");
    assert_eq!(
        notices.len(),
        1,
        "one notice, on the first call: {envelope}"
    );
    assert_eq!(notices[0]["code"], json!("protocol_downgraded"));
    assert_eq!(notices[0]["details"]["requested"], json!("2024-10-07"));
    assert_eq!(notices[0]["details"]["served"], json!("2026-07-28"));

    // And the handshake block reports both halves, so the fact survives the notice.
    assert_eq!(
        envelope["data"]["handshake"]["protocol_requested"],
        json!("2024-10-07")
    );
    assert_eq!(
        envelope["data"]["handshake"]["protocol_served"],
        json!("2026-07-28")
    );
}

/// Run one MCP session against `socket`: initialize, call `status`, return the
/// envelope from `structuredContent`.
fn call_status(socket: &Path, initialize_extras: &[(&str, &str)]) -> Value {
    let mut params = json!({"clientInfo": {"name": "wo011-test", "version": "0"}});
    for (key, value) in initialize_extras {
        params[*key] = json!(value);
    }

    let mut child = Command::new(MCP_BIN)
        .arg("--daemon-socket")
        .arg(socket)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the MCP binary starts");

    {
        let stdin = child.stdin.as_mut().expect("stdin is piped");
        writeln!(
            stdin,
            "{}",
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": params})
        )
        .expect("initialize is written");
        writeln!(
            stdin,
            "{}",
            json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "status", "arguments": {}}})
        )
        .expect("the tool call is written");
    }
    // Closing stdin ends the session, so the child exits on its own.
    drop(child.stdin.take());

    let output = child.wait_with_output().expect("the session ends");
    assert!(
        output.status.success(),
        "the server exited {:?}; stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut responses = Vec::new();
    for line in BufReader::new(output.stdout.as_slice()).lines() {
        let line = line.expect("stdout is text");
        if line.trim().is_empty() {
            continue;
        }
        responses
            .push(serde_json::from_str::<Value>(&line).expect("each line is one JSON-RPC message"));
    }
    assert_eq!(
        responses.len(),
        2,
        "one response per request: {responses:?}"
    );

    responses[1]["result"]["structuredContent"].clone()
}

/// A running daemon, stopped on drop so a failing assertion cannot leave one
/// behind holding a socket.
struct Daemon(Option<Child>);

impl Daemon {
    fn start(binary: &Path, socket: &Path) -> Self {
        let mut child = Command::new(binary)
            .arg("--socket")
            .arg(socket)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the daemon binary starts");

        // Wait for the socket rather than sleeping a guessed interval: the file
        // appearing IS the readiness signal, and waiting on the real signal is
        // what `WO007` proved matters (v0.1 polled an endpoint that 404s).
        for _ in 0..200 {
            if socket.exists() {
                return Daemon(Some(child));
            }
            if let Ok(Some(status)) = child.try_wait() {
                panic!("the daemon exited before binding its socket: {status:?}");
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let _ = child.kill();
        panic!("the daemon did not bind {} within 5s", socket.display());
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The daemon binary, built beside the one under test.
fn daemon_binary() -> Option<PathBuf> {
    let candidate = Path::new(MCP_BIN).parent()?.join("memexd-v2");
    candidate.exists().then_some(candidate)
}

/// A private directory for this test's sockets. A Unix socket path is limited to
/// ~104 bytes on macOS, so it stays short.
fn tempdir() -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "wqm-wo011-{}-{}",
        std::process::id(),
        // Distinct per test without a random source: the address of a local is
        // unique among live frames, which is all that is needed here.
        &format!("{:p}", &0u8)[2..8]
    ));
    std::fs::create_dir_all(&base).expect("the test directory is created");
    base
}
