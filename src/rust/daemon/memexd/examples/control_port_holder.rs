//! Test-only helper: bind the memexd control port and stay alive until
//! killed. Used by the integration test
//! `tests/control_port_cross_process.rs` to simulate a running memexd
//! holding the port from a *different* OS process.
//!
//! Usage:
//!   control_port_holder <port> [mode=host|docker]
//!
//! Behaviour:
//! * On successful bind: prints `BOUND <port>` to stdout, flushes, then
//!   sleeps until SIGTERM / SIGKILL / parent process exit.
//! * On bind failure: prints `BIND_FAILED: <error>` to stderr and exits 1.
//!
//! This binary intentionally avoids any other memexd machinery — no
//! tokio runtime, no logging subsystem — so the test only exercises the
//! control-port primitive.

use std::env;
use std::io::Write;
use std::process;
use std::thread;
use std::time::Duration;

use memexd::control_port::{acquire, DeploymentMode};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <port> [host|docker]", args[0]);
        process::exit(2);
    }

    let port: u16 = match args[1].parse() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("invalid port '{}': {}", args[1], e);
            process::exit(2);
        }
    };

    let mode = args
        .get(2)
        .map(|s| s.as_str())
        .map(|s| match s.to_ascii_lowercase().as_str() {
            "docker" => DeploymentMode::Docker,
            _ => DeploymentMode::Host,
        })
        .unwrap_or(DeploymentMode::Host);

    let _guard = match acquire(port, mode) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("BIND_FAILED: {e}");
            process::exit(1);
        }
    };

    // Signal readiness on stdout so the parent test can synchronise.
    println!("BOUND {port}");
    let _ = std::io::stdout().flush();

    // Stay alive — the integration test kills us with SIGTERM/SIGKILL.
    loop {
        thread::sleep(Duration::from_secs(60));
    }
}
