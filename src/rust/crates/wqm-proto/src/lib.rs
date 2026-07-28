//! wqm-proto -- the N24 gRPC contract source of truth (ARCH rev15 §9.1).
//!
//! Owns the `.proto` definitions, the code generated from them, and the [`Address`]
//! enum. It declares no wqm-crate dependencies (generated code only) so it can be
//! linked by every surface without dragging in a closure.
//!
//! `P04-GT001-WO011` brings forward the part of N24's slice (`P04-GT011`) the
//! walking skeleton needs: one service with one RPC, the daemon reporting itself.
//! The remaining services arrive with that slice; the debt is recorded in
//! `SCAFFOLD.md` §7.

use std::fmt;
use std::path::PathBuf;

/// The generated code. `clippy::all` is allowed here and nowhere else: the
/// workspace denies warnings, and a lint against machine-written code is a lint
/// against `tonic-prost-build`, not against this workspace.
#[allow(clippy::all, missing_docs)]
pub mod v1 {
    tonic::include_proto!("wqm.v1");
}

/// Where a daemon listens, and therefore where a client dials (ARCH rev15 §9.1:
/// "`Address` enum (`Uds(path) | Tcp(addr)`), UDS-0600 default").
///
/// **A Unix socket is the default for a reason the skeleton makes concrete.** A
/// TCP port is a scarce, machine-global name: two daemons that default to the same
/// port collide, which is exactly the hazard `P04-GT001-WO010` measured on v0.1's
/// control port. A socket path is a filesystem name, so it derives from the
/// deployment directory that already carries the `-v2` suffix and cannot collide
/// with production by construction. TCP stays in the enum because a remote daemon
/// is a declared deployment (ARCH §9.2), not because it is the norm.
///
/// There is deliberately **no `Default`**: an address is always supplied by the
/// caller, so no process can dial somewhere nobody named. That is CR-007's first
/// defect -- a default live endpoint -- applied to the product rather than only to
/// the test harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Address {
    /// A Unix domain socket at this path.
    Uds(PathBuf),
    /// A TCP endpoint, as `host:port`.
    Tcp(String),
}

impl Address {
    /// The URI a gRPC client dials. For a Unix socket the authority is a
    /// placeholder -- the connector ignores it and dials the path instead -- but
    /// tonic requires a well-formed URI, so it is spelled once, here.
    pub fn endpoint_uri(&self) -> String {
        match self {
            Address::Uds(_) => "http://[::]:50051".to_owned(),
            Address::Tcp(addr) => format!("http://{addr}"),
        }
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Address::Uds(path) => write!(f, "unix:{}", path.display()),
            Address::Tcp(addr) => write!(f, "tcp:{addr}"),
        }
    }
}
