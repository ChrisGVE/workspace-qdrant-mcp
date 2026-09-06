//! wqm-proto -- the N24 gRPC contract source of truth (ARCH rev15 §9.1).
//!
//! Owns the `.proto` definitions, the code generated from them, and the
//! [`TransportAddress`] enum. It declares no wqm-crate dependencies (generated
//! code only) so it can be linked by every surface without dragging in a closure.
//!
//! `P04-GT001-WO011` brings forward the part of N24's slice (`P04-GT011`) the
//! walking skeleton needs: one service with one RPC, the daemon reporting itself.
//! The remaining services arrive with that slice; the debt is recorded in
//! `SCAFFOLD.md` §7.

use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;

/// The generated code. `clippy::all` is allowed here and nowhere else: the
/// workspace denies warnings, and a lint against machine-written code is a lint
/// against `tonic-prost-build`, not against this workspace.
#[allow(clippy::all, missing_docs)]
pub mod v1 {
    tonic::include_proto!("wqm.v1");
}

/// Where a daemon listens, and therefore where a client dials: a Unix-domain
/// socket path, or a TCP endpoint (ARCH rev15 §9.1: "`Address` enum
/// (`Uds(path) | Tcp(addr)`), UDS-0600 default").
///
/// **A Unix socket is the default, and the type is what makes that expressible.**
/// A UDS is created owner-only (mode `0600`), so it carries user isolation that
/// plain loopback TCP does not have -- any local user can reach a loopback port.
/// A TCP port is also a scarce, machine-global name: two daemons that default to
/// the same port collide, which is exactly the hazard `P04-GT001-WO010` measured
/// on v0.1's control port. A socket path is a filesystem name, so it derives from
/// the deployment directory that already carries the `-v2` suffix and cannot
/// collide with production by construction. TCP stays in the enum because a
/// remote daemon is a declared deployment (ARCH §9.2), not because it is the norm.
/// With a single bare address type, "UDS by default" could not be stated at all,
/// and the safe default would not be the easy one.
///
/// **`Tcp` carries a [`SocketAddr`], not a string, and that is the point of the
/// variant.** The rule that governs a TCP bind is stated in terms of loopback: a
/// loopback bind requires a per-connection token, and a non-loopback bind without
/// TLS *and* a token is refused rather than warned about. "Is this loopback?" is a
/// question a `SocketAddr` can answer (`addr.ip().is_loopback()`) and a `String`
/// cannot. **The refusal itself is not here.** Enforcement lives in N48's serve
/// loop; this type carries no bind, no peer-cred check, no `0600` chmod and no
/// refusal logic. It only makes the two cases distinguishable, so that the loop
/// that does enforce has a fact to act on instead of a parse to attempt.
///
/// **In-memory only.** This is a runtime configuration value: there is no DDL, no
/// on-disk record and no wire-serialized form. A peer only ever sees the concrete
/// socket or endpoint the value resolves to.
///
/// There is deliberately **no `Default`**: an address is always supplied by the
/// caller, so no process can dial somewhere nobody named. That is CR-007's first
/// defect -- a default live endpoint -- applied to the product rather than only to
/// the test harness. The absence is asserted, not merely intended:
///
/// ```compile_fail
/// // CR-007: there is no default live endpoint. If this ever compiles, a
/// // `Default` impl has been added and the guarantee is gone.
/// let _ = wqm_proto::TransportAddress::default();
/// ```
///
/// # On the name
///
/// This type is `TransportAddress`, not `Address`, because `Address` is already
/// spoken for. N40's addressing type -- the `(Collection, KeepId, UnitLevel,
/// UnitId?)` tuple that names *what* a request is about -- is `C-ty-address` in
/// `wqm-common`, and it is held behind the query-language barrier. Two types
/// called `Address` in one workspace is precisely the single-producer collision
/// N8 exists to prevent, so the charter (§7.5) requires the transport one to ship
/// under a name that cannot be mistaken for it. This one names *where* a
/// connection goes; N40's names *what* it asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportAddress {
    /// A Unix domain socket at this path. The default transport; the socket file
    /// is created owner-only (mode `0600`) by the serve loop, not by this type.
    Uds(PathBuf),
    /// A TCP endpoint. Typed as a [`SocketAddr`] so that N48 can ask whether the
    /// bind is loopback rather than parse a string to find out.
    Tcp(SocketAddr),
}

/// Transition alias -- the concurrent `ui-module` workspace names this type as
/// `Address` at two sites (`wqm-tui/src/views/service.rs`,
/// `wqm-tui/src/widgets/daemon_status.rs`). It exists so that renaming the
/// canonical type does not break a tree this GT may not edit, and it is removed
/// the moment those two sites read `TransportAddress`. Do not use it in this
/// workspace; do not add a `#[deprecated]` attribute -- the other workspace denies
/// warnings, and a warning there is the breakage this avoids.
pub type Address = TransportAddress;

impl TransportAddress {
    /// The URI a gRPC client dials. For a Unix socket the authority is a
    /// placeholder -- the connector ignores it and dials the path instead -- but
    /// tonic requires a well-formed URI, so it is spelled once, here.
    pub fn endpoint_uri(&self) -> String {
        match self {
            TransportAddress::Uds(_) => "http://[::]:50051".to_owned(),
            TransportAddress::Tcp(addr) => format!("http://{addr}"),
        }
    }
}

impl fmt::Display for TransportAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportAddress::Uds(path) => write!(f, "unix:{}", path.display()),
            TransportAddress::Tcp(addr) => write!(f, "tcp:{addr}"),
        }
    }
}
