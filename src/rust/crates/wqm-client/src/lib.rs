//! wqm-client -- N49, the client seam (ARCH rev15 §9.1).
//!
//! Every non-daemon surface (S2 MCP, S3 CLI, S4 TUI) reads through this crate.
//! `P04-GT001-WO011` seeds it with the daemon-status leg the walking skeleton
//! crosses; the kernel read subset it embeds under decision B arrives with the
//! read slices.
//!
//! # Unreachability is data, not an error
//!
//! This is the crate's one load-bearing design decision, and it is inherited from
//! the sealed surface rather than invented here. MCP-SURFACE.md §4.3 excludes
//! `status` from `backend_unavailable` precisely because `status` "exists to
//! answer when the backend is down" -- a `status` that failed when the daemon was
//! unreachable could not report the one condition it exists to report, which is
//! I-3 recreated inside its own cure.
//!
//! So [`Client::status`] returns [`DaemonReport`], which has an *unreachable*
//! variant carrying why. A caller cannot accidentally treat "no daemon" as a
//! transport failure, because it never arrives as one.

use std::time::Duration;

use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use tonic::transport::{Channel, Endpoint};
use wqm_proto::v1::system_service_client::SystemServiceClient;
use wqm_proto::v1::{daemon_health, StatusRequest};
use wqm_proto::Address;

/// How long to wait for a connection before calling the daemon unreachable.
///
/// Short on purpose: `status` is the call an agent makes *because* something
/// seems wrong, so a long hang is itself a bad answer.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// A failure of this crate that is genuinely a failure -- as opposed to the
/// daemon being down, which is an answer (see [`DaemonReport`]).
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// The address could not be turned into something dialable at all. This is a
    /// malformed address, not an absent daemon.
    #[error("`{address}` is not a dialable address: {source}")]
    Address {
        /// The address as given.
        address: String,
        /// Why it could not be parsed.
        #[source]
        source: tonic::transport::Error,
    },

    /// The daemon answered, but with an error status.
    #[error("the daemon refused the call: {0}")]
    Rpc(#[from] tonic::Status),
}

/// What the daemon said about itself, or why it could not be asked.
///
/// Both variants are answers. The type exists so that "the daemon is down" cannot
/// be lost in an error path a caller might collapse into a generic failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonReport {
    /// The daemon answered.
    Reachable(DaemonStatus),
    /// The daemon could not be reached.
    Unreachable {
        /// The honest identifier for the cause, from the same vocabulary the
        /// internal metrics use (MCP-SURFACE.md §4.5 rule 1: one vocabulary for
        /// telemetry and for the agent; a constant string that does not vary with
        /// the cause cannot ship).
        reason: UnreachableReason,
        /// Where the client tried to reach it, so the answer is actionable.
        address: String,
    },
}

/// Why the daemon could not be reached. Each member is the identifier that
/// travels to the agent AND to telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnreachableReason {
    /// Nothing is listening: no socket, connection refused, or the connect timed
    /// out. From the agent's point of view these are one condition -- the daemon
    /// is not there -- and splitting them further would be precision it cannot act
    /// on differently.
    DaemonUnreachable,
}

impl UnreachableReason {
    /// The wire spelling, shared with the metric of the same name.
    pub const fn as_str(self) -> &'static str {
        match self {
            UnreachableReason::DaemonUnreachable => "daemon_unreachable",
        }
    }
}

/// The daemon's own report, in this crate's vocabulary rather than the wire's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonStatus {
    /// The daemon's condition.
    pub state: DaemonState,
    /// An open string qualifying `state`; empty when it needs none.
    pub detail: String,
    /// When the daemon entered this state, Unix seconds. `None` when unknown.
    pub since_unix_seconds: Option<i64>,
    /// Which build answered.
    pub version: String,
    /// The index subsystem's state, or `None` when this build has none. `None`
    /// is a different claim from a zero-filled block, and the difference is the
    /// point.
    pub index: Option<IndexState>,
}

/// The daemon's condition.
///
/// The full vocabulary is owed by CR-010 (the queue-failure engagement); these are
/// the members the current daemon can truthfully assert, plus the unspecified
/// value a future-dated daemon might send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonState {
    /// Serving, with nothing known to be wrong.
    Ok,
    /// The daemon sent a state this build does not know. Reported as-is rather
    /// than mapped onto `Ok`, because guessing would be the lie.
    Unrecognized(i32),
}

/// The index subsystem's state, when there is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexState {
    /// Files the daemon is tracking.
    pub files_tracked: u64,
    /// Items still queued.
    pub queue_pending: u64,
    /// Whether the index is caught up with its sources.
    pub complete: bool,
    /// How far behind, in seconds.
    pub lag_seconds: u64,
}

/// The client seam. Holds no connection: each call dials, so that a daemon
/// started or stopped between calls is observed rather than cached.
#[derive(Debug, Clone)]
pub struct Client {
    address: Address,
}

impl Client {
    /// A client aimed at `address`. Nothing is dialed here -- construction cannot
    /// fail, and reachability is reported by the call that needs it.
    pub fn new(address: Address) -> Self {
        Client { address }
    }

    /// Ask the daemon what it can currently do.
    ///
    /// Returns `Ok(DaemonReport::Unreachable { .. })` when no daemon answers.
    /// `Err` is reserved for a malformed address or a daemon that answered with
    /// an error -- both of which are faults, unlike an absent daemon.
    pub async fn status(&self, probe: bool) -> Result<DaemonReport, ClientError> {
        let channel = match self.connect().await {
            Ok(channel) => channel,
            Err(ConnectFailure::Unreachable) => {
                return Ok(DaemonReport::Unreachable {
                    reason: UnreachableReason::DaemonUnreachable,
                    address: self.address.to_string(),
                })
            }
            Err(ConnectFailure::Malformed(source)) => {
                return Err(ClientError::Address {
                    address: self.address.to_string(),
                    source,
                })
            }
        };

        let response = SystemServiceClient::new(channel)
            .status(StatusRequest { probe })
            .await?
            .into_inner();

        Ok(DaemonReport::Reachable(DaemonStatus {
            state: response
                .daemon
                .as_ref()
                .map(|d| match daemon_health::State::try_from(d.state) {
                    Ok(daemon_health::State::Ok) => DaemonState::Ok,
                    // Both an explicit UNSPECIFIED and a value this build has
                    // never heard of mean the same thing to a caller: the daemon
                    // did not make a health claim we can read.
                    _ => DaemonState::Unrecognized(d.state),
                })
                .unwrap_or(DaemonState::Unrecognized(0)),
            detail: response
                .daemon
                .as_ref()
                .map(|d| d.detail.clone())
                .unwrap_or_default(),
            since_unix_seconds: response
                .daemon
                .as_ref()
                .map(|d| d.since_unix_seconds)
                .filter(|s| *s != 0),
            version: response.daemon.map(|d| d.version).unwrap_or_default(),
            index: response.index.map(|i| IndexState {
                files_tracked: i.files_tracked,
                queue_pending: i.queue_pending,
                complete: i.complete,
                lag_seconds: i.lag_seconds,
            }),
        }))
    }

    /// Dial the daemon, distinguishing "nobody is there" from "that is not an
    /// address".
    async fn connect(&self) -> Result<Channel, ConnectFailure> {
        let endpoint = Endpoint::try_from(self.address.endpoint_uri())
            .map_err(ConnectFailure::Malformed)?
            .connect_timeout(CONNECT_TIMEOUT);

        match &self.address {
            Address::Uds(path) => {
                let path = path.clone();
                endpoint
                    .connect_with_connector(tower::service_fn(move |_: tonic::transport::Uri| {
                        let path = path.clone();
                        async move {
                            Ok::<_, std::io::Error>(TokioIo::new(UnixStream::connect(path).await?))
                        }
                    }))
                    .await
                    .map_err(|_| ConnectFailure::Unreachable)
            }
            Address::Tcp(_) => endpoint
                .connect()
                .await
                .map_err(|_| ConnectFailure::Unreachable),
        }
    }
}

/// Internal: the two ways a dial can end badly, kept apart so the public API can
/// route one to data and the other to an error.
enum ConnectFailure {
    /// Nothing answered.
    Unreachable,
    /// The address itself is not usable.
    Malformed(tonic::transport::Error),
}
