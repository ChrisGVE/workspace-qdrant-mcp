//! wqm-serve -- N48, the daemon's serving surface (ARCH rev15 §9.1).
//!
//! `P04-GT001-WO011` seeds this crate with the two things the walking skeleton
//! needs: a serve loop that binds N24's [`Address`], and the `SystemService`
//! implementation behind it. N48's own slice (`P04-GT054`) adds the interceptor,
//! the ProxiedRead lanes, supervision, slow-burn and the N33 mint half.
//!
//! # What the skeleton daemon may claim
//!
//! Only what it can observe. It knows it is running, when it started, and which
//! build it is -- so [`DaemonFacts`] carries exactly those, and the `index` block
//! is left absent rather than zero-filled, because "no index subsystem exists in
//! this build" and "the index is empty" are different claims and only one of them
//! is true.

use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::net::{TcpListener, UnixListener};
use tokio_stream::wrappers::{TcpListenerStream, UnixListenerStream};
use tonic::{Request, Response, Status};
use wqm_proto::v1::system_service_server::{SystemService, SystemServiceServer};
use wqm_proto::v1::{daemon_health, DaemonHealth, StatusRequest, StatusResponse};
use wqm_proto::Address;

/// Why a serve loop could not start, or stopped.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    /// The Unix socket path is already occupied.
    ///
    /// The loop refuses rather than unlinking it: an existing socket is usually
    /// another daemon, and silently removing it would take the machine's daemon
    /// down from inside a routine start-up. Cleaning up a genuinely stale socket
    /// is an operator action, and it is one they can see.
    #[error(
        "refusing to bind `{}`: the path already exists. Another daemon may be \
         serving there; if it is a stale socket, remove it and start again.",
        path.display()
    )]
    SocketPathOccupied {
        /// The path that was already present.
        path: PathBuf,
    },

    /// The listener could not be created.
    #[error("could not listen on {address}: {source}")]
    Listen {
        /// Where the daemon tried to listen.
        address: String,
        /// The underlying failure.
        #[source]
        source: io::Error,
    },

    /// The gRPC server itself failed.
    #[error("the serve loop failed: {0}")]
    Transport(#[from] tonic::transport::Error),
}

/// The facts the daemon can honestly assert about itself.
#[derive(Debug, Clone)]
pub struct DaemonFacts {
    /// When this daemon started, as a Unix timestamp in seconds.
    pub since_unix_seconds: i64,
    /// The daemon build's version.
    pub version: &'static str,
}

impl DaemonFacts {
    /// Facts for a daemon starting now.
    pub fn starting_now(version: &'static str) -> Self {
        let since_unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            // A clock before the epoch is not a reason to refuse to serve; it is
            // a reason to say "unknown", which is what zero means on this field.
            .unwrap_or(0);
        DaemonFacts {
            since_unix_seconds,
            version,
        }
    }
}

/// N48's `SystemService` implementation.
#[derive(Debug, Clone)]
struct SystemSurface {
    facts: DaemonFacts,
}

#[tonic::async_trait]
impl SystemService for SystemSurface {
    async fn status(
        &self,
        _request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        // `probe` is accepted and has no effect: this build has no subsystem to
        // probe, so probing and not probing return the same answer. Honouring the
        // flag by pretending to probe would be the lie; ignoring an argument the
        // wire carries is not, as long as the answer is identical either way.
        Ok(Response::new(StatusResponse {
            daemon: Some(DaemonHealth {
                state: daemon_health::State::Ok as i32,
                detail: String::new(),
                since_unix_seconds: self.facts.since_unix_seconds,
                version: self.facts.version.to_owned(),
            }),
            // Absent, not zero: there is no index subsystem in this build.
            index: None,
        }))
    }
}

/// Serve N24's services at `address` until `shutdown` completes.
///
/// A Unix socket is created with mode 0600 (ARCH rev15 §9.1's "UDS-0600 default")
/// and removed when the loop ends, so a clean shutdown leaves no path for the next
/// start to trip over.
pub async fn serve<F>(address: &Address, facts: DaemonFacts, shutdown: F) -> Result<(), ServeError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let service = SystemServiceServer::new(SystemSurface { facts });
    let server = tonic::transport::Server::builder().add_service(service);

    match address {
        Address::Uds(path) => {
            let listener = bind_uds(path)?;
            let result = server
                .serve_with_incoming_shutdown(UnixListenerStream::new(listener), shutdown)
                .await;
            // Best effort: a socket left behind after a crash is the operator's to
            // clear, but one left behind after a clean exit would be our bug.
            let _ = std::fs::remove_file(path);
            result.map_err(ServeError::from)
        }
        Address::Tcp(addr) => {
            let listener = TcpListener::bind(addr)
                .await
                .map_err(|source| ServeError::Listen {
                    address: addr.clone(),
                    source,
                })?;
            server
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
                .await
                .map_err(ServeError::from)
        }
    }
}

/// Bind a Unix socket at `path`, refusing an occupied path and tightening the
/// permissions to owner-only.
fn bind_uds(path: &Path) -> Result<UnixListener, ServeError> {
    if path.exists() {
        return Err(ServeError::SocketPathOccupied {
            path: path.to_path_buf(),
        });
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| ServeError::Listen {
            address: format!("unix:{}", path.display()),
            source,
        })?;
    }

    let listener = UnixListener::bind(path).map_err(|source| ServeError::Listen {
        address: format!("unix:{}", path.display()),
        source,
    })?;

    set_owner_only(path).map_err(|source| ServeError::Listen {
        address: format!("unix:{}", path.display()),
        source,
    })?;

    Ok(listener)
}

/// 0600 on the socket: the daemon is the sole writer of persistent state, and its
/// control surface is not a shared machine resource.
#[cfg(unix)]
fn set_owner_only(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn set_owner_only(_path: &Path) -> io::Result<()> {
    // A non-unix target has no UDS to tighten; the TCP arm is the one it uses.
    Ok(())
}
