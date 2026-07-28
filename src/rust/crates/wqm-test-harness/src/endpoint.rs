//! CR-007 defect (a): no test may reach a default live endpoint. **STRUCTURAL.**
//!
//! The whole cure is the absence of a constructor. There is no `new`, no `Default`,
//! no `FromStr`, and no `std::env` read anywhere in this file — so a store address
//! cannot arrive from ambient configuration, only from an owned container or from a
//! URL a caller typed into the call site.

use std::process::Command;
use std::time::Duration;

use testcontainers::core::{IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage};

/// The Qdrant image, pinned. A floating `latest` — what the v0.1 harness used —
/// makes a "hermetic" harness non-reproducible across machines and across time,
/// which is the one property it exists to provide.
const QDRANT_IMAGE: &str = "qdrant/qdrant";
const QDRANT_TAG: &str = "v1.18.3";

const QDRANT_HTTP_PORT: u16 = 6333;
const QDRANT_GRPC_PORT: u16 = 6334;

/// The line Qdrant prints once BOTH transports are up. Measured against
/// `qdrant/qdrant:v1.18.3`, not assumed: the container writes 23 lines to **stdout**
/// and nothing at all to stderr, and this is the last of its two "listening on"
/// lines (`Qdrant HTTP listening on 6333` precedes it). Waiting on the gRPC line
/// therefore means both ports are accepting by the time the endpoint is handed out.
///
/// The v0.1 harness got this wrong in the other direction and nobody noticed,
/// because nothing called it. A wait condition that cannot match is worse than no
/// wait: it fails slowly and blames the container.
const READY_LINE: &str = "Qdrant gRPC listening on";

/// How long to wait for a container to answer readiness before giving up.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

/// Where a test may talk to a store.
///
/// Deliberately opaque. The inner address is private and there is no constructor
/// that invents one, so every value of this type is traceable to either a container
/// this process owns or an explicit argument at a call site.
#[derive(Debug, Clone)]
pub struct StoreEndpoint {
    http_url: String,
    grpc_url: String,
}

impl StoreEndpoint {
    /// An endpoint the caller states outright.
    ///
    /// The URL is a required argument, never a fallback: this is the escape hatch
    /// for pointing a test at an already-running store, and it is deliberately one
    /// that shows up in a diff and in review. Nothing in this crate calls it.
    pub fn declared(http_url: impl Into<String>, grpc_url: impl Into<String>) -> Self {
        Self {
            http_url: http_url.into(),
            grpc_url: grpc_url.into(),
        }
    }

    /// The HTTP address of the store.
    pub fn http_url(&self) -> &str {
        &self.http_url
    }

    /// The gRPC address of the store.
    pub fn grpc_url(&self) -> &str {
        &self.grpc_url
    }
}

/// Why an ephemeral store could not be started.
///
/// Distinguishes "this environment has no Docker", which is a legitimate reason to
/// skip, from "Docker is here and the container failed", which is a real failure and
/// must not be laundered into a skip.
#[derive(Debug, thiserror::Error)]
pub enum EphemeralError {
    /// No usable container runtime. The caller may skip.
    #[error("no container runtime available: {0}")]
    NoRuntime(String),
    /// A runtime exists and starting the store still failed. The caller must fail.
    #[error("container runtime present but Qdrant did not start: {0}")]
    StartFailed(String),
}

/// A Qdrant instance owned by this process for the lifetime of one test.
///
/// Dropping it removes the container. Held by value rather than detached precisely
/// so that a panicking test still tears its store down.
#[derive(Debug)]
pub struct Ephemeral {
    #[allow(dead_code)] // held for Drop; the endpoint is what callers use
    container: ContainerAsync<GenericImage>,
    endpoint: StoreEndpoint,
}

impl Ephemeral {
    /// Start a throwaway Qdrant and wait until it answers readiness.
    pub async fn start() -> Result<Self, EphemeralError> {
        probe_runtime()?;

        // Readiness is delegated to testcontainers' own log wait rather than polled
        // over HTTP. The v0.1 harness polled `/health`, which current Qdrant answers
        // 404 — a wait condition that cannot succeed is worse than none, because it
        // fails slowly and blames the container.
        let image = GenericImage::new(QDRANT_IMAGE, QDRANT_TAG)
            .with_exposed_port(QDRANT_HTTP_PORT.tcp())
            .with_exposed_port(QDRANT_GRPC_PORT.tcp())
            .with_wait_for(WaitFor::message_on_stdout(READY_LINE));

        let container = tokio::time::timeout(STARTUP_TIMEOUT, image.start())
            .await
            .map_err(|_| {
                EphemeralError::StartFailed(format!(
                    "no readiness signal within {}s",
                    STARTUP_TIMEOUT.as_secs()
                ))
            })?
            .map_err(|e| EphemeralError::StartFailed(e.to_string()))?;

        let host = container
            .get_host()
            .await
            .map_err(|e| EphemeralError::StartFailed(e.to_string()))?
            .to_string();
        let http = container
            .get_host_port_ipv4(QDRANT_HTTP_PORT)
            .await
            .map_err(|e| EphemeralError::StartFailed(e.to_string()))?;
        let grpc = container
            .get_host_port_ipv4(QDRANT_GRPC_PORT)
            .await
            .map_err(|e| EphemeralError::StartFailed(e.to_string()))?;

        Ok(Self {
            container,
            endpoint: StoreEndpoint {
                http_url: format!("http://{host}:{http}"),
                grpc_url: format!("http://{host}:{grpc}"),
            },
        })
    }

    /// Where this container can be reached.
    pub fn endpoint(&self) -> &StoreEndpoint {
        &self.endpoint
    }

    /// The image this harness pins, for a deliverable that must not transcribe it.
    pub fn pinned_image() -> String {
        format!("{QDRANT_IMAGE}:{QDRANT_TAG}")
    }
}

/// Is there a container runtime we can use at all?
///
/// Separated from `start` so that "no Docker on this machine" is reported as its own
/// variant. Collapsing the two is how a broken container quietly becomes a skip.
fn probe_runtime() -> Result<(), EphemeralError> {
    match Command::new("docker").arg("info").output() {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => Err(EphemeralError::NoRuntime(format!(
            "`docker info` exited {}",
            out.status
        ))),
        Err(e) => Err(EphemeralError::NoRuntime(format!(
            "`docker` is not runnable: {e}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of defect (a) is what this type will NOT do. That is hard to assert
    /// directly, so assert the observable half: an endpoint's address is exactly what
    /// the caller supplied, with nothing filled in from anywhere else.
    #[test]
    fn a_declared_endpoint_carries_only_what_the_caller_passed() {
        let ep = StoreEndpoint::declared("http://example:1", "http://example:2");
        assert_eq!(ep.http_url(), "http://example:1");
        assert_eq!(ep.grpc_url(), "http://example:2");
    }

    /// Guards the pin itself. `latest` is the exact defect that made the v0.1
    /// harness non-hermetic, so a future edit back to a floating tag fails here.
    #[test]
    fn the_image_pin_is_a_version_not_a_floating_tag() {
        let pin = Ephemeral::pinned_image();
        assert!(pin.starts_with("qdrant/qdrant:"), "unexpected image: {pin}");
        let tag = pin.split(':').next_back().expect("tag");
        assert_ne!(
            tag, "latest",
            "a hermetic harness may not float its image tag"
        );
        assert!(
            tag.starts_with('v') && tag[1..].contains('.'),
            "tag should be an explicit version, got {tag}"
        );
    }
}
