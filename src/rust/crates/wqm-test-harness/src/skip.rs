//! CR-007 defect (b): a test that cannot run its assertions must FAIL or SKIP,
//! never pass. **RUNTIME-ONLY, with a floor.**
//!
//! Classified honestly. `cargo test` has no skip primitive — a function that returns
//! early returns `Ok`, and no type prevents a function from returning — so "never a
//! silent pass" cannot be made structural. `#[ignore]` is not the escape either;
//! `guard_no_skipped_tests.py` forbids it.
//!
//! What is enforced to the structural limit: the store arrives only as a closure
//! argument, so a test cannot assert against an unusable one; a skip is *announced*
//! on a machine-readable line with a mandatory reason; and [`SkipReason`] is
//! `#[must_use]` under a workspace that denies warnings, so discarding one does not
//! compile.
//!
//! The floor is CI gating on the *set* of announced skips rather than on a green run
//! (`P04-GT001-WO014`). The defect this cures is not one skipped test — it is a suite
//! drifting into asserting nothing while staying green, and only a set comparison
//! catches that.

use std::future::Future;
use std::io::Write;

use crate::endpoint::{Ephemeral, EphemeralError, StoreEndpoint};

/// The marker that makes a skip machine-readable. Grepped by CI, so it is a contract:
/// changing it is changing an interface.
pub const SKIP_MARKER: &str = "WQM-TEST-SKIP";

/// Why a test could not run its assertions.
///
/// `#[must_use]` on purpose: warnings are denied workspace-wide, so a caller who
/// receives one and drops it fails to compile.
#[must_use = "a skip must be announced -- dropping it is the silent pass CR-007 forbids"]
#[derive(Debug, Clone)]
pub struct SkipReason(String);

impl SkipReason {
    /// Build a reason. Empty text is refused: "skipped" with no cause is the same
    /// dead end as no message at all.
    pub fn new(reason: impl Into<String>) -> Self {
        let reason = reason.into();
        assert!(
            !reason.trim().is_empty(),
            "a skip reason may not be empty -- state what was unavailable"
        );
        Self(reason)
    }

    /// The reason text.
    pub fn text(&self) -> &str {
        &self.0
    }
}

/// Whether a store could be provided.
///
/// `#[must_use]` so that obtaining one and ignoring the answer does not compile.
#[must_use = "check whether a store was available before asserting anything"]
#[derive(Debug)]
pub enum StoreAvailability {
    /// A store is ready at this endpoint. Boxed because `Ephemeral` owns a container
    /// handle and dwarfs the skip variant; an enum that is large no matter which arm
    /// it holds costs every caller.
    Available(Box<Ephemeral>),
    /// No store; the test must be announced as skipped.
    Unavailable(SkipReason),
}

/// Run `body` against an ephemeral store, or announce a skip and do not run it.
///
/// This is the only way a test in this workspace obtains a store, which is what
/// prevents a test from asserting against one it never got. When no store can be
/// provided the closure is **not** invoked and a `WQM-TEST-SKIP` line is emitted.
///
/// A container runtime that is present but fails is **not** a skip — it propagates as
/// a panic. Collapsing "no Docker here" and "Docker is broken" into one outcome is
/// how a real failure becomes a green run.
pub async fn with_store<F, Fut>(test_name: &str, body: F)
where
    F: FnOnce(StoreEndpoint) -> Fut,
    Fut: Future<Output = ()>,
{
    match acquire().await {
        StoreAvailability::Available(store) => {
            let endpoint = store.endpoint().clone();
            body(endpoint).await;
            // `store` is dropped here, removing the container -- after the body, so a
            // failing assertion still tears its store down.
        }
        StoreAvailability::Unavailable(reason) => announce_skip(test_name, &reason),
    }
}

/// Try to obtain a store, mapping only "no runtime" to a skip.
async fn acquire() -> StoreAvailability {
    match Ephemeral::start().await {
        Ok(store) => StoreAvailability::Available(Box::new(store)),
        Err(EphemeralError::NoRuntime(why)) => {
            StoreAvailability::Unavailable(SkipReason::new(format!("no container runtime: {why}")))
        }
        // Deliberately a panic. A present-but-broken runtime is a failure to fix, not
        // a condition to tolerate (CR-007 defect (b): never pass).
        Err(e @ EphemeralError::StartFailed(_)) => {
            panic!("store unavailable and this is NOT a skippable condition: {e}")
        }
    }
}

/// Emit the skip so a human and CI both see it.
///
/// stderr always, because that is what a developer reads. Additionally appended to
/// `WQM_TEST_SKIP_LOG` when set, which is the seam CI uses to gate on the skip set
/// (`P04-GT001-WO014`). This is the one place the harness reads the environment, and
/// it reads a *sink* rather than any value a test observes -- defect (c) is about
/// inherited inputs, and a log destination is not one.
pub fn announce_skip(test_name: &str, reason: &SkipReason) {
    let line = format!("{SKIP_MARKER}: {test_name} -- {}", reason.text());
    let mut err = std::io::stderr();
    let _ = writeln!(err, "{line}");
    let _ = err.flush();

    if let Ok(path) = std::env::var("WQM_TEST_SKIP_LOG") {
        if !path.is_empty() {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = writeln!(f, "{line}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reason_keeps_its_text() {
        let r = SkipReason::new("no container runtime: docker absent");
        assert_eq!(r.text(), "no container runtime: docker absent");
    }

    #[test]
    #[should_panic(expected = "may not be empty")]
    fn an_empty_reason_is_refused() {
        let _ = SkipReason::new("   ");
    }

    /// The marker is a CI contract; pin its exact spelling so a rename is a visible
    /// change rather than a silently-unmatched grep.
    #[test]
    fn the_marker_is_stable() {
        assert_eq!(SKIP_MARKER, "WQM-TEST-SKIP");
    }

    /// A skip written to the log sink is the floor CI stands on, so prove the line is
    /// there and carries both the test name and the reason.
    #[test]
    fn an_announced_skip_reaches_the_log_sink() {
        let dir = std::env::temp_dir().join(format!("wqm-skiplog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("skips.txt");
        std::env::set_var("WQM_TEST_SKIP_LOG", &path);

        announce_skip("some_test", &SkipReason::new("no container runtime: none"));

        let body = std::fs::read_to_string(&path).expect("log written");
        assert!(body.contains(SKIP_MARKER), "marker missing: {body}");
        assert!(body.contains("some_test"), "test name missing: {body}");
        assert!(
            body.contains("no container runtime"),
            "reason missing: {body}"
        );

        std::env::remove_var("WQM_TEST_SKIP_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
