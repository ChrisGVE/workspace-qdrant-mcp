//! The hermetic test harness — CR-007's three defects, structurally prevented.
//!
//! CR-007 records three properties of *how* the v0.1 suite was built, each measured
//! at `P00-GT002` §4 rather than supposed. This crate is where the rebuild's cure
//! lives, and it exists before the first slice test because CR-007's recorded
//! adoption owner is "every P04 build WO from the first test onward".
//!
//! # The three defects, and how each is prevented in DP-8 terms
//!
//! DP-8 divides invariants into **structural** — the wrong thing is impossible to
//! express, so no discipline is required of the caller — and **runtime-only** —
//! the wrong thing is preventable to a limit, and the honest answer names both the
//! limit and the recovery floor. A cure misclassified as structural is worse than a
//! runtime-only one, because it stops being watched.
//!
//! ## (a) No test may reach a default live endpoint — **STRUCTURAL**
//!
//! In v0.1 the live-dependency tests defaulted `TEST_QDRANT_URL` to
//! `localhost:6333`, so a plain `cargo test` wrote to the production Qdrant. The
//! hazard is live against this very machine today.
//!
//! Here, [`StoreEndpoint`] has a private field and **no** public constructor: not
//! `new`, not `Default`, not `FromStr`, and no environment lookup anywhere in this
//! crate reads a URL. The only ways to obtain one are [`Ephemeral::start`], which
//! returns an endpoint pointing at a container this process owns and destroys, and
//! [`StoreEndpoint::declared`], which requires a caller to pass a URL as an
//! argument — an act that appears in the diff. There is no code path from ambient
//! configuration to a store handle, so "a test wrote to production because nobody
//! set a variable" is not a mistake this API can express.
//!
//! ## (b) A test that cannot run its assertions must FAIL or SKIP, never pass — **RUNTIME-ONLY**
//!
//! In v0.1, eleven tests reported green having executed zero assertions.
//!
//! This one **cannot** be made structural, and saying otherwise would be the
//! misclassification DP-8 warns about: `cargo test` has no skip primitive. A test
//! function that returns early returns `Ok`, and no type can prevent a function
//! from returning. `#[ignore]` is not the escape either — `guard_no_skipped_tests.py`
//! already forbids it.
//!
//! What is enforced instead, to the structural limit:
//!
//! - The assertions cannot be reached without a store, and a store arrives **only**
//!   as an argument to a closure this crate runs ([`with_store`]). A test cannot
//!   hold an unusable store and assert against it anyway.
//! - When no store can be provided, the closure is **never invoked** and the skip
//!   is *announced*: a `WQM-TEST-SKIP` line on stderr carrying the test name and a
//!   non-empty reason. Silence is not among the outcomes.
//! - [`SkipReason`] is `#[must_use]`, so discarding one is a warning, and warnings
//!   are denied workspace-wide. Ignoring a skip does not compile.
//!
//! **Recovery floor:** the announcement is machine-readable and appended to the
//! path in `WQM_TEST_SKIP_LOG` when set, so CI gates on the *set* of skips rather
//! than trusting a green run (`P04-GT001-WO014`). A test that quietly becomes a
//! no-op changes that set, which is the failure this defect is really about — not
//! one skip, but a suite that drifts into asserting nothing while staying green.
//!
//! ## (c) A test must establish its own environment, not inherit it — **STRUCTURAL**
//!
//! In v0.1, tests failed on any machine whose shell exported `XDG_*`, `QDRANT_URL`,
//! `WQM_LOG_DIR` or `OTEL_*`.
//!
//! [`HermeticEnv`] does not read the ambient environment and offers no way to. It
//! constructs its values from a temporary directory it owns, and the sensitive keys
//! are **always** written, never defaulted-if-absent — so an exported value is
//! overridden rather than consulted. The type carries no accessor that could return
//! an inherited value, which is what makes this structural rather than a
//! remember-to-clear convention.
//!
//! # What was salvaged from v0.1, and what was not
//!
//! CR-007 notes that a hermetic `QdrantTestContainer` already existed in the v0.1
//! tree with zero callers, and the charter's §0.2 salvage rule says keep on merit.
//! **The approach is kept; the file is not.** It is at ref `46f5df66b`,
//! `src/rust/daemon/shared-test-utils/src/containers.rs`, 296 lines, and it fails
//! on merit for reasons that were measured, not assumed:
//!
//! - It pinned the image to tag `latest`. A *hermetic* harness on a floating tag is
//!   a contradiction: two machines running the same test can run different Qdrant.
//! - Its readiness poll requested `/health`. Current Qdrant answers **404** there
//!   (`/healthz`, `/readyz` and `/livez` all answer 200 — verified against the live
//!   instance). Its thirty polls could only ever have exhausted and failed, so the
//!   harness could not have worked had anything called it. Zero callers is why
//!   nobody found out.
//! - Over half of it — `create_test_collection`, `delete_collection`,
//!   `insert_test_vectors` — was collection and point CRUD hand-rolled over the
//!   HTTP API: a second implementation of what the real client does, so a fixture
//!   built with it tests the fixture.
//! - `ContainerManager` added a named multi-container registry, with four unit tests
//!   that only exercised its `HashMap`, for a need no caller ever had.
//!
//! What survives is the shape: an ephemeral container per test, an owned handle
//! whose `Drop` removes it, and a readiness wait before the endpoint is handed out.

mod endpoint;
mod env;
mod skip;

pub mod scratchpad_fixture;

pub use endpoint::{Ephemeral, StoreEndpoint};
pub use env::HermeticEnv;
// `announce_skip` joins the re-exports at P04-GT001-WO011: the store-shaped
// `with_store` covers a test that needs a store, and the round-trip test needs the
// same announced-skip vocabulary for a precondition that is not a store (a
// sibling binary that has not been built). One skip format, one producer -- the
// alternative was a second hand-rolled `WQM-TEST-SKIP` line, which is how a
// marker CI gates on drifts.
pub use skip::{announce_skip, with_store, SkipReason, StoreAvailability};
