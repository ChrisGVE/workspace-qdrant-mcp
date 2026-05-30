//! Session lifecycle subsystem for the MCP server.
//!
//! Three submodules:
//!
//! - [`lifecycle`]: session init, project registration, cleanup.
//!   Exposes [`DaemonOps`] trait (test double in `lifecycle_tests.rs`).
//! - [`heartbeat`]: background heartbeat loop (abort-handle based).
//! - [`project_detect`]: git-root / branch / project-ID detection.

pub mod heartbeat;
pub mod lifecycle;
pub mod project_detect;

// Re-export the most-used types at the session level to avoid deep import paths.
pub use lifecycle::{
    cleanup_session, default_detect_fn, initialize_session, register_project, DaemonOps,
    RegisterResponse,
};

pub use heartbeat::start_heartbeat;
pub use project_detect::{detect_project, find_project_root, lookup_project_id, ProjectInfo};
