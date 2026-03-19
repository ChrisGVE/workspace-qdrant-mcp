//! WriteActor: serializes all gRPC write service state.db mutations.
//!
//! Architecture:
//! - `commands`: WriteCommand enum with variants for each gRPC write RPC
//! - `actor`: WriteActor task (receives commands) and WriteActorHandle (sends commands)
//! - `exec_*`: SQL execution methods split by service domain
//!
//! Internal daemon mutations (queue processor, file tracker) stay on the pool
//! for now and will be migrated in a future PR.

pub mod actor;
pub mod commands;
mod exec_admin;
mod exec_library;
mod exec_queue;
mod exec_tracking;
mod exec_watch;

pub use actor::{WriteActor, WriteActorHandle};
pub use commands::*;
