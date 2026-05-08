//! Integration tests for WriteActor commands.
//!
//! Each test creates an in-memory SQLite database, spawns a WriteActor,
//! and exercises the command through the WriteActorHandle API.
//!
//! Modules:
//! - `common`: shared test setup helpers
//! - `admin`: RenameTenantAdmin, RebalanceIdf, UpsertRuleMirror, DeleteRuleMirror
//! - `library`: AddLibrary, RemoveLibrary
//! - `queue`: EnqueueItem, RetryAll, RetryItem, CleanQueue, CancelItems, RemoveItem, CleanQueueByCollection
//! - `tracking`: LogSearchEvent
//! - `watch`: PauseWatchers, ResumeWatchers, EnableWatch, DisableWatch, ArchiveWatch, WatchLibrary

mod admin;
mod common;
mod library;
mod queue;
mod tracking;
mod watch;
