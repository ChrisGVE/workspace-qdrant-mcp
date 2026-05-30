// Integration test suite entry point (task-34).
//
// Gated by the `integration-tests` Cargo feature so the live-daemon/Qdrant
// tests are compiled **out** in normal `cargo test` runs.
//
// Enable with:
//   cargo test -p mcp-server --features integration-tests
//
// Runtime skip contract: every test probes the required service(s) at the
// start and calls `return` (with an `eprintln!` note) when unreachable.
// NO `#[ignore]` is used anywhere in this suite.
//
// Module layout (mirrors tests/integration/*.rs files):
//   helpers   — shared probe/env helpers
//   embedding — embedding provider status + retry/timeout policy
//   grep      — FTS text-search via daemon gRPC
//   retrieve  — Qdrant scroll/retrieve
//   list      — SQLite tracked_files live queries
//   rules     — rules_mirror: ingest + list + SQLite cross-check
//   store     — enqueue_item + idempotency_key cross-check
//   search    — embed + Qdrant dense search + RRF fusion
//   sqlite    — StateManager busy_timeout under concurrent writes

#![cfg(feature = "integration-tests")]

#[path = "integration/embedding.rs"]
mod embedding;
#[path = "integration/grep.rs"]
mod grep;
#[path = "integration/helpers.rs"]
mod helpers;
#[path = "integration/list.rs"]
mod list;
#[path = "integration/retrieve.rs"]
mod retrieve;
#[path = "integration/rules.rs"]
mod rules;
#[path = "integration/search.rs"]
mod search;
#[path = "integration/sqlite.rs"]
mod sqlite;
#[path = "integration/store.rs"]
mod store;
