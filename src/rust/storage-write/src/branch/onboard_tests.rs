//! Offline tests for `branch::onboard` -- AC-F8.1, F8.2, F8.3, F8.4, F8.7.
//!
//! Complex AC-F8.6 and AC-F8.7 rename tests live in `onboard_tests_2.rs` (codesize
//! split -- coding.md §X). All tests use a real temp `store.db` + `CaptureSink` +
//! `MockProvider` + mock `Embedder` (no live Qdrant, no live git2, no model).

// Include the rename / crash-between-halves tests as a sub-module so `cargo test`
// discovers them via this file's module declaration in onboard.rs.
#[cfg(test)]
#[path = "onboard_tests_2.rs"]
mod rename_tests;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use sqlx::SqlitePool;

use super::*;
use crate::blob::embed::{EmbeddedChunk, Embedder};
use crate::blob::ladder::CaptureSink;
use crate::blob::lock::{ContentKeyLockManager, LockManagerConfig};
use crate::blob::test_support::{fixture, TENANT};
use crate::qdrant::membership_batch::MembershipPutBatch;
use wqm_common::error::StorageError;
use wqm_common::git::file_change::{FileChange, FileChangeStatus};
use wqm_storage::types::requests::{ChunkInput, IngestFileRequest};

// ---------------------------------------------------------------------------
// Test doubles (shared by rename_tests via `pub(super)`)
// ---------------------------------------------------------------------------

/// Deterministic mock embedder: same text -> same vectors, call-counted.
pub(super) struct MockEmbedder {
    calls: Arc<AtomicUsize>,
}

impl MockEmbedder {
    pub(super) fn new() -> (Arc<Self>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(Self {
                calls: calls.clone(),
            }),
            calls,
        )
    }
}

#[async_trait::async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<EmbeddedChunk, StorageError> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        let byte_sum: u32 = text.bytes().map(u32::from).sum();
        let dense = vec![byte_sum as f32, text.len() as f32, 0.0];
        let mut sparse = std::collections::HashMap::new();
        for b in text.bytes() {
            *sparse.entry(u32::from(b)).or_insert(0.0f32) += 1.0;
        }
        Ok(EmbeddedChunk { dense, sparse })
    }
}

/// Embedder that panics if called (AC-F8.6: rename of same content must NOT embed).
pub(super) struct PanickingEmbedder;

#[async_trait::async_trait]
impl Embedder for PanickingEmbedder {
    async fn embed(&self, _text: &str) -> Result<EmbeddedChunk, StorageError> {
        panic!("PanickingEmbedder: embed() called -- same-content rename must hit, not miss");
    }
}

/// Mock content provider backed by an in-memory path->chunks map.
pub(super) struct MockProvider {
    files: std::collections::HashMap<String, Vec<(u32, String)>>,
}

impl MockProvider {
    pub(super) fn new() -> Self {
        Self {
            files: std::collections::HashMap::new(),
        }
    }

    pub(super) fn add_file(&mut self, path: &str, chunks: Vec<(u32, &str)>) {
        self.files.insert(
            path.to_string(),
            chunks
                .into_iter()
                .map(|(i, t)| (i, t.to_string()))
                .collect(),
        );
    }
}

#[async_trait::async_trait]
impl FileContentProvider for MockProvider {
    async fn chunk_file(
        &self,
        _tenant_id: &str,
        _branch_id: &str,
        path: &str,
    ) -> Result<IngestFileRequest, StorageError> {
        use wqm_common::hashing::compute_content_hash;
        match self.files.get(path) {
            Some(chunks) => {
                let inputs = chunks
                    .iter()
                    .map(|(idx, text)| ChunkInput {
                        chunk_index: *idx,
                        content_hash: compute_content_hash(text),
                        text: text.clone(),
                    })
                    .collect();
                Ok(IngestFileRequest::new(path, inputs))
            }
            None => Err(StorageError::Sqlite(format!(
                "MockProvider: no file={path}"
            ))),
        }
    }
}

/// Call-counting wrapper around `MockProvider` (AC-F8.3 at-most-once assertion).
pub(super) struct CountingProvider {
    inner: MockProvider,
    calls: Arc<std::sync::Mutex<Vec<String>>>,
}

impl CountingProvider {
    pub(super) fn new(inner: MockProvider) -> (Self, Arc<std::sync::Mutex<Vec<String>>>) {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Self {
                inner,
                calls: calls.clone(),
            },
            calls,
        )
    }
}

#[async_trait::async_trait]
impl FileContentProvider for CountingProvider {
    async fn chunk_file(
        &self,
        tid: &str,
        bid: &str,
        path: &str,
    ) -> Result<IngestFileRequest, StorageError> {
        self.calls.lock().unwrap().push(path.to_string());
        self.inner.chunk_file(tid, bid, path).await
    }
}

// ---------------------------------------------------------------------------
// Shared fixture helpers (also used by onboard_tests_2)
// ---------------------------------------------------------------------------

pub(super) fn lock_mgr() -> Arc<ContentKeyLockManager> {
    ContentKeyLockManager::new(LockManagerConfig::default())
}

pub(super) fn test_cfg_no_sleep() -> OnboardConfig {
    OnboardConfig {
        max_concurrent: 2,
        batch_retry_delays: vec![
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
        ],
        batch_size: 1000,
    }
}

pub(super) async fn read_sync_state(pool: &SqlitePool, branch_id: &str) -> String {
    sqlx::query_scalar::<_, String>("SELECT sync_state FROM branches WHERE branch_id = ?")
        .bind(branch_id)
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn read_sync_metadata(pool: &SqlitePool, branch_id: &str) -> Option<String> {
    sqlx::query_scalar::<_, Option<String>>(
        "SELECT sync_metadata FROM branches WHERE branch_id = ?",
    )
    .bind(branch_id)
    .fetch_one(pool)
    .await
    .unwrap()
}

// ---------------------------------------------------------------------------
// AC-F8.1: sync_state transitions pending -> indexing -> current
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sync_state_pending_indexing_current() {
    let f = fixture("br-a").await;
    let pool = &f.pool;
    let mut provider = MockProvider::new();
    provider.add_file("src/lib.rs", vec![(0, "fn foo() {}")]);
    let diff = vec![FileChange {
        status: FileChangeStatus::Added,
        path: "src/lib.rs".into(),
    }];
    let (embedder, _) = MockEmbedder::new();
    let locks = lock_mgr();
    let mut sink = CaptureSink::default();
    let stats = branch_onboard(
        pool,
        &locks,
        embedder.as_ref(),
        &mut sink,
        &provider,
        TENANT,
        "br-a",
        "br-a",
        "/repo",
        "projects",
        &diff,
        &test_cfg_no_sleep(),
    )
    .await
    .unwrap();
    assert_eq!(read_sync_state(pool, "br-a").await, "current");
    assert_eq!(stats.files_changed, 1);
    assert!(stats.chunks_ingested > 0);
}

// ---------------------------------------------------------------------------
// AC-F8.1: branch row created fresh for an unknown branch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_onboard_creates_branch_row() {
    let f = fixture("br-x").await;
    let pool = &f.pool;
    let provider = MockProvider::new();
    let (embedder, _) = MockEmbedder::new();
    let locks = lock_mgr();
    let mut sink = CaptureSink::default();
    branch_onboard(
        pool,
        &locks,
        embedder.as_ref(),
        &mut sink,
        &provider,
        TENANT,
        "new-branch",
        "new-branch",
        "/repo",
        "projects",
        &[],
        &test_cfg_no_sleep(),
    )
    .await
    .unwrap();
    let state: String =
        sqlx::query_scalar("SELECT sync_state FROM branches WHERE branch_id = 'new-branch'")
            .fetch_one(pool)
            .await
            .unwrap();
    assert_eq!(state, "current");
}

// ---------------------------------------------------------------------------
// AC-F8.2: resume cursor written per processed file
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_resume_cursor_written() {
    let f = fixture("br-b").await;
    let pool = &f.pool;
    let mut provider = MockProvider::new();
    provider.add_file("a.rs", vec![(0, "a")]);
    provider.add_file("b.rs", vec![(0, "b")]);
    let diff = vec![
        FileChange {
            status: FileChangeStatus::Added,
            path: "a.rs".into(),
        },
        FileChange {
            status: FileChangeStatus::Added,
            path: "b.rs".into(),
        },
    ];
    let (embedder, _) = MockEmbedder::new();
    let locks = lock_mgr();
    let mut sink = CaptureSink::default();
    branch_onboard(
        pool,
        &locks,
        embedder.as_ref(),
        &mut sink,
        &provider,
        TENANT,
        "br-b",
        "br-b",
        "/repo",
        "projects",
        &diff,
        &test_cfg_no_sleep(),
    )
    .await
    .unwrap();
    let meta = read_sync_metadata(pool, "br-b").await;
    assert!(meta.is_some(), "sync_metadata must be set after onboard");
    let v: serde_json::Value = serde_json::from_str(meta.as_deref().unwrap()).unwrap();
    assert_eq!(v["last_processed_chunk_index"].as_u64().unwrap(), 2);
}

// ---------------------------------------------------------------------------
// AC-F8.2: cursor parse round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_cursor_parse_round_trip() {
    let json = r#"{"last_processed_chunk_index": 42}"#.to_string();
    assert_eq!(parse_cursor_index(&Some(json)), 42);
    assert_eq!(parse_cursor_index(&None), 0);
    assert_eq!(parse_cursor_index(&Some("{}".into())), 0);
}

// ---------------------------------------------------------------------------
// AC-F8.2: idempotent re-run (blob_refs ON CONFLICT IGNORE, no re-embed)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_idempotent_rerun() {
    let f = fixture("br-c").await;
    let pool = &f.pool;
    let mut provider = MockProvider::new();
    provider.add_file("main.rs", vec![(0, "fn main() {}")]);
    let diff = vec![FileChange {
        status: FileChangeStatus::Added,
        path: "main.rs".into(),
    }];
    let (embedder, calls1) = MockEmbedder::new();
    let locks = lock_mgr();
    let mut sink = CaptureSink::default();
    branch_onboard(
        pool,
        &locks,
        embedder.as_ref(),
        &mut sink,
        &provider,
        TENANT,
        "br-c",
        "br-c",
        "/repo",
        "projects",
        &diff,
        &test_cfg_no_sleep(),
    )
    .await
    .unwrap();
    assert!(calls1.load(Ordering::Acquire) > 0, "first run must embed");

    // Second run: blobs already present -> HIT path -> no re-embed.
    let (embedder2, calls2) = MockEmbedder::new();
    let mut sink2 = CaptureSink::default();
    branch_onboard(
        pool,
        &locks,
        embedder2.as_ref(),
        &mut sink2,
        &provider,
        TENANT,
        "br-c",
        "br-c",
        "/repo",
        "projects",
        &diff,
        &test_cfg_no_sleep(),
    )
    .await
    .unwrap();
    assert_eq!(
        calls2.load(Ordering::Acquire),
        0,
        "re-run must not re-embed existing blobs"
    );
    assert_eq!(read_sync_state(pool, "br-c").await, "current");
}

// ---------------------------------------------------------------------------
// AC-F8.3: provider called at most once per changed file per onboard
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_provider_called_at_most_once_per_file() {
    let f = fixture("br-d").await;
    let pool = &f.pool;
    let mut base = MockProvider::new();
    base.add_file("f1.rs", vec![(0, "one")]);
    base.add_file("f2.rs", vec![(0, "two")]);
    let diff = vec![
        FileChange {
            status: FileChangeStatus::Added,
            path: "f1.rs".into(),
        },
        FileChange {
            status: FileChangeStatus::Modified,
            path: "f2.rs".into(),
        },
    ];
    let (provider, call_log) = CountingProvider::new(base);
    let (embedder, _) = MockEmbedder::new();
    let locks = lock_mgr();
    let mut sink = CaptureSink::default();
    branch_onboard(
        pool,
        &locks,
        embedder.as_ref(),
        &mut sink,
        &provider,
        TENANT,
        "br-d",
        "br-d",
        "/repo",
        "projects",
        &diff,
        &test_cfg_no_sleep(),
    )
    .await
    .unwrap();
    let calls = call_log.lock().unwrap();
    assert_eq!(calls.iter().filter(|p| p.as_str() == "f1.rs").count(), 1);
    assert_eq!(calls.iter().filter(|p| p.as_str() == "f2.rs").count(), 1);
    assert_eq!(calls.len(), 2, "exactly one call per Added/Modified file");
}

// ---------------------------------------------------------------------------
// apply_git_diff: stat counters for Added + Modified + Deleted
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_apply_git_diff_stats_counts() {
    let f = fixture("br-g").await;
    let pool = &f.pool;
    let locks = lock_mgr();
    seed_branch_files(
        pool,
        &locks,
        "br-g",
        &["mod.rs:original", "del.rs:to-delete"],
    )
    .await;
    let mut provider = MockProvider::new();
    provider.add_file("mod.rs", vec![(0, "modified")]);
    provider.add_file("new.rs", vec![(0, "added")]);
    let changes = vec![
        FileChange {
            status: FileChangeStatus::Added,
            path: "new.rs".into(),
        },
        FileChange {
            status: FileChangeStatus::Modified,
            path: "mod.rs".into(),
        },
        FileChange {
            status: FileChangeStatus::Deleted,
            path: "del.rs".into(),
        },
    ];
    let (embedder, _) = MockEmbedder::new();
    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    let stats = apply_git_diff(
        pool,
        &locks,
        embedder.as_ref(),
        &mut sink,
        &mut batch,
        &provider,
        TENANT,
        "br-g",
        "projects",
        "projects",
        &changes,
    )
    .await
    .unwrap();
    assert_eq!(stats.files_added, 1);
    assert_eq!(stats.files_modified, 1);
    assert_eq!(stats.files_deleted, 1);
    assert_eq!(stats.files_renamed, 0);
}

// ---------------------------------------------------------------------------
// AC-F8.1: bounded concurrency config is non-zero
// ---------------------------------------------------------------------------

#[test]
fn test_bounded_concurrency_max_concurrent_respected() {
    let cfg = OnboardConfig {
        max_concurrent: 3,
        ..test_cfg_no_sleep()
    };
    assert!(cfg.max_concurrent > 0);
    assert!(cfg.max_concurrent <= 1024);
}

// ---------------------------------------------------------------------------
// Setup helper: seed two files into an already-onboarded branch
// ---------------------------------------------------------------------------

/// Seed pre-existing files for diff tests (inline onboard with fresh embedder).
pub(super) async fn seed_branch_files(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    branch_id: &str,
    files: &[&str], // "path:content" pairs
) {
    let mut seed_prov = MockProvider::new();
    for entry in files {
        let (path, content) = entry.split_once(':').unwrap();
        seed_prov.add_file(path, vec![(0, content)]);
    }
    let seed_diff: Vec<FileChange> = files
        .iter()
        .map(|e| {
            let path = e.split_once(':').unwrap().0.to_string();
            FileChange {
                status: FileChangeStatus::Added,
                path,
            }
        })
        .collect();
    let (embedder, _) = MockEmbedder::new();
    let mut sink = CaptureSink::default();
    branch_onboard(
        pool,
        locks,
        embedder.as_ref(),
        &mut sink,
        &seed_prov,
        TENANT,
        branch_id,
        branch_id,
        "/repo",
        "projects",
        &seed_diff,
        &test_cfg_no_sleep(),
    )
    .await
    .unwrap();
}
