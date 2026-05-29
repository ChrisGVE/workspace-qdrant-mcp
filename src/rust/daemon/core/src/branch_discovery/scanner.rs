//! Branch discovery scanner: filesystem hash-scan and classification.

use std::collections::HashMap;
use std::path::Path;

use qdrant_client::qdrant::{Condition, Filter};
use sqlx::SqlitePool;
use tracing::{debug, info, warn};
use walkdir::WalkDir;

use crate::branch_switch::BranchUpdateContext;
use crate::patterns::exclusion::{should_exclude_directory, should_exclude_file};

use super::db;

/// Result of a branch discovery scan.
#[derive(Debug, Default)]
pub struct BranchDiscoveryResult {
    /// Files shared with existing branches (branch added, no re-embedding).
    pub shared_count: u64,
    /// Files unique to this branch (need embedding).
    pub novel_paths: Vec<String>,
    /// Inferred parent branch (branch with smallest diff), if any.
    pub parent_branch: Option<String>,
    /// Errors encountered during scanning.
    pub errors: u64,
}

/// Scans a project's filesystem for a new branch and classifies files
/// as shared (existing content) or novel (needs embedding).
pub struct DiscoveryScanner;

impl DiscoveryScanner {
    /// Run the full discovery algorithm for a new branch.
    ///
    /// Steps:
    /// 1. Hash-scan all files on the new branch's filesystem
    /// 2. Load known tracked_files hashes for this project
    /// 3. Classify: shared (hash match) or novel (no match)
    /// 4. For shared files: add branch to SQLite + Qdrant + search.db
    /// 5. Optionally infer parent branch
    /// 6. Return novel paths for the caller to enqueue for embedding
    pub async fn discover(
        pool: &SqlitePool,
        branch_ctx: &BranchUpdateContext,
        watch_folder_id: &str,
        tenant_id: &str,
        project_root: &Path,
        new_branch: &str,
    ) -> Result<BranchDiscoveryResult, String> {
        info!(
            "Starting branch discovery for '{}' in {} (root: {})",
            new_branch,
            watch_folder_id,
            project_root.display()
        );

        // 1. Hash-scan filesystem.
        //
        // `scan_filesystem` is a synchronous WalkDir + per-file hashing pass
        // that can take minutes on a large tree. Running it directly in this
        // async fn blocks the worker thread without yield points, which both
        // starves the runtime and prevents the per-item `tokio::time::timeout`
        // guard from firing. Offload to the blocking pool so the worker stays
        // cooperative and the timeout remains effective.
        let root = project_root.to_path_buf();
        let fs_files = tokio::task::spawn_blocking(move || scan_filesystem(&root))
            .await
            .map_err(|e| format!("branch discovery scan task failed: {e}"))??;
        info!(
            "Filesystem scan: {} files found for branch '{}'",
            fs_files.len(),
            new_branch
        );

        if fs_files.is_empty() {
            return Ok(BranchDiscoveryResult::default());
        }

        // 2. Load known hashes from tracked_files
        let known = db::load_known_files(pool, watch_folder_id).await?;
        debug!("Known tracked files: {} entries", known.len());

        // 3. Classify files
        let (shared, novel) = classify_files(&fs_files, &known, new_branch);
        info!(
            "Classification: {} shared, {} novel for branch '{}'",
            shared.len(),
            novel.len(),
            new_branch
        );

        let mut result = BranchDiscoveryResult {
            novel_paths: novel,
            ..Default::default()
        };

        // 4. Process shared files: add branch to SQLite + Qdrant + search.db
        if !shared.is_empty() {
            let root_str = project_root.to_string_lossy();
            result.shared_count =
                process_shared_files(pool, branch_ctx, &shared, tenant_id, new_branch, &root_str)
                    .await;
        }

        // 5. Infer parent branch (optional, informational)
        if !known.is_empty() {
            result.parent_branch = infer_parent_branch(&fs_files, &known);
        }

        info!(
            "Branch discovery complete for '{}': {} shared, {} novel, parent={:?}",
            new_branch,
            result.shared_count,
            result.novel_paths.len(),
            result.parent_branch
        );

        Ok(result)
    }
}

/// Scan the filesystem under project_root, returning (relative_path, file_hash) pairs.
fn scan_filesystem(project_root: &Path) -> Result<HashMap<String, String>, String> {
    let mut files = HashMap::new();

    let walker = WalkDir::new(project_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            if entry.file_type().is_dir() {
                let name = entry.file_name().to_string_lossy();
                !should_exclude_directory(&name)
            } else {
                true
            }
        });

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                debug!("WalkDir error (skipping): {}", e);
                continue;
            }
        };

        if !entry.file_type().is_file() {
            continue;
        }

        let abs_path = entry.path();
        let rel_path = match abs_path.strip_prefix(project_root) {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => continue,
        };

        if should_exclude_file(&rel_path) {
            continue;
        }

        match wqm_common::hashing::compute_file_hash(abs_path) {
            Ok(hash) => {
                files.insert(rel_path, hash);
            }
            Err(e) => {
                debug!("Failed to hash {}: {}", rel_path, e);
            }
        }
    }

    Ok(files)
}

/// Shared file info for batch processing.
pub struct SharedFile {
    pub file_id: i64,
    pub base_point: Option<String>,
    pub relative_path: String,
    pub file_hash: String,
    pub existing_branches: Vec<String>,
}

/// Classify filesystem files as shared or novel.
///
/// A file is SHARED if its (relative_path, file_hash) matches a known tracked_files entry
/// and the new_branch is not already in that entry's branches array.
/// A file is NOVEL if no matching entry exists.
pub fn classify_files(
    fs_files: &HashMap<String, String>,
    known: &HashMap<(String, String), db::KnownFile>,
    new_branch: &str,
) -> (Vec<SharedFile>, Vec<String>) {
    let mut shared = Vec::new();
    let mut novel = Vec::new();

    for (rel_path, file_hash) in fs_files {
        let key = (rel_path.clone(), file_hash.clone());
        match known.get(&key) {
            Some(kf) if !kf.branches.iter().any(|b| b == new_branch) => {
                shared.push(SharedFile {
                    file_id: kf.file_id,
                    base_point: kf.base_point.clone(),
                    relative_path: rel_path.clone(),
                    file_hash: file_hash.clone(),
                    existing_branches: kf.branches.clone(),
                });
            }
            Some(_) => {
                // Already has this branch — skip
            }
            None => {
                novel.push(rel_path.clone());
            }
        }
    }

    (shared, novel)
}

/// Process shared files: add branch to SQLite, Qdrant, and search.db.
async fn process_shared_files(
    pool: &SqlitePool,
    branch_ctx: &BranchUpdateContext,
    shared: &[SharedFile],
    tenant_id: &str,
    new_branch: &str,
    watch_root: &str,
) -> u64 {
    // 1. Batch SQLite update
    let file_ids: Vec<i64> = shared.iter().map(|s| s.file_id).collect();
    let sqlite_count = match db::batch_add_branch(pool, &file_ids, new_branch).await {
        Ok(c) => c,
        Err(e) => {
            warn!("Discovery: batch SQLite update failed: {}", e);
            return 0;
        }
    };

    // 2. Batch Qdrant updates (grouped by base_point, batches of 1000)
    let lock = branch_ctx.branch_locks.get(tenant_id);
    let _guard = lock.lock().await;

    let mut qdrant_updates: HashMap<String, Vec<String>> = HashMap::new();
    for sf in shared {
        if let Some(ref bp) = sf.base_point {
            let mut branches = sf.existing_branches.clone();
            branches.push(new_branch.to_string());
            branches.sort();
            branches.dedup();
            qdrant_updates.insert(bp.clone(), branches);
        }
    }

    let collection = "projects";
    let entries: Vec<_> = qdrant_updates.iter().collect();
    for chunk in entries.chunks(1000) {
        for (base_point, branches) in chunk {
            let filter = Filter::must([Condition::matches("base_point", base_point.to_string())]);
            let mut payload = HashMap::new();
            payload.insert("branches".to_string(), serde_json::json!(branches));

            if let Err(e) = branch_ctx
                .storage_client
                .set_payload_by_filter(collection, filter, payload)
                .await
            {
                warn!(
                    "Discovery: Qdrant update failed for base_point={}: {}",
                    base_point, e
                );
            }
        }
    }

    // 3. Insert file_metadata rows in search.db
    if let Some(ref sdb) = branch_ctx.search_db {
        let metadata_entries: Vec<(i64, &str, Option<&str>, Option<&str>, Option<&str>)> = shared
            .iter()
            .map(|sf| {
                (
                    sf.file_id,
                    sf.relative_path.as_str(),
                    sf.base_point.as_deref(),
                    Some(sf.relative_path.as_str()),
                    Some(sf.file_hash.as_str()),
                )
            })
            .collect();
        db::batch_insert_file_metadata(
            sdb.pool(),
            &metadata_entries,
            tenant_id,
            new_branch,
            watch_root,
        )
        .await;
    }

    sqlite_count
}

/// Infer the parent branch by finding the branch with the smallest diff
/// (most files in common with the new branch's filesystem).
pub fn infer_parent_branch(
    fs_files: &HashMap<String, String>,
    known: &HashMap<(String, String), db::KnownFile>,
) -> Option<String> {
    let mut branch_match_counts: HashMap<&str, usize> = HashMap::new();

    for (rel_path, file_hash) in fs_files {
        let key = (rel_path.clone(), file_hash.clone());
        if let Some(kf) = known.get(&key) {
            for branch in &kf.branches {
                *branch_match_counts.entry(branch.as_str()).or_insert(0) += 1;
            }
        }
    }

    branch_match_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(branch, _)| branch.to_string())
}
