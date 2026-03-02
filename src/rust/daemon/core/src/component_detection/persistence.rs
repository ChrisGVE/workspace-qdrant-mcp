//! SQLite persistence and backfill for detected components.

use std::path::Path;

use sqlx::SqlitePool;
use tracing::debug;

use super::detection::{assign_component, detect_components};
use super::{ComponentMap, ComponentSource};

/// Persist detected components to the project_components table.
pub async fn persist_components(
    pool: &SqlitePool,
    watch_folder_id: &str,
    components: &ComponentMap,
) -> Result<(), sqlx::Error> {
    let now = wqm_common::timestamps::now_utc();

    for component in components.values() {
        let component_id = format!("{}:{}", watch_folder_id, component.id);
        let patterns_json = serde_json::to_string(&component.patterns).unwrap_or_default();

        sqlx::query(
            "INSERT OR REPLACE INTO project_components
             (component_id, watch_folder_id, component_name, base_path, source, patterns, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
        )
        .bind(&component_id)
        .bind(watch_folder_id)
        .bind(&component.id)
        .bind(&component.base_path)
        .bind(component.source.as_str())
        .bind(&patterns_json)
        .bind(&now)
        .execute(pool)
        .await?;
    }

    debug!(
        "Persisted {} components for watch folder {}",
        components.len(),
        watch_folder_id
    );
    Ok(())
}

/// Load components from the project_components table.
pub async fn load_components(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<ComponentMap, sqlx::Error> {
    use sqlx::Row;

    let rows = sqlx::query(
        "SELECT component_name, base_path, source, patterns
         FROM project_components WHERE watch_folder_id = ?1",
    )
    .bind(watch_folder_id)
    .fetch_all(pool)
    .await?;

    let mut components = ComponentMap::new();
    for row in &rows {
        let name: String = row.get("component_name");
        let base_path: String = row.get("base_path");
        let source_str: String = row.get("source");
        let patterns_json: Option<String> = row.get("patterns");

        let source = match source_str.as_str() {
            "cargo" => ComponentSource::Cargo,
            "npm" => ComponentSource::Npm,
            _ => ComponentSource::Directory,
        };

        let patterns: Vec<String> = patterns_json
            .and_then(|j| serde_json::from_str(&j).ok())
            .unwrap_or_else(|| vec![format!("{}/**", base_path)]);

        components.insert(
            name.clone(),
            super::ComponentInfo {
                id: name,
                base_path,
                patterns,
                source,
            },
        );
    }

    Ok(components)
}

// ── Backfill ─────────────────────────────────────────────────────────────

/// Stats from a component backfill run.
#[derive(Debug, Clone, Default)]
pub struct BackfillStats {
    /// Watch folders processed
    pub folders_processed: u64,
    /// Files updated with component assignment
    pub files_updated: u64,
    /// Files that couldn't be assigned (no matching component)
    pub files_unmatched: u64,
    /// Errors during processing
    pub errors: u64,
}

/// Backfill component assignments for active watch folders.
///
/// For each enabled watch folder, detects components from the workspace
/// definition files, then assigns components to tracked_files.
///
/// When `force` is `false`, only files with `component IS NULL` are updated.
/// When `force` is `true`, all files are re-evaluated and stale
/// `project_components` entries are cleaned up before persisting.
///
/// When `tenant_id` is `Some`, only watch folders matching that tenant are processed.
/// Batches updates in transactions of `batch_size`.
pub async fn backfill_components(
    pool: &SqlitePool,
    batch_size: usize,
    force: bool,
    tenant_id: Option<&str>,
) -> Result<BackfillStats, String> {
    use sqlx::Row;

    let mut stats = BackfillStats::default();

    // Get enabled, non-archived watch folders (optionally filtered by tenant)
    let folders: Vec<(String, String)> = if let Some(tid) = tenant_id {
        sqlx::query_as(
            "SELECT watch_id, path FROM watch_folders \
             WHERE enabled = 1 AND is_archived = 0 AND tenant_id = ?1",
        )
        .bind(tid)
        .fetch_all(pool)
        .await
    } else {
        sqlx::query_as(
            "SELECT watch_id, path FROM watch_folders WHERE enabled = 1 AND is_archived = 0",
        )
        .fetch_all(pool)
        .await
    }
    .map_err(|e| format!("Failed to query watch_folders: {e}"))?;

    for (watch_id, path) in &folders {
        let project_path = Path::new(path.as_str());
        if !project_path.is_dir() {
            debug!("Skipping backfill for {}: path does not exist", watch_id);
            continue;
        }

        // Detect components for this project
        let components = detect_components(project_path);
        if components.is_empty() {
            debug!("No components detected for {}, skipping backfill", watch_id);
            stats.folders_processed += 1;
            continue;
        }

        // When force is true, delete stale project_components before persisting
        // so removed workspace members get cleaned up
        if force {
            if let Err(e) = sqlx::query(
                "DELETE FROM project_components WHERE watch_folder_id = ?1",
            )
            .bind(watch_id)
            .execute(pool)
            .await
            {
                debug!("Failed to clean stale components for {}: {}", watch_id, e);
            }
        }

        // Persist the detected components (idempotent via INSERT OR REPLACE)
        if let Err(e) = persist_components(pool, watch_id, &components).await {
            debug!("Failed to persist components for {}: {}", watch_id, e);
            stats.errors += 1;
        }

        // Find tracked files to update
        let files: Vec<(i64, String)> = if force {
            // Force mode: re-evaluate ALL files with a relative_path
            sqlx::query(
                "SELECT file_id, relative_path FROM tracked_files
                 WHERE watch_folder_id = ?1 AND relative_path IS NOT NULL",
            )
        } else {
            // Normal mode: only files with NULL component
            sqlx::query(
                "SELECT file_id, relative_path FROM tracked_files
                 WHERE watch_folder_id = ?1 AND component IS NULL AND relative_path IS NOT NULL",
            )
        }
        .bind(watch_id)
        .map(|row: sqlx::sqlite::SqliteRow| {
            let file_id: i64 = row.get("file_id");
            let rel_path: String = row.get("relative_path");
            (file_id, rel_path)
        })
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Failed to query tracked_files for {}: {e}", watch_id))?;

        if files.is_empty() {
            stats.folders_processed += 1;
            continue;
        }

        debug!(
            "Backfilling components for {}: {} files (force={})",
            watch_id,
            files.len(),
            force,
        );

        // Batch update in transactions
        for chunk in files.chunks(batch_size) {
            let mut tx = pool
                .begin()
                .await
                .map_err(|e| format!("Failed to begin transaction: {e}"))?;

            for (file_id, rel_path) in chunk {
                match assign_component(rel_path, &components) {
                    Some(comp) => {
                        if let Err(e) = sqlx::query(
                            "UPDATE tracked_files SET component = ?1 WHERE file_id = ?2",
                        )
                        .bind(&comp.id)
                        .bind(file_id)
                        .execute(&mut *tx)
                        .await
                        {
                            debug!("Failed to update file_id {}: {}", file_id, e);
                            stats.errors += 1;
                        } else {
                            stats.files_updated += 1;
                        }
                    }
                    None => {
                        stats.files_unmatched += 1;
                    }
                }
            }

            tx.commit()
                .await
                .map_err(|e| format!("Failed to commit batch: {e}"))?;
        }

        stats.folders_processed += 1;
    }

    Ok(stats)
}
