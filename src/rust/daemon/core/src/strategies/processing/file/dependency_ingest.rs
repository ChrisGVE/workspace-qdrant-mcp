//! Dependency extraction during file ingestion.
//!
//! When a dependency manifest (Cargo.toml, package.json, etc.) is ingested,
//! this module parses the file and stores the dependency set for the project.
//! Grouping recomputation is deferred to the idle task scheduler.
//!
//! Non-blocking: dependency errors are logged but never fail the ingestion pipeline.

use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::grouping::dependency;

/// Check if the ingested file is a dependency manifest, and if so, parse and
/// store the project's dependencies.
///
/// This is a fire-and-forget side effect: errors are logged and swallowed
/// so they never block the main file ingestion pipeline.
pub(super) async fn maybe_store_dependencies(
    pool: &SqlitePool,
    tenant_id: &str,
    file_path: &Path,
    abs_file_path: &str,
) {
    if !dependency::is_dependency_file(file_path) {
        return;
    }

    let filename = match file_path.file_name().and_then(|n| n.to_str()) {
        Some(name) => name,
        None => return,
    };

    // Read the file content from disk
    let content = match std::fs::read_to_string(abs_file_path) {
        Ok(c) => c,
        Err(e) => {
            warn!(
                "Failed to read dependency file {}: {} (skipping dependency parse)",
                abs_file_path, e
            );
            return;
        }
    };

    let deps = dependency::parse_dependencies(filename, &content);

    if deps.is_empty() {
        debug!(
            "No dependencies parsed from {} for tenant {}",
            filename, tenant_id
        );
        return;
    }

    match dependency::store_dependencies(pool, tenant_id, &deps).await {
        Ok(()) => {
            info!(
                "Stored {} dependencies from {} for tenant {}",
                deps.len(),
                filename,
                tenant_id
            );
        }
        Err(e) => {
            warn!(
                "Failed to store dependencies from {} for tenant {}: {}",
                filename, tenant_id, e
            );
        }
    }
}
