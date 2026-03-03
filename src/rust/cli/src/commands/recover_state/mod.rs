//! Recover state.db from Qdrant collections
//!
//! Scrolls all 4 canonical Qdrant collections and reconstructs:
//! - watch_folders: inferred from unique tenant_id + absolute_path prefixes
//! - tracked_files: one row per unique (tenant_id, file_path, branch)
//! - qdrant_chunks: one row per Qdrant point (for file-type points)
//! - rules_mirror: reconstructed from rules collection points

mod reconstruction;
mod schema;

use anyhow::Result;
use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
};

use crate::output;
use super::qdrant_helpers;

use reconstruction::{
    reconstruct_library_state, reconstruct_project_state, reconstruct_rules_state,
};
use schema::create_fresh_database;

/// All 4 canonical collections
const ALL_COLLECTIONS: &[&str] = &[
    COLLECTION_PROJECTS,
    COLLECTION_LIBRARIES,
    COLLECTION_RULES,
    COLLECTION_SCRATCHPAD,
];

/// Execute recover-state command
pub async fn execute(confirm: bool) -> Result<()> {
    output::section("State Recovery from Qdrant");

    if !confirm {
        output::info("This will rebuild state.db from Qdrant point payloads.");
        output::info("Existing state.db will be backed up to state.db.bak.");
        output::warning("Sparse vocabulary and keyword/tag data cannot be recovered.");
        output::info("They will be rebuilt by the daemon on restart.");
        println!();
        output::info("Run with --confirm to proceed.");
        return Ok(());
    }

    let db_path = crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    // Step 1: Backup existing database
    let bak_path = db_path.with_extension("db.bak");
    if db_path.exists() {
        std::fs::copy(&db_path, &bak_path)
            .map_err(|e| anyhow::anyhow!("Failed to backup state.db: {}", e))?;
        output::success(format!("Backed up to {}", bak_path.display()));
        std::fs::remove_file(&db_path)
            .map_err(|e| anyhow::anyhow!("Failed to remove old state.db: {}", e))?;
    }

    // Step 2: Create fresh database with full schema
    let conn = create_fresh_database(&db_path)?;
    output::success("Created fresh state.db with full schema");

    // Step 3: Connect to Qdrant
    let http_client = qdrant_helpers::build_qdrant_http_client()?;
    let base_url = qdrant_helpers::qdrant_base_url();
    output::kv("Qdrant URL", &base_url);
    output::separator();

    // Step 4: Scroll each collection and reconstruct
    let mut total_points = 0u64;
    let mut total_watch_folders = 0u64;
    let mut total_tracked_files = 0u64;
    let mut total_chunks = 0u64;
    let mut total_rules = 0u64;

    for collection in ALL_COLLECTIONS {
        output::info(format!("Scrolling {}...", collection));
        let points =
            qdrant_helpers::scroll_all_points(&http_client, &base_url, collection).await?;

        let count = points.len();
        total_points += count as u64;
        output::kv(&format!("  {} points", collection), &count.to_string());

        if points.is_empty() {
            continue;
        }

        match *collection {
            c if c == COLLECTION_PROJECTS => {
                let stats = reconstruct_project_state(&conn, &points)?;
                total_watch_folders += stats.watch_folders;
                total_tracked_files += stats.tracked_files;
                total_chunks += stats.chunks;
            }
            c if c == COLLECTION_LIBRARIES => {
                let stats = reconstruct_library_state(&conn, &points)?;
                total_watch_folders += stats.watch_folders;
                total_tracked_files += stats.tracked_files;
                total_chunks += stats.chunks;
            }
            c if c == COLLECTION_RULES => {
                total_rules += reconstruct_rules_state(&conn, &points)?;
            }
            _ => {
                // Scratchpad: no SQLite state needed, points exist only in Qdrant
            }
        }
    }

    // Step 5: Summary
    output::separator();
    output::section("Recovery Summary");
    output::kv("Total Qdrant points", &total_points.to_string());
    output::kv("Watch folders created", &total_watch_folders.to_string());
    output::kv("Tracked files created", &total_tracked_files.to_string());
    output::kv("Qdrant chunks mapped", &total_chunks.to_string());
    output::kv("Rules mirrored", &total_rules.to_string());
    output::separator();
    output::success("Recovery complete. Restart daemon to rebuild vocabulary and tags.");
    output::info("Verify with: wqm admin health");

    Ok(())
}
