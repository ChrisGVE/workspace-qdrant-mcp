//! Prune-logs subcommand handler

use anyhow::{Context, Result};
use tabled::Tabled;

use crate::output;
use crate::output::ColumnHints;

/// Format bytes into a human-readable string
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[derive(Tabled)]
struct PruneRow {
    #[tabled(rename = "File")]
    file: String,
    #[tabled(rename = "Size")]
    size: String,
    #[tabled(rename = "Age")]
    age: String,
}

impl ColumnHints for PruneRow {
    fn content_columns() -> &'static [usize] {
        &[]
    }
}

/// Prune old log files from the canonical log directory
pub fn execute(dry_run: bool, retention_hours: u64) -> Result<()> {
    let log_dir = wqm_common::paths::get_canonical_log_dir();

    output::section("Log Pruning");
    output::kv("Log directory", &log_dir.display().to_string());
    output::kv("Retention", &format!("{}h", retention_hours));
    if dry_run {
        output::info("Dry run — no files will be deleted");
    }
    output::separator();

    let result = workspace_qdrant_core::log_pruner::prune_now(&log_dir, retention_hours, dry_run)
        .context("Failed to prune logs")?;

    if result.candidates.is_empty() {
        output::success("No log files older than the retention period.");
        return Ok(());
    }

    let rows: Vec<PruneRow> = result
        .candidates
        .iter()
        .map(|c| PruneRow {
            file: c
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string(),
            size: format_bytes(c.size),
            age: wqm_common::duration_fmt::format_duration(c.age_hours * 3600.0, 0),
        })
        .collect();

    output::print_table_auto(&rows);
    output::separator();

    if dry_run {
        output::info(format!(
            "Would delete {} file(s), freeing {}",
            result.candidates.len(),
            format_bytes(result.candidates.iter().map(|c| c.size).sum()),
        ));
    } else {
        output::success(format!(
            "Deleted {} file(s), freed {}",
            result.files_deleted,
            format_bytes(result.bytes_freed),
        ));
    }

    Ok(())
}
