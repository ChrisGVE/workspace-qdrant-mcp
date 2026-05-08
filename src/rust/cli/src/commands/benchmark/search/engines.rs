//! Engine runners: FTS5 search.db and ripgrep execution.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{self, SearchOptions};

use super::types::{BenchQuery, EngineResult};

/// Run a query through FTS5 search.db.
pub(super) async fn run_fts5_query(
    db: &SearchDbManager,
    query: &BenchQuery,
    tenant_id: Option<&str>,
) -> Result<EngineResult> {
    let options = SearchOptions {
        tenant_id: tenant_id.map(|s| s.to_string()),
        max_results: 1000,
        ..Default::default()
    };

    let start = Instant::now();
    let results = if query.regex {
        text_search::search_regex(db, query.pattern, &options).await
    } else {
        text_search::search_exact(db, query.pattern, &options).await
    };
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    match results {
        Ok(r) => {
            let file_paths: HashSet<String> =
                r.matches.iter().map(|m| m.file_path.clone()).collect();
            Ok(EngineResult {
                latency_ms,
                match_count: r.matches.len(),
                file_paths,
            })
        }
        Err(e) => {
            eprintln!("  FTS5 error for '{}': {}", query.pattern, e);
            Ok(EngineResult {
                latency_ms,
                match_count: 0,
                file_paths: HashSet::new(),
            })
        }
    }
}

/// Run a query through ripgrep.
pub(super) fn run_rg_query(project_root: &PathBuf, query: &BenchQuery) -> Result<EngineResult> {
    let start = Instant::now();

    let mut cmd = Command::new("rg");
    cmd.arg("--no-heading")
        .arg("--with-filename")
        .arg("--line-number")
        .arg("--max-count=1000")
        .arg("--color=never");

    if !query.regex {
        cmd.arg("--fixed-strings");
    }

    cmd.arg(query.pattern).arg(project_root);

    let output = cmd.output().context("Failed to execute rg")?;
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut file_paths = HashSet::new();
    let mut match_count = 0;

    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }
        match_count += 1;
        if let Some(colon_pos) = line.find(':') {
            file_paths.insert(line[..colon_pos].to_string());
        }
    }

    Ok(EngineResult {
        latency_ms,
        match_count,
        file_paths,
    })
}

/// Resolve the project root path from state.db watch_folders.
pub(super) fn resolve_project_root(db_path: &PathBuf, tenant_id: Option<&str>) -> Result<PathBuf> {
    let conn = rusqlite::Connection::open(db_path).context("Failed to open state.db")?;
    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;

    let path: String = if let Some(tid) = tenant_id {
        conn.query_row(
            "SELECT path FROM watch_folders WHERE tenant_id = ?1 AND collection = 'projects' LIMIT 1",
            rusqlite::params![tid],
            |row| row.get(0),
        )
        .context(format!("No project found for tenant_id '{}'", tid))?
    } else {
        conn.query_row(
            "SELECT path FROM watch_folders WHERE collection = 'projects' ORDER BY is_active DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .context("No projects found in watch_folders")?
    };

    Ok(PathBuf::from(path))
}

/// Check if ripgrep is available in PATH.
pub(super) fn check_rg_available() -> bool {
    Command::new("rg")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
