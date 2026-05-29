//! `wqm admin ignore-candidates` — rank directories likely to belong in
//! `.wqmignore` or `global.wqmignore`.
//!
//! Aggregates rows from `tracked_files` by parent directory (truncated to a
//! configurable segment depth) and ranks each group by a composite score:
//!
//! ```text
//! score = file_count × (1 + 2·failure_rate + extension_homogeneity)
//! ```
//!
//! - `failure_rate` blends tree-sitter failures, LSP failures, and any
//!   `last_error` set during indexing (each contributes one signal per file).
//! - `extension_homogeneity` is the share of files contributed by the
//!   dominant extension; vendor / generated folders tend to be highly
//!   homogeneous (e.g. 95% `.json`, `.snap`, `.min.js`).
//!
//! The command only ranks — it never edits `wqmignore` files. Operators
//! decide which top entries warrant an ignore rule.
//!
//! See `docs/specs/14-future-development.md` for the planned follow-up
//! (Phase 2) that adds a `value_of_use` signal so the score reflects
//! cost-versus-usage rather than cost alone.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags};
use serde::Serialize;
use tabled::Tabled;

use crate::output::{self, ColumnHints, NumberLocale};

/// Local convenience for thousands-separated integer formatting.
fn fmt_int(n: u64) -> String {
    output::format_integer(n as i64, &NumberLocale::default())
}

/// One row from `tracked_files`, projected to the columns this command needs.
#[derive(Debug, Clone)]
struct FileRow {
    relative_path: String,
    chunk_count: i64,
    lsp_status: String,
    treesitter_status: String,
    has_error: bool,
    extension: Option<String>,
}

/// Aggregated stats for one parent directory.
#[derive(Debug, Clone, Default)]
struct DirStats {
    parent_dir: String,
    file_count: u64,
    indexed_count: u64,
    chunk_count_sum: i64,
    treesitter_failed: u64,
    lsp_failed: u64,
    error_count: u64,
    extensions: HashMap<String, u64>,
}

/// Final scored row for table output.
#[derive(Debug, Clone, Serialize, Tabled)]
struct IgnoreCandidateRow {
    #[tabled(rename = "Path")]
    #[serde(rename = "path")]
    path: String,
    #[tabled(rename = "Files")]
    #[serde(rename = "files")]
    files: String,
    #[tabled(rename = "Indexed")]
    #[serde(rename = "indexed")]
    indexed: String,
    #[tabled(rename = "Failed")]
    #[serde(rename = "failed")]
    failed: String,
    #[tabled(rename = "Top ext")]
    #[serde(rename = "top_extension")]
    top_extension: String,
    #[tabled(rename = "Score")]
    #[serde(rename = "score")]
    score: String,
}

impl ColumnHints for IgnoreCandidateRow {
    fn content_columns() -> &'static [usize] {
        &[0, 4] // Path + Top ext are content-flexible
    }

    fn numeric_columns() -> &'static [usize] {
        &[1, 2, 3, 5] // Files / Indexed / Failed / Score right-aligned
    }
}

/// Entry point dispatched from `admin/mod.rs`.
pub fn execute(
    top: usize,
    depth: usize,
    min_files: u64,
    json: bool,
    db_override: Option<PathBuf>,
) -> Result<()> {
    let (conn, db_path) = open_state_db_ro(db_override)?;

    let rows = query_tracked_files(&conn)?;

    if rows.is_empty() {
        output::warning(format!(
            "No usable rows in tracked_files at {}.",
            db_path.display()
        ));
        output::info(
            "If the daemon runs in Docker, the host CLI reads a different DB by default. \
             Set WQM_DATABASE_PATH or pass --db to point at the daemon's state.db.",
        );
        return Ok(());
    }

    let total_files = rows.len() as u64;
    let stats = aggregate_by_parent(&rows, depth.max(1));
    let mut scored = score_and_filter(stats, min_files);

    // Sort by score descending (NaN-safe: NaN falls to the bottom).
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if scored.len() > top {
        scored.truncate(top);
    }

    if json {
        print_json(&scored);
    } else {
        print_table(&scored, total_files, depth, min_files);
    }

    Ok(())
}

/// Open the resolved database read-only and return the connection + path.
fn open_state_db_ro(db_override: Option<PathBuf>) -> Result<(Connection, PathBuf)> {
    let db_path = match db_override {
        Some(p) => p,
        None => crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?,
    };

    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. \
             Hint: the daemon DB may live in a Docker volume; \
             pass --db <path> or set WQM_DATABASE_PATH.",
            db_path.display()
        );
    }

    let conn = Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .context("Failed to open state database read-only")?;

    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to configure SQLite connection")?;

    let table_exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master \
             WHERE type='table' AND name='tracked_files')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !table_exists {
        anyhow::bail!(
            "No tracked_files table in {}. Has the daemon ever run against this DB?",
            db_path.display()
        );
    }

    Ok((conn, db_path))
}

/// Query the projected columns from `tracked_files`. Only rows with a
/// non-empty `relative_path` are returned (legacy rows from pre-v37 schema
/// could have NULL `relative_path`).
fn query_tracked_files(conn: &Connection) -> Result<Vec<FileRow>> {
    let mut stmt = conn
        .prepare(
            "SELECT relative_path, chunk_count, lsp_status, treesitter_status, \
                    last_error, extension \
             FROM tracked_files \
             WHERE relative_path IS NOT NULL AND relative_path <> ''",
        )
        .context("Failed to prepare tracked_files query")?;

    let iter = stmt
        .query_map([], |row| {
            let last_error: Option<String> = row.get(4)?;
            Ok(FileRow {
                relative_path: row.get(0)?,
                chunk_count: row.get::<_, i64>(1).unwrap_or(0),
                lsp_status: row
                    .get::<_, Option<String>>(2)?
                    .unwrap_or_else(|| "none".to_string()),
                treesitter_status: row
                    .get::<_, Option<String>>(3)?
                    .unwrap_or_else(|| "none".to_string()),
                has_error: last_error
                    .as_ref()
                    .map(|e| !e.trim().is_empty())
                    .unwrap_or(false),
                extension: row.get::<_, Option<String>>(5)?,
            })
        })
        .context("Failed to query tracked_files")?;

    let mut out = Vec::new();
    for r in iter {
        out.push(r.context("Failed to read tracked_files row")?);
    }
    Ok(out)
}

/// Return the parent directory of `rel_path`, truncated to at most `depth`
/// segments. Root-level files (no parent) return ".".
///
/// `depth` is clamped to at least 1 by the caller.
fn parent_at_depth(rel_path: &str, depth: usize) -> String {
    let normalized = rel_path.replace('\\', "/");
    let trimmed = normalized.trim_start_matches('/').trim_end_matches('/');
    if trimmed.is_empty() {
        return ".".to_string();
    }
    let segments: Vec<&str> = trimmed.split('/').filter(|s| !s.is_empty()).collect();
    if segments.len() <= 1 {
        return ".".to_string();
    }
    let parent_count = (segments.len() - 1).min(depth.max(1));
    if parent_count == 0 {
        return ".".to_string();
    }
    segments[..parent_count].join("/")
}

/// Aggregate file rows into per-directory stats.
fn aggregate_by_parent(rows: &[FileRow], depth: usize) -> Vec<DirStats> {
    let mut map: HashMap<String, DirStats> = HashMap::new();

    for row in rows {
        let parent = parent_at_depth(&row.relative_path, depth);
        let entry = map.entry(parent.clone()).or_insert_with(|| DirStats {
            parent_dir: parent,
            ..DirStats::default()
        });
        entry.file_count += 1;
        if row.chunk_count > 0 {
            entry.indexed_count += 1;
        }
        entry.chunk_count_sum = entry.chunk_count_sum.saturating_add(row.chunk_count);
        if row.treesitter_status == "failed" {
            entry.treesitter_failed += 1;
        }
        if row.lsp_status == "failed" {
            entry.lsp_failed += 1;
        }
        if row.has_error {
            entry.error_count += 1;
        }
        if let Some(ext) = &row.extension {
            let trimmed = ext.trim_start_matches('.').to_lowercase();
            if !trimmed.is_empty() {
                *entry.extensions.entry(trimmed).or_default() += 1;
            }
        }
    }

    map.into_values().collect()
}

/// Composite score. See module docs for the formula.
fn compute_score(stats: &DirStats) -> f64 {
    if stats.file_count == 0 {
        return 0.0;
    }
    let denom = (stats.file_count as f64) * 3.0; // three failure signals per file
    let failures = stats.treesitter_failed + stats.lsp_failed + stats.error_count;
    let failure_rate = (failures as f64 / denom).min(1.0);

    let top_ext = stats.extensions.values().max().copied().unwrap_or(0);
    let homogeneity = top_ext as f64 / stats.file_count as f64;

    stats.file_count as f64 * (1.0 + 2.0 * failure_rate + homogeneity)
}

/// Human-readable label for the dominant extension, e.g. ".json (98%)".
fn top_extension_label(stats: &DirStats) -> String {
    if stats.extensions.is_empty() || stats.file_count == 0 {
        return "—".to_string();
    }
    let (ext, count) = stats
        .extensions
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
        .expect("extensions not empty");
    let pct = (*count as f64 / stats.file_count as f64) * 100.0;
    format!(".{} ({:.0}%)", ext, pct)
}

/// Drop directories below `min_files` and attach a score to each survivor.
fn score_and_filter(stats: Vec<DirStats>, min_files: u64) -> Vec<(DirStats, f64)> {
    stats
        .into_iter()
        .filter(|s| s.file_count >= min_files)
        .map(|s| {
            let score = compute_score(&s);
            (s, score)
        })
        .collect()
}

fn print_table(scored: &[(DirStats, f64)], total_files: u64, depth: usize, min_files: u64) {
    output::print_title("Ignore Candidates");
    output::print_blank();

    if scored.is_empty() {
        output::info(format!(
            "No directories with at least {} files (depth={}). \
             Try lowering --min-files or increasing --depth.",
            min_files, depth
        ));
        return;
    }

    let rows: Vec<IgnoreCandidateRow> = scored
        .iter()
        .map(|(s, score)| IgnoreCandidateRow {
            path: s.parent_dir.clone(),
            files: fmt_int(s.file_count),
            indexed: fmt_int(s.indexed_count),
            failed: fmt_int(s.treesitter_failed + s.lsp_failed + s.error_count),
            top_extension: top_extension_label(s),
            score: format!("{:.0}", score),
        })
        .collect();

    output::print_table_auto(&rows);
    output::print_blank();

    output::info(format!(
        "Aggregated {} tracked files into {} candidate dir(s) at depth={} (showing top {}).",
        fmt_int(total_files),
        fmt_int(scored.len() as u64),
        depth,
        scored.len(),
    ));
    output::info(
        "Score = file_count × (1 + 2·failure_rate + extension_homogeneity). \
         Higher means stronger candidate for .wqmignore — review each path before adding a rule.",
    );
}

fn print_json(scored: &[(DirStats, f64)]) {
    let arr: Vec<serde_json::Value> = scored
        .iter()
        .map(|(s, score)| {
            let top = s
                .extensions
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
                .map(|(e, c)| (e.clone(), *c))
                .unwrap_or_else(|| (String::new(), 0));
            serde_json::json!({
                "path": s.parent_dir,
                "file_count": s.file_count,
                "indexed_count": s.indexed_count,
                "chunk_count_sum": s.chunk_count_sum,
                "treesitter_failed": s.treesitter_failed,
                "lsp_failed": s.lsp_failed,
                "error_count": s.error_count,
                "top_extension": top.0,
                "top_extension_count": top.1,
                "score": score,
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&arr).unwrap());
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn row(rel: &str, chunks: i64, ts: &str, lsp: &str, err: bool, ext: &str) -> FileRow {
        FileRow {
            relative_path: rel.to_string(),
            chunk_count: chunks,
            lsp_status: lsp.to_string(),
            treesitter_status: ts.to_string(),
            has_error: err,
            extension: Some(ext.to_string()),
        }
    }

    #[test]
    fn parent_at_depth_truncates_segments() {
        assert_eq!(
            parent_at_depth("src/rust/cli/src/main.rs", 3),
            "src/rust/cli"
        );
        assert_eq!(parent_at_depth("src/rust/cli/src/main.rs", 1), "src");
        assert_eq!(
            parent_at_depth("src/rust/cli/src/main.rs", 10),
            "src/rust/cli/src"
        );
        assert_eq!(parent_at_depth("foo/bar.rs", 3), "foo");
        assert_eq!(parent_at_depth("/abs/path/x.rs", 2), "abs/path");
    }

    #[test]
    fn parent_at_depth_top_level_files_use_dot() {
        assert_eq!(parent_at_depth("foo.rs", 3), ".");
        assert_eq!(parent_at_depth("README.md", 3), ".");
    }

    #[test]
    fn parent_at_depth_handles_windows_separators() {
        assert_eq!(parent_at_depth("src\\rust\\cli\\main.rs", 2), "src/rust");
    }

    #[test]
    fn parent_at_depth_empty_returns_dot() {
        assert_eq!(parent_at_depth("", 3), ".");
        assert_eq!(parent_at_depth("/", 3), ".");
        assert_eq!(parent_at_depth("///", 3), ".");
    }

    #[test]
    fn aggregate_groups_by_depth() {
        let rows = vec![
            row("src/rust/cli/main.rs", 5, "done", "done", false, "rs"),
            row("src/rust/cli/lib.rs", 3, "done", "done", false, "rs"),
            row("src/rust/core/mod.rs", 4, "done", "done", false, "rs"),
            row("src/typescript/index.ts", 2, "done", "none", false, "ts"),
        ];

        let stats = aggregate_by_parent(&rows, 2);
        let by_path: HashMap<_, _> = stats
            .into_iter()
            .map(|s| (s.parent_dir.clone(), s))
            .collect();
        assert_eq!(by_path["src/rust"].file_count, 3);
        assert_eq!(by_path["src/typescript"].file_count, 1);
        assert_eq!(by_path["src/rust"].chunk_count_sum, 12);
    }

    #[test]
    fn aggregate_counts_failures_and_indexed() {
        let rows = vec![
            row("vendor/x.rs", 0, "failed", "none", false, "rs"),
            row("vendor/y.rs", 0, "failed", "failed", true, "rs"),
            row("vendor/z.rs", 1, "done", "done", false, "rs"),
        ];

        let stats = aggregate_by_parent(&rows, 1);
        assert_eq!(stats.len(), 1);
        let s = &stats[0];
        assert_eq!(s.parent_dir, "vendor");
        assert_eq!(s.file_count, 3);
        assert_eq!(s.indexed_count, 1);
        assert_eq!(s.treesitter_failed, 2);
        assert_eq!(s.lsp_failed, 1);
        assert_eq!(s.error_count, 1);
        assert_eq!(s.extensions.get("rs").copied(), Some(3));
    }

    #[test]
    fn score_rewards_failures_and_homogeneity() {
        let baseline = DirStats {
            parent_dir: "mixed".to_string(),
            file_count: 100,
            extensions: [("rs".to_string(), 50), ("toml".to_string(), 50)]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let s_base = compute_score(&baseline);

        let homogeneous = DirStats {
            parent_dir: "vendor".to_string(),
            file_count: 100,
            extensions: [("json".to_string(), 100)].into_iter().collect(),
            ..Default::default()
        };
        assert!(
            compute_score(&homogeneous) > s_base,
            "homogeneous folder should outscore mixed-extension folder of the same size"
        );

        let failing = DirStats {
            parent_dir: "broken".to_string(),
            file_count: 100,
            treesitter_failed: 100,
            extensions: [("rs".to_string(), 50), ("toml".to_string(), 50)]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        assert!(
            compute_score(&failing) > s_base,
            "folder with failures should outscore an otherwise identical clean folder"
        );
    }

    #[test]
    fn score_zero_for_empty_dir() {
        let s = DirStats {
            parent_dir: "empty".to_string(),
            file_count: 0,
            ..Default::default()
        };
        assert_eq!(compute_score(&s), 0.0);
    }

    #[test]
    fn score_caps_failure_rate_at_one() {
        // A pathological row with more failure events than 3·file_count
        // should not produce a divergent score.
        let pathological = DirStats {
            parent_dir: "weird".to_string(),
            file_count: 10,
            treesitter_failed: 20,
            lsp_failed: 20,
            error_count: 20,
            extensions: [("rs".to_string(), 10)].into_iter().collect(),
            ..Default::default()
        };
        let score = compute_score(&pathological);
        // file_count * (1 + 2*1 + 1) = file_count * 4 = 40
        assert!((score - 40.0).abs() < 1e-9);
    }

    #[test]
    fn score_and_filter_drops_small_dirs() {
        let stats = vec![
            DirStats {
                parent_dir: "big".to_string(),
                file_count: 100,
                ..Default::default()
            },
            DirStats {
                parent_dir: "tiny".to_string(),
                file_count: 3,
                ..Default::default()
            },
        ];
        let scored = score_and_filter(stats, 10);
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].0.parent_dir, "big");
    }

    #[test]
    fn top_extension_label_handles_empty_extensions() {
        let s = DirStats {
            parent_dir: "x".to_string(),
            file_count: 5,
            ..Default::default()
        };
        assert_eq!(top_extension_label(&s), "—");
    }

    #[test]
    fn top_extension_label_formats_percentage() {
        let s = DirStats {
            parent_dir: "x".to_string(),
            file_count: 100,
            extensions: [("rs".to_string(), 80), ("toml".to_string(), 20)]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        assert_eq!(top_extension_label(&s), ".rs (80%)");
    }

    // ─── Integration test against an in-memory SQLite ─────────────────────

    fn setup_in_memory_tracked_files() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        // Minimal subset of the post-v37 schema — only the columns this
        // command reads.
        conn.execute_batch(
            "CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                watch_folder_id TEXT NOT NULL,
                branch TEXT,
                relative_path TEXT,
                chunk_count INTEGER DEFAULT 0,
                lsp_status TEXT DEFAULT 'none',
                treesitter_status TEXT DEFAULT 'none',
                last_error TEXT,
                extension TEXT
            );",
        )
        .unwrap();
        conn
    }

    fn insert(
        conn: &Connection,
        rel: &str,
        chunks: i64,
        ts: &str,
        lsp: &str,
        err: Option<&str>,
        ext: &str,
    ) {
        conn.execute(
            "INSERT INTO tracked_files
                 (watch_folder_id, relative_path, chunk_count,
                  lsp_status, treesitter_status, last_error, extension)
             VALUES ('w', ?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![rel, chunks, lsp, ts, err, ext],
        )
        .unwrap();
    }

    #[test]
    fn query_tracked_files_returns_projected_rows() {
        let conn = setup_in_memory_tracked_files();
        insert(&conn, "src/a.rs", 5, "done", "done", None, "rs");
        insert(&conn, "src/b.rs", 0, "failed", "none", Some("oops"), "rs");
        insert(&conn, "", 0, "done", "none", None, "rs"); // filtered out
        insert(&conn, "vendor/c.json", 1, "skipped", "none", None, "json");

        let rows = query_tracked_files(&conn).unwrap();
        assert_eq!(rows.len(), 3, "rows with empty relative_path are skipped");

        let by_path: HashMap<_, _> = rows
            .iter()
            .map(|r| (r.relative_path.clone(), r.clone()))
            .collect();

        let b = &by_path["src/b.rs"];
        assert_eq!(b.chunk_count, 0);
        assert_eq!(b.treesitter_status, "failed");
        assert!(b.has_error, "non-empty last_error should set has_error");

        let a = &by_path["src/a.rs"];
        assert!(!a.has_error, "NULL last_error should leave has_error=false");
    }

    #[test]
    fn end_to_end_ranks_vendor_over_clean_src() {
        let rows = {
            let mut v = Vec::new();
            // Clean source code: 30 .rs files, no failures, mixed extensions
            for i in 0..20 {
                v.push(row(
                    &format!("src/rust/cli/mod_{i}.rs"),
                    5,
                    "done",
                    "done",
                    false,
                    "rs",
                ));
            }
            for i in 0..10 {
                v.push(row(
                    &format!("src/rust/cli/data_{i}.toml"),
                    1,
                    "done",
                    "none",
                    false,
                    "toml",
                ));
            }
            // Vendor folder: 40 homogeneous .json files, half tree-sitter failures
            for i in 0..40 {
                v.push(row(
                    &format!("vendor/dist/file_{i}.json"),
                    0,
                    if i % 2 == 0 { "failed" } else { "none" },
                    "none",
                    false,
                    "json",
                ));
            }
            v
        };

        let stats = aggregate_by_parent(&rows, 3);
        let scored = score_and_filter(stats, 5);

        let vendor_score = scored
            .iter()
            .find(|(s, _)| s.parent_dir == "vendor/dist")
            .map(|(_, sc)| *sc)
            .expect("vendor/dist should survive --min-files=5");
        let src_score = scored
            .iter()
            .find(|(s, _)| s.parent_dir == "src/rust/cli")
            .map(|(_, sc)| *sc)
            .expect("src/rust/cli should survive --min-files=5");

        assert!(
            vendor_score > src_score,
            "vendor/dist ({vendor_score}) should outrank src/rust/cli ({src_score})"
        );
    }
}
