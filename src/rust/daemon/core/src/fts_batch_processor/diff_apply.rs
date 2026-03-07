//! Diff-to-database application logic for code_lines updates.
//!
//! All FTS5 index updates are performed incrementally inline with code_lines
//! changes, avoiding the need for a full FTS5 rebuild after each file.

use std::collections::HashSet;

use crate::code_lines_schema::{initial_seq, INITIAL_SEQ_GAP};
use crate::code_lines_schema::{FTS5_DELETE_ROW_SQL, FTS5_INSERT_ROW_SQL};
use crate::line_diff::{DiffOp, DiffResult};
use crate::search_db::SearchDbError;

/// Per-file statistics from applying a diff.
#[derive(Debug, Default)]
pub(super) struct FileDiffStats {
    pub(super) lines_inserted: usize,
    pub(super) lines_updated: usize,
    pub(super) lines_deleted: usize,
    pub(super) lines_unchanged: usize,
    /// Internal: tracks retained line_ids during diff application (cleared before return).
    retained_ids: HashSet<i64>,
    /// Internal: tracks line_ids already deleted by explicit Deleted ops (cleared before return).
    explicitly_deleted_ids: HashSet<i64>,
}

/// Apply a diff result to the code_lines table within a transaction.
///
/// For files that already have code_lines, this applies the minimal set of
/// INSERT/UPDATE/DELETE operations. For new files (no existing lines),
/// all lines from `new_content` are inserted with standard seq gaps.
///
/// FTS5 index entries are maintained incrementally — no full rebuild needed.
pub(super) async fn apply_diff_to_code_lines(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
    diff: &DiffResult,
    new_content: &str,
) -> Result<FileDiffStats, SearchDbError> {
    let existing_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&mut **tx)
            .await?;

    if existing_count == 0 {
        return insert_all_lines(tx, file_id, new_content).await;
    }

    let existing_lines = fetch_existing_lines(tx, file_id).await?;
    let mut stats = FileDiffStats::default();
    let insertions = apply_diff_ops(tx, diff, &existing_lines, &mut stats).await?;
    let orphan_deleted =
        delete_orphaned_lines(tx, &existing_lines, &stats.retained_ids, &stats.explicitly_deleted_ids).await?;
    stats.lines_deleted += orphan_deleted;
    insert_pending_lines(tx, file_id, insertions, &mut stats).await?;
    renumber_after_changes(tx, file_id, &stats).await?;

    // Clear internal tracking fields before returning
    stats.retained_ids.clear();
    stats.explicitly_deleted_ids.clear();
    Ok(stats)
}

/// Insert all lines for first-time ingestion (no existing code_lines).
/// Also inserts corresponding FTS5 index entries.
async fn insert_all_lines(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
    content: &str,
) -> Result<FileDiffStats, SearchDbError> {
    let mut stats = FileDiffStats::default();
    let lines: Vec<&str> = content.split('\n').collect();
    for (i, line) in lines.iter().enumerate() {
        let seq = initial_seq(i);
        let line_number = (i + 1) as i64;
        let result = sqlx::query(
            "INSERT INTO code_lines (file_id, seq, content, line_number) VALUES (?1, ?2, ?3, ?4)",
        )
        .bind(file_id)
        .bind(seq)
        .bind(*line)
        .bind(line_number)
        .execute(&mut **tx)
        .await?;

        // Incrementally update FTS5 index
        let line_id = result.last_insert_rowid();
        sqlx::query(FTS5_INSERT_ROW_SQL)
            .bind(line_id)
            .bind(*line)
            .execute(&mut **tx)
            .await?;

        stats.lines_inserted += 1;
    }
    Ok(stats)
}

/// Fetch existing line_id, seq, and content values for a file, ordered by seq.
/// Content is needed for FTS5 incremental delete operations.
async fn fetch_existing_lines(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
) -> Result<Vec<(i64, f64, String)>, SearchDbError> {
    let rows =
        sqlx::query("SELECT line_id, seq, content FROM code_lines WHERE file_id = ?1 ORDER BY seq")
            .bind(file_id)
            .fetch_all(&mut **tx)
            .await?;

    Ok(rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            (
                r.get::<i64, _>("line_id"),
                r.get::<f64, _>("seq"),
                r.get::<String, _>("content"),
            )
        })
        .collect())
}

/// Process diff operations: update changed lines, collect insertions, delete removed lines.
/// Returns the list of pending insertions (new_index, content).
/// FTS5 index is updated incrementally for Changed and Deleted ops.
async fn apply_diff_ops(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    diff: &DiffResult,
    existing_lines: &[(i64, f64, String)],
    stats: &mut FileDiffStats,
) -> Result<Vec<(usize, String)>, SearchDbError> {
    let mut insertions: Vec<(usize, String)> = Vec::new();

    for op in &diff.ops {
        match op {
            DiffOp::Unchanged { old_index, .. } => {
                if let Some(&(line_id, _, _)) = existing_lines.get(*old_index) {
                    stats.retained_ids.insert(line_id);
                    stats.lines_unchanged += 1;
                }
            }
            DiffOp::Changed {
                old_index,
                new_content: content,
                ..
            } => {
                if let Some((line_id, _, old_content)) = existing_lines.get(*old_index) {
                    let line_id = *line_id;

                    // FTS5: delete old content, then insert new content
                    sqlx::query(FTS5_DELETE_ROW_SQL)
                        .bind(line_id)
                        .bind(old_content.as_str())
                        .execute(&mut **tx)
                        .await?;

                    sqlx::query("UPDATE code_lines SET content = ?1 WHERE line_id = ?2")
                        .bind(content.as_str())
                        .bind(line_id)
                        .execute(&mut **tx)
                        .await?;

                    sqlx::query(FTS5_INSERT_ROW_SQL)
                        .bind(line_id)
                        .bind(content.as_str())
                        .execute(&mut **tx)
                        .await?;

                    stats.retained_ids.insert(line_id);
                    stats.lines_updated += 1;
                }
            }
            DiffOp::Inserted {
                new_index,
                new_content: content,
            } => {
                insertions.push((*new_index, content.clone()));
            }
            DiffOp::Deleted { old_index } => {
                if let Some((line_id, _, old_content)) = existing_lines.get(*old_index) {
                    let line_id = *line_id;

                    // FTS5: delete before removing from code_lines
                    sqlx::query(FTS5_DELETE_ROW_SQL)
                        .bind(line_id)
                        .bind(old_content.as_str())
                        .execute(&mut **tx)
                        .await?;

                    sqlx::query("DELETE FROM code_lines WHERE line_id = ?1")
                        .bind(line_id)
                        .execute(&mut **tx)
                        .await?;
                    stats.lines_deleted += 1;
                    stats.explicitly_deleted_ids.insert(line_id);
                }
            }
        }
    }

    Ok(insertions)
}

/// Delete existing lines not referenced by any Unchanged or Changed op.
/// Also removes corresponding FTS5 index entries.
/// Returns the number of orphaned lines deleted.
async fn delete_orphaned_lines(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    existing_lines: &[(i64, f64, String)],
    retained_ids: &std::collections::HashSet<i64>,
    explicitly_deleted_ids: &std::collections::HashSet<i64>,
) -> Result<usize, SearchDbError> {
    let mut deleted = 0;
    for (line_id, _, old_content) in existing_lines {
        if !retained_ids.contains(line_id) && !explicitly_deleted_ids.contains(line_id) {
            // FTS5: delete before removing from code_lines
            sqlx::query(FTS5_DELETE_ROW_SQL)
                .bind(*line_id)
                .bind(old_content.as_str())
                .execute(&mut **tx)
                .await?;

            let result = sqlx::query("DELETE FROM code_lines WHERE line_id = ?1")
                .bind(*line_id)
                .execute(&mut **tx)
                .await?;
            if result.rows_affected() > 0 {
                deleted += 1;
            }
        }
    }
    Ok(deleted)
}

/// Insert pending lines, assigning seq values by appending after max_seq.
/// Also inserts corresponding FTS5 index entries.
async fn insert_pending_lines(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
    mut insertions: Vec<(usize, String)>,
    stats: &mut FileDiffStats,
) -> Result<(), SearchDbError> {
    insertions.sort_by_key(|(idx, _)| *idx);

    for (_, content) in &insertions {
        let max_seq: Option<f64> =
            sqlx::query_scalar("SELECT MAX(seq) FROM code_lines WHERE file_id = ?1")
                .bind(file_id)
                .fetch_optional(&mut **tx)
                .await?
                .flatten();

        let new_seq = match max_seq {
            Some(max) => max + INITIAL_SEQ_GAP,
            None => INITIAL_SEQ_GAP,
        };

        let result = sqlx::query(
            "INSERT INTO code_lines (file_id, seq, content, line_number) VALUES (?1, ?2, ?3, 0)",
        )
        .bind(file_id)
        .bind(new_seq)
        .bind(content.as_str())
        .execute(&mut **tx)
        .await?;

        // Incrementally update FTS5 index
        let line_id = result.last_insert_rowid();
        sqlx::query(FTS5_INSERT_ROW_SQL)
            .bind(line_id)
            .bind(content.as_str())
            .execute(&mut **tx)
            .await?;

        stats.lines_inserted += 1;
    }
    Ok(())
}

/// Renumber line_numbers sequentially after insertions or deletions.
async fn renumber_after_changes(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
    stats: &FileDiffStats,
) -> Result<(), SearchDbError> {
    if stats.lines_inserted == 0 && stats.lines_deleted == 0 {
        return Ok(());
    }

    let line_ids: Vec<i64> =
        sqlx::query_scalar("SELECT line_id FROM code_lines WHERE file_id = ?1 ORDER BY seq")
            .bind(file_id)
            .fetch_all(&mut **tx)
            .await?;

    for (i, line_id) in line_ids.iter().enumerate() {
        sqlx::query("UPDATE code_lines SET line_number = ?1 WHERE line_id = ?2")
            .bind((i + 1) as i64)
            .bind(*line_id)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}
