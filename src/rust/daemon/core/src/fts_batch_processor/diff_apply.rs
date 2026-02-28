//! Diff-to-database application logic for code_lines updates.

use crate::code_lines_schema::{initial_seq, INITIAL_SEQ_GAP};
use crate::line_diff::{DiffOp, DiffResult};
use crate::search_db::SearchDbError;

/// Per-file statistics from applying a diff.
#[derive(Debug, Default)]
pub(super) struct FileDiffStats {
    pub(super) lines_inserted: usize,
    pub(super) lines_updated: usize,
    pub(super) lines_deleted: usize,
    pub(super) lines_unchanged: usize,
}

/// Apply a diff result to the code_lines table within a transaction.
///
/// For files that already have code_lines, this applies the minimal set of
/// INSERT/UPDATE/DELETE operations. For new files (no existing lines),
/// all lines from `new_content` are inserted with standard seq gaps.
///
/// The `new_content` parameter is used as fallback for full rewrite when
/// the file has no existing code_lines (first ingestion).
pub(super) async fn apply_diff_to_code_lines(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
    diff: &DiffResult,
    new_content: &str,
) -> Result<FileDiffStats, SearchDbError> {
    let mut stats = FileDiffStats::default();

    // Check if the file already has code_lines
    let existing_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM code_lines WHERE file_id = ?1",
    )
    .bind(file_id)
    .fetch_one(&mut **tx)
    .await?;

    if existing_count == 0 {
        // First ingestion: insert all lines from new_content
        let lines: Vec<&str> = new_content.split('\n').collect();
        for (i, line) in lines.iter().enumerate() {
            let seq = initial_seq(i);
            let line_number = (i + 1) as i64;
            sqlx::query("INSERT INTO code_lines (file_id, seq, content, line_number) VALUES (?1, ?2, ?3, ?4)")
                .bind(file_id)
                .bind(seq)
                .bind(*line)
                .bind(line_number)
                .execute(&mut **tx)
                .await?;
            stats.lines_inserted += 1;
        }
        return Ok(stats);
    }

    // File has existing lines — build a map from old line index to (line_id, seq)
    let rows = sqlx::query(
        "SELECT line_id, seq FROM code_lines WHERE file_id = ?1 ORDER BY seq",
    )
    .bind(file_id)
    .fetch_all(&mut **tx)
    .await?;

    let existing_lines: Vec<(i64, f64)> = rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            (r.get::<i64, _>("line_id"), r.get::<f64, _>("seq"))
        })
        .collect();

    // Track which existing line_ids to delete (those not referenced by Unchanged or Changed)
    let mut retained_line_ids = std::collections::HashSet::new();

    // Track insertions that need seq assignment after we know the final layout
    let mut insertions: Vec<(usize, String)> = Vec::new(); // (new_index, content)

    for op in &diff.ops {
        match op {
            DiffOp::Unchanged { old_index, .. } => {
                if let Some(&(line_id, _)) = existing_lines.get(*old_index) {
                    retained_line_ids.insert(line_id);
                    stats.lines_unchanged += 1;
                }
            }
            DiffOp::Changed {
                old_index,
                new_content: content,
                ..
            } => {
                if let Some(&(line_id, _)) = existing_lines.get(*old_index) {
                    // Update content in place (seq stays the same)
                    sqlx::query("UPDATE code_lines SET content = ?1 WHERE line_id = ?2")
                        .bind(content.as_str())
                        .bind(line_id)
                        .execute(&mut **tx)
                        .await?;
                    retained_line_ids.insert(line_id);
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
                if let Some(&(line_id, _)) = existing_lines.get(*old_index) {
                    sqlx::query("DELETE FROM code_lines WHERE line_id = ?1")
                        .bind(line_id)
                        .execute(&mut **tx)
                        .await?;
                    stats.lines_deleted += 1;
                }
            }
        }
    }

    // Delete any existing lines not in the retained set (orphaned by the diff)
    for &(line_id, _) in &existing_lines {
        if !retained_line_ids.contains(&line_id) {
            // Already handled by DiffOp::Deleted above, but this catches edge cases
            // where the diff algorithm doesn't produce explicit Delete ops
            let result = sqlx::query("DELETE FROM code_lines WHERE line_id = ?1")
                .bind(line_id)
                .execute(&mut **tx)
                .await?;
            if result.rows_affected() > 0 {
                stats.lines_deleted += 1;
            }
        }
    }

    // Handle insertions: assign seq values based on surrounding lines
    // Sort insertions by new_index so we can process them in order
    let mut sorted_insertions = insertions;
    sorted_insertions.sort_by_key(|(idx, _)| *idx);

    for (_, content) in &sorted_insertions {
        // For simplicity in the transactional context, append at end with max_seq + gap
        let max_seq: Option<f64> = sqlx::query_scalar(
            "SELECT MAX(seq) FROM code_lines WHERE file_id = ?1",
        )
        .bind(file_id)
        .fetch_optional(&mut **tx)
        .await?
        .flatten();

        let new_seq = match max_seq {
            Some(max) => max + INITIAL_SEQ_GAP,
            None => INITIAL_SEQ_GAP,
        };

        // line_number=0 as placeholder; renumbered below after all insertions
        sqlx::query("INSERT INTO code_lines (file_id, seq, content, line_number) VALUES (?1, ?2, ?3, 0)")
            .bind(file_id)
            .bind(new_seq)
            .bind(content.as_str())
            .execute(&mut **tx)
            .await?;
        stats.lines_inserted += 1;
    }

    // Renumber line_numbers for this file after diff operations if any insertions/deletions occurred
    if stats.lines_inserted > 0 || stats.lines_deleted > 0 {
        let line_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT line_id FROM code_lines WHERE file_id = ?1 ORDER BY seq",
        )
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
    }

    Ok(stats)
}
