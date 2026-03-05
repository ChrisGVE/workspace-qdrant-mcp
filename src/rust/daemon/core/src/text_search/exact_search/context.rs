//! Context line retrieval for exact search matches.
//!
//! Groups matches by file, fetches surrounding lines from code_lines,
//! and attaches them as `context_before` / `context_after` on each match.

use std::collections::HashMap;

use sqlx::Row;

use super::super::types::SearchMatch;
use crate::search_db::{SearchDbError, SearchDbManager};

/// Attach context lines (before/after) to search matches.
///
/// Groups matches by file_id, fetches the needed line ranges from code_lines,
/// then distributes context to each match. Uses a single query per file for
/// efficiency.
pub(crate) async fn attach_context_lines(
    search_db: &SearchDbManager,
    matches: &mut [SearchMatch],
    context_lines: usize,
) -> Result<(), SearchDbError> {
    if matches.is_empty() || context_lines == 0 {
        return Ok(());
    }

    let context_n = context_lines as i64;

    // Group match indices by file_id
    let mut file_matches: HashMap<i64, Vec<usize>> = HashMap::new();
    for (idx, m) in matches.iter().enumerate() {
        file_matches.entry(m.file_id).or_default().push(idx);
    }

    let pool = search_db.pool();

    for (file_id, match_indices) in &file_matches {
        let min_line = match_indices
            .iter()
            .map(|&i| matches[i].line_number)
            .min()
            .unwrap();
        let max_line = match_indices
            .iter()
            .map(|&i| matches[i].line_number)
            .max()
            .unwrap();

        let range_start = (min_line - context_n).max(1);
        let range_end = max_line + context_n;

        let rows = sqlx::query(
            r#"
SELECT line_number, content
FROM code_lines
WHERE file_id = ?1 AND line_number BETWEEN ?2 AND ?3
ORDER BY line_number"#,
        )
        .bind(file_id)
        .bind(range_start)
        .bind(range_end)
        .fetch_all(pool)
        .await?;

        let line_map: HashMap<i64, String> = rows
            .iter()
            .map(|row| {
                (
                    row.get::<i64, _>("line_number"),
                    row.get::<String, _>("content"),
                )
            })
            .collect();

        for &idx in match_indices {
            let match_line = matches[idx].line_number;

            let before_start = (match_line - context_n).max(1);
            for ln in before_start..match_line {
                if let Some(content) = line_map.get(&ln) {
                    matches[idx].context_before.push(content.clone());
                }
            }

            for ln in (match_line + 1)..=(match_line + context_n) {
                if let Some(content) = line_map.get(&ln) {
                    matches[idx].context_after.push(content.clone());
                }
            }
        }
    }

    Ok(())
}
