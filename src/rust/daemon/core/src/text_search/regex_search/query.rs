//! SQL query building and FTS5 candidate-count probe for regex search.

use crate::search_db::SearchDbError;
use crate::text_search::types::SearchOptions;

/// Lightweight FTS5-only candidate count probe.
pub(super) async fn fts5_exceeds_threshold(
    pool: &sqlx::SqlitePool,
    fts5_query: &str,
    threshold: i64,
) -> Result<bool, SearchDbError> {
    let result: Option<i64> = sqlx::query_scalar(
        "SELECT rowid FROM code_lines_fts WHERE content MATCH ?1 LIMIT 1 OFFSET ?2",
    )
    .bind(fts5_query)
    .bind(threshold)
    .fetch_optional(pool)
    .await?;
    Ok(result.is_some())
}

/// Build SQL query for regex search.
///
/// Returns `(sql, use_fts)` where `use_fts` indicates whether the FTS5 MATCH
/// clause was included (and therefore the first bind parameter is the FTS query).
pub(super) fn build_regex_search_query(
    fts5_query: &Option<String>,
    options: &SearchOptions,
) -> (String, bool) {
    let use_fts = fts5_query.is_some();

    let candidates_cte = if use_fts {
        r#"
WITH candidates AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    JOIN code_lines_fts fts ON cl.line_id = fts.rowid
    WHERE fts.content MATCH ?1
)"#
        .to_string()
    } else {
        r#"
WITH candidates AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
)"#
        .to_string()
    };

    let mut sql = format!(
        "{}
SELECT c.line_id, c.file_id, c.line_number, c.content,
       fm.file_path, fm.tenant_id, fm.branch, fm.size_bytes
FROM candidates c
JOIN file_metadata fm ON c.file_id = fm.file_id
WHERE 1=1",
        candidates_cte
    );

    let mut next_param = if use_fts { 2 } else { 1 };

    if options.tenant_id.is_some() {
        sql.push_str(&format!(" AND fm.tenant_id = ?{}", next_param));
        next_param += 1;
    }
    if options.branch.is_some() {
        sql.push_str(&format!(" AND fm.branch = ?{}", next_param));
        next_param += 1;
    }
    if options.path_prefix.is_some() {
        sql.push_str(&format!(
            " AND fm.file_path LIKE ?{} ESCAPE '\\'",
            next_param
        ));
    }

    sql.push_str("\nORDER BY c.file_id, c.line_number");

    (sql, use_fts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_search::types::SearchOptions;

    #[test]
    fn test_build_regex_query_with_fts() {
        let fts_query = Some("\"async\" OR \"await\"".to_string());
        let options = SearchOptions::default();
        let (sql, use_fts) = build_regex_search_query(&fts_query, &options);
        assert!(use_fts);
        assert!(sql.contains("MATCH ?1"));
        assert!(sql.contains("candidates"));
        assert!(sql.contains("cl.line_number"));
        assert!(!sql.contains("SELECT COUNT(*)"));
    }

    #[test]
    fn test_build_regex_query_without_fts() {
        let options = SearchOptions::default();
        let (sql, use_fts) = build_regex_search_query(&None, &options);
        assert!(!use_fts);
        assert!(!sql.contains("MATCH"));
        assert!(sql.contains("candidates"));
        assert!(!sql.contains("all_lines"));
    }

    #[test]
    fn test_build_regex_query_with_scope_filters() {
        let fts_query = Some("\"test\"".to_string());
        let options = SearchOptions {
            tenant_id: Some("proj1".to_string()),
            branch: Some("main".to_string()),
            path_prefix: Some("src/".to_string()),
            ..Default::default()
        };
        let (sql, _) = build_regex_search_query(&fts_query, &options);
        assert!(sql.contains("fm.tenant_id = ?2"));
        assert!(sql.contains("fm.branch = ?3"));
        assert!(sql.contains("fm.file_path LIKE ?4"));
    }
}
