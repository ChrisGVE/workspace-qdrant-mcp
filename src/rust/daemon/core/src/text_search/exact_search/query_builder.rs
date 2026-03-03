//! SQL query construction for exact substring search.
//!
//! Builds parameterized SQLite queries that use FTS5 trigram pre-filtering
//! combined with INSTR verification for exact substring matching.

use super::super::types::SearchOptions;

/// Build the search SQL query based on options.
///
/// Returns `(sql_string, uses_fts5)`. When patterns are < 3 chars, FTS5
/// trigram index cannot be used and we fall back to INSTR-only scan.
pub(super) fn build_search_query(
    fts5_pattern: &Option<String>,
    options: &SearchOptions,
) -> (String, bool) {
    let use_fts = fts5_pattern.is_some();

    let matching_cte = if use_fts {
        if options.case_insensitive {
            r#"
WITH matching AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    JOIN code_lines_fts fts ON cl.line_id = fts.rowid
    WHERE fts.content MATCH ?1 AND INSTR(LOWER(cl.content), ?2) > 0
)"#
            .to_string()
        } else {
            r#"
WITH matching AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    JOIN code_lines_fts fts ON cl.line_id = fts.rowid
    WHERE fts.content MATCH ?1 AND INSTR(cl.content, ?2) > 0
)"#
            .to_string()
        }
    } else if options.case_insensitive {
        r#"
WITH matching AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    WHERE INSTR(LOWER(cl.content), ?1) > 0
)"#
        .to_string()
    } else {
        r#"
WITH matching AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    WHERE INSTR(cl.content, ?1) > 0
)"#
        .to_string()
    };

    let mut sql = format!(
        "{}
SELECT m.line_id, m.file_id, m.line_number, m.content,
       fm.file_path, fm.tenant_id, fm.branch
FROM matching m
JOIN file_metadata fm ON m.file_id = fm.file_id
WHERE 1=1",
        matching_cte
    );

    let mut next_param = if use_fts { 3 } else { 2 };

    if options.tenant_id.is_some() {
        sql.push_str(&format!(" AND fm.tenant_id = ?{}", next_param));
        next_param += 1;
    }
    if options.branch.is_some() {
        sql.push_str(&format!(" AND fm.branch = ?{}", next_param));
        next_param += 1;
    }
    if options.path_prefix.is_some() {
        sql.push_str(&format!(" AND fm.file_path LIKE ?{} ESCAPE '\\'", next_param));
    }

    sql.push_str("\nORDER BY m.file_id, m.line_number");

    (sql, use_fts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_query_with_fts5() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions::default();
        let (sql, use_fts) = build_search_query(&pattern, &options);
        assert!(use_fts);
        assert!(sql.contains("MATCH ?1"));
        assert!(sql.contains("INSTR(cl.content, ?2)"));
        assert!(sql.contains("matching"));
        assert!(sql.contains("cl.line_number"));
        assert!(!sql.contains("SELECT COUNT(*)"));
    }

    #[test]
    fn test_build_query_without_fts5() {
        let options = SearchOptions::default();
        let (sql, use_fts) = build_search_query(&None, &options);
        assert!(!use_fts);
        assert!(!sql.contains("MATCH"));
        assert!(sql.contains("INSTR(cl.content, ?1)"));
    }

    #[test]
    fn test_build_query_with_tenant_filter() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions {
            tenant_id: Some("proj1".to_string()),
            ..Default::default()
        };
        let (sql, _) = build_search_query(&pattern, &options);
        assert!(sql.contains("fm.tenant_id = ?3"));
    }

    #[test]
    fn test_build_query_case_insensitive() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions {
            case_insensitive: true,
            ..Default::default()
        };
        let (sql, _) = build_search_query(&pattern, &options);
        assert!(sql.contains("INSTR(LOWER(cl.content), ?2)"));
    }

    #[test]
    fn test_build_query_with_all_filters() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions {
            tenant_id: Some("proj1".to_string()),
            branch: Some("main".to_string()),
            path_prefix: Some("src/".to_string()),
            ..Default::default()
        };
        let (sql, _) = build_search_query(&pattern, &options);
        assert!(sql.contains("fm.tenant_id = ?3"));
        assert!(sql.contains("fm.branch = ?4"));
        assert!(sql.contains("fm.file_path LIKE ?5"));
    }
}
