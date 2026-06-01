//! Filter-clause builder for `tracked_files` queries.
//!
//! Mirrors `buildFilterClause` in `tracked-files-queries/tracked-files.ts:47-98`.

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Options for filtering `tracked_files` rows.
///
/// Mirrors `ListTrackedFilesOptions` in `tracked-files-queries/tracked-files.ts`.
#[derive(Debug, Clone, Default)]
pub struct ListTrackedFilesOptions {
    pub watch_folder_id: String,
    pub path: Option<String>,
    pub file_type: Option<String>,
    pub language: Option<String>,
    pub extension: Option<String>,
    pub include_tests: Option<bool>,
    pub branch: Option<String>,
    pub limit: Option<usize>,
    /// Glob pattern (e.g. `"*.rs"`) — `**` is translated to `*` for SQLite GLOB.
    pub glob: Option<String>,
    /// Component base-path prefixes (OR logic).
    pub component_base_paths: Option<Vec<String>>,
    /// Keyset pagination cursor: `relative_path > cursor`.
    pub after_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal representation
// ---------------------------------------------------------------------------

/// Built WHERE clause and positional params.
pub(super) struct FilterClause {
    pub conditions: Vec<String>,
    pub params: Vec<FilterParam>,
}

/// A single parameter value.  We use an enum so the caller can pass a
/// heterogeneous list to rusqlite without boxing.
#[derive(Debug, Clone)]
pub(super) enum FilterParam {
    Text(String),
    Int(i64),
}

impl rusqlite::types::ToSql for FilterParam {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        match self {
            FilterParam::Text(s) => s.to_sql(),
            FilterParam::Int(i) => i.to_sql(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public builder
// ---------------------------------------------------------------------------

/// Build the `WHERE` conditions and positional params from filter options.
///
/// Always starts with `watch_folder_id = ?`.
pub(super) fn build_filter_clause(options: &ListTrackedFilesOptions) -> FilterClause {
    let mut conditions = vec!["watch_folder_id = ?".to_string()];
    let mut params: Vec<FilterParam> = vec![FilterParam::Text(options.watch_folder_id.clone())];

    if let Some(path) = &options.path {
        conditions.push("relative_path LIKE ?".to_string());
        params.push(FilterParam::Text(format!("{path}/%")));
    }

    if let Some(ft) = &options.file_type {
        conditions.push("file_type = ?".to_string());
        params.push(FilterParam::Text(ft.clone()));
    }

    if let Some(lang) = &options.language {
        conditions.push("language = ?".to_string());
        params.push(FilterParam::Text(lang.clone()));
    }

    if let Some(ext) = &options.extension {
        conditions.push("extension = ?".to_string());
        params.push(FilterParam::Text(ext.clone()));
    }

    let include_tests = options.include_tests.unwrap_or(true);
    if !include_tests {
        conditions.push("is_test = 0".to_string());
    }

    if let Some(branch) = &options.branch {
        conditions.push(
            "EXISTS (SELECT 1 FROM json_each(branches) WHERE json_each.value = ?)".to_string(),
        );
        params.push(FilterParam::Text(branch.clone()));
    }

    if let Some(glob) = &options.glob {
        // Translate ** → * for SQLite GLOB (matches TS logic).
        let sqlite_glob = glob.replace("**", "*");
        conditions.push("relative_path GLOB ?".to_string());
        params.push(FilterParam::Text(sqlite_glob));
    }

    if let Some(base_paths) = &options.component_base_paths {
        if !base_paths.is_empty() {
            let clauses: Vec<String> = base_paths
                .iter()
                .map(|_| "(relative_path = ? OR relative_path LIKE ?)".to_string())
                .collect();
            conditions.push(format!("({})", clauses.join(" OR ")));
            for bp in base_paths {
                params.push(FilterParam::Text(bp.clone()));
                params.push(FilterParam::Text(format!("{bp}/%")));
            }
        }
    }

    if let Some(after) = &options.after_path {
        conditions.push("relative_path > ?".to_string());
        params.push(FilterParam::Text(after.clone()));
    }

    FilterClause { conditions, params }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base_opts(id: &str) -> ListTrackedFilesOptions {
        ListTrackedFilesOptions {
            watch_folder_id: id.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn minimal_options_single_condition() {
        let clause = build_filter_clause(&base_opts("wid1"));
        assert_eq!(clause.conditions, vec!["watch_folder_id = ?"]);
        assert_eq!(clause.params.len(), 1);
    }

    #[test]
    fn path_filter_adds_like_condition() {
        let mut opts = base_opts("w");
        opts.path = Some("src/rust".to_string());
        let clause = build_filter_clause(&opts);
        assert!(clause
            .conditions
            .iter()
            .any(|c| c.contains("relative_path LIKE")));
        let param = clause
            .params
            .iter()
            .find(|p| matches!(p, FilterParam::Text(s) if s.ends_with("/%")));
        assert!(param.is_some());
    }

    #[test]
    fn exclude_tests_adds_condition() {
        let mut opts = base_opts("w");
        opts.include_tests = Some(false);
        let clause = build_filter_clause(&opts);
        assert!(clause.conditions.contains(&"is_test = 0".to_string()));
    }

    #[test]
    fn include_tests_true_no_condition() {
        let mut opts = base_opts("w");
        opts.include_tests = Some(true);
        let clause = build_filter_clause(&opts);
        assert!(!clause.conditions.contains(&"is_test = 0".to_string()));
    }

    #[test]
    fn glob_translates_double_star() {
        let mut opts = base_opts("w");
        opts.glob = Some("src/**/*.rs".to_string());
        let clause = build_filter_clause(&opts);
        let glob_param = clause.params.iter().find_map(|p| {
            if let FilterParam::Text(s) = p {
                if s.contains("*.rs") {
                    Some(s.clone())
                } else {
                    None
                }
            } else {
                None
            }
        });
        assert_eq!(glob_param.unwrap(), "src/*/*.rs");
    }

    #[test]
    fn component_base_paths_or_logic() {
        let mut opts = base_opts("w");
        opts.component_base_paths = Some(vec!["src/rust".to_string(), "src/ts".to_string()]);
        let clause = build_filter_clause(&opts);
        // Should produce one compound OR condition — the outer wrapper adds one extra
        // level of parens: "((relative_path = ? ...) OR (relative_path = ? ...))"
        let compound = clause
            .conditions
            .iter()
            .find(|c| c.contains("relative_path = ?") && c.contains("relative_path LIKE ?"));
        assert!(compound.is_some());
        // Two base paths × 2 params each = 4 extra params + 1 for watch_folder_id
        assert_eq!(clause.params.len(), 5);
    }

    #[test]
    fn after_path_cursor_added() {
        let mut opts = base_opts("w");
        opts.after_path = Some("src/main.rs".to_string());
        let clause = build_filter_clause(&opts);
        assert!(clause
            .conditions
            .iter()
            .any(|c| c.contains("relative_path > ?")));
    }
}
