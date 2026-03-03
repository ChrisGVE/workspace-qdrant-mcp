/// Database query helpers for resolving file paths before grep scanning.

use sqlx::Row;

use crate::search_db::{SearchDbManager, SearchDbError};
use crate::text_search::{compile_glob_matcher, resolve_path_filter, SearchOptions};

use super::types::FileInfo;

/// Query file_metadata for file paths matching scope filters.
pub(super) async fn query_file_paths(
    search_db: &SearchDbManager,
    options: &SearchOptions,
    glob_matcher: Option<&Box<dyn Fn(&str) -> bool + Send + Sync>>,
) -> Result<Vec<FileInfo>, SearchDbError> {
    let mut sql = String::from(
        "SELECT file_path, tenant_id, branch FROM file_metadata WHERE 1=1",
    );
    let mut next_param = 1;

    if options.tenant_id.is_some() {
        sql.push_str(&format!(" AND tenant_id = ?{}", next_param));
        next_param += 1;
    }
    if options.branch.is_some() {
        sql.push_str(&format!(" AND branch = ?{}", next_param));
        next_param += 1;
    }
    if options.path_prefix.is_some() {
        sql.push_str(&format!(" AND file_path LIKE ?{} ESCAPE '\\'", next_param));
    }

    sql.push_str(" ORDER BY file_path");

    let pool = search_db.pool();
    let mut query = sqlx::query(&sql);

    if let Some(ref tid) = options.tenant_id {
        query = query.bind(tid);
    }
    if let Some(ref branch) = options.branch {
        query = query.bind(branch);
    }
    if let Some(ref prefix) = options.path_prefix {
        query = query.bind(format!("{}%", prefix));
    }

    let rows = query.fetch_all(pool).await?;

    let mut files = Vec::with_capacity(rows.len());
    for row in rows {
        let file_path: String = row.get("file_path");
        // Apply glob filter
        if let Some(matcher) = glob_matcher {
            if !matcher(&file_path) {
                continue;
            }
        }
        files.push(FileInfo {
            file_path,
            tenant_id: row.get("tenant_id"),
            branch: row.get("branch"),
        });
    }

    Ok(files)
}

/// Resolve path glob filter options and return a compiled matcher if needed.
pub(super) fn resolve_and_compile(
    options: &SearchOptions,
) -> Result<
    (SearchOptions, Option<Box<dyn Fn(&str) -> bool + Send + Sync>>),
    SearchDbError,
> {
    let (glob_pattern, effective_options) = resolve_path_filter(options);
    let glob_matcher = glob_pattern
        .as_deref()
        .map(compile_glob_matcher)
        .transpose()?;
    Ok((effective_options, glob_matcher))
}
