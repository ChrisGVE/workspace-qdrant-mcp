//! Tenant name resolution utilities
//!
//! Provides mapping from tenant_id to human-readable project names.
//! Used across commands that display tenant IDs (rules, scratchpad, watch, queue).

use std::collections::HashMap;
use std::path::Path;

use wqm_common::schema::sqlite::watch_folders as wf_schema;

use crate::config::get_database_path_checked;

/// Build a tenant_id -> project name mapping from watch_folders.
///
/// Extracts the last path component as the project name. Returns an
/// empty map if the database is unavailable.
pub fn load_project_names() -> HashMap<String, String> {
    let mut map = HashMap::new();
    let db_path = match get_database_path_checked() {
        Ok(p) => p,
        Err(_) => return map,
    };
    let conn = match rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        Ok(c) => c,
        Err(_) => return map,
    };
    let _ = conn.execute_batch("PRAGMA busy_timeout=5000;");
    let sql = format!(
        "SELECT {}, {} FROM {} WHERE {} = 'projects'",
        wf_schema::TENANT_ID.name,
        wf_schema::PATH.name,
        wf_schema::TABLE.name,
        wf_schema::COLLECTION.name,
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return map,
    };
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    });
    if let Ok(rows) = rows {
        for row in rows.flatten() {
            let (tenant_id, path) = row;
            let name = Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            map.insert(tenant_id, name);
        }
    }
    map
}

/// Resolve a tenant_id to a project name, falling back to the ID itself.
pub fn resolve_tenant_name(tenant_id: &str, names: &HashMap<String, String>) -> String {
    names
        .get(tenant_id)
        .cloned()
        .unwrap_or_else(|| tenant_id.to_string())
}
