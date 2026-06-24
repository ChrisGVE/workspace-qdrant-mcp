//! SQLite store discovery and read-consistent copy helpers (F20 / AC-F20.1).
//!
//! Enumerates all SQLite truth stores under the data directory:
//!   - `<data_dir>/state.db`            (central index)
//!   - `<data_dir>/projects/<id>/store.db`
//!   - `<data_dir>/global/store.db`
//!   - `<data_dir>/libraries/store.db`
//!
//! Each store is copied read-consistently via `VACUUM INTO '<dest>'`
//! (rusqlite) so a concurrent writer cannot tear the backup snapshot.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::manifest::StoreEntry;

/// A discovered SQLite store ready for backup.
#[derive(Debug, Clone)]
pub(crate) struct DiscoveredStore {
    /// Absolute path of the source store file.
    pub abs_path: PathBuf,
    /// Path relative to the data directory (used as the in-archive member name
    /// and in the manifest).
    pub rel_path: String,
    /// Tenant / namespace identifier extracted from the directory path; `None`
    /// for `state.db` and `global/store.db`.
    pub tenant_id: Option<String>,
}

/// Discover all SQLite truth stores under `data_dir`.
///
/// Only files that actually exist are returned; missing optional stores
/// (e.g. `global/store.db` before first use) are silently skipped.
pub(crate) fn discover_stores(data_dir: &Path) -> Vec<DiscoveredStore> {
    let mut stores: Vec<DiscoveredStore> = Vec::new();

    // 1. Central state.db
    let state_db = data_dir.join("state.db");
    if state_db.exists() {
        stores.push(DiscoveredStore {
            abs_path: state_db,
            rel_path: "state.db".into(),
            tenant_id: None,
        });
    }

    // 2. global/store.db
    let global_db = data_dir.join("global").join("store.db");
    if global_db.exists() {
        stores.push(DiscoveredStore {
            abs_path: global_db,
            rel_path: "global/store.db".into(),
            tenant_id: None,
        });
    }

    // 3. libraries/store.db
    let libraries_db = data_dir.join("libraries").join("store.db");
    if libraries_db.exists() {
        stores.push(DiscoveredStore {
            abs_path: libraries_db,
            rel_path: "libraries/store.db".into(),
            tenant_id: None,
        });
    }

    // 4. projects/<tenant_id>/store.db
    let projects_dir = data_dir.join("projects");
    if projects_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&projects_dir) {
            let mut project_stores: Vec<DiscoveredStore> = entries
                .flatten()
                .filter_map(|entry| {
                    let tenant_path = entry.path();
                    if !tenant_path.is_dir() {
                        return None;
                    }
                    let tenant_id = tenant_path.file_name()?.to_str()?.to_string();
                    let store = tenant_path.join("store.db");
                    if !store.exists() {
                        return None;
                    }
                    let rel = format!("projects/{}/store.db", tenant_id);
                    Some(DiscoveredStore {
                        abs_path: store,
                        rel_path: rel,
                        tenant_id: Some(tenant_id),
                    })
                })
                .collect();
            // Sort for deterministic archive member order.
            project_stores.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
            stores.extend(project_stores);
        }
    }

    stores
}

/// Total byte size of all discovered stores (used for the pre-flight check).
pub(crate) fn total_store_bytes(stores: &[DiscoveredStore]) -> u64 {
    stores
        .iter()
        .filter_map(|s| std::fs::metadata(&s.abs_path).ok())
        .map(|m| m.len())
        .sum()
}

/// Copy `src` to `dest` using `VACUUM INTO` for read-consistency.
///
/// `VACUUM INTO` copies the database with a shared read lock, ensuring no
/// concurrent WAL writer can produce a torn page in the backup copy.
pub(crate) fn vacuum_into(src: &Path, dest: &Path) -> Result<()> {
    let conn = rusqlite::Connection::open(src)
        .with_context(|| format!("open source db: {}", src.display()))?;

    let dest_str = dest
        .to_str()
        .with_context(|| format!("non-UTF-8 dest path: {}", dest.display()))?;

    conn.execute_batch(&format!("VACUUM INTO '{}'", dest_str.replace('\'', "''")))
        .with_context(|| {
            format!(
                "VACUUM INTO failed: src={}, dest={}",
                src.display(),
                dest.display()
            )
        })?;

    Ok(())
}

/// Convert a `DiscoveredStore` to its `StoreEntry` for the manifest.
///
/// Attempts to read `content_key_version` from the store's schema_meta table;
/// returns `None` gracefully when the table or column is absent (pre-F20 stores).
pub(crate) fn to_store_entry(store: &DiscoveredStore) -> StoreEntry {
    let ckv = read_content_key_version(&store.abs_path);
    StoreEntry {
        rel_path: store.rel_path.clone(),
        tenant_id: store.tenant_id.clone(),
        content_key_version: ckv,
    }
}

fn read_content_key_version(path: &Path) -> Option<u32> {
    let conn = rusqlite::Connection::open(path).ok()?;
    // The schema_meta table may not exist in pre-F20 stores.
    conn.query_row(
        "SELECT value FROM schema_meta WHERE key = 'content_key_version' LIMIT 1",
        [],
        |row| row.get::<_, u32>(0),
    )
    .ok()
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
#[path = "stores_tests.rs"]
mod tests;
