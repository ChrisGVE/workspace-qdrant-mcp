//! Schema definition for the `watch_folders` SQLite table.

use crate::schema::{FieldDef, TableDef};

pub const TABLE: TableDef = TableDef { name: "watch_folders" };

// Primary identification
pub const WATCH_ID: FieldDef = FieldDef::categorical("watch_id");
pub const PATH: FieldDef = FieldDef::content("path");
pub const COLLECTION: FieldDef = FieldDef::categorical("collection");
pub const TENANT_ID: FieldDef = FieldDef::categorical("tenant_id");

// Hierarchy (submodules)
pub const PARENT_WATCH_ID: FieldDef = FieldDef::categorical("parent_watch_id");
pub const SUBMODULE_PATH: FieldDef = FieldDef::content("submodule_path");

// Project-specific
pub const GIT_REMOTE_URL: FieldDef = FieldDef::content("git_remote_url");
pub const REMOTE_HASH: FieldDef = FieldDef::categorical("remote_hash");
pub const DISAMBIGUATION_PATH: FieldDef = FieldDef::content("disambiguation_path");
pub const IS_ACTIVE: FieldDef = FieldDef::categorical("is_active");
pub const LAST_ACTIVITY_AT: FieldDef = FieldDef::categorical("last_activity_at");
pub const IS_PAUSED: FieldDef = FieldDef::categorical("is_paused");
pub const PAUSE_START_TIME: FieldDef = FieldDef::categorical("pause_start_time");
pub const IS_ARCHIVED: FieldDef = FieldDef::categorical("is_archived");

// Library-specific
pub const LIBRARY_MODE: FieldDef = FieldDef::categorical("library_mode");

// Shared configuration
pub const FOLLOW_SYMLINKS: FieldDef = FieldDef::categorical("follow_symlinks");
pub const ENABLED: FieldDef = FieldDef::categorical("enabled");
pub const CLEANUP_ON_DISABLE: FieldDef = FieldDef::categorical("cleanup_on_disable");

// Timestamps
pub const CREATED_AT: FieldDef = FieldDef::categorical("created_at");
pub const UPDATED_AT: FieldDef = FieldDef::categorical("updated_at");
pub const LAST_SCAN: FieldDef = FieldDef::categorical("last_scan");

/// All columns in definition order.
pub const ALL_COLUMNS: &[FieldDef] = &[
    WATCH_ID, PATH, COLLECTION, TENANT_ID,
    PARENT_WATCH_ID, SUBMODULE_PATH,
    GIT_REMOTE_URL, REMOTE_HASH, DISAMBIGUATION_PATH,
    IS_ACTIVE, LAST_ACTIVITY_AT, IS_PAUSED, PAUSE_START_TIME, IS_ARCHIVED,
    LIBRARY_MODE,
    FOLLOW_SYMLINKS, ENABLED, CLEANUP_ON_DISABLE,
    CREATED_AT, UPDATED_AT, LAST_SCAN,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FieldKind;

    #[test]
    fn table_name() {
        assert_eq!(TABLE.name, "watch_folders");
    }

    #[test]
    fn column_count() {
        assert_eq!(ALL_COLUMNS.len(), 21);
    }

    #[test]
    fn content_columns_are_content() {
        let content: Vec<&str> = ALL_COLUMNS.iter()
            .filter(|f| matches!(f.kind, FieldKind::Content))
            .map(|f| f.name)
            .collect();
        assert!(content.contains(&"path"));
        assert!(content.contains(&"submodule_path"));
        assert!(content.contains(&"git_remote_url"));
        assert!(content.contains(&"disambiguation_path"));
    }

    #[test]
    fn no_duplicate_names() {
        let names: Vec<&str> = ALL_COLUMNS.iter().map(|f| f.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len(), "duplicate column names found");
    }
}
