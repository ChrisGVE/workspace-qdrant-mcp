//! Schema definition for the `tracked_files` SQLite table.

use crate::schema::{FieldDef, TableDef};

pub const TABLE: TableDef = TableDef { name: "tracked_files" };

pub const FILE_ID: FieldDef = FieldDef::categorical("file_id");
pub const WATCH_FOLDER_ID: FieldDef = FieldDef::categorical("watch_folder_id");
pub const FILE_PATH: FieldDef = FieldDef::content("file_path");
pub const BRANCH: FieldDef = FieldDef::categorical("branch");
pub const FILE_TYPE: FieldDef = FieldDef::categorical("file_type");
pub const LANGUAGE: FieldDef = FieldDef::categorical("language");
pub const FILE_MTIME: FieldDef = FieldDef::categorical("file_mtime");
pub const FILE_HASH: FieldDef = FieldDef::categorical("file_hash");
pub const CHUNK_COUNT: FieldDef = FieldDef::categorical("chunk_count");
pub const CHUNKING_METHOD: FieldDef = FieldDef::categorical("chunking_method");
pub const LSP_STATUS: FieldDef = FieldDef::categorical("lsp_status");
pub const TREESITTER_STATUS: FieldDef = FieldDef::categorical("treesitter_status");
pub const LAST_ERROR: FieldDef = FieldDef::content("last_error");
pub const NEEDS_RECONCILE: FieldDef = FieldDef::categorical("needs_reconcile");
pub const RECONCILE_REASON: FieldDef = FieldDef::content("reconcile_reason");
pub const COLLECTION: FieldDef = FieldDef::categorical("collection");
pub const CREATED_AT: FieldDef = FieldDef::categorical("created_at");
pub const UPDATED_AT: FieldDef = FieldDef::categorical("updated_at");

/// All columns in definition order.
pub const ALL_COLUMNS: &[FieldDef] = &[
    FILE_ID, WATCH_FOLDER_ID, FILE_PATH, BRANCH,
    FILE_TYPE, LANGUAGE, FILE_MTIME, FILE_HASH,
    CHUNK_COUNT, CHUNKING_METHOD, LSP_STATUS, TREESITTER_STATUS,
    LAST_ERROR, NEEDS_RECONCILE, RECONCILE_REASON,
    COLLECTION, CREATED_AT, UPDATED_AT,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FieldKind;

    #[test]
    fn table_name() {
        assert_eq!(TABLE.name, "tracked_files");
    }

    #[test]
    fn column_count() {
        assert_eq!(ALL_COLUMNS.len(), 18);
    }

    #[test]
    fn content_columns_are_content() {
        let content: Vec<&str> = ALL_COLUMNS.iter()
            .filter(|f| matches!(f.kind, FieldKind::Content))
            .map(|f| f.name)
            .collect();
        assert!(content.contains(&"file_path"));
        assert!(content.contains(&"last_error"));
        assert!(content.contains(&"reconcile_reason"));
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
