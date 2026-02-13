//! Schema definition for the `qdrant_chunks` SQLite table.

use crate::schema::{FieldDef, TableDef};

pub const TABLE: TableDef = TableDef { name: "qdrant_chunks" };

pub const CHUNK_ID: FieldDef = FieldDef::categorical("chunk_id");
pub const FILE_ID: FieldDef = FieldDef::categorical("file_id");
pub const POINT_ID: FieldDef = FieldDef::categorical("point_id");
pub const CHUNK_INDEX: FieldDef = FieldDef::categorical("chunk_index");
pub const CONTENT_HASH: FieldDef = FieldDef::categorical("content_hash");
pub const CHUNK_TYPE: FieldDef = FieldDef::categorical("chunk_type");
pub const SYMBOL_NAME: FieldDef = FieldDef::content("symbol_name");
pub const START_LINE: FieldDef = FieldDef::categorical("start_line");
pub const END_LINE: FieldDef = FieldDef::categorical("end_line");
pub const CREATED_AT: FieldDef = FieldDef::categorical("created_at");

/// All columns in definition order.
pub const ALL_COLUMNS: &[FieldDef] = &[
    CHUNK_ID, FILE_ID, POINT_ID, CHUNK_INDEX,
    CONTENT_HASH, CHUNK_TYPE, SYMBOL_NAME,
    START_LINE, END_LINE, CREATED_AT,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FieldKind;

    #[test]
    fn table_name() {
        assert_eq!(TABLE.name, "qdrant_chunks");
    }

    #[test]
    fn column_count() {
        assert_eq!(ALL_COLUMNS.len(), 10);
    }

    #[test]
    fn content_columns_are_content() {
        let content: Vec<&str> = ALL_COLUMNS.iter()
            .filter(|f| matches!(f.kind, FieldKind::Content))
            .map(|f| f.name)
            .collect();
        assert!(content.contains(&"symbol_name"));
        assert_eq!(content.len(), 1);
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
