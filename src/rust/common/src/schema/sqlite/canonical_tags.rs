//! Schema definition for the `canonical_tags` SQLite table.

use crate::schema::{FieldDef, TableDef};

pub const TABLE: TableDef = TableDef {
    name: "canonical_tags",
};

pub const CANONICAL_ID: FieldDef = FieldDef::categorical("canonical_id");
pub const CANONICAL_NAME: FieldDef = FieldDef::content("canonical_name");
pub const CENTROID_VECTOR_ID: FieldDef = FieldDef::categorical("centroid_vector_id");
pub const LEVEL: FieldDef = FieldDef::categorical("level");
pub const PARENT_ID: FieldDef = FieldDef::categorical("parent_id");
pub const TENANT_ID: FieldDef = FieldDef::categorical("tenant_id");
pub const COLLECTION: FieldDef = FieldDef::categorical("collection");
pub const CREATED_AT: FieldDef = FieldDef::categorical("created_at");

/// All columns in definition order.
pub const ALL_COLUMNS: &[FieldDef] = &[
    CANONICAL_ID,
    CANONICAL_NAME,
    CENTROID_VECTOR_ID,
    LEVEL,
    PARENT_ID,
    TENANT_ID,
    COLLECTION,
    CREATED_AT,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_name() {
        assert_eq!(TABLE.name, "canonical_tags");
    }

    #[test]
    fn column_count() {
        assert_eq!(ALL_COLUMNS.len(), 8);
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
