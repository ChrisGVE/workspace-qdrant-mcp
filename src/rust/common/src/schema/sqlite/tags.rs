//! Schema definition for the `tags` SQLite table.

use crate::schema::{FieldDef, TableDef};

pub const TABLE: TableDef = TableDef { name: "tags" };

pub const TAG_ID: FieldDef = FieldDef::categorical("tag_id");
pub const DOC_ID: FieldDef = FieldDef::categorical("doc_id");
pub const TAG: FieldDef = FieldDef::content("tag");
pub const TAG_TYPE: FieldDef = FieldDef::categorical("tag_type");
pub const SCORE: FieldDef = FieldDef::categorical("score");
pub const DIVERSITY_SCORE: FieldDef = FieldDef::categorical("diversity_score");
pub const BASKET_ID: FieldDef = FieldDef::categorical("basket_id");
pub const COLLECTION: FieldDef = FieldDef::categorical("collection");
pub const TENANT_ID: FieldDef = FieldDef::categorical("tenant_id");
pub const CREATED_AT: FieldDef = FieldDef::categorical("created_at");

/// All columns in definition order.
pub const ALL_COLUMNS: &[FieldDef] = &[
    TAG_ID,
    DOC_ID,
    TAG,
    TAG_TYPE,
    SCORE,
    DIVERSITY_SCORE,
    BASKET_ID,
    COLLECTION,
    TENANT_ID,
    CREATED_AT,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_name() {
        assert_eq!(TABLE.name, "tags");
    }

    #[test]
    fn column_count() {
        assert_eq!(ALL_COLUMNS.len(), 10);
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
