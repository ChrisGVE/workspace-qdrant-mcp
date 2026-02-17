//! Schema definition for the `keywords` SQLite table.

use crate::schema::{FieldDef, TableDef};

pub const TABLE: TableDef = TableDef { name: "keywords" };

pub const KEYWORD_ID: FieldDef = FieldDef::categorical("keyword_id");
pub const DOC_ID: FieldDef = FieldDef::categorical("doc_id");
pub const KEYWORD: FieldDef = FieldDef::content("keyword");
pub const SCORE: FieldDef = FieldDef::categorical("score");
pub const SEMANTIC_SCORE: FieldDef = FieldDef::categorical("semantic_score");
pub const LEXICAL_SCORE: FieldDef = FieldDef::categorical("lexical_score");
pub const STABILITY_COUNT: FieldDef = FieldDef::categorical("stability_count");
pub const COLLECTION: FieldDef = FieldDef::categorical("collection");
pub const TENANT_ID: FieldDef = FieldDef::categorical("tenant_id");
pub const CREATED_AT: FieldDef = FieldDef::categorical("created_at");

/// All columns in definition order.
pub const ALL_COLUMNS: &[FieldDef] = &[
    KEYWORD_ID, DOC_ID, KEYWORD, SCORE, SEMANTIC_SCORE,
    LEXICAL_SCORE, STABILITY_COUNT, COLLECTION, TENANT_ID, CREATED_AT,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_name() {
        assert_eq!(TABLE.name, "keywords");
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
