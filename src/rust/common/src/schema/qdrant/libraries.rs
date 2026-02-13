//! Payload field definitions for the `libraries` Qdrant collection.
//!
//! The libraries collection stores reference documentation and library content,
//! isolated by `library_name`.

use crate::schema::FieldDef;

// Core identification (library_name is the tenant key)
pub const LIBRARY_NAME: FieldDef = FieldDef::categorical("library_name");
pub const DOCUMENT_ID: FieldDef = FieldDef::categorical("document_id");
pub const ITEM_TYPE: FieldDef = FieldDef::categorical("item_type");

// Content
pub const CONTENT: FieldDef = FieldDef::content("content");
pub const SOURCE_TYPE: FieldDef = FieldDef::categorical("source_type");
pub const MAIN_TAG: FieldDef = FieldDef::categorical("main_tag");
pub const FULL_TAG: FieldDef = FieldDef::categorical("full_tag");

// Library metadata
pub const BRANCH: FieldDef = FieldDef::categorical("branch");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_match_daemon() {
        assert_eq!(LIBRARY_NAME.name, "library_name");
        assert_eq!(CONTENT.name, "content");
        assert_eq!(DOCUMENT_ID.name, "document_id");
    }
}
