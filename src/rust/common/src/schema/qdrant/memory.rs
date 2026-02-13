//! Payload field definitions for the `memory` Qdrant collection.
//!
//! The memory collection stores behavioral rules and LLM memory entries.

use crate::schema::FieldDef;

// Core identification
pub const LABEL: FieldDef = FieldDef::categorical("label");
pub const DOCUMENT_ID: FieldDef = FieldDef::categorical("document_id");

// Content
pub const CONTENT: FieldDef = FieldDef::content("content");
pub const TITLE: FieldDef = FieldDef::content("title");

// Classification
pub const SCOPE: FieldDef = FieldDef::categorical("scope");
pub const PROJECT_ID: FieldDef = FieldDef::categorical("project_id");
pub const SOURCE_TYPE: FieldDef = FieldDef::categorical("source_type");
pub const PRIORITY: FieldDef = FieldDef::categorical("priority");
pub const TAGS: FieldDef = FieldDef::list("tags");
pub const ENABLED: FieldDef = FieldDef::categorical("enabled");

// Metadata
pub const ITEM_TYPE: FieldDef = FieldDef::categorical("item_type");
pub const BRANCH: FieldDef = FieldDef::categorical("branch");
pub const TENANT_ID: FieldDef = FieldDef::categorical("tenant_id");

// Timestamps
pub const CREATED_AT: FieldDef = FieldDef::categorical("created_at");
pub const UPDATED_AT: FieldDef = FieldDef::categorical("updated_at");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FieldKind;

    #[test]
    fn field_names_match_daemon() {
        assert_eq!(LABEL.name, "label");
        assert_eq!(TITLE.name, "title");
        assert_eq!(CONTENT.name, "content");
        assert_eq!(SCOPE.name, "scope");
        assert_eq!(PROJECT_ID.name, "project_id");
        assert_eq!(PRIORITY.name, "priority");
        assert_eq!(TAGS.name, "tags");
    }

    #[test]
    fn tags_is_list_kind() {
        assert_eq!(TAGS.kind, FieldKind::List);
    }

    #[test]
    fn title_and_content_are_content_kind() {
        assert_eq!(TITLE.kind, FieldKind::Content);
        assert_eq!(CONTENT.kind, FieldKind::Content);
    }
}
