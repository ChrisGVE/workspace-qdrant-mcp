//! Canonical schema definitions for SQLite tables and Qdrant payload fields.
//!
//! Single source of truth for field names and classifications used across
//! daemon, CLI, and MCP server. All definitions are `const` with zero
//! runtime cost.
//!
//! # Field Classification
//!
//! Each field has a [`FieldKind`] that drives table layout in the CLI:
//! - **Categorical**: IDs, enums, dates, booleans, numbers, hashes — constrained
//!   to their widest atomic value
//! - **Content**: Variable-length text (paths, descriptions, error messages) —
//!   receives extra width
//! - **List**: Comma-separated or array values (tags, scopes)

pub mod qdrant;
pub mod sqlite;

/// Classification of a field for table layout purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// Fixed-width data: IDs, enums, dates, booleans, numbers, hashes.
    Categorical,
    /// Variable-length text that benefits from extra width.
    Content,
    /// Comma-separated or array values.
    List,
}

/// A named field with a layout classification.
#[derive(Debug, Clone, Copy)]
pub struct FieldDef {
    pub name: &'static str,
    pub kind: FieldKind,
}

impl FieldDef {
    /// Create a new categorical field.
    pub const fn categorical(name: &'static str) -> Self {
        Self { name, kind: FieldKind::Categorical }
    }

    /// Create a new content field.
    pub const fn content(name: &'static str) -> Self {
        Self { name, kind: FieldKind::Content }
    }

    /// Create a new list field.
    pub const fn list(name: &'static str) -> Self {
        Self { name, kind: FieldKind::List }
    }
}

/// A named table definition.
#[derive(Debug, Clone, Copy)]
pub struct TableDef {
    pub name: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_def_constructors() {
        let cat = FieldDef::categorical("id");
        assert_eq!(cat.name, "id");
        assert_eq!(cat.kind, FieldKind::Categorical);

        let content = FieldDef::content("description");
        assert_eq!(content.name, "description");
        assert_eq!(content.kind, FieldKind::Content);

        let list = FieldDef::list("tags");
        assert_eq!(list.name, "tags");
        assert_eq!(list.kind, FieldKind::List);
    }

    #[test]
    fn test_field_kind_equality() {
        assert_eq!(FieldKind::Categorical, FieldKind::Categorical);
        assert_ne!(FieldKind::Categorical, FieldKind::Content);
        assert_ne!(FieldKind::Content, FieldKind::List);
    }
}
