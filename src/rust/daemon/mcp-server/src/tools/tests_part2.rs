//! Schema parity tests, part 2: rules, store, grep, list, embedding, and
//! structural schema checks.
//!
//! Included from `tests.rs` via `#[path = "tests_part2.rs"] mod part2;`.

use serde_json::Value;

use super::{enum_vals, prop, props, required, schema_for};
use crate::tools::definitions::list_tools;

// ---------------------------------------------------------------------------
// rules schema
// ---------------------------------------------------------------------------

#[test]
fn rules_required_is_action_only() {
    let s = schema_for("rules");
    assert_eq!(required(&s), vec!["action"]);
}

#[test]
fn rules_action_enum_values() {
    let s = schema_for("rules");
    let enums = enum_vals(prop(&s, "action"));
    assert_eq!(enums, vec!["add", "update", "remove", "list"]);
}

#[test]
fn rules_scope_enum_values() {
    let s = schema_for("rules");
    let enums = enum_vals(prop(&s, "scope"));
    assert_eq!(enums, vec!["global", "project"]);
}

#[test]
fn rules_tags_is_array_of_string() {
    let s = schema_for("rules");
    let p = prop(&s, "tags");
    assert_eq!(p.get("type").and_then(|v| v.as_str()), Some("array"));
    let items = p.get("items").expect("tags must have items");
    assert_eq!(items.get("type").and_then(|v| v.as_str()), Some("string"));
}

#[test]
fn rules_all_properties_present() {
    let s = schema_for("rules");
    for field in &[
        "action",
        "content",
        "label",
        "scope",
        "projectId",
        "title",
        "tags",
        "priority",
        "limit",
    ] {
        assert!(
            props(&s).get(field).is_some(),
            "rules is missing '{}'",
            field
        );
    }
}

// ---------------------------------------------------------------------------
// store schema
// ---------------------------------------------------------------------------

#[test]
fn store_no_required_fields() {
    let s = schema_for("store");
    assert!(
        s.get("required").is_none(),
        "store should have no required array"
    );
}

#[test]
fn store_type_enum_values() {
    let s = schema_for("store");
    let enums = enum_vals(prop(&s, "type"));
    assert_eq!(enums, vec!["library", "url", "scratchpad", "project"]);
}

#[test]
fn store_source_type_enum_values() {
    let s = schema_for("store");
    let enums = enum_vals(prop(&s, "sourceType"));
    assert_eq!(
        enums,
        vec!["user_input", "web", "file", "scratchbook", "note"]
    );
}

#[test]
fn store_tags_is_array_of_string() {
    let s = schema_for("store");
    let p = prop(&s, "tags");
    assert_eq!(p.get("type").and_then(|v| v.as_str()), Some("array"));
    let items = p.get("items").expect("tags must have items");
    assert_eq!(items.get("type").and_then(|v| v.as_str()), Some("string"));
}

#[test]
fn store_metadata_is_object_with_additional_string_props() {
    let s = schema_for("store");
    let p = prop(&s, "metadata");
    assert_eq!(p.get("type").and_then(|v| v.as_str()), Some("object"));
    let ap = p
        .get("additionalProperties")
        .expect("metadata must have additionalProperties");
    assert_eq!(ap.get("type").and_then(|v| v.as_str()), Some("string"));
}

#[test]
fn store_all_properties_present() {
    let s = schema_for("store");
    for field in &[
        "type",
        "content",
        "libraryName",
        "forProject",
        "path",
        "name",
        "title",
        "url",
        "filePath",
        "tags",
        "sourceType",
        "metadata",
    ] {
        assert!(
            props(&s).get(field).is_some(),
            "store is missing '{}'",
            field
        );
    }
}

// ---------------------------------------------------------------------------
// grep schema
// ---------------------------------------------------------------------------

#[test]
fn grep_required_is_pattern_only() {
    let s = schema_for("grep");
    assert_eq!(required(&s), vec!["pattern"]);
}

#[test]
fn grep_scope_enum_values() {
    let s = schema_for("grep");
    let enums = enum_vals(prop(&s, "scope"));
    assert_eq!(enums, vec!["project", "all"]);
}

#[test]
fn grep_boolean_fields() {
    let s = schema_for("grep");
    for field in &["regex", "caseSensitive"] {
        let p = prop(&s, field);
        assert_eq!(
            p.get("type").and_then(|v| v.as_str()),
            Some("boolean"),
            "grep field '{}' should be boolean",
            field
        );
    }
}

#[test]
fn grep_all_properties_present() {
    let s = schema_for("grep");
    for field in &[
        "pattern",
        "regex",
        "caseSensitive",
        "pathGlob",
        "scope",
        "contextLines",
        "maxResults",
        "branch",
        "projectId",
    ] {
        assert!(
            props(&s).get(field).is_some(),
            "grep is missing '{}'",
            field
        );
    }
}

// ---------------------------------------------------------------------------
// list schema
// ---------------------------------------------------------------------------

#[test]
fn list_no_required_fields() {
    let s = schema_for("list");
    assert!(
        s.get("required").is_none(),
        "list should have no required array"
    );
}

#[test]
fn list_format_enum_values() {
    let s = schema_for("list");
    let enums = enum_vals(prop(&s, "format"));
    assert_eq!(enums, vec!["tree", "summary", "flat"]);
}

#[test]
fn list_all_properties_present() {
    let s = schema_for("list");
    for field in &[
        "path",
        "depth",
        "format",
        "fileType",
        "language",
        "extension",
        "pattern",
        "includeTests",
        "limit",
        "projectId",
        "component",
        "branch",
    ] {
        assert!(
            props(&s).get(field).is_some(),
            "list is missing '{}'",
            field
        );
    }
}

// ---------------------------------------------------------------------------
// embedding schema
// ---------------------------------------------------------------------------

#[test]
fn embedding_schema_type_is_object() {
    let s = schema_for("embedding");
    assert_eq!(s.get("type").and_then(|v| v.as_str()), Some("object"));
}

#[test]
fn embedding_properties_is_empty_object() {
    let s = schema_for("embedding");
    let p = s.get("properties").expect("embedding must have properties");
    assert!(
        p.as_object().map(|o| o.is_empty()).unwrap_or(false),
        "embedding properties must be empty object"
    );
}

#[test]
fn embedding_no_required_fields() {
    let s = schema_for("embedding");
    assert!(
        s.get("required").is_none(),
        "embedding should have no required array"
    );
}

// ---------------------------------------------------------------------------
// Schema structure: all schemas have type:object and properties
// ---------------------------------------------------------------------------

#[test]
fn all_schemas_are_type_object() {
    let tools = list_tools();
    for tool in &tools {
        let schema = Value::Object(tool.input_schema.as_ref().clone());
        assert_eq!(
            schema.get("type").and_then(|v| v.as_str()),
            Some("object"),
            "tool '{}' schema must have type:object",
            tool.name
        );
    }
}

#[test]
fn all_schemas_have_properties() {
    let tools = list_tools();
    for tool in &tools {
        let schema = Value::Object(tool.input_schema.as_ref().clone());
        assert!(
            schema.get("properties").is_some(),
            "tool '{}' schema must have properties",
            tool.name
        );
    }
}

// ---------------------------------------------------------------------------
// No unexpected top-level schema keys (parity check: TS emits no $schema,
// no $defs, no allOf, no additionalProperties at root)
// ---------------------------------------------------------------------------

#[test]
fn no_draft_2020_schema_key() {
    let tools = list_tools();
    for tool in &tools {
        let schema = Value::Object(tool.input_schema.as_ref().clone());
        assert!(
            schema.get("$schema").is_none(),
            "tool '{}' schema must NOT have $schema key (parity with TS)",
            tool.name
        );
        assert!(
            schema.get("$defs").is_none(),
            "tool '{}' schema must NOT have $defs key (parity with TS)",
            tool.name
        );
    }
}
