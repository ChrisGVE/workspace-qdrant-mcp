//! Schema parity tests: verify Rust tool definitions match the TypeScript server.
//!
//! Expected schemas are captured directly from the TS source (verified identical
//! to the runtime output, since the TS files are plain object literals — no
//! runtime transformation). Collection enum values come from wqm_common constants.

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use crate::tools::definitions::list_tools;

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// Extract the inputSchema of a named tool from the list.
    fn schema_for(name: &str) -> Value {
        let tools = list_tools();
        let tool = tools.iter().find(|t| t.name == name).unwrap_or_else(|| {
            panic!("tool '{}' not found in list_tools()", name);
        });
        Value::Object(tool.input_schema.as_ref().clone())
    }

    fn props(schema: &Value) -> &Value {
        schema
            .get("properties")
            .expect("schema must have 'properties'")
    }

    fn prop<'a>(schema: &'a Value, field: &str) -> &'a Value {
        props(schema)
            .get(field)
            .unwrap_or_else(|| panic!("property '{}' not found", field))
    }

    fn required(schema: &Value) -> Vec<String> {
        match schema.get("required") {
            None => vec![],
            Some(arr) => arr
                .as_array()
                .expect("required must be an array")
                .iter()
                .map(|v| {
                    v.as_str()
                        .expect("required element must be a string")
                        .to_string()
                })
                .collect(),
        }
    }

    fn enum_vals(prop_value: &Value) -> Vec<String> {
        prop_value
            .get("enum")
            .expect("property must have 'enum'")
            .as_array()
            .expect("enum must be an array")
            .iter()
            .map(|v| v.as_str().expect("enum value must be a string").to_string())
            .collect()
    }

    // ---------------------------------------------------------------------------
    // tools/list: order and count
    // ---------------------------------------------------------------------------

    #[test]
    fn tools_list_count_and_order() {
        let tools = list_tools();
        assert_eq!(tools.len(), 7, "expected exactly 7 tools");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert_eq!(
            names,
            &[
                "search",
                "retrieve",
                "rules",
                "store",
                "grep",
                "list",
                "embedding"
            ],
            "tool order must match TS getToolDefinitions()"
        );
    }

    // ---------------------------------------------------------------------------
    // Tool names and descriptions
    // ---------------------------------------------------------------------------

    #[test]
    fn tool_names_match_ts() {
        let tools = list_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert!(names.contains(&"search"));
        assert!(names.contains(&"retrieve"));
        assert!(names.contains(&"rules"));
        assert!(names.contains(&"store"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"list"));
        assert!(names.contains(&"embedding"));
    }

    #[test]
    fn tool_descriptions_match_ts() {
        let tools = list_tools();
        let desc = |name: &str| {
            tools
                .iter()
                .find(|t| t.name == name)
                .and_then(|t| t.description.as_deref())
                .unwrap_or_else(|| panic!("tool '{}' has no description", name))
                .to_string()
        };

        assert!(desc("search").contains("hybrid semantic and keyword search"));
        assert!(desc("retrieve").contains("Retrieve documents by ID"));
        assert!(desc("rules").contains("Manage behavioral rules"));
        assert!(desc("store").contains("Store content or register a project"));
        assert!(desc("grep").contains("exact substring or regex pattern matching"));
        assert!(desc("list").contains("List project files and folder structure"));
        assert!(desc("embedding").contains("active embedding provider"));
    }

    // ---------------------------------------------------------------------------
    // search schema
    // ---------------------------------------------------------------------------

    #[test]
    fn search_schema_type_is_object() {
        let s = schema_for("search");
        assert_eq!(s.get("type").and_then(|v| v.as_str()), Some("object"));
    }

    #[test]
    fn search_required_is_query_only() {
        let s = schema_for("search");
        assert_eq!(required(&s), vec!["query"]);
    }

    #[test]
    fn search_query_is_string() {
        let s = schema_for("search");
        let q = prop(&s, "query");
        assert_eq!(q.get("type").and_then(|v| v.as_str()), Some("string"));
    }

    #[test]
    fn search_collection_enum_values() {
        let s = schema_for("search");
        let enums = enum_vals(prop(&s, "collection"));
        assert_eq!(enums, vec!["projects", "libraries", "rules", "scratchpad"]);
    }

    #[test]
    fn search_mode_enum_values() {
        let s = schema_for("search");
        let enums = enum_vals(prop(&s, "mode"));
        assert_eq!(enums, vec!["hybrid", "semantic", "keyword"]);
    }

    #[test]
    fn search_scope_enum_values() {
        let s = schema_for("search");
        let enums = enum_vals(prop(&s, "scope"));
        assert_eq!(enums, vec!["project", "group", "all"]);
    }

    #[test]
    fn search_limit_is_number() {
        let s = schema_for("search");
        let p = prop(&s, "limit");
        assert_eq!(p.get("type").and_then(|v| v.as_str()), Some("number"));
    }

    #[test]
    fn search_tags_is_array_of_string() {
        let s = schema_for("search");
        let p = prop(&s, "tags");
        assert_eq!(p.get("type").and_then(|v| v.as_str()), Some("array"));
        let items = p.get("items").expect("tags must have items");
        assert_eq!(items.get("type").and_then(|v| v.as_str()), Some("string"));
    }

    #[test]
    fn search_boolean_fields_are_boolean_type() {
        let s = schema_for("search");
        for field in &[
            "includeLibraries",
            "exact",
            "includeGraphContext",
            "diverse",
        ] {
            let p = prop(&s, field);
            assert_eq!(
                p.get("type").and_then(|v| v.as_str()),
                Some("boolean"),
                "field '{}' should be boolean",
                field
            );
        }
    }

    #[test]
    fn search_all_properties_present() {
        let s = schema_for("search");
        let expected_props = [
            "query",
            "collection",
            "mode",
            "scope",
            "limit",
            "projectId",
            "libraryName",
            "libraryPath",
            "branch",
            "fileType",
            "scoreThreshold",
            "includeLibraries",
            "tag",
            "tags",
            "pathGlob",
            "component",
            "exact",
            "contextLines",
            "includeGraphContext",
            "diverse",
        ];
        for name in &expected_props {
            assert!(
                props(&s).get(name).is_some(),
                "search is missing property '{}'",
                name
            );
        }
    }

    // ---------------------------------------------------------------------------
    // retrieve schema
    // ---------------------------------------------------------------------------

    #[test]
    fn retrieve_no_required_fields() {
        let s = schema_for("retrieve");
        assert!(
            s.get("required").is_none(),
            "retrieve should have no required array"
        );
    }

    #[test]
    fn retrieve_collection_enum_values() {
        let s = schema_for("retrieve");
        let enums = enum_vals(prop(&s, "collection"));
        assert_eq!(enums, vec!["projects", "libraries", "rules", "scratchpad"]);
    }

    #[test]
    fn retrieve_filter_is_object_with_additional_string_props() {
        let s = schema_for("retrieve");
        let p = prop(&s, "filter");
        assert_eq!(p.get("type").and_then(|v| v.as_str()), Some("object"));
        let ap = p
            .get("additionalProperties")
            .expect("filter must have additionalProperties");
        assert_eq!(ap.get("type").and_then(|v| v.as_str()), Some("string"));
    }

    #[test]
    fn retrieve_all_properties_present() {
        let s = schema_for("retrieve");
        for field in &[
            "documentId",
            "collection",
            "filter",
            "limit",
            "offset",
            "projectId",
            "libraryName",
        ] {
            assert!(
                props(&s).get(field).is_some(),
                "retrieve is missing '{}'",
                field
            );
        }
    }

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
}
