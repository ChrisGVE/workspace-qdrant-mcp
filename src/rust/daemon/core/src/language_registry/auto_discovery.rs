//! Heuristic pattern discovery from tree-sitter node-types.json.
//!
//! For languages without bundled YAML definitions, this module analyzes a
//! grammar's `node-types.json` file and infers `SemanticPatterns` using
//! name-based heuristics. The generated patterns are candidates for user
//! review — they may need refinement.

use serde::Deserialize;

use super::types::{
    DocstringStyle, FunctionPatternGroup, MethodPatternGroup, PatternGroup, SemanticPatterns,
};

/// A node type entry from tree-sitter's `node-types.json`.
#[derive(Debug, Deserialize)]
struct NodeType {
    #[serde(rename = "type")]
    node_type: String,
    named: bool,
    #[serde(default)]
    #[allow(dead_code)]
    fields: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    children: Option<serde_json::Value>,
}

/// Result of auto-discovery.
#[derive(Debug)]
pub struct DiscoveredPatterns {
    /// Inferred semantic patterns.
    pub patterns: SemanticPatterns,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
    /// Human-readable notes about what was discovered.
    pub notes: Vec<String>,
}

/// Analyze `node-types.json` content and infer semantic patterns.
///
/// Returns `None` if the JSON is invalid or no useful patterns are found.
pub fn discover_patterns(node_types_json: &str) -> Option<DiscoveredPatterns> {
    let node_types: Vec<NodeType> = serde_json::from_str(node_types_json).ok()?;

    let named_types: Vec<&str> = node_types
        .iter()
        .filter(|n| n.named)
        .map(|n| n.node_type.as_str())
        .collect();

    if named_types.is_empty() {
        return None;
    }

    let mut patterns = SemanticPatterns::default();
    let mut notes = Vec::new();
    let mut matched = 0;

    // Functions
    let fn_types = find_matching(&named_types, FUNCTION_HINTS);
    if !fn_types.is_empty() {
        notes.push(format!("Functions: {}", fn_types.join(", ")));
        let async_types = find_matching(&named_types, ASYNC_FUNCTION_HINTS);
        patterns.function = FunctionPatternGroup {
            node_types: fn_types,
            async_node_types: async_types,
        };
        matched += 1;
    }

    // Classes
    let class_types = find_matching(&named_types, CLASS_HINTS);
    if !class_types.is_empty() {
        notes.push(format!("Classes: {}", class_types.join(", ")));
        patterns.class = PatternGroup {
            node_types: class_types,
        };
        matched += 1;
    }

    // Methods (same as functions, context determined at runtime)
    let method_types = find_matching(&named_types, METHOD_HINTS);
    if !method_types.is_empty() {
        notes.push(format!("Methods: {}", method_types.join(", ")));
        patterns.method = MethodPatternGroup {
            node_types: method_types,
            context: Some("inside_class".to_string()),
        };
        matched += 1;
    }

    // Structs
    let struct_types = find_matching(&named_types, STRUCT_HINTS);
    if !struct_types.is_empty() {
        notes.push(format!("Structs: {}", struct_types.join(", ")));
        patterns.struct_def = PatternGroup {
            node_types: struct_types,
        };
        matched += 1;
    }

    // Enums
    let enum_types = find_matching(&named_types, ENUM_HINTS);
    if !enum_types.is_empty() {
        notes.push(format!("Enums: {}", enum_types.join(", ")));
        patterns.enum_def = PatternGroup {
            node_types: enum_types,
        };
        matched += 1;
    }

    // Traits/Interfaces
    let trait_types = find_matching(&named_types, TRAIT_HINTS);
    if !trait_types.is_empty() {
        notes.push(format!("Traits/Interfaces: {}", trait_types.join(", ")));
        patterns.trait_def = PatternGroup {
            node_types: trait_types.clone(),
        };
        patterns.interface = PatternGroup {
            node_types: trait_types,
        };
        matched += 1;
    }

    // Modules
    let module_types = find_matching(&named_types, MODULE_HINTS);
    if !module_types.is_empty() {
        notes.push(format!("Modules: {}", module_types.join(", ")));
        patterns.module = PatternGroup {
            node_types: module_types,
        };
        matched += 1;
    }

    // Preamble (imports/includes)
    let preamble_types = find_matching(&named_types, PREAMBLE_HINTS);
    if !preamble_types.is_empty() {
        notes.push(format!("Preamble: {}", preamble_types.join(", ")));
        patterns.preamble = PatternGroup {
            node_types: preamble_types,
        };
        matched += 1;
    }

    // Constants
    let const_types = find_matching(&named_types, CONSTANT_HINTS);
    if !const_types.is_empty() {
        patterns.constant = PatternGroup {
            node_types: const_types,
        };
        matched += 1;
    }

    // Macros
    let macro_types = find_matching(&named_types, MACRO_HINTS);
    if !macro_types.is_empty() {
        patterns.macro_def = PatternGroup {
            node_types: macro_types,
        };
        matched += 1;
    }

    // Type aliases
    let type_types = find_matching(&named_types, TYPE_ALIAS_HINTS);
    if !type_types.is_empty() {
        patterns.type_alias = PatternGroup {
            node_types: type_types,
        };
        matched += 1;
    }

    // Name node heuristic
    if named_types.contains(&"identifier") {
        patterns.name_node = Some("identifier".to_string());
    } else if named_types.contains(&"name") {
        patterns.name_node = Some("name".to_string());
    }

    // Body node heuristic
    if named_types.contains(&"block") {
        patterns.body_node = Some("block".to_string());
    } else if named_types.contains(&"body") {
        patterns.body_node = Some("body".to_string());
    } else if named_types.contains(&"statement_block") {
        patterns.body_node = Some("statement_block".to_string());
    }

    // Comment nodes
    let comment_types = find_matching(&named_types, COMMENT_HINTS);
    if !comment_types.is_empty() {
        patterns.comment_nodes = comment_types;
    }

    // Docstring style heuristic
    patterns.docstring_style =
        if named_types.contains(&"string_literal") || named_types.contains(&"string") {
            // Might have Python-style docstrings
            DocstringStyle::PrecedingComments // safer default
        } else {
            DocstringStyle::PrecedingComments
        };

    if matched == 0 {
        return None;
    }

    let confidence = (matched as f32 / 8.0).min(1.0);
    notes.push(format!(
        "Confidence: {:.0}% ({matched} categories matched)",
        confidence * 100.0
    ));

    Some(DiscoveredPatterns {
        patterns,
        confidence,
        notes,
    })
}

/// Generate a YAML string from discovered patterns.
pub fn patterns_to_yaml(
    language: &str,
    discovered: &DiscoveredPatterns,
) -> Result<String, serde_yaml_ng::Error> {
    let mut yaml = format!(
        "# Auto-discovered patterns for {language}\n\
         # Confidence: {:.0}%\n\
         # Review and adjust before using in production.\n\
         #\n",
        discovered.confidence * 100.0
    );

    for note in &discovered.notes {
        yaml.push_str(&format!("# {note}\n"));
    }
    yaml.push('\n');

    let patterns_yaml = serde_yaml_ng::to_string(&discovered.patterns)?;
    yaml.push_str(&patterns_yaml);

    Ok(yaml)
}

// ─── Heuristic keyword lists ────────────────────────────────────────────────

/// Match named node types against a list of substring hints.
fn find_matching(named_types: &[&str], hints: &[&str]) -> Vec<String> {
    named_types
        .iter()
        .filter(|t| hints.iter().any(|h| t.contains(h)))
        .map(|t| t.to_string())
        .collect()
}

const FUNCTION_HINTS: &[&str] = &[
    "function_definition",
    "function_declaration",
    "function_item",
    "func_declaration",
    "func_definition",
    "method_definition",
    "method_declaration",
    "procedure_declaration",
    "subroutine",
    "defun",
    "def_",
    "fun_declaration",
];

const ASYNC_FUNCTION_HINTS: &[&str] = &["async_function", "async_method", "coroutine_definition"];

const METHOD_HINTS: &[&str] = &[
    "method_definition",
    "method_declaration",
    "method_item",
    "instance_method",
    "class_method",
];

const CLASS_HINTS: &[&str] = &[
    "class_definition",
    "class_declaration",
    "class_specifier",
    "class_item",
    "object_declaration",
];

const STRUCT_HINTS: &[&str] = &[
    "struct_item",
    "struct_specifier",
    "struct_definition",
    "struct_declaration",
    "record_declaration",
    "record_definition",
    "data_declaration",
    "type_declaration",
];

const ENUM_HINTS: &[&str] = &[
    "enum_item",
    "enum_specifier",
    "enum_declaration",
    "enum_definition",
];

const TRAIT_HINTS: &[&str] = &[
    "trait_item",
    "trait_declaration",
    "interface_declaration",
    "interface_definition",
    "protocol_declaration",
    "abstract_class",
    "typeclass",
];

const MODULE_HINTS: &[&str] = &[
    "module_declaration",
    "module_definition",
    "mod_item",
    "namespace_definition",
    "namespace_declaration",
    "package_declaration",
];

const PREAMBLE_HINTS: &[&str] = &[
    "import_statement",
    "import_declaration",
    "import_from_statement",
    "use_declaration",
    "include_directive",
    "require_statement",
    "using_declaration",
    "using_directive",
    "preproc_include",
    "extern_crate",
    "open_statement",
];

const CONSTANT_HINTS: &[&str] = &[
    "const_item",
    "const_declaration",
    "constant_declaration",
    "static_item",
    "let_declaration",
    "val_declaration",
];

const MACRO_HINTS: &[&str] = &[
    "macro_definition",
    "macro_declaration",
    "preproc_def",
    "define_directive",
];

const TYPE_ALIAS_HINTS: &[&str] = &[
    "type_item",
    "type_alias",
    "type_declaration",
    "typedef",
    "type_synonym",
    "newtype",
];

const COMMENT_HINTS: &[&str] = &["comment", "line_comment", "block_comment", "doc_comment"];

#[cfg(test)]
mod tests {
    use super::*;

    const RUST_NODE_TYPES: &str = r#"[
        {"type": "source_file", "named": true},
        {"type": "function_item", "named": true, "fields": {}},
        {"type": "struct_item", "named": true, "fields": {}},
        {"type": "enum_item", "named": true, "fields": {}},
        {"type": "trait_item", "named": true, "fields": {}},
        {"type": "impl_item", "named": true, "fields": {}},
        {"type": "mod_item", "named": true, "fields": {}},
        {"type": "use_declaration", "named": true, "fields": {}},
        {"type": "const_item", "named": true, "fields": {}},
        {"type": "static_item", "named": true, "fields": {}},
        {"type": "type_item", "named": true, "fields": {}},
        {"type": "macro_definition", "named": true, "fields": {}},
        {"type": "identifier", "named": true},
        {"type": "block", "named": true},
        {"type": "line_comment", "named": true},
        {"type": "block_comment", "named": true},
        {"type": ";", "named": false}
    ]"#;

    #[test]
    fn test_discover_rust_patterns() {
        let result = discover_patterns(RUST_NODE_TYPES);
        assert!(result.is_some());

        let discovered = result.unwrap();
        assert!(discovered.confidence > 0.5);
        assert!(discovered
            .patterns
            .function
            .node_types
            .contains(&"function_item".to_string()));
        assert!(discovered
            .patterns
            .struct_def
            .node_types
            .contains(&"struct_item".to_string()));
        assert!(discovered
            .patterns
            .enum_def
            .node_types
            .contains(&"enum_item".to_string()));
        assert!(discovered
            .patterns
            .trait_def
            .node_types
            .contains(&"trait_item".to_string()));
        assert!(discovered
            .patterns
            .module
            .node_types
            .contains(&"mod_item".to_string()));
        assert!(discovered
            .patterns
            .preamble
            .node_types
            .contains(&"use_declaration".to_string()));
        assert_eq!(
            discovered.patterns.name_node,
            Some("identifier".to_string())
        );
        assert_eq!(discovered.patterns.body_node, Some("block".to_string()));
    }

    const PYTHON_NODE_TYPES: &str = r#"[
        {"type": "module", "named": true},
        {"type": "function_definition", "named": true, "fields": {}},
        {"type": "class_definition", "named": true, "fields": {}},
        {"type": "import_statement", "named": true},
        {"type": "import_from_statement", "named": true},
        {"type": "identifier", "named": true},
        {"type": "block", "named": true},
        {"type": "comment", "named": true},
        {"type": "string", "named": true}
    ]"#;

    #[test]
    fn test_discover_python_patterns() {
        let result = discover_patterns(PYTHON_NODE_TYPES);
        assert!(result.is_some());

        let discovered = result.unwrap();
        assert!(discovered
            .patterns
            .function
            .node_types
            .contains(&"function_definition".to_string()));
        assert!(discovered
            .patterns
            .class
            .node_types
            .contains(&"class_definition".to_string()));
        assert!(discovered.patterns.preamble.node_types.len() >= 2);
    }

    #[test]
    fn test_discover_empty() {
        assert!(discover_patterns("[]").is_none());
        assert!(discover_patterns("invalid json").is_none());
    }

    #[test]
    fn test_discover_no_named_types() {
        let json = r#"[{"type": ";", "named": false}, {"type": "(", "named": false}]"#;
        assert!(discover_patterns(json).is_none());
    }

    #[test]
    fn test_patterns_to_yaml() {
        let discovered = discover_patterns(RUST_NODE_TYPES).unwrap();
        let yaml = patterns_to_yaml("rust", &discovered).unwrap();
        assert!(yaml.contains("Auto-discovered"));
        assert!(yaml.contains("function_item"));
    }
}
