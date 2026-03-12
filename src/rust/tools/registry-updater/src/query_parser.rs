//! Tree-sitter query file parser for bootstrapping semantic patterns.
//!
//! Parses `.scm` query files from nvim-treesitter `queries/<language>/`
//! directories to extract AST node types for the semantic_patterns YAML
//! schema. This enables automatic generation of function/class/module
//! patterns for new languages.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use workspace_qdrant_core::language_registry::types::{
    DocstringStyle, FunctionPatternGroup, MethodPatternGroup, PatternGroup, SemanticPatterns,
};

/// Extracted semantic patterns from tree-sitter query files.
#[derive(Debug, Default)]
pub struct ExtractedPatterns {
    pub function_nodes: Vec<String>,
    pub class_nodes: Vec<String>,
    pub method_nodes: Vec<String>,
    pub module_nodes: Vec<String>,
    pub struct_nodes: Vec<String>,
    pub enum_nodes: Vec<String>,
    pub interface_nodes: Vec<String>,
    pub preamble_nodes: Vec<String>,
    pub constant_nodes: Vec<String>,
    pub type_alias_nodes: Vec<String>,
    pub name_node: Option<String>,
    pub body_node: Option<String>,
    pub docstring_style: DocstringStyle,
}

/// Parse a tree-sitter query (.scm) file and extract semantic patterns.
///
/// Captures like `@function.definition`, `@class.definition`,
/// `@module.definition` are mapped to their corresponding node types.
pub fn parse_query_file(content: &str) -> ExtractedPatterns {
    let mut patterns = ExtractedPatterns::default();

    // Strip comment lines before parsing (comments start with ;)
    let cleaned: String = content
        .lines()
        .filter(|line| !line.trim_start().starts_with(';'))
        .collect::<Vec<_>>()
        .join("\n");

    // Strategy: tree-sitter queries are S-expressions spanning multiple lines.
    // We need to associate the opening `(node_type` with any `@capture` that
    // appears before the next top-level `(`. We track the most recent top-level
    // node type and match it against all @captures found before the next one.

    // Step 1: Find all top-level node types: lines starting with `(`
    // Step 2: Find all @captures and associate with the most recent node type

    let node_re = regex::Regex::new(r"\((\w+)").unwrap();
    let capture_re = regex::Regex::new(r"@([\w.]+)").unwrap();

    // Track nesting depth to identify top-level patterns
    let mut current_top_node: Option<String> = None;
    let mut depth: i32 = 0;

    for line in cleaned.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // Blank line resets context at depth 0
            if depth <= 0 {
                current_top_node = None;
                depth = 0;
            }
            continue;
        }

        // Track parenthesis depth
        let opens = trimmed.chars().filter(|&c| c == '(').count() as i32;
        let closes = trimmed.chars().filter(|&c| c == ')').count() as i32;

        // If we're at depth 0 and see a `(node_type`, capture it as top-level
        if depth == 0 {
            if let Some(cap) = node_re.captures(trimmed) {
                current_top_node = Some(cap[1].to_string());
            }
        }

        depth += opens - closes;
        if depth < 0 {
            depth = 0;
        }

        // Extract @captures on this line and associate with current_top_node
        if let Some(ref node_type) = current_top_node {
            for cap in capture_re.captures_iter(trimmed) {
                let capture = &cap[1];
                classify_capture(node_type, capture, &mut patterns);
            }
        }

        // Reset at depth 0
        if depth == 0 {
            current_top_node = None;
        }
    }

    // Deduplicate all vectors
    dedup_preserve_order(&mut patterns.function_nodes);
    dedup_preserve_order(&mut patterns.class_nodes);
    dedup_preserve_order(&mut patterns.method_nodes);
    dedup_preserve_order(&mut patterns.module_nodes);
    dedup_preserve_order(&mut patterns.struct_nodes);
    dedup_preserve_order(&mut patterns.enum_nodes);
    dedup_preserve_order(&mut patterns.interface_nodes);
    dedup_preserve_order(&mut patterns.preamble_nodes);
    dedup_preserve_order(&mut patterns.constant_nodes);
    dedup_preserve_order(&mut patterns.type_alias_nodes);

    patterns
}

/// Classify a @capture and add the associated node type to the right pattern group.
fn classify_capture(node_type: &str, capture: &str, patterns: &mut ExtractedPatterns) {
    match capture {
        // Function definitions
        c if c.starts_with("function") || c == "definition.function" => {
            if !patterns.function_nodes.contains(&node_type.to_string()) {
                patterns.function_nodes.push(node_type.to_string());
            }
        }
        // Class definitions
        c if c.starts_with("class") || c == "definition.class" => {
            if !patterns.class_nodes.contains(&node_type.to_string()) {
                patterns.class_nodes.push(node_type.to_string());
            }
        }
        // Method definitions
        c if c.starts_with("method") || c == "definition.method" => {
            if !patterns.method_nodes.contains(&node_type.to_string()) {
                patterns.method_nodes.push(node_type.to_string());
            }
        }
        // Module definitions
        c if c.starts_with("module") || c == "definition.module" || c == "scope.module" => {
            if !patterns.module_nodes.contains(&node_type.to_string()) {
                patterns.module_nodes.push(node_type.to_string());
            }
        }
        // Import/include (preamble)
        c if c.starts_with("import") || c.starts_with("include") || c == "keyword.import" => {
            if !patterns.preamble_nodes.contains(&node_type.to_string()) {
                patterns.preamble_nodes.push(node_type.to_string());
            }
        }
        // Struct definitions
        c if c.starts_with("struct") || c == "definition.struct" => {
            if !patterns.struct_nodes.contains(&node_type.to_string()) {
                patterns.struct_nodes.push(node_type.to_string());
            }
        }
        // Enum definitions
        c if c.starts_with("enum") || c == "definition.enum" => {
            if !patterns.enum_nodes.contains(&node_type.to_string()) {
                patterns.enum_nodes.push(node_type.to_string());
            }
        }
        // Interface definitions
        c if c.starts_with("interface") || c == "definition.interface" => {
            if !patterns.interface_nodes.contains(&node_type.to_string()) {
                patterns.interface_nodes.push(node_type.to_string());
            }
        }
        // Constants
        c if c.starts_with("constant") || c == "definition.constant" => {
            if !patterns.constant_nodes.contains(&node_type.to_string()) {
                patterns.constant_nodes.push(node_type.to_string());
            }
        }
        // Type aliases
        c if c.starts_with("type") && c.contains("alias") => {
            if !patterns.type_alias_nodes.contains(&node_type.to_string()) {
                patterns.type_alias_nodes.push(node_type.to_string());
            }
        }
        _ => {}
    }
}

/// Convert extracted patterns to the SemanticPatterns YAML type.
pub fn to_semantic_patterns(extracted: &ExtractedPatterns) -> SemanticPatterns {
    SemanticPatterns {
        function: FunctionPatternGroup {
            node_types: extracted.function_nodes.clone(),
            async_node_types: Vec::new(),
        },
        class: PatternGroup {
            node_types: extracted.class_nodes.clone(),
        },
        method: MethodPatternGroup {
            node_types: extracted.method_nodes.clone(),
            context: if extracted.method_nodes.is_empty() {
                None
            } else {
                Some("inside_class".to_string())
            },
        },
        module: PatternGroup {
            node_types: extracted.module_nodes.clone(),
        },
        struct_def: PatternGroup {
            node_types: extracted.struct_nodes.clone(),
        },
        enum_def: PatternGroup {
            node_types: extracted.enum_nodes.clone(),
        },
        interface: PatternGroup {
            node_types: extracted.interface_nodes.clone(),
        },
        preamble: PatternGroup {
            node_types: extracted.preamble_nodes.clone(),
        },
        constant: PatternGroup {
            node_types: extracted.constant_nodes.clone(),
        },
        type_alias: PatternGroup {
            node_types: extracted.type_alias_nodes.clone(),
        },
        name_node: extracted.name_node.clone(),
        body_node: extracted.body_node.clone(),
        docstring_style: extracted.docstring_style,
        // Fields not bootstrapped from queries
        trait_def: PatternGroup::default(),
        macro_def: PatternGroup::default(),
        impl_block: PatternGroup::default(),
        comment_nodes: Vec::new(),
        decorated_wrapper: None,
        root_wrappers: Vec::new(),
    }
}

/// Fetch query files for a language from nvim-treesitter GitHub repo.
#[allow(dead_code)]
pub async fn fetch_query_files(language: &str) -> Result<HashMap<String, String>> {
    let base_url = format!(
        "https://api.github.com/repos/nvim-treesitter/nvim-treesitter/contents/queries/{}",
        language
    );

    let client = reqwest::Client::builder()
        .user_agent("workspace-qdrant-mcp/registry-updater")
        .build()?;

    let response = client.get(&base_url).send().await?;
    if !response.status().is_success() {
        return Ok(HashMap::new());
    }

    #[derive(serde::Deserialize)]
    struct GithubFile {
        name: String,
        download_url: Option<String>,
    }

    let files: Vec<GithubFile> = response.json().await.unwrap_or_default();
    let mut result = HashMap::new();

    for file in files {
        if !file.name.ends_with(".scm") {
            continue;
        }
        if let Some(url) = file.download_url {
            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    if let Ok(content) = resp.text().await {
                        result.insert(file.name, content);
                    }
                }
                _ => continue,
            }
        }
    }

    Ok(result)
}

/// Bootstrap semantic patterns for a language from its query files.
#[allow(dead_code)]
pub async fn bootstrap_patterns(language: &str) -> Result<Option<SemanticPatterns>> {
    let files = fetch_query_files(language)
        .await
        .with_context(|| format!("Fetching query files for {language}"))?;

    if files.is_empty() {
        return Ok(None);
    }

    let mut combined = ExtractedPatterns::default();

    // Parse all query files and combine
    for (filename, content) in &files {
        let extracted = parse_query_file(content);

        // Merge extracted patterns into combined
        for node in extracted.function_nodes {
            if !combined.function_nodes.contains(&node) {
                combined.function_nodes.push(node);
            }
        }
        for node in extracted.class_nodes {
            if !combined.class_nodes.contains(&node) {
                combined.class_nodes.push(node);
            }
        }
        for node in extracted.method_nodes {
            if !combined.method_nodes.contains(&node) {
                combined.method_nodes.push(node);
            }
        }
        for node in extracted.module_nodes {
            if !combined.module_nodes.contains(&node) {
                combined.module_nodes.push(node);
            }
        }
        for node in extracted.struct_nodes {
            if !combined.struct_nodes.contains(&node) {
                combined.struct_nodes.push(node);
            }
        }
        for node in extracted.enum_nodes {
            if !combined.enum_nodes.contains(&node) {
                combined.enum_nodes.push(node);
            }
        }
        for node in extracted.interface_nodes {
            if !combined.interface_nodes.contains(&node) {
                combined.interface_nodes.push(node);
            }
        }
        for node in extracted.preamble_nodes {
            if !combined.preamble_nodes.contains(&node) {
                combined.preamble_nodes.push(node);
            }
        }
        for node in extracted.constant_nodes {
            if !combined.constant_nodes.contains(&node) {
                combined.constant_nodes.push(node);
            }
        }
        for node in extracted.type_alias_nodes {
            if !combined.type_alias_nodes.contains(&node) {
                combined.type_alias_nodes.push(node);
            }
        }

        let _ = filename; // Used for tracing in debug builds
    }

    // Only return patterns if we found something meaningful
    if combined.function_nodes.is_empty()
        && combined.class_nodes.is_empty()
        && combined.module_nodes.is_empty()
    {
        return Ok(None);
    }

    Ok(Some(to_semantic_patterns(&combined)))
}

/// Deduplicate a vector while preserving insertion order.
fn dedup_preserve_order(vec: &mut Vec<String>) {
    let mut seen = HashSet::new();
    vec.retain(|item| seen.insert(item.clone()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_query_python() {
        let scm = r#"
; Functions
(function_definition
  name: (identifier) @function.definition)

; Classes
(class_definition
  name: (identifier) @class.definition)

; Imports
(import_statement) @import
(import_from_statement) @import
"#;

        let result = parse_query_file(scm);
        assert!(result
            .function_nodes
            .contains(&"function_definition".to_string()));
        assert!(result.class_nodes.contains(&"class_definition".to_string()));
    }

    #[test]
    fn test_parse_query_rust() {
        let scm = r#"
(function_item
  name: (identifier) @function.definition)

(struct_item
  name: (type_identifier) @struct.definition)

(enum_item
  name: (type_identifier) @enum.definition)

(impl_item) @class.definition

(use_declaration) @import
"#;

        let result = parse_query_file(scm);
        assert!(result.function_nodes.contains(&"function_item".to_string()));
        assert!(result.struct_nodes.contains(&"struct_item".to_string()));
        assert!(result.enum_nodes.contains(&"enum_item".to_string()));
        assert!(result.class_nodes.contains(&"impl_item".to_string()));
    }

    #[test]
    fn test_parse_skips_comments() {
        let scm = r#"
; This is a comment
; (function_definition) @function.definition
(class_definition) @class.definition
"#;

        let result = parse_query_file(scm);
        assert!(result.function_nodes.is_empty());
        assert!(!result.class_nodes.is_empty());
    }

    #[test]
    fn test_to_semantic_patterns() {
        let extracted = ExtractedPatterns {
            function_nodes: vec!["function_definition".to_string()],
            class_nodes: vec!["class_definition".to_string()],
            method_nodes: vec!["method_definition".to_string()],
            ..Default::default()
        };

        let patterns = to_semantic_patterns(&extracted);
        assert_eq!(patterns.function.node_types, vec!["function_definition"]);
        assert_eq!(patterns.class.node_types, vec!["class_definition"]);
        assert_eq!(patterns.method.node_types, vec!["method_definition"]);
        assert_eq!(patterns.method.context, Some("inside_class".to_string()));
    }

    #[test]
    fn test_dedup_preserve_order() {
        let mut v = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "c".to_string(),
            "b".to_string(),
        ];
        dedup_preserve_order(&mut v);
        assert_eq!(v, vec!["a", "b", "c"]);
    }
}
