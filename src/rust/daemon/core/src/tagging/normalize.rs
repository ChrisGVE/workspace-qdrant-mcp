//! Concept normalization for tags.
//!
//! Canonicalizes tags from all tiers (path-derived, taxonomy, LLM, structural)
//! to eliminate synonyms and formatting inconsistencies before storage.
//!
//! Normalization steps:
//! 1. Strip leading/trailing whitespace
//! 2. Lowercase
//! 3. Replace separators (spaces, underscores) with hyphens
//! 4. Collapse consecutive hyphens
//! 5. Strip leading/trailing hyphens
//! 6. Map common abbreviations to canonical forms

use std::collections::HashMap;
use std::sync::LazyLock;

// ── Abbreviation map ────────────────────────────────────────────────────

/// Common abbreviations mapped to their canonical expanded forms.
static ABBREVIATION_MAP: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("ml", "machine-learning");
    m.insert("js", "javascript");
    m.insert("ts", "typescript");
    m.insert("db", "database");
    m.insert("api", "api");
    m.insert("ui", "user-interface");
    m.insert("ux", "user-experience");
    m.insert("ai", "artificial-intelligence");
    m.insert("nlp", "natural-language-processing");
    m.insert("cv", "computer-vision");
    m.insert("k8s", "kubernetes");
    m.insert("tf", "tensorflow");
    m.insert("py", "python");
    m
});

// ── Core normalizer ─────────────────────────────────────────────────────

/// Normalize a single tag to its canonical form.
///
/// If the tag has a prefix (e.g., `path:`, `dep:`, `tax:`, `llm:`),
/// only the value portion is normalized and the prefix is preserved.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(normalize_tag("  Machine_Learning  "), "machine-learning");
/// assert_eq!(normalize_tag("ML"), "machine-learning");
/// assert_eq!(normalize_tag("dep:ML"), "dep:machine-learning");
/// assert_eq!(normalize_tag("path:my_module"), "path:my-module");
/// ```
pub fn normalize_tag(tag: &str) -> String {
    let trimmed = tag.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Split on first colon to preserve prefix
    if let Some((prefix, value)) = trimmed.split_once(':') {
        let normalized_value = normalize_value(value);
        if normalized_value.is_empty() {
            return String::new();
        }
        format!("{}:{}", prefix.to_lowercase(), normalized_value)
    } else {
        normalize_value(trimmed)
    }
}

/// Normalize the value portion of a tag (without prefix).
fn normalize_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Lowercase and replace separators
    let mut result = String::with_capacity(trimmed.len());
    let mut prev_was_hyphen = false;

    for ch in trimmed.chars() {
        if ch == '_' || ch == ' ' {
            if !result.is_empty() && !prev_was_hyphen {
                result.push('-');
                prev_was_hyphen = true;
            }
        } else if ch == '-' {
            if !result.is_empty() && !prev_was_hyphen {
                result.push('-');
                prev_was_hyphen = true;
            }
        } else if ch.is_alphanumeric() {
            result.push(ch.to_ascii_lowercase());
            prev_was_hyphen = false;
        }
        // Skip non-alphanumeric, non-separator characters
    }

    // Strip trailing hyphen
    if result.ends_with('-') {
        result.pop();
    }

    // Apply abbreviation mapping on the full normalized value
    if let Some(&canonical) = ABBREVIATION_MAP.get(result.as_str()) {
        return canonical.to_string();
    }

    result
}

/// Normalize a batch of tags, deduplicating after normalization.
///
/// Tags that normalize to the same canonical form are deduplicated,
/// keeping the first occurrence.
pub fn normalize_tags(tags: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::with_capacity(tags.len());

    for tag in tags {
        let normalized = normalize_tag(tag);
        if !normalized.is_empty() && seen.insert(normalized.clone()) {
            result.push(normalized);
        }
    }

    result
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Whitespace stripping ────────────────────────────────────────

    #[test]
    fn test_strips_leading_trailing_whitespace() {
        assert_eq!(normalize_tag("  web-server  "), "web-server");
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(normalize_tag(""), "");
    }

    #[test]
    fn test_whitespace_only() {
        assert_eq!(normalize_tag("   "), "");
    }

    // ── Lowercasing ─────────────────────────────────────────────────

    #[test]
    fn test_lowercase_uppercase() {
        assert_eq!(normalize_tag("WEB-SERVER"), "web-server");
    }

    #[test]
    fn test_lowercase_mixed_case() {
        assert_eq!(normalize_tag("MachineLearning"), "machinelearning");
    }

    #[test]
    fn test_already_lowercase() {
        assert_eq!(normalize_tag("database"), "database");
    }

    // ── Separator normalization ─────────────────────────────────────

    #[test]
    fn test_underscores_to_hyphens() {
        assert_eq!(normalize_tag("machine_learning"), "machine-learning");
    }

    #[test]
    fn test_spaces_to_hyphens() {
        assert_eq!(normalize_tag("machine learning"), "machine-learning");
    }

    #[test]
    fn test_mixed_separators() {
        assert_eq!(
            normalize_tag("machine_learning model"),
            "machine-learning-model"
        );
    }

    #[test]
    fn test_consecutive_separators_collapsed() {
        assert_eq!(normalize_tag("machine__learning"), "machine-learning");
        assert_eq!(normalize_tag("machine - learning"), "machine-learning");
        assert_eq!(normalize_tag("machine_ _learning"), "machine-learning");
    }

    #[test]
    fn test_leading_trailing_separators_stripped() {
        assert_eq!(normalize_tag("_machine_"), "machine");
        assert_eq!(normalize_tag("-web-server-"), "web-server");
    }

    // ── Abbreviation mapping ────────────────────────────────────────

    #[test]
    fn test_ml_to_machine_learning() {
        assert_eq!(normalize_tag("ml"), "machine-learning");
        assert_eq!(normalize_tag("ML"), "machine-learning");
    }

    #[test]
    fn test_js_to_javascript() {
        assert_eq!(normalize_tag("js"), "javascript");
        assert_eq!(normalize_tag("JS"), "javascript");
    }

    #[test]
    fn test_ts_to_typescript() {
        assert_eq!(normalize_tag("ts"), "typescript");
        assert_eq!(normalize_tag("TS"), "typescript");
    }

    #[test]
    fn test_db_to_database() {
        assert_eq!(normalize_tag("db"), "database");
        assert_eq!(normalize_tag("DB"), "database");
    }

    #[test]
    fn test_api_stays_api() {
        assert_eq!(normalize_tag("api"), "api");
        assert_eq!(normalize_tag("API"), "api");
    }

    #[test]
    fn test_ui_to_user_interface() {
        assert_eq!(normalize_tag("ui"), "user-interface");
        assert_eq!(normalize_tag("UI"), "user-interface");
    }

    #[test]
    fn test_ux_to_user_experience() {
        assert_eq!(normalize_tag("ux"), "user-experience");
        assert_eq!(normalize_tag("UX"), "user-experience");
    }

    #[test]
    fn test_ai_to_artificial_intelligence() {
        assert_eq!(normalize_tag("ai"), "artificial-intelligence");
        assert_eq!(normalize_tag("AI"), "artificial-intelligence");
    }

    #[test]
    fn test_nlp_to_natural_language_processing() {
        assert_eq!(normalize_tag("nlp"), "natural-language-processing");
        assert_eq!(normalize_tag("NLP"), "natural-language-processing");
    }

    #[test]
    fn test_cv_to_computer_vision() {
        assert_eq!(normalize_tag("cv"), "computer-vision");
        assert_eq!(normalize_tag("CV"), "computer-vision");
    }

    #[test]
    fn test_k8s_to_kubernetes() {
        assert_eq!(normalize_tag("k8s"), "kubernetes");
        assert_eq!(normalize_tag("K8S"), "kubernetes");
    }

    #[test]
    fn test_tf_to_tensorflow() {
        assert_eq!(normalize_tag("tf"), "tensorflow");
        assert_eq!(normalize_tag("TF"), "tensorflow");
    }

    #[test]
    fn test_py_to_python() {
        assert_eq!(normalize_tag("py"), "python");
        assert_eq!(normalize_tag("PY"), "python");
    }

    // ── Abbreviation NOT applied when part of a compound tag ────────

    #[test]
    fn test_abbreviation_not_applied_to_compound() {
        // "ml-pipeline" should NOT be expanded, only exact matches
        assert_eq!(normalize_tag("ml-pipeline"), "ml-pipeline");
    }

    #[test]
    fn test_abbreviation_not_applied_to_longer_word() {
        // "database" should not be affected by "db" mapping
        assert_eq!(normalize_tag("database"), "database");
    }

    // ── Prefix preservation ─────────────────────────────────────────

    #[test]
    fn test_prefix_path_preserved() {
        assert_eq!(normalize_tag("path:my_module"), "path:my-module");
    }

    #[test]
    fn test_prefix_dep_preserved() {
        assert_eq!(
            normalize_tag("dep:Machine_Learning"),
            "dep:machine-learning"
        );
    }

    #[test]
    fn test_prefix_tax_preserved() {
        assert_eq!(normalize_tag("tax:WEB_DEVELOPMENT"), "tax:web-development");
    }

    #[test]
    fn test_prefix_llm_preserved() {
        assert_eq!(normalize_tag("llm:Data Pipeline"), "llm:data-pipeline");
    }

    #[test]
    fn test_prefix_language_preserved() {
        assert_eq!(normalize_tag("language:Rust"), "language:rust");
    }

    #[test]
    fn test_prefix_abbreviation_expansion() {
        assert_eq!(normalize_tag("dep:ML"), "dep:machine-learning");
        assert_eq!(normalize_tag("llm:DB"), "llm:database");
    }

    #[test]
    fn test_prefix_lowercased() {
        assert_eq!(normalize_tag("PATH:my_module"), "path:my-module");
    }

    // ── Batch normalization with deduplication ───────────────────────

    #[test]
    fn test_batch_normalization() {
        let tags = vec![
            "ML".to_string(),
            "machine_learning".to_string(),
            "web-server".to_string(),
        ];
        let normalized = normalize_tags(&tags);
        assert_eq!(normalized, vec!["machine-learning", "web-server"]);
    }

    #[test]
    fn test_batch_deduplication() {
        let tags = vec![
            "web_server".to_string(),
            "web-server".to_string(),
            "Web Server".to_string(),
        ];
        let normalized = normalize_tags(&tags);
        assert_eq!(normalized, vec!["web-server"]);
    }

    #[test]
    fn test_batch_empty_filtered() {
        let tags = vec!["".to_string(), "  ".to_string(), "valid-tag".to_string()];
        let normalized = normalize_tags(&tags);
        assert_eq!(normalized, vec!["valid-tag"]);
    }

    #[test]
    fn test_batch_preserves_order() {
        let tags = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let normalized = normalize_tags(&tags);
        assert_eq!(normalized, vec!["alpha", "beta", "gamma"]);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_single_char_preserved() {
        // Single characters are valid after normalization
        assert_eq!(normalize_tag("a"), "a");
    }

    #[test]
    fn test_numeric_tag() {
        assert_eq!(normalize_tag("v2"), "v2");
    }

    #[test]
    fn test_hyphenated_already_canonical() {
        assert_eq!(normalize_tag("web-server"), "web-server");
    }

    #[test]
    fn test_special_chars_stripped() {
        assert_eq!(normalize_tag("web@server!"), "webserver");
    }

    #[test]
    fn test_prefix_with_empty_value() {
        assert_eq!(normalize_tag("prefix:"), "");
    }

    #[test]
    fn test_prefix_with_whitespace_value() {
        assert_eq!(normalize_tag("prefix:  "), "");
    }

    #[test]
    fn test_colon_only() {
        assert_eq!(normalize_tag(":"), "");
    }
}
