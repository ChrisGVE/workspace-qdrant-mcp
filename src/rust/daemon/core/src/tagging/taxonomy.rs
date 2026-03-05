//! Taxonomy loading and parsing for Tier 2 embedding-based tagging.
//!
//! Loads concept taxonomy entries from YAML files with category-to-terms mappings.

use std::path::Path;

/// A single taxonomy entry: a concept term with its category.
#[derive(Debug, Clone)]
pub struct TaxonomyEntry {
    /// The concept term (e.g. "machine learning algorithms").
    pub term: String,
    /// The category this term belongs to (e.g. "machine-learning").
    pub category: String,
}

/// Load taxonomy entries from a YAML string.
///
/// Expected format:
/// ```yaml
/// categories:
///   category-name:
///     - "term one"
///     - "term two"
/// ```
pub fn load_taxonomy(yaml_content: &str) -> Result<Vec<TaxonomyEntry>, String> {
    let doc: serde_yaml_ng::Value = serde_yaml_ng::from_str(yaml_content)
        .map_err(|e| format!("taxonomy YAML parse error: {}", e))?;

    let categories = doc
        .get("categories")
        .and_then(|c| c.as_mapping())
        .ok_or("taxonomy YAML missing 'categories' mapping")?;

    let mut entries = Vec::new();

    for (key, value) in categories {
        let category = key
            .as_str()
            .ok_or("taxonomy category key must be a string")?
            .to_string();

        let terms = value
            .as_sequence()
            .ok_or_else(|| format!("category '{}' must be a sequence", category))?;

        for term_val in terms {
            let term = term_val
                .as_str()
                .ok_or_else(|| format!("term in '{}' must be a string", category))?
                .to_string();
            entries.push(TaxonomyEntry {
                term,
                category: category.clone(),
            });
        }
    }

    Ok(entries)
}

/// Load taxonomy from the bundled asset file.
pub fn load_taxonomy_from_file(path: &Path) -> Result<Vec<TaxonomyEntry>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read taxonomy file: {}", e))?;
    load_taxonomy(&content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_taxonomy_basic() {
        let yaml = r#"
categories:
  programming-languages:
    - rust programming
    - python programming
  databases:
    - relational database
"#;
        let entries = load_taxonomy(yaml).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].category, "programming-languages");
        assert_eq!(entries[0].term, "rust programming");
        assert_eq!(entries[2].category, "databases");
    }

    #[test]
    fn test_load_taxonomy_empty() {
        let yaml = "categories: {}";
        let entries = load_taxonomy(yaml).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_load_taxonomy_invalid() {
        let result = load_taxonomy("not valid yaml: [");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_taxonomy_missing_categories() {
        let result = load_taxonomy("other_key: value");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_bundled_taxonomy() {
        let yaml = include_str!("../../../../../../assets/taxonomy.yaml");
        let entries = load_taxonomy(yaml).unwrap();
        // Should have ~180 entries
        assert!(
            entries.len() >= 150,
            "Bundled taxonomy should have ~150+ entries, got {}",
            entries.len()
        );
        // Check a known entry exists
        assert!(
            entries.iter().any(|e| e.term == "rust programming"),
            "Should contain 'rust programming'"
        );
        // Verify all entries have non-empty term and category
        for entry in &entries {
            assert!(!entry.term.is_empty(), "Term should not be empty");
            assert!(!entry.category.is_empty(), "Category should not be empty");
        }
    }
}
