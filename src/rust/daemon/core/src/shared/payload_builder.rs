//! Typed payload builder for Qdrant point construction.
//!
//! Replaces scattered `HashMap::new()` + manual `.insert()` chains
//! with a fluent builder API that enforces field name consistency.

use std::collections::HashMap;
use serde_json::Value;

/// Fluent builder for Qdrant point payloads.
///
/// Instead of manually constructing `HashMap<String, Value>` with string
/// literals for keys, use typed setter methods that guarantee field name
/// consistency across all processing strategies.
///
/// # Example
/// ```ignore
/// let payload = PayloadBuilder::new()
///     .tenant_id("project-abc")
///     .content("fn main() {}")
///     .branch("main")
///     .item_type("file")
///     .build();
/// ```
pub struct PayloadBuilder {
    inner: HashMap<String, Value>,
}

impl PayloadBuilder {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    // ── Core fields (present on most points) ────────────────────────

    pub fn tenant_id(mut self, id: &str) -> Self {
        self.inner.insert("tenant_id".into(), serde_json::json!(id));
        self
    }

    pub fn content(mut self, text: &str) -> Self {
        self.inner
            .insert("content".into(), serde_json::json!(text));
        self
    }

    pub fn document_id(mut self, id: &str) -> Self {
        self.inner
            .insert("document_id".into(), serde_json::json!(id));
        self
    }

    pub fn branch(mut self, b: &str) -> Self {
        self.inner.insert("branch".into(), serde_json::json!(b));
        self
    }

    pub fn item_type(mut self, t: &str) -> Self {
        self.inner
            .insert("item_type".into(), serde_json::json!(t));
        self
    }

    pub fn source_type(mut self, t: &str) -> Self {
        self.inner
            .insert("source_type".into(), serde_json::json!(t));
        self
    }

    // ── File-specific fields ────────────────────────────────────────

    pub fn file_path(mut self, p: &str) -> Self {
        self.inner
            .insert("file_path".into(), serde_json::json!(p));
        self
    }

    pub fn relative_path(mut self, p: &str) -> Self {
        self.inner
            .insert("relative_path".into(), serde_json::json!(p));
        self
    }

    pub fn absolute_path(mut self, p: &str) -> Self {
        self.inner
            .insert("absolute_path".into(), serde_json::json!(p));
        self
    }

    pub fn file_extension(mut self, ext: &str) -> Self {
        self.inner
            .insert("file_extension".into(), serde_json::json!(ext));
        self
    }

    pub fn file_hash(mut self, hash: &str) -> Self {
        self.inner
            .insert("file_hash".into(), serde_json::json!(hash));
        self
    }

    pub fn document_type(mut self, dt: &str) -> Self {
        self.inner
            .insert("document_type".into(), serde_json::json!(dt));
        self
    }

    pub fn language(mut self, lang: &str) -> Self {
        self.inner
            .insert("language".into(), serde_json::json!(lang));
        self
    }

    pub fn chunk_index(mut self, idx: usize) -> Self {
        self.inner
            .insert("chunk_index".into(), serde_json::json!(idx));
        self
    }

    pub fn base_point(mut self, bp: &str) -> Self {
        self.inner
            .insert("base_point".into(), serde_json::json!(bp));
        self
    }

    // ── Memory-specific fields ──────────────────────────────────────

    pub fn label(mut self, l: &str) -> Self {
        self.inner.insert("label".into(), serde_json::json!(l));
        self
    }

    pub fn scope(mut self, s: &str) -> Self {
        self.inner.insert("scope".into(), serde_json::json!(s));
        self
    }

    pub fn title(mut self, t: &str) -> Self {
        self.inner.insert("title".into(), serde_json::json!(t));
        self
    }

    pub fn project_id(mut self, id: &str) -> Self {
        self.inner
            .insert("project_id".into(), serde_json::json!(id));
        self
    }

    // ── Tag fields ──────────────────────────────────────────────────

    pub fn main_tag(mut self, tag: &str) -> Self {
        self.inner
            .insert("main_tag".into(), serde_json::json!(tag));
        self
    }

    pub fn full_tag(mut self, tag: &str) -> Self {
        self.inner
            .insert("full_tag".into(), serde_json::json!(tag));
        self
    }

    // ── Generic setters for less-common fields ──────────────────────

    /// Insert an arbitrary key-value pair.
    pub fn field(mut self, key: impl Into<String>, value: Value) -> Self {
        self.inner.insert(key.into(), value);
        self
    }

    /// Insert a string value for an arbitrary key.
    pub fn field_str(mut self, key: impl Into<String>, value: &str) -> Self {
        self.inner.insert(key.into(), serde_json::json!(value));
        self
    }

    /// Insert an optional string value (skips if `None`).
    pub fn field_opt(mut self, key: impl Into<String>, value: Option<&str>) -> Self {
        if let Some(v) = value {
            self.inner.insert(key.into(), serde_json::json!(v));
        }
        self
    }

    /// Insert all entries from an existing HashMap (for chunk metadata etc.).
    pub fn extend(mut self, entries: HashMap<String, Value>) -> Self {
        self.inner.extend(entries);
        self
    }

    /// Consume the builder and return the payload HashMap.
    pub fn build(self) -> HashMap<String, Value> {
        self.inner
    }
}

impl Default for PayloadBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_builder_basic() {
        let payload = PayloadBuilder::new()
            .tenant_id("test-tenant")
            .content("hello world")
            .branch("main")
            .item_type("file")
            .build();

        assert_eq!(payload["tenant_id"], serde_json::json!("test-tenant"));
        assert_eq!(payload["content"], serde_json::json!("hello world"));
        assert_eq!(payload["branch"], serde_json::json!("main"));
        assert_eq!(payload["item_type"], serde_json::json!("file"));
    }

    #[test]
    fn test_payload_builder_optional_fields() {
        let payload = PayloadBuilder::new()
            .tenant_id("t")
            .field_opt("label", Some("my-label"))
            .field_opt("missing", None)
            .build();

        assert_eq!(payload["label"], serde_json::json!("my-label"));
        assert!(!payload.contains_key("missing"));
    }

    #[test]
    fn test_payload_builder_extend() {
        let mut extra = HashMap::new();
        extra.insert("chunk_symbol".to_string(), serde_json::json!("main"));
        extra.insert("chunk_start_line".to_string(), serde_json::json!(1));

        let payload = PayloadBuilder::new()
            .tenant_id("t")
            .extend(extra)
            .build();

        assert_eq!(payload["chunk_symbol"], serde_json::json!("main"));
        assert_eq!(payload["chunk_start_line"], serde_json::json!(1));
        assert_eq!(payload.len(), 3);
    }

    #[test]
    fn test_payload_builder_file_fields() {
        let payload = PayloadBuilder::new()
            .file_path("/src/main.rs")
            .relative_path("src/main.rs")
            .absolute_path("/home/user/project/src/main.rs")
            .file_extension("rs")
            .document_type("code")
            .language("rust")
            .base_point("abc123")
            .build();

        assert_eq!(payload.len(), 7);
        assert_eq!(payload["language"], serde_json::json!("rust"));
    }
}
