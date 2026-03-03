//! Parent-unit record structure for Qdrant.
//!
//! Parent records store full structural units (pages, chapters, code files)
//! without vectors. They serve as expansion targets: when a search returns
//! a chunk, the parent record provides the full surrounding context.
//!
//! Parent records live in the same collection as chunks, discriminated by
//! `record_type = "parent"` vs `record_type = "chunk"`.

mod code_parents;
mod types;

// Re-export everything so callers continue to use `crate::parent_unit::*`
// without any path changes.
pub use code_parents::{create_code_parents, CodeParentMapping};
pub use types::{
    code_block_parent, code_file_parent, epub_section_parent, parent_point_id, pdf_page_parent,
    sha256_hex, text_section_parent, ParentUnitRecord, RECORD_TYPE_CHUNK, RECORD_TYPE_PARENT,
    UNIT_TYPE_CODE_BLOCK, UNIT_TYPE_CODE_FILE, UNIT_TYPE_DOCX_SECTION, UNIT_TYPE_EPUB_SECTION,
    UNIT_TYPE_PDF_PAGE, UNIT_TYPE_TEXT_SECTION,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_page_parent_creation() {
        let parent = pdf_page_parent("doc-123", "fingerprint-abc", 5, "Page five content here.");
        assert_eq!(parent.doc_id, "doc-123");
        assert_eq!(parent.doc_fingerprint, "fingerprint-abc");
        assert_eq!(parent.unit_type, UNIT_TYPE_PDF_PAGE);
        assert_eq!(parent.unit_locator["page"], 5);
        assert_eq!(parent.unit_text, "Page five content here.");
        assert_eq!(parent.unit_char_len, 23);
        assert!(!parent.unit_hash.is_empty());
        assert!(!parent.point_id.is_empty());
    }

    #[test]
    fn test_epub_section_parent_creation() {
        let parent = epub_section_parent(
            "doc-456",
            "fp-xyz",
            "ch3",
            Some("Chapter Three"),
            "The content of chapter three.",
        );
        assert_eq!(parent.unit_type, UNIT_TYPE_EPUB_SECTION);
        assert_eq!(parent.unit_locator["spine_id"], "ch3");
        assert_eq!(parent.unit_locator["chapter_title"], "Chapter Three");
    }

    #[test]
    fn test_epub_section_no_title() {
        let parent =
            epub_section_parent("doc-456", "fp-xyz", "ch1", None, "Untitled chapter content.");
        assert_eq!(parent.unit_locator["spine_id"], "ch1");
        assert!(parent.unit_locator.get("chapter_title").is_none());
    }

    #[test]
    fn test_code_file_parent_creation() {
        let parent = code_file_parent(
            "doc-789",
            "fp-code",
            "src/main.rs",
            "fn main() { println!(\"hello\"); }",
        );
        assert_eq!(parent.unit_type, UNIT_TYPE_CODE_FILE);
        assert_eq!(parent.unit_locator["file_path"], "src/main.rs");
    }

    #[test]
    fn test_text_section_parent_creation() {
        let parent =
            text_section_parent("doc-txt", "fp-text", "Introduction", 0, "This is the introduction.");
        assert_eq!(parent.unit_type, UNIT_TYPE_TEXT_SECTION);
        assert_eq!(parent.unit_locator["section_title"], "Introduction");
        assert_eq!(parent.unit_locator["section_index"], 0);
    }

    #[test]
    fn test_unit_hash_deterministic() {
        let text = "Deterministic hashing test.";
        let hash1 = sha256_hex(text);
        let hash2 = sha256_hex(text);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA256 hex = 64 chars
    }

    #[test]
    fn test_unit_hash_changes_with_content() {
        let hash1 = sha256_hex("Version 1");
        let hash2 = sha256_hex("Version 2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_parent_point_id_deterministic() {
        let locator = serde_json::json!({"page": 1});
        let id1 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &locator);
        let id2 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &locator);
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 32); // UUID hex without dashes
    }

    #[test]
    fn test_parent_point_id_unique_across_pages() {
        let loc1 = serde_json::json!({"page": 1});
        let loc2 = serde_json::json!({"page": 2});
        let id1 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &loc1);
        let id2 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &loc2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_parent_point_id_unique_across_unit_types() {
        let locator = serde_json::json!({"page": 1});
        let id1 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &locator);
        let id2 = parent_point_id("doc-1", UNIT_TYPE_EPUB_SECTION, &locator);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_to_payload_fields() {
        let parent = pdf_page_parent("doc-1", "fp-1", 3, "Page three.");
        let payload = parent.to_payload("my-lib", Some("My Doc"), "page_based", "pdf");

        assert_eq!(payload["library_name"], "my-lib");
        assert_eq!(payload["record_type"], RECORD_TYPE_PARENT);
        assert_eq!(payload["doc_id"], "doc-1");
        assert_eq!(payload["doc_fingerprint"], "fp-1");
        assert_eq!(payload["doc_type"], "page_based");
        assert_eq!(payload["source_format"], "pdf");
        assert_eq!(payload["doc_title"], "My Doc");
        assert_eq!(payload["unit_type"], UNIT_TYPE_PDF_PAGE);
        assert_eq!(payload["chunk_text_raw"], "Page three.");
        assert!(payload.contains_key("unit_hash"));
        assert!(payload.contains_key("unit_locator"));
    }

    #[test]
    fn test_to_payload_without_title() {
        let parent = pdf_page_parent("doc-1", "fp-1", 1, "Content.");
        let payload = parent.to_payload("lib", None, "page_based", "pdf");
        assert!(!payload.contains_key("doc_title"));
    }

    #[test]
    fn test_code_block_parent_creation() {
        let parent = code_block_parent(
            "doc-code",
            "fp-code",
            "src/lib.rs",
            "MyStruct",
            "struct",
            10,
            50,
            "struct MyStruct { ... }",
        );
        assert_eq!(parent.unit_type, UNIT_TYPE_CODE_BLOCK);
        assert_eq!(parent.unit_locator["file_path"], "src/lib.rs");
        assert_eq!(parent.unit_locator["block_name"], "MyStruct");
        assert_eq!(parent.unit_locator["block_kind"], "struct");
        assert_eq!(parent.unit_locator["start_line"], 10);
        assert_eq!(parent.unit_locator["end_line"], 50);
        assert!(!parent.point_id.is_empty());
    }

    #[test]
    fn test_code_block_parent_unique_per_block() {
        let p1 = code_block_parent("doc", "fp", "f.rs", "Foo", "class", 1, 10, "class Foo");
        let p2 = code_block_parent("doc", "fp", "f.rs", "Bar", "class", 20, 30, "class Bar");
        assert_ne!(p1.point_id, p2.point_id);
    }

    #[test]
    fn test_is_container_type() {
        use crate::tree_sitter::types::ChunkType;
        use crate::parent_unit::code_parents::is_container_type;
        assert!(is_container_type(ChunkType::Class));
        assert!(is_container_type(ChunkType::Struct));
        assert!(is_container_type(ChunkType::Trait));
        assert!(is_container_type(ChunkType::Interface));
        assert!(is_container_type(ChunkType::Impl));
        assert!(is_container_type(ChunkType::Module));
        assert!(is_container_type(ChunkType::Enum));
        // Non-containers
        assert!(!is_container_type(ChunkType::Function));
        assert!(!is_container_type(ChunkType::Method));
        assert!(!is_container_type(ChunkType::Preamble));
        assert!(!is_container_type(ChunkType::Text));
        assert!(!is_container_type(ChunkType::Constant));
    }

    #[test]
    fn test_create_code_parents_file_only() {
        use crate::tree_sitter::types::{ChunkType, SemanticChunk};
        // Two top-level functions, no containers
        let chunks = vec![
            SemanticChunk::new(ChunkType::Function, "foo", "fn foo() {}", 1, 3, "rust", "lib.rs"),
            SemanticChunk::new(ChunkType::Function, "bar", "fn bar() {}", 5, 8, "rust", "lib.rs"),
        ];
        let mapping =
            create_code_parents("doc", "fp", "lib.rs", "fn foo() {}\nfn bar() {}", &chunks);

        assert_eq!(mapping.file_parent.unit_type, UNIT_TYPE_CODE_FILE);
        assert!(mapping.block_parents.is_empty());
        assert_eq!(mapping.chunk_parent_ids.len(), 2);
        // Both map to the file parent
        assert_eq!(mapping.chunk_parent_ids[0], mapping.file_parent.point_id);
        assert_eq!(mapping.chunk_parent_ids[1], mapping.file_parent.point_id);
    }

    #[test]
    fn test_create_code_parents_with_class() {
        use crate::tree_sitter::types::{ChunkType, SemanticChunk};
        let chunks = vec![
            SemanticChunk::new(
                ChunkType::Class,
                "MyClass",
                "class MyClass { ... }",
                1,
                20,
                "python",
                "app.py",
            ),
            SemanticChunk::new(
                ChunkType::Method,
                "process",
                "def process(self):",
                3,
                8,
                "python",
                "app.py",
            )
            .with_parent("MyClass"),
            SemanticChunk::new(
                ChunkType::Method,
                "validate",
                "def validate(self):",
                10,
                15,
                "python",
                "app.py",
            )
            .with_parent("MyClass"),
            SemanticChunk::new(
                ChunkType::Function,
                "helper",
                "def helper():",
                22,
                25,
                "python",
                "app.py",
            ),
        ];
        let mapping = create_code_parents("doc", "fp", "app.py", "full file text", &chunks);

        // One block parent for MyClass
        assert_eq!(mapping.block_parents.len(), 1);
        assert_eq!(mapping.block_parents[0].unit_locator["block_name"], "MyClass");
        assert_eq!(mapping.block_parents[0].unit_type, UNIT_TYPE_CODE_BLOCK);

        // Class chunk → file parent (the class IS a container, references file)
        assert_eq!(mapping.chunk_parent_ids[0], mapping.file_parent.point_id);
        // Methods → block parent (MyClass)
        assert_eq!(mapping.chunk_parent_ids[1], mapping.block_parents[0].point_id);
        assert_eq!(mapping.chunk_parent_ids[2], mapping.block_parents[0].point_id);
        // Top-level function → file parent
        assert_eq!(mapping.chunk_parent_ids[3], mapping.file_parent.point_id);
    }

    #[test]
    fn test_create_code_parents_with_impl_block() {
        use crate::tree_sitter::types::{ChunkType, SemanticChunk};
        let chunks = vec![
            SemanticChunk::new(
                ChunkType::Struct,
                "Config",
                "struct Config {}",
                1,
                5,
                "rust",
                "config.rs",
            ),
            SemanticChunk::new(
                ChunkType::Impl,
                "Config",
                "impl Config { ... }",
                7,
                30,
                "rust",
                "config.rs",
            ),
            SemanticChunk::new(
                ChunkType::Method,
                "new",
                "fn new() -> Self",
                8,
                15,
                "rust",
                "config.rs",
            )
            .with_parent("Config"),
            SemanticChunk::new(
                ChunkType::Method,
                "validate",
                "fn validate(&self)",
                17,
                25,
                "rust",
                "config.rs",
            )
            .with_parent("Config"),
        ];
        let mapping = create_code_parents("doc", "fp", "config.rs", "full text", &chunks);

        // Struct and Impl are both containers; both create block parents
        // but they share the same name "Config", so the map stores the last one (Impl)
        assert_eq!(mapping.block_parents.len(), 2);
        // Methods reference the "Config" block parent (last wins in HashMap)
        let config_id = mapping.chunk_parent_ids[2].clone();
        assert_eq!(mapping.chunk_parent_ids[3], config_id);
        assert_ne!(config_id, mapping.file_parent.point_id);
    }

    #[test]
    fn test_create_code_parents_multiple_classes() {
        use crate::tree_sitter::types::{ChunkType, SemanticChunk};
        let chunks = vec![
            SemanticChunk::new(ChunkType::Class, "Foo", "class Foo", 1, 10, "python", "m.py"),
            SemanticChunk::new(ChunkType::Method, "run", "def run(self):", 3, 8, "python", "m.py")
                .with_parent("Foo"),
            SemanticChunk::new(ChunkType::Class, "Bar", "class Bar", 12, 20, "python", "m.py"),
            SemanticChunk::new(
                ChunkType::Method,
                "start",
                "def start(self):",
                14,
                18,
                "python",
                "m.py",
            )
            .with_parent("Bar"),
        ];
        let mapping = create_code_parents("doc", "fp", "m.py", "full", &chunks);

        assert_eq!(mapping.block_parents.len(), 2);
        // run → Foo's block parent
        // start → Bar's block parent
        assert_ne!(mapping.chunk_parent_ids[1], mapping.chunk_parent_ids[3]);
        // Both classes → file parent
        assert_eq!(mapping.chunk_parent_ids[0], mapping.file_parent.point_id);
        assert_eq!(mapping.chunk_parent_ids[2], mapping.file_parent.point_id);
    }
}
