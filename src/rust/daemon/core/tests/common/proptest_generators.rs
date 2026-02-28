//! Shared proptest generators for property-based integration tests
//!
//! Provides reusable random data generators for file content, document types,
//! chunking configurations, embeddings, and error scenarios.

use proptest::prelude::*;

use workspace_qdrant_core::{ChunkingConfig, DocumentType};

// ============================================================================
// CUSTOM PROPTEST GENERATORS
// ============================================================================

/// Generate random file content with various encodings and formats
prop_compose! {
    pub fn arb_file_content()(
        content_type in prop_oneof![
            "text", "binary", "mixed", "unicode", "empty", "large"
        ],
        size in 0..100000usize,
    ) -> String {
        match content_type.as_str() {
            "text" => {
                let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t.,!?;:-_()[]{}\"'@#$%^&*+=|\\/<>~`"
                    .chars().collect();
                (0..size).map(|_| chars[fastrand::usize(..chars.len())]).collect()
            },
            "binary" => {
                (0..size).map(|_| fastrand::u8(..) as char).collect()
            },
            "mixed" => {
                let mut content = String::new();
                for _ in 0..size {
                    match fastrand::u8(..4) {
                        0 => content.push_str("Hello, World! "),
                        1 => content.push(fastrand::u8(..) as char),
                        2 => content.push_str("\u{1f680}\u{1f525}\u{1f4af}\n"),
                        _ => content.push_str(&format!("Number: {}\t", fastrand::u64(..))),
                    }
                }
                content
            },
            "unicode" => {
                let unicode_ranges = [
                    0x0000..0x007F,  // Basic Latin
                    0x0080..0x00FF,  // Latin-1 Supplement
                    0x0100..0x017F,  // Latin Extended-A
                    0x1F600..0x1F64F, // Emoticons
                    0x4E00..0x9FFF,  // CJK Unified Ideographs
                ];
                (0..size).map(|_| {
                    let range = &unicode_ranges[fastrand::usize(..unicode_ranges.len())];
                    let code_point = fastrand::u32(range.clone());
                    std::char::from_u32(code_point).unwrap_or('?')
                }).collect()
            },
            "empty" => String::new(),
            "large" => {
                let base_text = "This is a large document with repeated content. ";
                base_text.repeat(size / base_text.len() + 1)
            },
            _ => "default content".to_string()
        }
    }
}

/// Generate random document types
pub fn arb_document_type() -> impl Strategy<Value = DocumentType> {
    prop_oneof![
        Just(DocumentType::Pdf),
        Just(DocumentType::Epub),
        Just(DocumentType::Docx),
        Just(DocumentType::Text),
        Just(DocumentType::Markdown),
        any::<String>().prop_map(DocumentType::Code),
        Just(DocumentType::Unknown),
    ]
}

/// Generate random chunking configurations
prop_compose! {
    pub fn arb_chunking_config()(
        chunk_size in 1..10000usize,
        overlap_size in 0..1000usize,
        preserve_paragraphs in any::<bool>(),
    ) -> ChunkingConfig {
        ChunkingConfig {
            chunk_size,
            overlap_size: std::cmp::min(overlap_size, chunk_size / 2),
            preserve_paragraphs,
            ..ChunkingConfig::default()
        }
    }
}

/// Generate random file extensions
pub fn arb_file_extension() -> impl Strategy<Value = String> {
    prop_oneof![
        "txt", "md", "pdf", "docx", "epub", "rs", "py", "js", "json", "yaml", "xml",
        "html", "css", "cpp", "java", "go", "rb", "php", "sh", "sql", "toml", "unknown"
    ]
    .prop_map(|s| s.to_string())
}

/// Generate malformed or edge case file content
prop_compose! {
    pub fn arb_malformed_content()(
        corruption_type in prop_oneof![
            "truncated", "oversized", "null_bytes", "invalid_utf8",
            "control_chars", "bom", "mixed_encodings"
        ],
        base_content in arb_file_content(),
    ) -> Vec<u8> {
        match corruption_type.as_str() {
            "truncated" => base_content.as_bytes()[..base_content.len() / 2].to_vec(),
            "oversized" => {
                let mut content = base_content.into_bytes();
                content.extend(vec![b'X'; 10_000_000]); // 10MB of X's
                content
            },
            "null_bytes" => {
                let mut content = base_content.into_bytes();
                for i in (0..content.len()).step_by(10) {
                    if i < content.len() {
                        content[i] = 0;
                    }
                }
                content
            },
            "invalid_utf8" => vec![0xFF, 0xFE, 0xFD, 0xFC, 0x80, 0x81, 0x82],
            "control_chars" => (0..256).map(|i| (i % 32) as u8).collect(),
            "bom" => {
                let mut content = vec![0xEF, 0xBB, 0xBF]; // UTF-8 BOM
                content.extend(base_content.into_bytes());
                content
            },
            "mixed_encodings" => {
                let mut content = base_content.into_bytes();
                // Add some Latin-1 bytes that aren't valid UTF-8
                content.extend(&[0xC0, 0xC1, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF]);
                content
            },
            _ => base_content.into_bytes()
        }
    }
}

/// Generate random vector dimensions for embedding tests
prop_compose! {
    pub fn arb_embedding_dimensions()(
        dims in prop_oneof![
            1..10usize,      // Very small
            384..385usize,   // Common size
            768..769usize,   // Common size
            1536..1537usize, // Common size
            10000..50000usize, // Very large
        ]
    ) -> usize {
        dims
    }
}

/// Generate random error scenarios
#[derive(Debug, Clone)]
pub enum ErrorScenario {
    NetworkTimeout,
    DiskFull,
    PermissionDenied,
    CorruptedData,
    OutOfMemory,
    InvalidFormat,
    ResourceExhausted,
    ConcurrencyLimit,
}

impl Arbitrary for ErrorScenario {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: ()) -> Self::Strategy {
        prop_oneof![
            Just(ErrorScenario::NetworkTimeout),
            Just(ErrorScenario::DiskFull),
            Just(ErrorScenario::PermissionDenied),
            Just(ErrorScenario::CorruptedData),
            Just(ErrorScenario::OutOfMemory),
            Just(ErrorScenario::InvalidFormat),
            Just(ErrorScenario::ResourceExhausted),
            Just(ErrorScenario::ConcurrencyLimit),
        ]
        .boxed()
    }
}
