//! Unit tests for the semantic chunker module.

use std::path::PathBuf;
use std::sync::Arc;

use super::splitting::{handle_oversized_chunks, safe_char_boundary, text_chunk_fallback};
use super::SemanticChunker;
use super::DEFAULT_MAX_CHUNK_SIZE;
use crate::tokenizer::ModelTokenizer;
use crate::tree_sitter::types::{ChunkType, SemanticChunk};

#[test]
fn test_text_chunk_fallback_small() {
    let source = "hello\nworld";
    let path = PathBuf::from("test.txt");
    let chunks = text_chunk_fallback(source, &path, 8000);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].chunk_type, ChunkType::Text);
    assert_eq!(chunks[0].start_line, 1);
    // end_line is the last line processed
    assert!(chunks[0].end_line >= 1);
}

#[test]
fn test_text_chunk_fallback_large() {
    let source = "line\n".repeat(10000);
    let path = PathBuf::from("test.txt");
    let chunks = text_chunk_fallback(&source, &path, 100); // Small max size

    assert!(chunks.len() > 1);
    // All chunks should be text type
    for chunk in &chunks {
        assert_eq!(chunk.chunk_type, ChunkType::Text);
    }
}

#[test]
fn test_chunker_creation() {
    let chunker = SemanticChunker::new(4000);
    assert_eq!(chunker.max_chunk_size, 4000);

    let chunker = SemanticChunker::default();
    assert_eq!(chunker.max_chunk_size, DEFAULT_MAX_CHUNK_SIZE);
}

#[test]
fn test_split_chunk_with_overlap() {
    // Use a chunk size that gives target_size > FRAGMENT_OVERLAP (500)
    // max_chunk_size * 4 = target_size, so we need max_chunk_size > 125
    let chunker = SemanticChunker::new(200); // target_size = 800 chars

    // Create content larger than target_size to trigger splitting
    // 31 chars * 100 = 3100 chars, which is > 800
    let chunk = SemanticChunk::new(
        ChunkType::Function,
        "big_fn",
        "line1\nline2\nline3\nline4\nline5\n".repeat(100),
        1,
        500,
        "rust",
        "test.rs",
    )
    .with_docstring("A big function")
    .with_signature("fn big_fn()");

    let fragments = chunker.split_oversized_chunk(&chunk);

    assert!(
        fragments.len() > 1,
        "Expected multiple fragments, got {}",
        fragments.len()
    );
    // First fragment should have docstring and signature
    assert!(fragments[0].docstring.is_some());
    assert!(fragments[0].signature.is_some());
    // Subsequent fragments should not
    if fragments.len() > 1 {
        assert!(fragments[1].docstring.is_none());
    }
    // All should be marked as fragments
    for frag in &fragments {
        assert!(frag.is_fragment);
        assert!(frag.fragment_index.is_some());
        assert!(frag.total_fragments.is_some());
    }
}

#[test]
fn test_safe_char_boundary() {
    // ASCII: all byte indices are char boundaries
    assert_eq!(safe_char_boundary("hello", 3), 3);
    assert_eq!(safe_char_boundary("hello", 10), 5); // beyond end

    // Box-drawing char (U+2500, 3 bytes)
    let s = "ab\u{2500}cd";
    // Layout: a(0) b(1) \u{2500}(2,3,4) c(5) d(6)
    assert_eq!(safe_char_boundary(s, 2), 2); // start of \u{2500}
    assert_eq!(safe_char_boundary(s, 3), 2); // inside \u{2500} -> back to 2
    assert_eq!(safe_char_boundary(s, 4), 2); // inside \u{2500} -> back to 2
    assert_eq!(safe_char_boundary(s, 5), 5); // start of c
}

#[test]
fn test_split_chunk_with_multibyte_utf8() {
    // Reproduce the exact crash scenario: box-drawing chars in code comments
    let chunker = SemanticChunker::new(200); // target_size = 800 bytes

    // Build content with box-drawing chars that exceeds target_size
    let line =
        "    // \u{2500}\u{2500} parse_relative_duration \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n";
    let content = line.repeat(20);
    assert!(content.len() > 800, "Content must exceed target_size");

    let chunk = SemanticChunk::new(
        ChunkType::Function,
        "test_fn",
        &content,
        1,
        20,
        "rust",
        "test.rs",
    );

    // This should NOT panic even with multi-byte characters
    let fragments = chunker.split_oversized_chunk(&chunk);
    assert!(!fragments.is_empty());
    // Verify all fragment content is valid UTF-8 (implicit in &str type)
    for frag in &fragments {
        assert!(!frag.content.is_empty());
    }
}

#[test]
fn test_chunker_with_tokenizer_builder() {
    // Builder is plumbed regardless of cache state.
    let chunker = SemanticChunker::new(100);
    assert!(chunker.tokenizer().is_none());

    let Some(tk) = ModelTokenizer::from_model_cache(None).ok() else {
        eprintln!("Skipping cache-dependent assertion: HF model not cached");
        return;
    };

    let chunker = SemanticChunker::new(100).with_tokenizer(Arc::new(tk));
    assert!(chunker.tokenizer().is_some());
}

#[test]
fn test_oversize_gate_uses_real_tokenizer() {
    // Goal: a chunk where `content.len() / 4` underestimates real token count.
    // Without a tokenizer, the heuristic passes the gate; with a tokenizer,
    // the real count fails the gate and fragmentation kicks in.

    let Some(tokenizer) = ModelTokenizer::from_model_cache(None).ok() else {
        eprintln!("Skipping: HF model not cached (expected on CI without download)");
        return;
    };
    let tk = Arc::new(tokenizer);

    // Token-dense Rust: operators + short identifiers + punctuation tokenize
    // to many more pieces than `len/4` predicts.
    let content = "let _x: usize = ();\n".repeat(50);
    let real_tokens = tk.count_tokens(&content).expect("count_tokens");
    let heuristic_tokens = content.len() / 4;

    if real_tokens <= heuristic_tokens {
        // Tokenizer vocabulary on this host doesn't show the divergence —
        // skip rather than fail (different model versions may differ).
        eprintln!(
            "Skipping: tokenizer produced {real_tokens} tokens \
             vs heuristic {heuristic_tokens} (no divergence)"
        );
        return;
    }

    // Budget sits strictly between heuristic and real: heuristic gate passes,
    // real gate fails.
    let budget = heuristic_tokens + 1;
    assert!(budget < real_tokens, "budget setup invariant");

    let chunk = SemanticChunk::new(
        ChunkType::Function,
        "tokens_heavy",
        content.clone(),
        1,
        50,
        "rust",
        "test.rs",
    );

    // Without tokenizer: len/4 fallback says we're under budget, no split.
    let without_tk = handle_oversized_chunks(vec![chunk.clone()], "", budget, None);
    assert_eq!(
        without_tk.len(),
        1,
        "Heuristic gate (len/4={heuristic_tokens}) <= budget={budget}; \
         expected no split but got {} fragments",
        without_tk.len()
    );
    assert!(!without_tk[0].is_fragment);

    // With tokenizer: real count exceeds budget, split kicks in.
    let with_tk = handle_oversized_chunks(vec![chunk], "", budget, Some(&tk));
    assert!(
        with_tk.len() > 1,
        "Real gate ({real_tokens} > budget={budget}); expected split, got {} fragment(s)",
        with_tk.len()
    );
    assert!(with_tk.iter().all(|f| f.is_fragment));
}

#[test]
fn test_oversize_gate_no_split_when_under_budget() {
    // Regression guard: a small chunk that fits under both heuristic and
    // real budgets must not be fragmented, with or without a tokenizer.
    let content = "fn small() {}".to_string();

    let chunk = SemanticChunk::new(
        ChunkType::Function,
        "small",
        content,
        1,
        1,
        "rust",
        "test.rs",
    );

    let without_tk = handle_oversized_chunks(vec![chunk.clone()], "", 1000, None);
    assert_eq!(without_tk.len(), 1);
    assert!(!without_tk[0].is_fragment);

    if let Some(tokenizer) = ModelTokenizer::from_model_cache(None).ok() {
        let tk = Arc::new(tokenizer);
        let with_tk = handle_oversized_chunks(vec![chunk], "", 1000, Some(&tk));
        assert_eq!(with_tk.len(), 1);
        assert!(!with_tk[0].is_fragment);
    }
}
