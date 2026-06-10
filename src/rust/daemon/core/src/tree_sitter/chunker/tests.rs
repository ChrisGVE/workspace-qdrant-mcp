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

    if let Ok(tokenizer) = ModelTokenizer::from_model_cache(None) {
        let tk = Arc::new(tokenizer);
        let with_tk = handle_oversized_chunks(vec![chunk], "", 1000, Some(&tk));
        assert_eq!(with_tk.len(), 1);
        assert!(!with_tk[0].is_fragment);
    }
}

#[test]
fn test_text_chunk_fallback_empty_source() {
    // Empty source should produce no chunks (no content to chunk).
    // Current behavior: returns a single empty chunk because the
    // small-file branch handles `source.len() <= target_chars` first.
    let path = PathBuf::from("empty.txt");
    let chunks = text_chunk_fallback("", &path, 8000);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].chunk_type, ChunkType::Text);
    assert!(chunks[0].content.is_empty());
}

#[test]
fn test_text_chunk_fallback_no_newlines_large() {
    // Content with no newlines that exceeds target_chars hits the
    // `for line in source.lines()` loop with a single line. Since the
    // line itself is bigger than the budget, the loop still emits one
    // chunk containing that whole line — verify we don't drop content.
    let source = "x".repeat(10_000);
    let path = PathBuf::from("oneliner.txt");
    let chunks = text_chunk_fallback(&source, &path, 100); // target 400 chars

    assert!(!chunks.is_empty(), "Must not drop content");
    // Reassembling all chunk contents should round-trip the source
    // (with newlines reinserted by the splitter only when content had
    // them — no newlines here, so the single chunk contains everything).
    let reassembled: String = chunks.iter().map(|c| c.content.as_str()).collect();
    assert_eq!(reassembled.len(), source.len());
}

#[test]
fn test_text_chunk_fallback_language_from_extension() {
    let chunks = text_chunk_fallback("println!(\"hi\");", &PathBuf::from("a.rs"), 8000);
    assert_eq!(chunks[0].language, "rs");

    let chunks = text_chunk_fallback("print('hi')", &PathBuf::from("a.py"), 8000);
    assert_eq!(chunks[0].language, "py");

    // Missing extension falls back to "text".
    let chunks = text_chunk_fallback("plain", &PathBuf::from("README"), 8000);
    assert_eq!(chunks[0].language, "text");
}

#[test]
fn test_text_chunk_fallback_line_numbers_contiguous() {
    // Chunks emitted by the line-by-line splitter must cover the
    // source with monotonically increasing, contiguous line ranges
    // (chunk[i].end_line + 1 == chunk[i+1].start_line).
    let source = (1..=200)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n");
    let chunks = text_chunk_fallback(&source, &PathBuf::from("file.txt"), 50);

    assert!(chunks.len() > 1, "expected multiple chunks");
    for pair in chunks.windows(2) {
        assert_eq!(
            pair[0].end_line + 1,
            pair[1].start_line,
            "chunks not contiguous: {} -> {}",
            pair[0].end_line,
            pair[1].start_line
        );
    }
}

#[test]
fn test_safe_char_boundary_emoji_4byte() {
    // 4-byte UTF-8 (emoji): U+1F600 "😀" encodes as F0 9F 98 80.
    let s = "a\u{1F600}b";
    // Layout: a(0) 😀(1,2,3,4) b(5)
    assert_eq!(safe_char_boundary(s, 0), 0);
    assert_eq!(safe_char_boundary(s, 1), 1); // start of emoji
    assert_eq!(safe_char_boundary(s, 2), 1); // inside emoji -> back to 1
    assert_eq!(safe_char_boundary(s, 3), 1);
    assert_eq!(safe_char_boundary(s, 4), 1);
    assert_eq!(safe_char_boundary(s, 5), 5); // start of b
}

#[test]
fn test_safe_char_boundary_empty_and_zero() {
    assert_eq!(safe_char_boundary("", 0), 0);
    assert_eq!(safe_char_boundary("", 100), 0); // beyond len of empty
    assert_eq!(safe_char_boundary("abc", 0), 0);
}

#[test]
fn test_handle_oversized_chunks_empty_input() {
    let result = handle_oversized_chunks(Vec::new(), "", 1000, None);
    assert!(result.is_empty());
}

#[test]
fn test_handle_oversized_chunks_mixed_batch_preserves_order() {
    // Small + large + small: small chunks pass through unchanged,
    // large is split into fragments. Order must be preserved.
    let small_a = SemanticChunk::new(
        ChunkType::Function,
        "small_a",
        "fn a() {}",
        1,
        1,
        "rust",
        "a.rs",
    );
    let large = SemanticChunk::new(
        ChunkType::Function,
        "big",
        "let v = compute();\n".repeat(150), // ~2850 chars, ~712 tokens via len/4
        1,
        150,
        "rust",
        "big.rs",
    );
    let small_b = SemanticChunk::new(
        ChunkType::Function,
        "small_b",
        "fn b() {}",
        1,
        1,
        "rust",
        "b.rs",
    );

    // max_chunk_size=200 → target_size=800 chars, step_size=300 (> 0),
    // so the splitter actually advances past the first fragment.
    let result = handle_oversized_chunks(
        vec![small_a.clone(), large.clone(), small_b.clone()],
        "",
        200,
        None,
    );

    // First should be the unchanged small_a.
    assert_eq!(result[0].symbol_name, "small_a");
    assert!(!result[0].is_fragment);

    // Last should be the unchanged small_b.
    let last = result.last().expect("non-empty");
    assert_eq!(last.symbol_name, "small_b");
    assert!(!last.is_fragment);

    // Everything between must be `big` fragments.
    let middle = &result[1..result.len() - 1];
    assert!(
        !middle.is_empty(),
        "expected `big` to split into >=1 fragment"
    );
    for frag in middle {
        assert_eq!(frag.symbol_name, "big");
        assert!(frag.is_fragment);
    }
}

#[test]
fn test_split_preserves_parent_symbol_on_all_fragments() {
    // Methods carry parent_symbol; when a long method is fragmented,
    // every fragment must keep the parent attribution so downstream
    // consumers (graph, search) don't lose the class context.
    let chunker = SemanticChunker::new(200); // target 800 chars

    let chunk = SemanticChunk::new(
        ChunkType::Method,
        "process",
        "let v = compute();\n".repeat(150), // ~2850 chars
        10,
        160,
        "rust",
        "svc.rs",
    )
    .with_parent("AuthService");

    let fragments = chunker.split_oversized_chunk(&chunk);
    assert!(fragments.len() > 1, "expected fragmentation");
    for frag in &fragments {
        assert_eq!(
            frag.parent_symbol.as_deref(),
            Some("AuthService"),
            "fragment {:?} lost parent_symbol",
            frag.fragment_index
        );
    }
}

#[test]
fn test_split_fragment_indices_sequential() {
    // Fragment indices must form 0, 1, 2, ... with no gaps or duplicates.
    let chunker = SemanticChunker::new(200);
    let chunk = SemanticChunk::new(
        ChunkType::Function,
        "long_fn",
        "let _x = 0;\n".repeat(400), // 4800 chars, will produce several fragments
        1,
        400,
        "rust",
        "f.rs",
    );

    let fragments = chunker.split_oversized_chunk(&chunk);
    assert!(
        fragments.len() >= 3,
        "need multiple fragments to test ordering"
    );

    let indices: Vec<usize> = fragments
        .iter()
        .map(|f| f.fragment_index.expect("set on fragments"))
        .collect();
    let expected: Vec<usize> = (0..fragments.len()).collect();
    assert_eq!(
        indices, expected,
        "fragment indices must be 0..N sequential"
    );

    // All fragments report the same total_fragments value.
    let totals: std::collections::HashSet<usize> = fragments
        .iter()
        .map(|f| f.total_fragments.expect("set on fragments"))
        .collect();
    assert_eq!(
        totals.len(),
        1,
        "total_fragments must agree across fragments"
    );

    // Later fragments never leak docstring/signature — those belong only
    // to the first fragment (regression for the as_fragment() logic).
    for frag in &fragments[1..] {
        assert!(
            frag.docstring.is_none(),
            "non-first fragment leaked docstring"
        );
        assert!(
            frag.signature.is_none(),
            "non-first fragment leaked signature"
        );
    }
}

/// Regression: a long run without line breaks must not loop forever.
///
/// Reproduces the daemon OOM where `split_chunk_with_overlap` allocated
/// fragments without bound (~10 GB Vec + ~14 GB of fragment Strings) on
/// minified / single-line content. `find_line_boundary` snapped `actual_end`
/// back to the same early newline every iteration while the overlap pull-back
/// failed to advance `start`. Must now terminate with a bounded number of
/// gap-free fragments.
#[test]
fn test_split_no_newline_tail_terminates() {
    let chunker = SemanticChunker::new(200); // target_size = 800, step_size = 300

    // Early newline followed by a long unbroken run — the exact stall trigger.
    let content = format!("preamble\n{}", "x".repeat(60_000));
    assert!(content.len() > 800);

    let chunk = SemanticChunk::new(
        ChunkType::Function,
        "minified",
        &content,
        1,
        1,
        "javascript",
        "bundle.min.js",
    );

    let fragments = chunker.split_oversized_chunk(&chunk);

    assert!(fragments.len() >= 2, "long content should split");
    // Bounded: count is ~content.len()/step_size, never per-character explosion.
    assert!(
        fragments.len() < content.len() / 100,
        "fragment count {} is unbounded for {}-char input",
        fragments.len(),
        content.len()
    );
    // Coverage: first fragment starts at the beginning, last reaches the end.
    assert!(
        content.starts_with(fragments[0].content.as_str()),
        "first fragment must cover the start"
    );
    assert!(
        content.ends_with(fragments.last().unwrap().content.as_str()),
        "last fragment must cover the end"
    );
}

/// Regression: a single giant line with no newline at all.
#[test]
fn test_split_single_giant_line_terminates() {
    let chunker = SemanticChunker::new(200);
    let content = "a".repeat(100_000); // no '\n' anywhere

    let chunk = SemanticChunk::new(ChunkType::Text, "blob", &content, 1, 1, "json", "data.json");

    let fragments = chunker.split_oversized_chunk(&chunk);
    assert!(fragments.len() >= 2);
    assert!(
        fragments.len() < content.len() / 100,
        "fragment count {} unbounded",
        fragments.len()
    );
    assert!(content.ends_with(fragments.last().unwrap().content.as_str()));
}
