//! Grammar Pipeline Diagnostic Test
//!
//! Stratified test that exercises the grammar pipeline for all 25 bookshelf
//! languages in isolation, answering:
//!   Q1: Can we download and load grammars when not cached?
//!   Q2: Are there compilation/loading errors?
//!
//! For each language, tests:
//!   1. Grammar cache status (cached or needs download)
//!   2. Grammar load/download success
//!   3. Bundled YAML semantic patterns availability
//!   4. Semantic chunking on a bookshelf source file
//!   5. Timing for each phase

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::language_registry::providers::registry::RegistryProvider;
use workspace_qdrant_core::tree_sitter::{extract_chunks_with_provider, GrammarManager};

/// Bookshelf languages and their representative source files.
const BOOKSHELF: &[(&str, &str)] = &[
    ("ada", "ada/src/utils.adb"),
    ("c", "c/src/storage.c"),
    ("clojure", "clojure/src/bookshelf/storage.clj"),
    ("cpp", "cpp/src/utils.cpp"),
    ("elixir", "elixir/lib/storage.ex"),
    ("erlang", "erlang/src/storage.erl"),
    ("fortran", "fortran/src/storage.f90"),
    ("go", "go/models.go"),
    ("haskell", "haskell/src/Storage.hs"),
    ("java", "java/src/bookshelf/Storage.java"),
    ("javascript", "javascript/src/models.js"),
    ("lisp", "lisp/src/utils.lisp"),
    ("lua", "lua/models.lua"),
    ("ocaml", "ocaml/bin/main.ml"),
    ("odin", "odin/main.odin"),
    ("pascal", "pascal/src/storage.pas"),
    ("perl", "perl/main.pl"),
    ("python", "python/bookshelf/models.py"),
    ("ruby", "ruby/main.rb"),
    ("rust", "rust/src/main.rs"),
    ("scala", "scala/src/main/scala/bookshelf/Storage.scala"),
    ("bash", "shell/utils.sh"),
    ("swift", "swift/Sources/Bookshelf/Storage.swift"),
    ("typescript", "typescript/src/models.ts"),
    ("zig", "zig/src/main.zig"),
];

#[derive(Debug)]
struct LanguageResult {
    language: String,
    was_cached: bool,
    grammar_load_ms: u64,
    grammar_load_ok: bool,
    grammar_load_err: Option<String>,
    has_bundled_patterns: bool,
    chunk_count: usize,
    has_semantic_chunks: bool,
    chunk_ms: u64,
    chunk_err: Option<String>,
    chunk_types: Vec<String>,
}

fn test_base_dir() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../../tests/language-support")
        .canonicalize()
        .expect("tests/language-support directory must exist")
}

/// Load bundled YAML patterns keyed by language id.
fn load_bundled_patterns() -> HashMap<String, bool> {
    let mut map = HashMap::new();
    match RegistryProvider::new() {
        Ok(provider) => {
            for def in provider.definitions() {
                let has = def.semantic_patterns.is_some();
                map.insert(def.id(), has);
            }
        }
        Err(e) => {
            eprintln!("Failed to load bundled provider: {}", e);
        }
    }
    map
}

#[tokio::test]
async fn grammar_pipeline_diagnostic() {
    let base = test_base_dir();

    // Use system grammar cache
    let cache_dir = dirs::home_dir().unwrap().join(".workspace-qdrant/grammars");

    let config = GrammarConfig {
        cache_dir,
        auto_download: true,
        ..Default::default()
    };

    let mut manager = GrammarManager::new(config);
    let bundled = load_bundled_patterns();

    let mut results: Vec<LanguageResult> = Vec::new();

    for (lang, file_rel) in BOOKSHELF {
        let file_path = base.join(file_rel);

        // Phase 1: Check cache status before loading
        let was_cached = manager.get_loaded_grammar(lang).is_some() || {
            // Check if .dylib exists in cache without loading
            let cache_paths = manager.cache_paths();
            cache_paths.grammar_path(lang).exists()
        };

        // Phase 2: Load/download grammar (timed)
        let t_load = Instant::now();
        let load_result = manager.get_grammar(lang).await;
        let grammar_load_ms = t_load.elapsed().as_millis() as u64;

        let (grammar_load_ok, grammar_load_err) = match &load_result {
            Ok(_) => (true, None),
            Err(e) => (false, Some(format!("{}", e))),
        };

        // Phase 3: Check bundled patterns
        let has_bundled_patterns = bundled.get(*lang).copied().unwrap_or(false);

        // Phase 4: Semantic chunking (timed)
        let (chunk_count, has_semantic, chunk_ms, chunk_err, chunk_types) = if grammar_load_ok
            && file_path.exists()
        {
            let provider = manager.create_language_provider();
            let source = std::fs::read_to_string(&file_path).unwrap_or_default();
            let t_chunk = Instant::now();
            match extract_chunks_with_provider(&source, &file_path, 4096, Some(Arc::new(provider)))
            {
                Ok(chunks) => {
                    let elapsed = t_chunk.elapsed().as_millis() as u64;
                    let types: Vec<String> = chunks
                        .iter()
                        .map(|c| format!("{:?}", c.chunk_type))
                        .collect();
                    let semantic = chunks.iter().any(|c| {
                        let ct = format!("{:?}", c.chunk_type);
                        ct != "Text" && ct != "Unknown"
                    });
                    (chunks.len(), semantic, elapsed, None, types)
                }
                Err(e) => (
                    0,
                    false,
                    t_chunk.elapsed().as_millis() as u64,
                    Some(format!("{}", e)),
                    vec![],
                ),
            }
        } else if !file_path.exists() {
            (0, false, 0, Some("file not found".to_string()), vec![])
        } else {
            (0, false, 0, Some("grammar not loaded".to_string()), vec![])
        };

        results.push(LanguageResult {
            language: lang.to_string(),
            was_cached,
            grammar_load_ms,
            grammar_load_ok,
            grammar_load_err,
            has_bundled_patterns,
            chunk_count,
            has_semantic_chunks: has_semantic,
            chunk_ms,
            chunk_err,
            chunk_types,
        });
    }

    // === Generate report ===
    let mut report = String::new();
    report.push_str("=== Grammar Pipeline Diagnostic Report ===\n\n");

    // Summary table header
    report.push_str(&format!(
        "{:<12} {:>7} {:>8} {:>8} {:>8} {:>6} {:>8}  {}\n",
        "Language", "Cached", "LoadMs", "LoadOK", "Patterns", "Chunks", "ChunkMs", "Status"
    ));
    report.push_str(&"-".repeat(100));
    report.push('\n');

    let mut total_load_ms: u64 = 0;
    let mut total_chunk_ms: u64 = 0;
    let mut grammar_ok = 0;
    let mut grammar_fail = 0;
    let mut semantic_ok = 0;
    let mut text_fallback = 0;
    let mut total_fail = 0;

    for r in &results {
        let status = if r.has_semantic_chunks {
            "SEMANTIC"
        } else if r.chunk_count > 0 {
            "TEXT_FALLBACK"
        } else if r.grammar_load_ok {
            "NO_CHUNKS"
        } else {
            "GRAMMAR_FAIL"
        };

        report.push_str(&format!(
            "{:<12} {:>7} {:>7}ms {:>8} {:>8} {:>6} {:>7}ms  {}\n",
            r.language,
            if r.was_cached { "yes" } else { "no" },
            r.grammar_load_ms,
            if r.grammar_load_ok { "OK" } else { "FAIL" },
            if r.has_bundled_patterns { "yes" } else { "NO" },
            r.chunk_count,
            r.chunk_ms,
            status,
        ));

        total_load_ms += r.grammar_load_ms;
        total_chunk_ms += r.chunk_ms;

        if r.grammar_load_ok {
            grammar_ok += 1;
        } else {
            grammar_fail += 1;
        }
        if r.has_semantic_chunks {
            semantic_ok += 1;
        } else if r.chunk_count > 0 {
            text_fallback += 1;
        } else {
            total_fail += 1;
        }
    }

    report.push_str(&"-".repeat(100));
    report.push_str(&format!(
        "\n\nSummary:\n  Grammar load OK: {}/{}\n  Grammar load FAIL: {}\n  Semantic chunks: {}\n  Text fallback: {}\n  No chunks: {}\n  Total grammar load time: {}ms\n  Total chunk time: {}ms\n",
        grammar_ok,
        results.len(),
        grammar_fail,
        semantic_ok,
        text_fallback,
        total_fail,
        total_load_ms,
        total_chunk_ms,
    ));

    // Detailed errors
    let errors: Vec<&LanguageResult> = results
        .iter()
        .filter(|r| r.grammar_load_err.is_some() || r.chunk_err.is_some())
        .collect();
    if !errors.is_empty() {
        report.push_str("\nDetailed Errors:\n");
        for r in &errors {
            if let Some(e) = &r.grammar_load_err {
                report.push_str(&format!("  {} grammar: {}\n", r.language, e));
            }
            if let Some(e) = &r.chunk_err {
                report.push_str(&format!("  {} chunking: {}\n", r.language, e));
            }
        }
    }

    // Chunk type details for semantic languages
    report.push_str("\nSemantic Chunk Details:\n");
    for r in &results {
        if r.has_semantic_chunks {
            report.push_str(&format!(
                "  {}: {} chunks [{}]\n",
                r.language,
                r.chunk_count,
                r.chunk_types.join(", ")
            ));
        }
    }

    // Languages needing attention
    let no_patterns: Vec<&str> = results
        .iter()
        .filter(|r| r.grammar_load_ok && !r.has_bundled_patterns)
        .map(|r| r.language.as_str())
        .collect();
    if !no_patterns.is_empty() {
        report.push_str(&format!(
            "\nGrammar loads OK but NO bundled YAML patterns (will always text-fallback): {:?}\n",
            no_patterns
        ));
    }

    // Write report
    std::fs::write("/tmp/grammar_pipeline_diagnostic.txt", &report).ok();
    print!("{}", report);

    // Assertions: no unexpected grammar failures
    let known_grammar_failures: &[&str] = &[];
    let known_chunk_failures: &[&str] = &[];

    let unexpected_grammar_failures: Vec<String> = results
        .iter()
        .filter(|r| !r.grammar_load_ok && !known_grammar_failures.contains(&r.language.as_str()))
        .map(|r| {
            format!(
                "{}: {}",
                r.language,
                r.grammar_load_err.as_deref().unwrap_or("unknown")
            )
        })
        .collect();

    assert!(
        unexpected_grammar_failures.is_empty(),
        "Unexpected grammar load failures: {:?}",
        unexpected_grammar_failures
    );

    let unexpected_chunk_failures: Vec<String> = results
        .iter()
        .filter(|r| {
            r.grammar_load_ok
                && r.chunk_count == 0
                && !known_chunk_failures.contains(&r.language.as_str())
        })
        .map(|r| {
            format!(
                "{}: {}",
                r.language,
                r.chunk_err.as_deref().unwrap_or("no chunks produced")
            )
        })
        .collect();

    assert!(
        unexpected_chunk_failures.is_empty(),
        "Unexpected chunking failures (grammar loaded but no chunks): {:?}",
        unexpected_chunk_failures
    );
}
