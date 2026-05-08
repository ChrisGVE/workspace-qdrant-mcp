//! TF-IDF lexical candidate extraction with n-gram filtering.
//!
//! Generates keyword candidates from document text using:
//! 1. N-gram extraction (1-3 words)
//! 2. Sublinear TF with L2 normalization
//! 3. Strong pattern-based filters (IDs, hashes, versions, paths)
//! 4. Code and prose boilerplate stoplists

mod filters;

use std::collections::HashMap;

use filters::{is_stopword, JunkPatterns};

/// A keyword candidate with its score components.
#[derive(Debug, Clone)]
pub struct LexicalCandidate {
    /// The candidate phrase (lowercased, trimmed)
    pub phrase: String,
    /// Raw term frequency in the document
    pub raw_tf: u32,
    /// Sublinear TF score: 1 + log(tf) if tf > 0, else 0
    pub tf_score: f64,
    /// Number of words in the phrase (1, 2, or 3)
    pub ngram_size: u8,
}

/// Configuration for lexical candidate extraction.
#[derive(Debug, Clone)]
pub struct LexicalConfig {
    /// Maximum number of candidates to return per document
    pub max_candidates: usize,
    /// Minimum term frequency to consider
    pub min_tf: u32,
    /// Maximum n-gram size (1, 2, or 3)
    pub max_ngram: u8,
    /// Whether this is code content (enables code stoplists)
    pub is_code: bool,
}

impl Default for LexicalConfig {
    fn default() -> Self {
        Self {
            max_candidates: 200,
            min_tf: 1,
            max_ngram: 3,
            is_code: false,
        }
    }
}

/// Tokenize text into words suitable for n-gram construction.
///
/// Splits on whitespace and common punctuation, lowercases,
/// and filters single-char tokens and common stopwords.
fn tokenize_for_ngrams(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || "(){}[]<>;:,\"'`~!@#$%^&*+=|".contains(c))
        .map(|s| s.trim_matches(|c: char| c == '.' || c == '-' || c == '_' || c == '/'))
        .filter(|s| s.len() > 1)
        .map(|s| s.to_lowercase())
        .collect()
}

/// Extract n-grams (1 to max_n words) from a token sequence.
fn extract_ngrams(tokens: &[String], max_n: u8) -> Vec<(String, u8)> {
    let mut ngrams = Vec::new();
    let max_n = max_n as usize;

    for n in 1..=max_n.min(tokens.len()) {
        for window in tokens.windows(n) {
            let phrase = window.join(" ");
            if phrase.len() > 1 {
                ngrams.push((phrase, n as u8));
            }
        }
    }

    ngrams
}

/// Extract TF-IDF lexical candidates from document text.
///
/// Returns candidates sorted by TF score descending, limited to `config.max_candidates`.
pub fn extract_candidates(text: &str, config: &LexicalConfig) -> Vec<LexicalCandidate> {
    if text.is_empty() {
        return Vec::new();
    }

    let junk = JunkPatterns::new();

    let tokens = tokenize_for_ngrams(text);
    if tokens.is_empty() {
        return Vec::new();
    }

    let ngrams = extract_ngrams(&tokens, config.max_ngram);
    let mut freq_map: HashMap<String, (u32, u8)> = HashMap::new();
    for (phrase, n) in ngrams {
        let entry = freq_map.entry(phrase).or_insert((0, n));
        entry.0 += 1;
    }

    let mut candidates: Vec<LexicalCandidate> = freq_map
        .into_iter()
        .filter(|(phrase, (tf, _))| {
            if *tf < config.min_tf {
                return false;
            }
            if !phrase.contains(' ') && is_stopword(phrase, config.is_code) {
                return false;
            }
            if junk.is_junk(phrase) {
                return false;
            }
            if phrase.contains(' ') {
                let words: Vec<&str> = phrase.split(' ').collect();
                if words.iter().all(|w| is_stopword(w, config.is_code)) {
                    return false;
                }
            }
            true
        })
        .map(|(phrase, (tf, n))| {
            let tf_score = if tf > 0 { 1.0 + (tf as f64).ln() } else { 0.0 };
            LexicalCandidate {
                phrase,
                raw_tf: tf,
                tf_score,
                ngram_size: n,
            }
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.tf_score
            .partial_cmp(&a.tf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.ngram_size.cmp(&a.ngram_size))
    });

    candidates.truncate(config.max_candidates);

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_candidates_basic() {
        let text = "The quick brown fox jumps over the lazy dog. \
                     The quick brown fox is very fast. \
                     The quick brown fox loves to jump.";
        let config = LexicalConfig {
            max_candidates: 50,
            min_tf: 1,
            max_ngram: 2,
            is_code: false,
        };
        let candidates = extract_candidates(text, &config);
        assert!(!candidates.is_empty());

        let quick = candidates.iter().find(|c| c.phrase == "quick");
        assert!(quick.is_some(), "Expected 'quick' as a candidate");
        assert_eq!(quick.unwrap().raw_tf, 3);
    }

    #[test]
    fn test_extract_candidates_code() {
        let text = r#"
            pub fn process_file(path: &str) -> Result<DocumentResult> {
                let embedding = generate_embedding(path);
                let chunks = chunk_document(path);
                for chunk in chunks {
                    let vector = generate_embedding(&chunk.content);
                    storage.insert_point(vector, chunk);
                }
            }
            pub fn generate_embedding(text: &str) -> Vec<f32> {
                model.encode(text)
            }
        "#;
        let config = LexicalConfig {
            max_candidates: 50,
            min_tf: 1,
            max_ngram: 2,
            is_code: true,
        };
        let candidates = extract_candidates(text, &config);

        let has_pub = candidates.iter().any(|c| c.phrase == "pub");
        assert!(!has_pub, "'pub' should be filtered by code stoplist");

        let has_fn = candidates.iter().any(|c| c.phrase == "fn");
        assert!(!has_fn, "'fn' should be filtered by code stoplist");

        let has_embedding = candidates.iter().any(|c| c.phrase.contains("embedding"));
        assert!(has_embedding, "Expected 'embedding' related candidate");
    }

    #[test]
    fn test_extract_candidates_empty_input() {
        let candidates = extract_candidates("", &LexicalConfig::default());
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_extract_candidates_filters_hashes() {
        let text = "commit abc123def456 was merged into main branch abc123def456";
        let config = LexicalConfig::default();
        let candidates = extract_candidates(text, &config);

        let has_hash = candidates.iter().any(|c| c.phrase == "abc123def456");
        assert!(!has_hash, "Hex hashes should be filtered");
    }

    #[test]
    fn test_extract_candidates_filters_versions() {
        let text = "upgrade to v2.3.1 from version v1.0.0 for better performance";
        let config = LexicalConfig::default();
        let candidates = extract_candidates(text, &config);

        let has_version = candidates
            .iter()
            .any(|c| c.phrase.starts_with("v2.3") || c.phrase.starts_with("v1.0"));
        assert!(!has_version, "Version strings should be filtered");
    }

    #[test]
    fn test_extract_candidates_max_limit() {
        let text: String = (0..1000)
            .map(|i| format!("unique_word_{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let config = LexicalConfig {
            max_candidates: 10,
            ..LexicalConfig::default()
        };
        let candidates = extract_candidates(&text, &config);
        assert!(
            candidates.len() <= 10,
            "Should respect max_candidates limit"
        );
    }

    #[test]
    fn test_extract_candidates_tf_scoring() {
        let text = "vector vector vector database database search";
        let config = LexicalConfig {
            max_candidates: 10,
            min_tf: 1,
            max_ngram: 1,
            is_code: false,
        };
        let candidates = extract_candidates(text, &config);

        let vector = candidates.iter().find(|c| c.phrase == "vector").unwrap();
        let database = candidates.iter().find(|c| c.phrase == "database").unwrap();
        let search = candidates.iter().find(|c| c.phrase == "search").unwrap();

        assert!(
            vector.tf_score > database.tf_score,
            "vector (tf=3) should score > database (tf=2)"
        );
        assert!(
            database.tf_score > search.tf_score,
            "database (tf=2) should score > search (tf=1)"
        );
    }

    #[test]
    fn test_extract_candidates_bigrams() {
        let text = "vector search is great. vector search outperforms keyword matching. \
                     vector search uses embeddings.";
        let config = LexicalConfig {
            max_candidates: 50,
            min_tf: 1,
            max_ngram: 2,
            is_code: false,
        };
        let candidates = extract_candidates(text, &config);

        let bigram = candidates.iter().find(|c| c.phrase == "vector search");
        assert!(bigram.is_some(), "Expected 'vector search' bigram");
        assert_eq!(bigram.unwrap().ngram_size, 2);
        assert_eq!(bigram.unwrap().raw_tf, 3);
    }

    #[test]
    fn test_extract_candidates_filters_all_stopword_bigrams() {
        let text = "this is that was this was";
        let config = LexicalConfig {
            max_candidates: 50,
            min_tf: 1,
            max_ngram: 2,
            is_code: false,
        };
        let candidates = extract_candidates(text, &config);

        assert!(
            candidates.is_empty(),
            "All-stopword phrases should be filtered"
        );
    }

    #[test]
    fn test_tokenize_for_ngrams_basic() {
        let tokens = tokenize_for_ngrams("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_extract_ngrams_unigrams() {
        let tokens: Vec<String> = vec!["hello".into(), "world".into()];
        let ngrams = extract_ngrams(&tokens, 1);
        assert_eq!(ngrams.len(), 2);
        assert!(ngrams.iter().any(|(p, n)| p == "hello" && *n == 1));
        assert!(ngrams.iter().any(|(p, n)| p == "world" && *n == 1));
    }

    #[test]
    fn test_extract_ngrams_bigrams() {
        let tokens: Vec<String> = vec!["hello".into(), "world".into(), "foo".into()];
        let ngrams = extract_ngrams(&tokens, 2);
        // unigrams: hello, world, foo
        // bigrams: "hello world", "world foo"
        assert_eq!(ngrams.len(), 5);
        assert!(ngrams.iter().any(|(p, n)| p == "hello world" && *n == 2));
        assert!(ngrams.iter().any(|(p, n)| p == "world foo" && *n == 2));
    }
}
