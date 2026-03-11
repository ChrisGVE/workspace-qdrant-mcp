//! BM25 sparse vector generation and tokenization for hybrid search.
//!
//! Provides token-level BM25 scoring with IDF weighting for sparse vector
//! generation, and a junk-filtering tokenizer tuned for source code text.

use once_cell::sync::Lazy;
use regex::Regex;

use super::types::SparseEmbedding;

/// Tokenize text for BM25 sparse vector generation.
///
/// Splits on whitespace and punctuation, trims trailing separators, lowercases,
/// and filters out junk tokens (hex hashes, version strings, paths, pure digits,
/// single characters). Matches the quality level of the keyword extraction pipeline.
pub fn tokenize_for_bm25(text: &str) -> Vec<String> {
    static JUNK: Lazy<BM25JunkFilter> = Lazy::new(BM25JunkFilter::new);

    text.split(|c: char| c.is_whitespace() || "(){}[]<>;:,.\"'`~!@#$%^&*+=|\\".contains(c))
        .map(|s| s.trim_matches(|c: char| c == '-' || c == '_' || c == '/'))
        .filter(|s| s.len() > 1)
        .map(|s| s.to_lowercase())
        .filter(|s| !JUNK.is_junk(s))
        .collect()
}

/// Junk pattern filter for BM25 tokens.
struct BM25JunkFilter {
    hex_hash: Regex,
    version: Regex,
    path: Regex,
    hex_literal: Regex,
    all_digits: Regex,
}

impl BM25JunkFilter {
    fn new() -> Self {
        Self {
            hex_hash: Regex::new(r"^[a-f0-9]{8,}$").unwrap(),
            version: Regex::new(r"^v?\d+\.\d+").unwrap(),
            path: Regex::new(r"[/\\]").unwrap(),
            hex_literal: Regex::new(r"^0x[a-f0-9]+$").unwrap(),
            all_digits: Regex::new(r"^\d+$").unwrap(),
        }
    }

    fn is_junk(&self, token: &str) -> bool {
        self.hex_hash.is_match(token)
            || self.version.is_match(token)
            || self.path.is_match(token)
            || self.hex_literal.is_match(token)
            || self.all_digits.is_match(token)
    }
}

/// BM25 for sparse vector generation with IDF weighting
///
/// Uses true BM25 scoring: `IDF * (k1 * tf) / (tf + k1)` where
/// `IDF = ln((N - df + 0.5) / (df + 0.5))`.
///
/// Document frequency statistics are tracked per-term to weight rare terms
/// higher than common terms like "function" or "return".
#[derive(Debug)]
pub struct BM25 {
    k1: f32,
    pub(super) vocab: std::collections::HashMap<String, u32>,
    pub(super) next_vocab_id: u32,
    /// Document frequency: term_id → number of documents containing the term
    pub(super) doc_freq: std::collections::HashMap<u32, u32>,
    /// Total number of documents added to the corpus
    total_docs: u32,
    /// Term IDs modified since the last persist (new terms or incremented doc_freq).
    /// Only these need to be written to SQLite on the next persist() call.
    pub(super) dirty_ids: std::collections::HashSet<u32>,
}

impl BM25 {
    pub fn new(k1: f32) -> Self {
        Self {
            k1,
            vocab: std::collections::HashMap::new(),
            next_vocab_id: 0,
            doc_freq: std::collections::HashMap::new(),
            total_docs: 0,
            dirty_ids: std::collections::HashSet::new(),
        }
    }

    /// Restore BM25 state from persisted data (e.g., SQLite on startup)
    pub fn from_persisted(
        k1: f32,
        vocab: std::collections::HashMap<String, u32>,
        doc_freq: std::collections::HashMap<u32, u32>,
        total_docs: u32,
    ) -> Self {
        let next_vocab_id = vocab.values().max().map(|v| v + 1).unwrap_or(0);
        Self {
            k1,
            vocab,
            next_vocab_id,
            doc_freq,
            total_docs,
            dirty_ids: std::collections::HashSet::new(),
        }
    }

    pub fn add_document(&mut self, tokens: &[String]) {
        self.total_docs += 1;

        // Track unique terms in this document for document frequency
        let mut seen_in_doc: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for token in tokens {
            let vocab_id = if let Some(&id) = self.vocab.get(token) {
                id
            } else {
                let id = self.next_vocab_id;
                self.vocab.insert(token.clone(), id);
                self.next_vocab_id += 1;
                id
            };
            seen_in_doc.insert(vocab_id);
        }

        // Increment document frequency for each unique term in this document
        // and mark them dirty so persist() only writes what changed.
        for term_id in seen_in_doc {
            *self.doc_freq.entry(term_id).or_insert(0) += 1;
            self.dirty_ids.insert(term_id);
        }
    }

    /// Drain the set of dirty term IDs and return their (term, term_id, doc_count) triples.
    ///
    /// Used by `LexiconManager::persist()` to write only the terms modified since
    /// the last persist, instead of the entire 3M+ vocabulary.
    pub(crate) fn take_dirty_entries(&mut self) -> Vec<(String, u32, u32)> {
        if self.dirty_ids.is_empty() {
            return Vec::new();
        }
        // Build a reverse map: term_id → term (only for dirty IDs)
        let id_to_term: std::collections::HashMap<u32, &str> = self
            .vocab
            .iter()
            .filter(|(_, &id)| self.dirty_ids.contains(&id))
            .map(|(term, &id)| (id, term.as_str()))
            .collect();

        let mut entries = Vec::with_capacity(self.dirty_ids.len());
        for &id in &self.dirty_ids {
            if let Some(term) = id_to_term.get(&id) {
                let df = self.doc_freq.get(&id).copied().unwrap_or(0);
                entries.push(((*term).to_string(), id, df));
            }
        }
        self.dirty_ids.clear();
        entries
    }

    pub fn generate_sparse_vector(&self, tokens: &[String]) -> SparseEmbedding {
        let mut term_freq: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        for token in tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
        }

        let mut indices = Vec::new();
        let mut values = Vec::new();
        let n = self.total_docs as f32;

        for (term, tf) in term_freq {
            if let Some(&vocab_id) = self.vocab.get(&term) {
                let tf = tf as f32;
                let df = self.doc_freq.get(&vocab_id).copied().unwrap_or(0) as f32;

                // IDF: ln((N - df + 0.5) / (df + 0.5)), floored at 0
                let idf = if n > 0.0 && df > 0.0 {
                    ((n - df + 0.5) / (df + 0.5)).ln().max(0.0)
                } else {
                    // No corpus stats yet — fall back to TF-only
                    1.0
                };

                // BM25 score: IDF * (k1 * tf) / (tf + k1)
                let score = idf * (self.k1 * tf) / (tf + self.k1);
                if score > 0.0 {
                    indices.push(vocab_id);
                    values.push(score);
                }
            }
        }

        SparseEmbedding {
            indices,
            values,
            vocab_size: self.next_vocab_id as usize,
        }
    }

    /// Evict all hapax legomena (terms with document_frequency == 1) from the
    /// in-memory vocabulary.
    ///
    /// Hapax terms have maximal IDF (`ln(N)`) but zero discriminative value
    /// — they appear in exactly one document, so every query containing them
    /// matches only that document. They dominate vocabulary size without
    /// contributing useful signal.
    ///
    /// Eviction is self-rebalancing: if an evicted term later appears in a
    /// second document it is re-added with df=2 and is no longer a hapax.
    ///
    /// Returns the number of evicted terms.
    pub fn evict_hapax(&mut self) -> usize {
        let hapax_ids: std::collections::HashSet<u32> = self
            .doc_freq
            .iter()
            .filter(|(_, &df)| df == 1)
            .map(|(id, _)| *id)
            .collect();

        if hapax_ids.is_empty() {
            return 0;
        }

        let count = hapax_ids.len();

        // Remove from doc_freq and dirty tracking
        for &id in &hapax_ids {
            self.doc_freq.remove(&id);
            self.dirty_ids.remove(&id);
        }

        // Remove from vocab (single O(vocab_size) scan using retain)
        self.vocab.retain(|_, id| !hapax_ids.contains(id));

        count
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the total number of documents in the corpus
    pub fn total_docs(&self) -> u32 {
        self.total_docs
    }

    /// Get the vocabulary for persistence
    pub fn vocab(&self) -> &std::collections::HashMap<String, u32> {
        &self.vocab
    }

    /// Get the document frequency map for persistence
    pub fn doc_freq(&self) -> &std::collections::HashMap<u32, u32> {
        &self.doc_freq
    }
}
