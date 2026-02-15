//! Model-aware tokenizer for accurate token counting and token-based chunking.
//!
//! Uses the HuggingFace `tokenizers` crate with the same tokenizer as the
//! embedding model (all-MiniLM-L6-v2, WordPiece/BERT). This ensures token
//! counts match what the model actually processes.

use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// Errors from tokenizer operations
#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Failed to load tokenizer: {0}")]
    LoadError(String),

    #[error("Tokenization failed: {0}")]
    EncodeError(String),
}

/// Model-aware tokenizer for token counting and chunking.
///
/// Wraps the HuggingFace `tokenizers` crate to provide accurate token
/// counts matching the embedding model's tokenizer (all-MiniLM-L6-v2).
#[derive(Clone)]
pub struct ModelTokenizer {
    inner: Arc<Tokenizer>,
}

impl std::fmt::Debug for ModelTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelTokenizer").finish()
    }
}

impl ModelTokenizer {
    /// Load the tokenizer from a tokenizer.json file.
    ///
    /// The tokenizer.json is typically found in the HuggingFace model cache
    /// directory for the embedding model.
    pub fn from_file(path: &std::path::Path) -> Result<Self, TokenizerError> {
        let mut tokenizer = Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::LoadError(format!("{}: {}", path.display(), e)))?;
        // Disable padding and truncation for accurate token counting.
        // The model's tokenizer.json may have these pre-configured for
        // inference, but we need raw token counts.
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None)
            .map_err(|e| TokenizerError::LoadError(format!("Failed to disable truncation: {}", e)))?;
        info!("Loaded tokenizer from {}", path.display());
        Ok(Self {
            inner: Arc::new(tokenizer),
        })
    }

    /// Load the tokenizer from the HuggingFace cache for all-MiniLM-L6-v2.
    ///
    /// Searches common cache locations for the model's tokenizer.json file.
    pub fn from_model_cache(cache_dir: Option<&PathBuf>) -> Result<Self, TokenizerError> {
        let candidates = Self::tokenizer_candidates(cache_dir);
        for path in &candidates {
            if path.exists() {
                return Self::from_file(path);
            }
        }

        Err(TokenizerError::LoadError(format!(
            "tokenizer.json not found in any candidate path. Searched: {:?}",
            candidates
        )))
    }

    /// Count the number of tokens in the given text.
    pub fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| TokenizerError::EncodeError(e.to_string()))?;
        Ok(encoding.get_ids().len())
    }

    /// Split text into chunks of approximately `target_tokens` tokens each,
    /// with `overlap_tokens` tokens of overlap between adjacent chunks.
    ///
    /// Chunks are aligned to paragraph boundaries (double newline) where
    /// possible. If a paragraph exceeds the target, it is split at word
    /// boundaries.
    pub fn chunk_by_tokens(
        &self,
        text: &str,
        target_tokens: usize,
        overlap_tokens: usize,
    ) -> Result<Vec<TokenChunk>, TokenizerError> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Split into paragraphs first
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut chunks: Vec<TokenChunk> = Vec::new();
        let mut current_text = String::new();
        let mut current_tokens = 0usize;
        let mut char_offset = 0usize;
        let mut chunk_start_char = 0usize;

        for (i, paragraph) in paragraphs.iter().enumerate() {
            let para_tokens = self.count_tokens(paragraph)?;

            // If this single paragraph exceeds target, split it by words
            if para_tokens > target_tokens && current_text.is_empty() {
                let word_chunks = self.split_paragraph_by_tokens(
                    paragraph, target_tokens, overlap_tokens, char_offset,
                )?;
                chunks.extend(word_chunks);
                char_offset += paragraph.len();
                if i < paragraphs.len() - 1 {
                    char_offset += 2; // "\n\n" separator
                }
                chunk_start_char = char_offset;
                continue;
            }

            // Check if adding this paragraph would exceed the target
            let separator = if current_text.is_empty() { "" } else { "\n\n" };
            let combined_tokens = if current_text.is_empty() {
                para_tokens
            } else {
                self.count_tokens(&format!("{}{}{}", current_text, separator, paragraph))?
            };

            if combined_tokens > target_tokens && !current_text.is_empty() {
                // Flush current chunk
                let token_count = current_tokens;
                chunks.push(TokenChunk {
                    text: std::mem::take(&mut current_text),
                    token_count,
                    char_start: chunk_start_char,
                    char_end: char_offset.saturating_sub(2), // before the separator
                });

                // Apply overlap: re-include the end of the previous chunk
                if overlap_tokens > 0 {
                    if let Some(overlap_text) = self.extract_overlap_suffix(
                        chunks.last().map(|c| c.text.as_str()).unwrap_or(""),
                        overlap_tokens,
                    )? {
                        current_text = overlap_text;
                        chunk_start_char = char_offset.saturating_sub(current_text.len());
                    } else {
                        chunk_start_char = char_offset;
                    }
                } else {
                    chunk_start_char = char_offset;
                }

                // Add current paragraph
                if !current_text.is_empty() {
                    current_text.push_str("\n\n");
                }
                current_text.push_str(paragraph);
                current_tokens = self.count_tokens(&current_text)?;
            } else {
                if !current_text.is_empty() {
                    current_text.push_str("\n\n");
                }
                current_text.push_str(paragraph);
                current_tokens = combined_tokens;
            }

            char_offset += paragraph.len();
            if i < paragraphs.len() - 1 {
                char_offset += 2; // "\n\n" separator
            }
        }

        // Flush remaining text
        if !current_text.is_empty() {
            let token_count = self.count_tokens(&current_text)?;
            chunks.push(TokenChunk {
                text: current_text,
                token_count,
                char_start: chunk_start_char,
                char_end: char_offset,
            });
        }

        debug!("Split text into {} token-based chunks", chunks.len());
        Ok(chunks)
    }

    /// Split a single paragraph into chunks by word boundaries
    fn split_paragraph_by_tokens(
        &self,
        text: &str,
        target_tokens: usize,
        overlap_tokens: usize,
        base_char_offset: usize,
    ) -> Result<Vec<TokenChunk>, TokenizerError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut current_words: Vec<&str> = Vec::new();
        let mut word_char_offset = 0usize;
        let mut chunk_start = 0usize;

        for word in &words {
            // Skip leading whitespace to find word position
            while word_char_offset < text.len()
                && text[word_char_offset..].starts_with(char::is_whitespace)
            {
                word_char_offset += text[word_char_offset..]
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(1);
            }

            current_words.push(word);
            let combined = current_words.join(" ");
            let combined_tokens = self.count_tokens(&combined)?;

            if combined_tokens > target_tokens && current_words.len() > 1 {
                // Pop the last word and flush
                current_words.pop();
                let chunk_text = current_words.join(" ");
                let token_count = self.count_tokens(&chunk_text)?;
                let chunk_end = word_char_offset;

                chunks.push(TokenChunk {
                    text: chunk_text,
                    token_count,
                    char_start: base_char_offset + chunk_start,
                    char_end: base_char_offset + chunk_end,
                });

                // Start new chunk with overlap
                if overlap_tokens > 0 {
                    let overlap_start = current_words.len().saturating_sub(
                        (overlap_tokens * current_words.len()) / token_count.max(1)
                    );
                    current_words = current_words[overlap_start..].to_vec();
                } else {
                    current_words.clear();
                }
                current_words.push(word);
                chunk_start = word_char_offset;
            }

            word_char_offset += word.len();
        }

        // Flush remaining
        if !current_words.is_empty() {
            let chunk_text = current_words.join(" ");
            let token_count = self.count_tokens(&chunk_text)?;
            chunks.push(TokenChunk {
                text: chunk_text,
                token_count,
                char_start: base_char_offset + chunk_start,
                char_end: base_char_offset + text.len(),
            });
        }

        Ok(chunks)
    }

    /// Extract the last `target_tokens` tokens worth of text from a string
    fn extract_overlap_suffix(
        &self,
        text: &str,
        target_tokens: usize,
    ) -> Result<Option<String>, TokenizerError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(None);
        }

        // Binary search for the right number of trailing words
        let mut lo = 1usize;
        let mut hi = words.len();
        let mut best = 0usize;

        while lo <= hi {
            let mid = (lo + hi) / 2;
            let suffix = words[words.len() - mid..].join(" ");
            let tokens = self.count_tokens(&suffix)?;
            if tokens <= target_tokens {
                best = mid;
                lo = mid + 1;
            } else {
                if mid == 0 { break; }
                hi = mid - 1;
            }
        }

        if best == 0 {
            Ok(None)
        } else {
            Ok(Some(words[words.len() - best..].join(" ")))
        }
    }

    /// List candidate paths for the tokenizer.json file
    fn tokenizer_candidates(cache_dir: Option<&PathBuf>) -> Vec<PathBuf> {
        let model_subpath = "models--sentence-transformers--all-MiniLM-L6-v2";
        let mut candidates = Vec::new();

        // If user provided a cache dir, check it first
        if let Some(dir) = cache_dir {
            // Direct path (fastembed stores differently from HF hub)
            candidates.push(dir.join("sentence-transformers--all-MiniLM-L6-v2/tokenizer.json"));
            candidates.push(dir.join("fast-all-MiniLM-L6-v2/tokenizer.json"));
        }

        // Standard HuggingFace cache: find the latest snapshot
        let hf_cache = dirs::home_dir()
            .map(|h| h.join(".cache/huggingface/hub"))
            .unwrap_or_else(|| PathBuf::from("/tmp"));

        let snapshots_dir = hf_cache.join(model_subpath).join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
            for entry in entries.flatten() {
                let tokenizer_path = entry.path().join("tokenizer.json");
                if tokenizer_path.exists() {
                    candidates.push(tokenizer_path);
                }
            }
        }

        // Fallback: fastembed's own cache locations
        if let Some(home) = dirs::home_dir() {
            candidates.push(
                home.join(".cache/fastembed/models/fast-all-MiniLM-L6-v2/tokenizer.json")
            );
        }

        candidates
    }
}

/// A chunk of text with token count and character offsets
#[derive(Debug, Clone)]
pub struct TokenChunk {
    /// The chunk text
    pub text: String,
    /// Number of tokens in this chunk
    pub token_count: usize,
    /// Character offset of chunk start in the source text
    pub char_start: usize,
    /// Character offset of chunk end in the source text
    pub char_end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_tokenizer() -> Option<ModelTokenizer> {
        ModelTokenizer::from_model_cache(None).ok()
    }

    #[test]
    fn test_tokenizer_loads_from_cache() {
        let tokenizer = get_test_tokenizer();
        assert!(tokenizer.is_some(), "Tokenizer should load from HF cache (requires model to be cached)");
    }

    #[test]
    fn test_count_tokens_basic() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return, // Skip if model not cached
        };

        let count = tokenizer.count_tokens("Hello world").unwrap();
        assert!(count > 0);
        assert!(count <= 5); // "Hello" and "world" should be 2-3 tokens
    }

    #[test]
    fn test_count_tokens_empty() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let count = tokenizer.count_tokens("").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_tokens_subwords() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        // WordPiece should split unknown words into subwords
        let count = tokenizer.count_tokens("uncharacteristically").unwrap();
        assert!(count > 1, "Long word should be split into multiple subword tokens");
    }

    #[test]
    fn test_chunk_by_tokens_empty() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let chunks = tokenizer.chunk_by_tokens("", 100, 10).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_by_tokens_short_text() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let text = "This is a short text.";
        let chunks = tokenizer.chunk_by_tokens(text, 100, 10).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
        assert!(chunks[0].token_count <= 100);
    }

    #[test]
    fn test_chunk_by_tokens_respects_target() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        // Generate text that should produce multiple chunks at 20-token target
        let sentences: Vec<String> = (0..50)
            .map(|i| format!("Sentence number {} has some words in it.", i))
            .collect();
        let text = sentences.join("\n\n");

        let chunks = tokenizer.chunk_by_tokens(&text, 20, 2).unwrap();
        assert!(chunks.len() > 1, "Should split into multiple chunks");

        for chunk in &chunks {
            // Allow some tolerance (paragraphs may push slightly over)
            assert!(
                chunk.token_count <= 30,
                "Chunk has {} tokens, expected <= 30 (target=20 with tolerance)",
                chunk.token_count
            );
        }
    }

    #[test]
    fn test_chunk_by_tokens_paragraph_alignment() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let chunks = tokenizer.chunk_by_tokens(text, 100, 0).unwrap();
        // All paragraphs fit in one chunk
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("First"));
        assert!(chunks[0].text.contains("Third"));
    }
}
