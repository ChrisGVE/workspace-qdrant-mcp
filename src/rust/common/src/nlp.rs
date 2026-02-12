//! NLP utilities shared between daemon services
//!
//! Provides tokenization and stopword filtering for BM25 sparse vector generation.

/// Common English stopwords filtered during tokenization
pub const ENGLISH_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how"
];

/// Simple tokenization for BM25 sparse vector generation
///
/// Splits on whitespace and punctuation, lowercases, removes stopwords
/// and single-character tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty() && s.len() > 1)
        .filter(|s| !ENGLISH_STOPWORDS.contains(s))
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello World, this is a test!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Stopwords and single-char tokens should be filtered
        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_tokenize_code() {
        let tokens = tokenize("fn process_file(path: &str) -> Result<()>");
        assert!(tokens.contains(&"fn".to_string()));
        assert!(tokens.contains(&"process_file".to_string()));
        assert!(tokens.contains(&"path".to_string()));
        assert!(tokens.contains(&"result".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_only_stopwords() {
        let tokens = tokenize("the and or but");
        assert!(tokens.is_empty());
    }
}
