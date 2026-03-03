//! Keyword basket assignment.
//!
//! Assigns each keyword to the nearest tag based on embedding cosine
//! similarity. Keywords too distant from all tags go into a misc basket.

use super::keyword_selector::SelectedKeyword;
use super::semantic_rerank::cosine_similarity;
use super::tag_selector::SelectedTag;

/// A keyword assigned to a basket with similarity metadata.
#[derive(Debug, Clone)]
pub struct AssignedKeyword {
    /// The keyword phrase
    pub phrase: String,
    /// Keyword score from selection
    pub score: f64,
    /// Cosine similarity to the assigned tag
    pub similarity_to_tag: f64,
}

/// A basket of keywords grouped under a tag.
#[derive(Debug, Clone)]
pub struct KeywordBasket {
    /// The tag this basket belongs to (None for misc basket)
    pub tag: Option<String>,
    /// Tag index in the original tags list (None for misc)
    pub tag_index: Option<usize>,
    /// Keywords assigned to this basket
    pub keywords: Vec<AssignedKeyword>,
}

/// Configuration for basket assignment.
#[derive(Debug, Clone)]
pub struct BasketConfig {
    /// Minimum similarity to assign a keyword to a tag.
    /// Keywords below this threshold go to the misc basket.
    pub min_similarity: f64,
}

impl Default for BasketConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.40,
        }
    }
}

/// Return the index and similarity of the nearest tag to a keyword vector.
fn find_nearest_tag(kw_vec: &[f32], tag_vectors: &[Vec<f32>]) -> (usize, f64) {
    let mut best_tag_idx = 0;
    let mut best_sim = f64::NEG_INFINITY;

    for (ti, tv) in tag_vectors.iter().enumerate() {
        let sim = cosine_similarity(kw_vec, tv);
        if sim > best_sim {
            best_sim = sim;
            best_tag_idx = ti;
        }
    }

    (best_tag_idx, best_sim)
}

/// Assign keywords to tag baskets based on embedding similarity.
///
/// # Arguments
/// * `keywords` - Selected keywords with scores
/// * `keyword_vectors` - Embeddings for each keyword (parallel with keywords)
/// * `tags` - Selected tags
/// * `tag_vectors` - Embeddings for each tag (parallel with tags)
/// * `config` - Basket configuration
///
/// Returns one basket per tag (possibly empty) plus a misc basket for orphans.
pub fn assign_baskets(
    keywords: &[SelectedKeyword],
    keyword_vectors: &[Vec<f32>],
    tags: &[SelectedTag],
    tag_vectors: &[Vec<f32>],
    config: &BasketConfig,
) -> Vec<KeywordBasket> {
    assert_eq!(
        keywords.len(),
        keyword_vectors.len(),
        "keywords and keyword_vectors must have same length"
    );
    assert_eq!(
        tags.len(),
        tag_vectors.len(),
        "tags and tag_vectors must have same length"
    );

    // Initialize one basket per tag
    let mut baskets: Vec<KeywordBasket> = tags
        .iter()
        .enumerate()
        .map(|(i, t)| KeywordBasket {
            tag: Some(t.phrase.clone()),
            tag_index: Some(i),
            keywords: Vec::new(),
        })
        .collect();

    // Misc basket for orphans
    let mut misc = KeywordBasket {
        tag: None,
        tag_index: None,
        keywords: Vec::new(),
    };

    for (ki, kw) in keywords.iter().enumerate() {
        if tags.is_empty() {
            misc.keywords.push(AssignedKeyword {
                phrase: kw.phrase.clone(),
                score: kw.score,
                similarity_to_tag: 0.0,
            });
            continue;
        }

        let (best_tag_idx, best_sim) =
            find_nearest_tag(&keyword_vectors[ki], tag_vectors);

        let assigned = AssignedKeyword {
            phrase: kw.phrase.clone(),
            score: kw.score,
            similarity_to_tag: best_sim,
        };

        if best_sim >= config.min_similarity {
            baskets[best_tag_idx].keywords.push(assigned);
        } else {
            misc.keywords.push(assigned);
        }
    }

    // Sort keywords within each basket by score descending
    for basket in &mut baskets {
        basket
            .keywords
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }
    misc.keywords
        .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Only include misc basket if it has keywords
    if !misc.keywords.is_empty() {
        baskets.push(misc);
    }

    baskets
}

/// Convert baskets to a JSON-friendly map: {tag_name: [keyword1, keyword2, ...]}.
///
/// Misc basket appears under the key "__misc__".
pub fn baskets_to_map(baskets: &[KeywordBasket]) -> std::collections::HashMap<String, Vec<String>> {
    let mut map = std::collections::HashMap::new();
    for basket in baskets {
        let key = basket
            .tag
            .clone()
            .unwrap_or_else(|| "__misc__".to_string());
        let kws: Vec<String> = basket.keywords.iter().map(|k| k.phrase.clone()).collect();
        if !kws.is_empty() {
            map.insert(key, kws);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyword_extraction::keyword_selector::SelectedKeyword;
    use crate::keyword_extraction::tag_selector::{SelectedTag, TagType};

    fn make_keyword(phrase: &str, score: f64) -> SelectedKeyword {
        SelectedKeyword {
            phrase: phrase.to_string(),
            score,
            semantic_score: 0.5,
            lexical_score: 1.0,
            stability_count: 2,
            ngram_size: phrase.split(' ').count() as u8,
        }
    }

    fn make_tag(phrase: &str) -> SelectedTag {
        SelectedTag {
            phrase: phrase.to_string(),
            tag_type: TagType::Concept,
            score: 0.8,
            diversity_score: 0.9,
            semantic_score: 0.7,
            ngram_size: phrase.split(' ').count() as u8,
        }
    }

    #[test]
    fn test_assign_baskets_basic() {
        let keywords = vec![
            make_keyword("vector search", 0.9),
            make_keyword("embedding model", 0.8),
            make_keyword("grpc protocol", 0.7),
        ];
        let tags = vec![make_tag("vector indexing"), make_tag("networking")];

        // keyword 0 similar to tag 0, keyword 2 similar to tag 1
        let kw_vecs = vec![
            vec![0.9, 0.1, 0.0], // close to tag 0
            vec![0.8, 0.2, 0.0], // close to tag 0
            vec![0.0, 0.1, 0.9], // close to tag 1
        ];
        let tag_vecs = vec![
            vec![1.0, 0.0, 0.0], // tag 0: vector indexing
            vec![0.0, 0.0, 1.0], // tag 1: networking
        ];

        let config = BasketConfig::default();
        let baskets = assign_baskets(&keywords, &kw_vecs, &tags, &tag_vecs, &config);

        // Should have 2 tag baskets (no misc since all have good similarity)
        assert!(baskets.len() >= 2);
        assert_eq!(baskets[0].tag.as_deref(), Some("vector indexing"));
        assert_eq!(baskets[1].tag.as_deref(), Some("networking"));

        // "vector search" and "embedding model" should be in basket 0
        let b0_phrases: Vec<&str> = baskets[0].keywords.iter().map(|k| k.phrase.as_str()).collect();
        assert!(b0_phrases.contains(&"vector search"));
        assert!(b0_phrases.contains(&"embedding model"));

        // "grpc protocol" should be in basket 1
        let b1_phrases: Vec<&str> = baskets[1].keywords.iter().map(|k| k.phrase.as_str()).collect();
        assert!(b1_phrases.contains(&"grpc protocol"));
    }

    #[test]
    fn test_assign_baskets_misc() {
        let keywords = vec![make_keyword("random noise", 0.5)];
        let tags = vec![make_tag("vector indexing")];

        // Keyword orthogonal to tag → goes to misc
        let kw_vecs = vec![vec![0.0, 1.0, 0.0]];
        let tag_vecs = vec![vec![1.0, 0.0, 0.0]];

        let config = BasketConfig {
            min_similarity: 0.40,
        };
        let baskets = assign_baskets(&keywords, &kw_vecs, &tags, &tag_vecs, &config);

        // Should have tag basket (empty) + misc basket
        let misc = baskets.iter().find(|b| b.tag.is_none());
        assert!(misc.is_some(), "Should have misc basket");
        assert_eq!(misc.unwrap().keywords.len(), 1);
        assert_eq!(misc.unwrap().keywords[0].phrase, "random noise");
    }

    #[test]
    fn test_assign_baskets_no_tags() {
        let keywords = vec![
            make_keyword("keyword_a", 0.9),
            make_keyword("keyword_b", 0.8),
        ];
        let kw_vecs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let tags: Vec<SelectedTag> = vec![];
        let tag_vecs: Vec<Vec<f32>> = vec![];

        let config = BasketConfig::default();
        let baskets = assign_baskets(&keywords, &kw_vecs, &tags, &tag_vecs, &config);

        // All keywords go to misc
        assert_eq!(baskets.len(), 1);
        assert!(baskets[0].tag.is_none());
        assert_eq!(baskets[0].keywords.len(), 2);
    }

    #[test]
    fn test_assign_baskets_no_keywords() {
        let keywords: Vec<SelectedKeyword> = vec![];
        let kw_vecs: Vec<Vec<f32>> = vec![];
        let tags = vec![make_tag("tag_a")];
        let tag_vecs = vec![vec![1.0, 0.0]];

        let config = BasketConfig::default();
        let baskets = assign_baskets(&keywords, &kw_vecs, &tags, &tag_vecs, &config);

        // One empty tag basket, no misc
        assert_eq!(baskets.len(), 1);
        assert_eq!(baskets[0].tag.as_deref(), Some("tag_a"));
        assert!(baskets[0].keywords.is_empty());
    }

    #[test]
    fn test_assign_baskets_sorted_by_score() {
        let keywords = vec![
            make_keyword("low_score", 0.3),
            make_keyword("high_score", 0.9),
            make_keyword("mid_score", 0.6),
        ];
        let tags = vec![make_tag("tag_a")];

        // All keywords similar to tag
        let kw_vecs = vec![
            vec![0.9, 0.1],
            vec![0.8, 0.2],
            vec![0.85, 0.15],
        ];
        let tag_vecs = vec![vec![1.0, 0.0]];

        let config = BasketConfig::default();
        let baskets = assign_baskets(&keywords, &kw_vecs, &tags, &tag_vecs, &config);

        let scores: Vec<f64> = baskets[0].keywords.iter().map(|k| k.score).collect();
        assert!(
            scores.windows(2).all(|w| w[0] >= w[1]),
            "Keywords should be sorted by score descending: {:?}",
            scores
        );
    }

    #[test]
    fn test_baskets_to_map() {
        let baskets = vec![
            KeywordBasket {
                tag: Some("vector search".to_string()),
                tag_index: Some(0),
                keywords: vec![
                    AssignedKeyword {
                        phrase: "embedding".to_string(),
                        score: 0.9,
                        similarity_to_tag: 0.8,
                    },
                    AssignedKeyword {
                        phrase: "similarity".to_string(),
                        score: 0.7,
                        similarity_to_tag: 0.6,
                    },
                ],
            },
            KeywordBasket {
                tag: None,
                tag_index: None,
                keywords: vec![AssignedKeyword {
                    phrase: "orphan".to_string(),
                    score: 0.3,
                    similarity_to_tag: 0.1,
                }],
            },
        ];

        let map = baskets_to_map(&baskets);
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get("vector search").unwrap(),
            &vec!["embedding".to_string(), "similarity".to_string()]
        );
        assert_eq!(
            map.get("__misc__").unwrap(),
            &vec!["orphan".to_string()]
        );
    }

    #[test]
    fn test_baskets_to_map_empty_baskets_excluded() {
        let baskets = vec![
            KeywordBasket {
                tag: Some("empty_tag".to_string()),
                tag_index: Some(0),
                keywords: vec![],
            },
            KeywordBasket {
                tag: Some("has_keywords".to_string()),
                tag_index: Some(1),
                keywords: vec![AssignedKeyword {
                    phrase: "word".to_string(),
                    score: 0.5,
                    similarity_to_tag: 0.6,
                }],
            },
        ];

        let map = baskets_to_map(&baskets);
        assert_eq!(map.len(), 1, "Empty baskets should be excluded from map");
        assert!(map.contains_key("has_keywords"));
    }
}
