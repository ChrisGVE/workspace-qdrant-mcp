//! SQLite persistence for keyword/tag extraction results.
//!
//! Writes extraction results to the `keywords`, `tags`, and `keyword_baskets`
//! tables after the extraction pipeline completes. This is non-fatal: failures
//! are logged but never block the ingestion pipeline.

use std::collections::HashMap;

use sqlx::SqlitePool;
use tracing::{debug, warn};

use crate::keyword_extraction::pipeline::ExtractionResult;

/// Persist extraction results to SQLite.
///
/// Writes keywords, tags (concept + structural), and keyword baskets for a
/// single document. Operates within a transaction for atomicity. Old records
/// for the same `doc_id` are deleted first (replace-on-re-index).
pub(super) async fn persist_extraction(
    pool: &SqlitePool,
    doc_id: &str,
    tenant_id: &str,
    collection: &str,
    extraction: &ExtractionResult,
) {
    if extraction.keywords.is_empty()
        && extraction.tags.is_empty()
        && extraction.structural_tags.is_empty()
    {
        debug!("No keywords/tags to persist for doc_id={}", doc_id);
        return;
    }

    if let Err(e) = persist_inner(pool, doc_id, tenant_id, collection, extraction).await {
        warn!(
            "Failed to persist keywords/tags for doc_id={}: {}",
            doc_id, e
        );
    }
}

async fn persist_inner(
    pool: &SqlitePool,
    doc_id: &str,
    tenant_id: &str,
    collection: &str,
    extraction: &ExtractionResult,
) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;

    delete_old_records(doc_id, &mut tx).await?;
    insert_keywords(doc_id, tenant_id, collection, &extraction.keywords, &mut tx).await?;
    let mut tag_ids = insert_tags(doc_id, tenant_id, collection, &extraction.tags, &mut tx).await?;
    let struct_tag_ids = insert_tags(
        doc_id,
        tenant_id,
        collection,
        &extraction.structural_tags,
        &mut tx,
    )
    .await?;
    tag_ids.extend(struct_tag_ids);
    insert_baskets(tenant_id, &extraction.baskets, &tag_ids, &mut tx).await?;

    tx.commit().await?;

    debug!(
        "Persisted {} keywords, {} tags, {} structural tags for doc_id={}",
        extraction.keywords.len(),
        extraction.tags.len(),
        extraction.structural_tags.len(),
        doc_id,
    );

    Ok(())
}

/// Delete all existing keyword/tag/basket records for a document within a transaction.
async fn delete_old_records(
    doc_id: &str,
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "DELETE FROM keyword_baskets WHERE tag_id IN (SELECT tag_id FROM tags WHERE doc_id = ?1)",
    )
    .bind(doc_id)
    .execute(&mut **tx)
    .await?;
    sqlx::query("DELETE FROM tags WHERE doc_id = ?1")
        .bind(doc_id)
        .execute(&mut **tx)
        .await?;
    sqlx::query("DELETE FROM keywords WHERE doc_id = ?1")
        .bind(doc_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

/// Insert keyword rows within a transaction.
async fn insert_keywords(
    doc_id: &str,
    tenant_id: &str,
    collection: &str,
    keywords: &[crate::keyword_extraction::keyword_selector::SelectedKeyword],
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
) -> Result<(), sqlx::Error> {
    for kw in keywords {
        sqlx::query(
            "INSERT INTO keywords \
             (doc_id, keyword, score, semantic_score, lexical_score, stability_count, collection, tenant_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )
        .bind(doc_id)
        .bind(&kw.phrase)
        .bind(kw.score)
        .bind(kw.semantic_score)
        .bind(kw.lexical_score)
        .bind(kw.stability_count as i32)
        .bind(collection)
        .bind(tenant_id)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Insert tag rows (concept or structural) within a transaction.
/// Returns a map of tag phrase → tag_id for use by basket insertion.
async fn insert_tags(
    doc_id: &str,
    tenant_id: &str,
    collection: &str,
    tags: &[crate::keyword_extraction::tag_selector::SelectedTag],
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
) -> Result<HashMap<String, i64>, sqlx::Error> {
    let mut tag_ids = HashMap::new();
    for tag in tags {
        sqlx::query(
            "INSERT INTO tags \
             (doc_id, tag, tag_type, score, diversity_score, collection, tenant_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )
        .bind(doc_id)
        .bind(&tag.phrase)
        .bind(tag.tag_type.as_str())
        .bind(tag.score)
        .bind(tag.diversity_score)
        .bind(collection)
        .bind(tenant_id)
        .execute(&mut **tx)
        .await?;

        let tag_id: i64 = sqlx::query_scalar("SELECT last_insert_rowid()")
            .fetch_one(&mut **tx)
            .await?;
        tag_ids.insert(tag.phrase.clone(), tag_id);
    }
    Ok(tag_ids)
}

/// Insert keyword basket rows and back-link basket_id into the parent tag row.
/// Uses the pre-computed tag_ids map instead of querying tags inside the transaction.
async fn insert_baskets(
    tenant_id: &str,
    baskets: &[crate::keyword_extraction::basket_assignment::KeywordBasket],
    tag_ids: &HashMap<String, i64>,
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
) -> Result<(), sqlx::Error> {
    for basket in baskets {
        let tag_name = match &basket.tag {
            Some(name) => name.as_str(),
            None => continue, // skip misc basket (no parent tag)
        };

        let tag_id = match tag_ids.get(tag_name) {
            Some(&id) => id,
            None => continue,
        };

        let kw_phrases: Vec<&str> = basket.keywords.iter().map(|k| k.phrase.as_str()).collect();
        let keywords_json = serde_json::to_string(&kw_phrases).unwrap_or_else(|_| "[]".to_string());

        sqlx::query(
            "INSERT INTO keyword_baskets (tag_id, keywords_json, tenant_id) VALUES (?1, ?2, ?3)",
        )
        .bind(tag_id)
        .bind(&keywords_json)
        .bind(tenant_id)
        .execute(&mut **tx)
        .await?;

        let basket_id: i64 = sqlx::query_scalar("SELECT last_insert_rowid()")
            .fetch_one(&mut **tx)
            .await?;
        sqlx::query("UPDATE tags SET basket_id = ?1 WHERE tag_id = ?2")
            .bind(basket_id)
            .bind(tag_id)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

/// Delete keyword/tag records for a document (used during file deletion).
pub(super) async fn delete_extraction(pool: &SqlitePool, doc_id: &str) {
    if let Err(e) = delete_inner(pool, doc_id).await {
        warn!(
            "Failed to delete keywords/tags for doc_id={}: {}",
            doc_id, e
        );
    }
}

async fn delete_inner(pool: &SqlitePool, doc_id: &str) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query(
        "DELETE FROM keyword_baskets WHERE tag_id IN (SELECT tag_id FROM tags WHERE doc_id = ?1)",
    )
    .bind(doc_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query("DELETE FROM tags WHERE doc_id = ?1")
        .bind(doc_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("DELETE FROM keywords WHERE doc_id = ?1")
        .bind(doc_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    debug!("Deleted keywords/tags for doc_id={}", doc_id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyword_extraction::basket_assignment::{AssignedKeyword, KeywordBasket};
    use crate::keyword_extraction::keyword_selector::SelectedKeyword;
    use crate::keyword_extraction::tag_selector::{SelectedTag, TagType};

    async fn create_test_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:")
            .await
            .expect("Failed to create test pool");
        sqlx::query(crate::keywords_schema::CREATE_KEYWORDS_SQL)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(crate::keywords_schema::CREATE_TAGS_SQL)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(crate::keywords_schema::CREATE_KEYWORD_BASKETS_SQL)
            .execute(&pool)
            .await
            .unwrap();
        for idx in crate::keywords_schema::CREATE_KEYWORDS_INDEXES_SQL {
            sqlx::query(idx).execute(&pool).await.unwrap();
        }
        for idx in crate::keywords_schema::CREATE_TAGS_INDEXES_SQL {
            sqlx::query(idx).execute(&pool).await.unwrap();
        }
        for idx in crate::keywords_schema::CREATE_KEYWORD_BASKETS_INDEXES_SQL {
            sqlx::query(idx).execute(&pool).await.unwrap();
        }
        pool
    }

    fn sample_extraction() -> ExtractionResult {
        ExtractionResult {
            summary_vector: None,
            gist_indices: vec![],
            keywords: vec![
                SelectedKeyword {
                    phrase: "async runtime".into(),
                    score: 0.85,
                    semantic_score: 0.9,
                    lexical_score: 0.8,
                    stability_count: 3,
                    ngram_size: 2,
                },
                SelectedKeyword {
                    phrase: "tokio".into(),
                    score: 0.72,
                    semantic_score: 0.75,
                    lexical_score: 0.7,
                    stability_count: 5,
                    ngram_size: 1,
                },
            ],
            tags: vec![SelectedTag {
                phrase: "concurrency".into(),
                tag_type: TagType::Concept,
                score: 0.88,
                diversity_score: 0.95,
                semantic_score: 0.9,
                ngram_size: 1,
            }],
            structural_tags: vec![SelectedTag {
                phrase: "lang:rust".into(),
                tag_type: TagType::Structural,
                score: 1.0,
                diversity_score: 1.0,
                semantic_score: 1.0,
                ngram_size: 1,
            }],
            baskets: vec![KeywordBasket {
                tag: Some("concurrency".into()),
                tag_index: Some(0),
                keywords: vec![
                    AssignedKeyword {
                        phrase: "async runtime".into(),
                        score: 0.85,
                        similarity_to_tag: 0.78,
                    },
                    AssignedKeyword {
                        phrase: "tokio".into(),
                        score: 0.72,
                        similarity_to_tag: 0.65,
                    },
                ],
            }],
        }
    }

    #[tokio::test]
    async fn test_persist_keywords_and_tags() {
        let pool = create_test_pool().await;
        let extraction = sample_extraction();

        persist_extraction(&pool, "doc-1", "tenant-1", "projects", &extraction).await;

        let kw_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM keywords WHERE doc_id = 'doc-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(kw_count, 2);

        let tag_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tags WHERE doc_id = 'doc-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(tag_count, 2); // 1 concept + 1 structural

        let basket_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM keyword_baskets")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(basket_count, 1);
    }

    #[tokio::test]
    async fn test_persist_replaces_on_reindex() {
        let pool = create_test_pool().await;
        let extraction = sample_extraction();

        // Persist twice — second call should replace, not duplicate
        persist_extraction(&pool, "doc-1", "tenant-1", "projects", &extraction).await;
        persist_extraction(&pool, "doc-1", "tenant-1", "projects", &extraction).await;

        let kw_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM keywords WHERE doc_id = 'doc-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(kw_count, 2); // not 4
    }

    #[tokio::test]
    async fn test_delete_extraction() {
        let pool = create_test_pool().await;
        let extraction = sample_extraction();

        persist_extraction(&pool, "doc-1", "tenant-1", "projects", &extraction).await;
        delete_extraction(&pool, "doc-1").await;

        let kw_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM keywords WHERE doc_id = 'doc-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(kw_count, 0);

        let tag_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tags WHERE doc_id = 'doc-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(tag_count, 0);

        let basket_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM keyword_baskets")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(basket_count, 0);
    }

    #[tokio::test]
    async fn test_persist_empty_extraction_is_noop() {
        let pool = create_test_pool().await;
        let extraction = ExtractionResult {
            summary_vector: None,
            gist_indices: vec![],
            keywords: vec![],
            tags: vec![],
            structural_tags: vec![],
            baskets: vec![],
        };

        persist_extraction(&pool, "doc-1", "tenant-1", "projects", &extraction).await;

        let kw_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM keywords")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(kw_count, 0);
    }

    #[tokio::test]
    async fn test_keyword_scores_persisted_correctly() {
        let pool = create_test_pool().await;
        let extraction = sample_extraction();

        persist_extraction(&pool, "doc-1", "tenant-1", "projects", &extraction).await;

        let (score, semantic, lexical, stability): (f64, f64, f64, i32) = sqlx::query_as(
            "SELECT score, semantic_score, lexical_score, stability_count FROM keywords WHERE keyword = 'async runtime'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert!((score - 0.85).abs() < 0.001);
        assert!((semantic - 0.9).abs() < 0.001);
        assert!((lexical - 0.8).abs() < 0.001);
        assert_eq!(stability, 3);
    }

    #[tokio::test]
    async fn test_basket_keywords_json() {
        let pool = create_test_pool().await;
        let extraction = sample_extraction();

        persist_extraction(&pool, "doc-1", "tenant-1", "projects", &extraction).await;

        let json: String = sqlx::query_scalar(
            "SELECT kb.keywords_json FROM keyword_baskets kb JOIN tags t ON kb.tag_id = t.tag_id WHERE t.tag = 'concurrency'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        let keywords: Vec<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(keywords.len(), 2);
        assert!(keywords.contains(&"async runtime".to_string()));
        assert!(keywords.contains(&"tokio".to_string()));
    }

    #[tokio::test]
    async fn test_tenant_collection_isolation() {
        let pool = create_test_pool().await;
        let extraction = sample_extraction();

        // Different tenants produce different doc_ids (via generate_document_id)
        persist_extraction(&pool, "doc-A", "tenant-A", "projects", &extraction).await;
        persist_extraction(&pool, "doc-B", "tenant-B", "projects", &extraction).await;

        let count_a: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM keywords WHERE tenant_id = 'tenant-A'")
                .fetch_one(&pool)
                .await
                .unwrap();
        let count_b: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM keywords WHERE tenant_id = 'tenant-B'")
                .fetch_one(&pool)
                .await
                .unwrap();

        assert_eq!(count_a, 2);
        assert_eq!(count_b, 2);
    }
}
