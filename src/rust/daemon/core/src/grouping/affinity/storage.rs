/// Embedding blob serialization and project embedding CRUD operations.
use sqlx::{Row, SqlitePool};
use tracing::{debug, warn};

// ---- Blob serialization ----------------------------------------------------

/// Serialize an f32 vector to a byte blob (little-endian).
pub(crate) fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Deserialize a byte blob back to an f32 vector.
pub(crate) fn blob_to_embedding(blob: &[u8], dim: usize) -> Option<Vec<f32>> {
    if blob.len() != dim * 4 {
        return None;
    }
    let mut vec = Vec::with_capacity(dim);
    for chunk in blob.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().ok()?;
        vec.push(f32::from_le_bytes(arr));
    }
    Some(vec)
}

// ---- Embedding CRUD --------------------------------------------------------

/// Store or update a project's aggregate embedding.
pub async fn store_project_embedding(
    pool: &SqlitePool,
    tenant_id: &str,
    embedding: &[f32],
    chunk_count: usize,
    label: Option<&str>,
) -> Result<(), sqlx::Error> {
    let now = wqm_common::timestamps::now_utc();
    let blob = embedding_to_blob(embedding);
    let dim = embedding.len() as i64;
    let chunk_count = chunk_count as i64;

    sqlx::query(
        r#"
        INSERT OR REPLACE INTO project_embeddings
            (tenant_id, embedding, dim, chunk_count, label, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(tenant_id)
    .bind(&blob)
    .bind(dim)
    .bind(chunk_count)
    .bind(label)
    .bind(&now)
    .execute(pool)
    .await?;

    debug!(
        tenant_id,
        dim = embedding.len(),
        chunk_count,
        "Stored project aggregate embedding"
    );

    Ok(())
}

/// Load a single project's aggregate embedding.
pub async fn load_project_embedding(
    pool: &SqlitePool,
    tenant_id: &str,
) -> Result<Option<Vec<f32>>, sqlx::Error> {
    let row = sqlx::query("SELECT embedding, dim FROM project_embeddings WHERE tenant_id = ?")
        .bind(tenant_id)
        .fetch_optional(pool)
        .await?;

    Ok(row.and_then(|r| {
        let blob: Vec<u8> = r.get("embedding");
        let dim: i64 = r.get("dim");
        blob_to_embedding(&blob, dim as usize)
    }))
}

/// Load all project aggregate embeddings.
///
/// Returns a list of (tenant_id, embedding) pairs ordered by tenant_id.
pub async fn load_all_project_embeddings(
    pool: &SqlitePool,
) -> Result<Vec<(String, Vec<f32>)>, sqlx::Error> {
    let rows =
        sqlx::query("SELECT tenant_id, embedding, dim FROM project_embeddings ORDER BY tenant_id")
            .fetch_all(pool)
            .await?;

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let tenant: String = row.get("tenant_id");
        let blob: Vec<u8> = row.get("embedding");
        let dim: i64 = row.get("dim");
        if let Some(emb) = blob_to_embedding(&blob, dim as usize) {
            results.push((tenant, emb));
        } else {
            warn!(
                tenant_id = tenant.as_str(),
                "Corrupt embedding blob, skipping"
            );
        }
    }

    Ok(results)
}

/// Delete a project's aggregate embedding.
pub async fn delete_project_embedding(
    pool: &SqlitePool,
    tenant_id: &str,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query("DELETE FROM project_embeddings WHERE tenant_id = ?")
        .bind(tenant_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// ---- Affinity label storage ------------------------------------------------

/// Store a label for an affinity group.
pub(crate) async fn store_affinity_label(
    pool: &SqlitePool,
    group_id: &str,
    label: &str,
    category: &str,
    score: f64,
) -> Result<(), sqlx::Error> {
    let now = wqm_common::timestamps::now_utc();

    sqlx::query(
        r#"
        INSERT OR REPLACE INTO affinity_labels
            (group_id, label, category, score, updated_at)
        VALUES (?, ?, ?, ?, ?)
        "#,
    )
    .bind(group_id)
    .bind(label)
    .bind(category)
    .bind(score)
    .bind(&now)
    .execute(pool)
    .await?;

    Ok(())
}

/// Load the label for an affinity group.
pub async fn load_affinity_label(
    pool: &SqlitePool,
    group_id: &str,
) -> Result<Option<(String, String, f64)>, sqlx::Error> {
    let row = sqlx::query("SELECT label, category, score FROM affinity_labels WHERE group_id = ?")
        .bind(group_id)
        .fetch_optional(pool)
        .await?;

    Ok(row.map(|r| {
        (
            r.get::<String, _>("label"),
            r.get::<String, _>("category"),
            r.get::<f64, _>("score"),
        )
    }))
}
