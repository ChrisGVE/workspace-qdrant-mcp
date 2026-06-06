use sha2::{Digest, Sha256};
use sqlx::SqlitePool;
use uuid::Uuid;

/// Enqueue a `folder|scan` item for every enabled `watch_folders` row.
///
/// Each row IS a watch-folder root, so the scan must target that root. The
/// folder-scan strategy resolves the root from `watch_folders` by
/// `(tenant_id, collection)` and treats `payload.folder_path` as **relative**
/// to it: `None` => scan the root, `Some(rel)` => `root.join(rel)`. Passing the
/// absolute root path as `folder_path` made the strategy join it onto the root
/// (`root.join(absolute_root)`), producing a doubled, non-existent path like
/// `/repo/x/repo/x` — every reembed scan then logged "target is not a
/// directory", enqueued zero files, and the re-embed "completed" without
/// re-ingesting anything. Emit `folder_path: null` so the strategy scans the
/// actual root (and takes the git-index fast path for project scans).
pub(super) async fn enqueue_folder_scans(pool: &SqlitePool, now: &str) -> Result<u32, sqlx::Error> {
    let folders = sqlx::query_as::<_, (String, String)>(
        "SELECT collection, tenant_id FROM watch_folders WHERE enabled = 1",
    )
    .fetch_all(pool)
    .await?;

    let mut count = 0u32;
    for (collection, tenant_id) in &folders {
        let payload = serde_json::json!({
            "folder_path": null,
            "recursive": true,
            "recursive_depth": 10,
            "patterns": [],
            "ignore_patterns": []
        })
        .to_string();
        let idem_input = format!("folder|scan|{}|{}|{}", tenant_id, collection, payload);
        let mut hasher = Sha256::new();
        hasher.update(idem_input.as_bytes());
        let idem_key: String = hasher
            .finalize()
            .iter()
            .take(16)
            .map(|b| format!("{:02x}", b))
            .collect();
        let qid = Uuid::new_v4().to_string();
        let res = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'folder', 'scan', ?3, ?4, 'pending', ?5, ?6, ?7)",
        )
        .bind(&qid)
        .bind(&idem_key)
        .bind(tenant_id)
        .bind(collection)
        .bind(&payload)
        .bind(now)
        .bind(now)
        .execute(pool)
        .await?;
        if res.rows_affected() > 0 {
            count += 1;
        }
    }
    Ok(count)
}

/// Enqueue a `text|add` item for every row in `rules_mirror`.
pub(super) async fn enqueue_rules_mirror(pool: &SqlitePool, now: &str) -> Result<u32, sqlx::Error> {
    let rules = sqlx::query_as::<_, (String, String, Option<String>, Option<String>)>(
        "SELECT rule_id, rule_text, scope, tenant_id FROM rules_mirror",
    )
    .fetch_all(pool)
    .await?;

    let mut count = 0u32;
    for (rule_id, rule_text, scope, tenant_id_opt) in &rules {
        let tenant_id = tenant_id_opt
            .clone()
            .unwrap_or_else(|| "_system".to_string());
        let payload = serde_json::json!({
            "rule_id": rule_id,
            "content": rule_text,
            "source_type": "rule",
            "scope": scope,
        })
        .to_string();
        let idem_input = format!("text|add|{}|rules|{}", tenant_id, payload);
        let mut hasher = Sha256::new();
        hasher.update(idem_input.as_bytes());
        let idem_key: String = hasher
            .finalize()
            .iter()
            .take(16)
            .map(|b| format!("{:02x}", b))
            .collect();
        let qid = Uuid::new_v4().to_string();
        let res = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'text', 'add', ?3, 'rules', 'pending', ?4, ?5, ?6)",
        )
        .bind(&qid)
        .bind(&idem_key)
        .bind(&tenant_id)
        .bind(&payload)
        .bind(now)
        .bind(now)
        .execute(pool)
        .await?;
        if res.rows_affected() > 0 {
            count += 1;
        }
    }
    Ok(count)
}

/// Enqueue a `text|add` item for every row in `scratchpad_mirror`.
pub(super) async fn enqueue_scratchpad_mirror(
    pool: &SqlitePool,
    now: &str,
) -> Result<u32, sqlx::Error> {
    let entries = sqlx::query_as::<_, (String, Option<String>, String, Option<String>, String)>(
        "SELECT scratchpad_id, title, content, tags, tenant_id FROM scratchpad_mirror",
    )
    .fetch_all(pool)
    .await?;

    let mut count = 0u32;
    for (id, title, content, tags, tenant_id) in &entries {
        let payload = serde_json::json!({
            "scratchpad_id": id,
            "content": content,
            "title": title.clone().unwrap_or_default(),
            "tags": tags.clone().unwrap_or_else(|| "[]".to_string()),
            "source_type": "scratchpad",
        })
        .to_string();
        let idem_input = format!("text|add|{}|scratchpad|{}", tenant_id, payload);
        let mut hasher = Sha256::new();
        hasher.update(idem_input.as_bytes());
        let idem_key: String = hasher
            .finalize()
            .iter()
            .take(16)
            .map(|b| format!("{:02x}", b))
            .collect();
        let qid = Uuid::new_v4().to_string();
        let res = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'text', 'add', ?3, 'scratchpad', 'pending', ?4, ?5, ?6)",
        )
        .bind(&qid)
        .bind(&idem_key)
        .bind(tenant_id)
        .bind(&payload)
        .bind(now)
        .bind(now)
        .execute(pool)
        .await?;
        if res.rows_affected() > 0 {
            count += 1;
        }
    }
    Ok(count)
}
