//! Library operations for the tenant strategy.
//!
//! Handles `ItemType::Tenant` items routed to the `libraries` collection:
//! Add (ensure collection), Scan (walk directory, enqueue files), and
//! Delete (tenant-scoped point removal).

use std::path::Path;
use std::sync::Arc;

use tracing::{debug, info, warn};
use walkdir::WalkDir;

use crate::allowed_extensions::AllowedExtensions;
use crate::context::ProcessingContext;
use crate::file_classification::classify_file_type;
use crate::patterns::exclusion::{should_exclude_directory, should_exclude_file};
use crate::queue_operations::QueueManager;
use crate::specs::parse_payload;
use crate::storage::DocumentPoint;
use crate::storage::StorageClient;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    FilePayload, ItemType, LibraryContentPayload, LibraryPayload, QueueOperation, UnifiedQueueItem,
};
use wqm_common::constants::COLLECTION_LIBRARIES;
use wqm_common::paths::{CanonicalPath, RelativePath};

/// Process library item -- create/manage library collections.
pub(crate) async fn process_library_item(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
) -> UnifiedProcessorResult<()> {
    info!(
        "Processing library item: {} (op={:?})",
        item.queue_id, item.op
    );

    match item.op {
        QueueOperation::Add => handle_library_add(ctx, item).await?,
        QueueOperation::Scan => handle_library_scan(ctx, item).await?,
        QueueOperation::Delete => handle_library_delete(ctx, item).await?,
        _ => {
            warn!(
                "Unsupported operation {:?} for library item {}",
                item.op, item.queue_id
            );
        }
    }

    info!(
        "Successfully processed library item {} (op={:?})",
        item.queue_id, item.op
    );

    Ok(())
}

/// Add-op dispatcher: content-bearing payloads embed and write; registration
/// payloads only ensure the destination collection exists.
async fn handle_library_add(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
) -> UnifiedProcessorResult<()> {
    // Try the content-bearing payload first (MCP `store` enqueues this shape).
    if let Ok(content_payload) = parse_payload::<LibraryContentPayload>(item) {
        if !content_payload.content.trim().is_empty() {
            return write_library_content_point(ctx, item, &content_payload).await;
        }
    }
    // Fall back to registration semantics: ensure the collection exists.
    let _payload: LibraryPayload = parse_payload(item)?;
    crate::shared::ensure_collection(
        &ctx.storage_client,
        &item.collection,
        ctx.embedding_generator.dense_dim() as u64,
    )
    .await
    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
    Ok(())
}

/// Scan-op handler: walks the library directory and enqueues files.
async fn handle_library_scan(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
) -> UnifiedProcessorResult<()> {
    let pool = ctx.queue_manager.pool();
    let folder_path: Option<String> = sqlx::query_scalar(
        "SELECT path FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2",
    )
    .bind(&item.tenant_id)
    .bind(COLLECTION_LIBRARIES)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("Failed to lookup library path: {}", e))
    })?;

    match folder_path {
        Some(path) => {
            scan_library_directory(
                item,
                &path,
                &ctx.queue_manager,
                &ctx.storage_client,
                &ctx.allowed_extensions,
                ctx.embedding_generator.dense_dim() as u64,
            )
            .await?;
        }
        None => {
            warn!("Library '{}' not found in watch_folders", item.tenant_id);
        }
    }
    Ok(())
}

/// Delete-op handler: tenant-scoped point removal from the libraries
/// collection.
async fn handle_library_delete(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
) -> UnifiedProcessorResult<()> {
    if ctx
        .storage_client
        .collection_exists(&item.collection)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
    {
        info!(
            "Deleting library data for tenant={} from collection={}",
            item.tenant_id, item.collection
        );
        ctx.storage_client
            .delete_points_by_tenant(&item.collection, &item.tenant_id)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
    }
    Ok(())
}

/// Embed content via the shared pipeline and upsert a single point to the
/// libraries collection. Mirrors the projects content-add flow but scopes
/// the payload by `library_name` instead of branch/project.
async fn write_library_content_point(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &LibraryContentPayload,
) -> UnifiedProcessorResult<()> {
    crate::shared::ensure_collection(
        &ctx.storage_client,
        &item.collection,
        ctx.embedding_generator.dense_dim() as u64,
    )
    .await
    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    let embed_result = crate::shared::embedding_pipeline::embed_with_sparse(
        &ctx.embedding_generator,
        &ctx.embedding_semaphore,
        &payload.content,
        "bge-small-en-v1.5",
    )
    .await?;

    let point_payload = build_library_content_payload(item, payload);
    let point = DocumentPoint {
        id: crate::generate_point_id(&item.tenant_id, &item.branch, &payload.document_id, 0),
        dense_vector: embed_result.dense_vector,
        sparse_vector: embed_result.sparse_vector,
        payload: point_payload,
    };

    ctx.storage_client
        .insert_points_batch(&item.collection, vec![point], Some(1))
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    info!(
        "Wrote library content point: tenant={} library={} document_id={} -> {}",
        item.tenant_id, payload.library_name, payload.document_id, item.collection
    );
    Ok(())
}

/// Build the Qdrant payload map for a library content point.
fn build_library_content_payload(
    item: &UnifiedQueueItem,
    payload: &LibraryContentPayload,
) -> std::collections::HashMap<String, serde_json::Value> {
    let mut p = std::collections::HashMap::new();
    p.insert("content".to_string(), serde_json::json!(payload.content));
    p.insert(
        "document_id".to_string(),
        serde_json::json!(payload.document_id),
    );
    p.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
    p.insert(
        "library_name".to_string(),
        serde_json::json!(payload.library_name),
    );
    p.insert(
        "source_type".to_string(),
        serde_json::json!(payload.source_type),
    );
    p.insert("item_type".to_string(), serde_json::json!("content"));
    p.insert("branches".to_string(), serde_json::json!([&item.branch]));

    if let Some(metadata) = &payload.metadata {
        for (k, v) in metadata.iter() {
            // Avoid clobbering reserved keys; expose user-supplied metadata flat.
            if !p.contains_key(k) {
                p.insert(k.clone(), serde_json::json!(v));
            }
        }
    }
    p
}

/// Scan a library directory and enqueue files for ingestion (Task 523).
///
/// Similar to scan_project_directory but for library folders:
/// - Uses `tenant_id` as library name
/// - Targets the `libraries` collection
/// - No branch tracking (libraries are not Git repos)
pub(crate) async fn scan_library_directory(
    item: &UnifiedQueueItem,
    folder_path: &str,
    queue_manager: &Arc<QueueManager>,
    storage_client: &Arc<StorageClient>,
    allowed_extensions: &Arc<AllowedExtensions>,
    vector_size: u64,
) -> UnifiedProcessorResult<()> {
    let library_root = Path::new(folder_path);

    if !library_root.exists() {
        return Err(UnifiedProcessorError::FileNotFound(format!(
            "Library path does not exist: {}",
            folder_path
        )));
    }
    if !library_root.is_dir() {
        return Err(UnifiedProcessorError::InvalidPayload(format!(
            "Library path is not a directory: {}",
            folder_path
        )));
    }

    crate::shared::ensure_collection(storage_client, &item.collection, vector_size)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    info!(
        "Scanning library directory: {} (tenant_id={})",
        folder_path, item.tenant_id
    );

    let start_time = std::time::Instant::now();
    let (files_queued, files_excluded, errors) =
        walk_and_enqueue(item, library_root, queue_manager, allowed_extensions).await?;

    update_last_scan(item, queue_manager).await;

    info!(
        "Library scan complete: {} files queued, {} excluded, {} errors in {:?} (library={})",
        files_queued,
        files_excluded,
        errors,
        start_time.elapsed(),
        folder_path
    );

    Ok(())
}

/// Walk the library directory tree and enqueue each eligible file.
///
/// Returns `(files_queued, files_excluded, errors)`.
async fn walk_and_enqueue(
    item: &UnifiedQueueItem,
    library_root: &Path,
    queue_manager: &Arc<QueueManager>,
    allowed_extensions: &Arc<AllowedExtensions>,
) -> UnifiedProcessorResult<(u64, u64, u64)> {
    let mut files_queued = 0u64;
    let mut files_excluded = 0u64;
    let mut errors = 0u64;

    for entry in WalkDir::new(library_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() && e.depth() > 0 {
                !should_exclude_directory(&e.file_name().to_string_lossy())
            } else {
                true
            }
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        match enqueue_library_file(item, path, library_root, queue_manager, allowed_extensions)
            .await?
        {
            FileEnqueueResult::Queued => {
                files_queued += 1;
                if files_queued % 100 == 0 {
                    tokio::task::yield_now().await;
                }
            }
            FileEnqueueResult::Excluded => files_excluded += 1,
            FileEnqueueResult::Error => errors += 1,
            FileEnqueueResult::Deduplicated => {}
        }
    }

    Ok((files_queued, files_excluded, errors))
}

/// Result of attempting to enqueue a single library file.
enum FileEnqueueResult {
    Queued,
    Excluded,
    Deduplicated,
    Error,
}

/// Evaluate a single file entry: apply filters, build payload, and enqueue.
///
/// Returns `Err` only for hard failures (serialization); transient I/O issues
/// (metadata read, enqueue) are counted as `Error` and logged.
async fn enqueue_library_file(
    item: &UnifiedQueueItem,
    path: &Path,
    library_root: &Path,
    queue_manager: &Arc<QueueManager>,
    allowed_extensions: &Arc<AllowedExtensions>,
) -> UnifiedProcessorResult<FileEnqueueResult> {
    let rel_path = path
        .strip_prefix(library_root)
        .unwrap_or(path)
        .to_string_lossy();
    let abs_path = path.to_string_lossy();

    // Root-relative check only (#97): checking the absolute path would let
    // hidden components ABOVE the library root (e.g. `~/.config/...`) exclude
    // every file of an explicitly registered library.
    if should_exclude_file(&rel_path) {
        return Ok(FileEnqueueResult::Excluded);
    }
    if !allowed_extensions.is_allowed(&abs_path, &item.collection) {
        return Ok(FileEnqueueResult::Excluded);
    }

    let metadata = match path.metadata() {
        Ok(m) => m,
        Err(e) => {
            warn!("Failed to get metadata for {}: {}", abs_path, e);
            return Ok(FileEnqueueResult::Error);
        }
    };

    const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;
    if metadata.len() > MAX_FILE_SIZE {
        debug!(
            "Skipping large file: {} ({} bytes)",
            abs_path,
            metadata.len()
        );
        return Ok(FileEnqueueResult::Excluded);
    }

    let library_root_canonical =
        match CanonicalPath::from_user_input(&library_root.to_string_lossy()) {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "Library root {} failed canonical validation: {}",
                    library_root.display(),
                    e
                );
                return Ok(FileEnqueueResult::Error);
            }
        };
    let abs_canonical = match CanonicalPath::from_user_input(&abs_path) {
        Ok(a) => a,
        Err(e) => {
            warn!("File path {} failed canonical validation: {}", abs_path, e);
            return Ok(FileEnqueueResult::Error);
        }
    };
    let relative =
        match RelativePath::from_absolute_and_root(&abs_canonical, &library_root_canonical) {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "File {} not under library root {} ({}); skipping",
                    abs_path,
                    library_root.display(),
                    e
                );
                return Ok(FileEnqueueResult::Error);
            }
        };

    let file_payload = FilePayload {
        file_path: relative,
        file_type: Some(classify_file_type(path).as_str().to_string()),
        file_hash: None,
        size_bytes: Some(metadata.len()),
        old_path: None,
    };

    let payload_json = serde_json::to_string(&file_payload).map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!("Failed to serialize FilePayload: {}", e))
    })?;

    match queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            &item.tenant_id,
            &item.collection,
            &payload_json,
            Some(""),
            None,
        )
        .await
    {
        Ok((queue_id, true)) => {
            debug!(
                "Queued library file for ingestion: {} (queue_id={})",
                abs_path, queue_id
            );
            Ok(FileEnqueueResult::Queued)
        }
        Ok((_, false)) => {
            debug!("Library file already in queue (deduplicated): {}", abs_path);
            Ok(FileEnqueueResult::Deduplicated)
        }
        Err(e) => {
            warn!("Failed to queue library file {}: {}", abs_path, e);
            Ok(FileEnqueueResult::Error)
        }
    }
}

/// Update `last_scan` timestamp on the library's watch_folder row (best-effort).
async fn update_last_scan(item: &UnifiedQueueItem, queue_manager: &Arc<QueueManager>) {
    let result = sqlx::query(
        "UPDATE watch_folders SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE tenant_id = ?1 AND collection = ?2",
    )
    .bind(&item.tenant_id)
    .bind(COLLECTION_LIBRARIES)
    .execute(queue_manager.pool())
    .await;

    if let Err(e) = result {
        warn!(
            "Failed to update last_scan for library {}: {}",
            item.tenant_id, e
        );
    }
}
