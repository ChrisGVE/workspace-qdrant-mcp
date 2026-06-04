//! Library ingest subcommand

use std::path::PathBuf;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use wqm_common::classification::extension_to_document_type;
use wqm_common::constants::COLLECTION_LIBRARIES;
use wqm_common::paths::{CanonicalPath, RelativePath};
use wqm_common::payloads::{ChunkingConfigPayload, LibraryDocumentPayload};

use super::helpers::{
    canonical_from_cli_path, classify_document_extension, signal_daemon_ingest_queue,
};
use crate::data::db::connect_readonly;
use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::EnqueueItemRequest;
use crate::output;
use crate::output::style::home_to_tilde;

/// Ingest a single document into a library
pub async fn execute(
    file: &PathBuf,
    library: &str,
    chunk_tokens: usize,
    overlap_tokens: usize,
) -> Result<()> {
    output::section(format!("Ingest Document into Library: {}", library));

    // Validate file exists
    if !file.exists() {
        output::error(format!("File does not exist: {}", file.display()));
        return Ok(());
    }

    if !file.is_file() {
        output::error(format!("Path is not a file: {}", file.display()));
        return Ok(());
    }

    let abs_path = canonical_from_cli_path(file)?;
    let abs_path_str = abs_path.as_str().to_string();
    let abs_path_buf = PathBuf::from(&abs_path_str);

    let (source_format, document_type) = classify_extension(&abs_path_buf)?;

    // Resolve the registered library root from SQLite so the persisted
    // document_path is relative to that root (per docs/specs/16-path-abstraction.md §3.3).
    let library_root = lookup_library_root(library).with_context(|| {
        format!(
            "Library '{}' is not registered. Run `wqm library add {} <path>` first.",
            library, library
        )
    })?;

    let relative_document_path = RelativePath::from_absolute_and_root(&abs_path, &library_root)
        .with_context(|| {
            format!(
                "File {} is not inside library '{}' root {}",
                home_to_tilde(abs_path.as_str()),
                library,
                home_to_tilde(library_root.as_str()),
            )
        })?;

    output::kv("  File", home_to_tilde(&abs_path_str));
    output::kv("  Library", library);
    output::kv("  Library Root", home_to_tilde(library_root.as_str()));
    output::kv("  Relative Path", relative_document_path.as_str());
    output::kv("  Format", source_format);
    output::kv("  Type", document_type);
    output::kv("  Chunk Target", format!("{} tokens", chunk_tokens));
    output::kv("  Chunk Overlap", format!("{} tokens", overlap_tokens));

    // Generate doc_id (UUID v5 from library_name + relative path).
    let doc_id =
        wqm_common::document_id::generate_document_id(library, relative_document_path.as_str());
    output::kv("  Doc ID", &doc_id);

    // Calculate doc_fingerprint (SHA256 of file bytes)
    let file_bytes =
        std::fs::read(&abs_path_buf).context("Failed to read file for fingerprinting")?;
    let mut hasher = Sha256::new();
    hasher.update(&file_bytes);
    let doc_fingerprint = format!("{:x}", hasher.finalize());

    // Build payload
    let payload = LibraryDocumentPayload {
        document_path: relative_document_path,
        library_name: library.to_string(),
        document_type: document_type.to_string(),
        source_format: source_format.to_string(),
        doc_id: doc_id.clone(),
        doc_fingerprint: Some(doc_fingerprint),
        library_path: None,
        source_project_id: None,
        chunking_config: Some(ChunkingConfigPayload {
            chunk_target_tokens: chunk_tokens,
            chunk_overlap_tokens: overlap_tokens,
        }),
    };

    let payload_json =
        serde_json::to_string(&payload).context("Failed to serialize library document payload")?;

    enqueue_document(library, &payload_json).await?;
    signal_daemon_ingest_queue().await;

    Ok(())
}

/// Resolve the canonical root path for a registered library tag.
///
/// Reads the `watch_folders` row whose `tenant_id` matches the supplied
/// library tag and whose `collection` is the libraries collection.
fn lookup_library_root(library: &str) -> Result<CanonicalPath> {
    let conn = connect_readonly().context("Could not open state database")?;
    let path: String = conn
        .query_row(
            "SELECT path FROM watch_folders \
             WHERE tenant_id = ?1 AND collection = ?2 \
             LIMIT 1",
            rusqlite::params![library, COLLECTION_LIBRARIES],
            |row| row.get(0),
        )
        .context("Could not find library in watch_folders")?;
    CanonicalPath::from_user_input(&path).map_err(|e| anyhow::anyhow!("library root invalid: {e}"))
}

/// Extract and classify the file extension
fn classify_extension(abs_path: &PathBuf) -> Result<(&'static str, &'static str)> {
    let ext = abs_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let ext = match ext {
        Some(e) => e,
        None => {
            output::error("File has no extension. Cannot determine document format.");
            anyhow::bail!("File has no extension");
        }
    };

    match classify_document_extension(&ext) {
        Some(result) => Ok(result),
        None => {
            if extension_to_document_type(&ext).is_some() {
                output::error(format!(
                    "Extension '.{}' is recognized but not supported for library document ingestion.",
                    ext
                ));
            } else {
                output::error(format!(
                    "Unsupported file extension: '.{}'. Supported: pdf, docx, doc, pptx, ppt, \
                     pages, key, odt, odp, ods, rtf, epub, html, htm, md, txt",
                    ext
                ));
            }
            anyhow::bail!("Unsupported file extension: .{}", ext);
        }
    }
}

/// Enqueue the document for ingestion via gRPC
async fn enqueue_document(library: &str, payload_json: &str) -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    match client
        .queue_write()
        .enqueue_item(EnqueueItemRequest {
            item_type: "file".to_string(),
            op: "add".to_string(),
            tenant_id: library.to_string(),
            collection: COLLECTION_LIBRARIES.to_string(),
            payload_json: payload_json.to_string(),
            branch: String::new(),
            metadata_json: None,
        })
        .await
    {
        Ok(resp) => {
            let inner = resp.into_inner();
            if inner.is_new {
                output::success(format!(
                    "Document queued for ingestion (queue_id: {})",
                    inner.queue_id
                ));
            } else {
                output::info("Document already queued for ingestion (same content)");
            }
        }
        Err(e) => {
            output::error(format!("Failed to enqueue document: {}", e));
        }
    }

    Ok(())
}
