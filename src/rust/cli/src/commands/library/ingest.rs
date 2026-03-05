//! Library ingest subcommand

use std::path::PathBuf;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use wqm_common::classification::extension_to_document_type;
use wqm_common::constants::COLLECTION_LIBRARIES;
use wqm_common::payloads::{ChunkingConfigPayload, LibraryDocumentPayload};

use super::helpers::{classify_document_extension, signal_daemon_ingest_queue};
use crate::output;
use crate::queue::{ItemType, QueueOperation, UnifiedQueueClient};

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

    let abs_path = file
        .canonicalize()
        .context("Could not resolve absolute path")?;
    let abs_path_str = abs_path.to_string_lossy().to_string();

    let (source_format, document_type) = classify_extension(&abs_path)?;

    output::kv("  File", &abs_path_str);
    output::kv("  Library", library);
    output::kv("  Format", source_format);
    output::kv("  Type", document_type);
    output::kv("  Chunk Target", &format!("{} tokens", chunk_tokens));
    output::kv("  Chunk Overlap", &format!("{} tokens", overlap_tokens));

    // Generate doc_id (UUID v5 from library_name + path)
    let doc_id = workspace_qdrant_core::generate_document_id(library, &abs_path_str);
    output::kv("  Doc ID", &doc_id);

    // Calculate doc_fingerprint (SHA256 of file bytes)
    let file_bytes = std::fs::read(&abs_path).context("Failed to read file for fingerprinting")?;
    let mut hasher = Sha256::new();
    hasher.update(&file_bytes);
    let doc_fingerprint = format!("{:x}", hasher.finalize());

    // Build payload
    let payload = LibraryDocumentPayload {
        document_path: abs_path_str,
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

    enqueue_document(library, &payload_json)?;
    signal_daemon_ingest_queue().await;

    Ok(())
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

/// Enqueue the document for ingestion
fn enqueue_document(library: &str, payload_json: &str) -> Result<()> {
    match UnifiedQueueClient::connect() {
        Ok(client) => {
            match client.enqueue(
                ItemType::File,
                QueueOperation::Add,
                library,
                COLLECTION_LIBRARIES,
                payload_json,
                "",
                None,
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::info("Document already queued for ingestion (same content)");
                    } else {
                        output::success(format!(
                            "Document queued for ingestion (queue_id: {})",
                            result.queue_id
                        ));
                    }
                    Ok(())
                }
                Err(e) => {
                    output::error(format!("Failed to enqueue document: {}", e));
                    Ok(())
                }
            }
        }
        Err(e) => {
            output::error(format!("Could not connect to queue database: {}", e));
            output::info("Ensure daemon has been started at least once: wqm service start");
            Ok(())
        }
    }
}
