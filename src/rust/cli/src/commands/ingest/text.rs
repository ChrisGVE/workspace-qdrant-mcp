//! Text ingest subcommand handler

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::IngestTextRequest;
use crate::output;

pub async fn ingest_text(content: &str, collection: &str, title: Option<String>) -> Result<()> {
    output::section("Ingest Text");

    let preview = if content.len() > 50 {
        format!("{}...", &content[..50])
    } else {
        content.to_string()
    };

    output::kv("Content", &preview);
    output::kv("Collection", collection);
    if let Some(t) = &title {
        output::kv("Title", t);
    }
    output::separator();

    let mut client = ensure_daemon_available().await?;

    output::info("Ingesting text via daemon...");

    let request = IngestTextRequest {
        content: content.to_string(),
        collection_basename: collection.to_string(),
        tenant_id: String::new(), // Auto-detected by daemon
        document_id: title,
        metadata: std::collections::HashMap::new(),
        chunk_text: true,
    };

    let result = client.document().ingest_text(request).await?.into_inner();

    if result.success {
        output::success("Text ingested successfully");
        output::kv("Document ID", &result.document_id);
        output::kv("Chunks Created", result.chunks_created.to_string());
    } else {
        output::error(format!("Ingestion failed: {}", result.error_message));
    }

    Ok(())
}
