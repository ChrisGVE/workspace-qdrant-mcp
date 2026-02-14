//! Scratchpad command - persistent LLM scratch space
//!
//! Manages scratchpad entries stored in the `scratchpad` collection.
//! Subcommands: add, list

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use tabled::Tabled;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{RefreshSignalRequest, QueueType};
use crate::output::{self, ColumnHints};
use crate::queue::{ScratchpadPayload, UnifiedQueueClient};

/// Scratchpad command arguments
#[derive(Args)]
pub struct ScratchArgs {
    #[command(subcommand)]
    command: ScratchCommand,
}

/// Scratchpad subcommands
#[derive(Subcommand)]
enum ScratchCommand {
    /// Add a scratchpad entry
    Add {
        /// Content text
        content: String,

        /// Optional title
        #[arg(short, long)]
        title: Option<String>,

        /// Tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,

        /// Project ID or path (defaults to _global_)
        #[arg(short, long)]
        project: Option<String>,
    },

    /// List scratchpad entries
    List {
        /// Project ID or path (defaults to showing all)
        #[arg(short, long)]
        project: Option<String>,

        /// Maximum entries to show
        #[arg(short = 'n', long, default_value = "50")]
        limit: usize,

        /// Show detailed info including full content
        #[arg(short, long)]
        verbose: bool,

        /// Output format: table (default) or json
        #[arg(short, long, default_value = "table")]
        format: String,
    },
}

/// Execute scratchpad command
pub async fn execute(args: ScratchArgs) -> Result<()> {
    match args.command {
        ScratchCommand::Add {
            content,
            title,
            tags,
            project,
        } => add_entry(content, title, tags, project).await,
        ScratchCommand::List {
            project,
            limit,
            verbose,
            format,
        } => list_entries(project, limit, verbose, &format).await,
    }
}

// ─── Add implementation ─────────────────────────────────────────────────────

async fn add_entry(
    content: String,
    title: Option<String>,
    tags: Option<String>,
    project: Option<String>,
) -> Result<()> {
    let tenant_id = resolve_tenant_id(project.as_deref())?;

    let tag_vec: Vec<String> = tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();

    let payload = ScratchpadPayload {
        content: content.clone(),
        title: title.clone(),
        tags: tag_vec.clone(),
        source_type: "scratchpad".to_string(),
    };

    let queue = UnifiedQueueClient::connect()?;
    let result = queue.enqueue_scratchpad(&tenant_id, &payload, 0)?;

    // Signal daemon to process queue
    if let Ok(mut client) = DaemonClient::connect_default().await {
        let request = RefreshSignalRequest {
            queue_type: QueueType::IngestQueue as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        let _ = client.system().send_refresh_signal(request).await;
    }

    output::section("Scratchpad Entry Queued");
    output::kv("Queue ID", &result.queue_id);
    output::kv("Tenant", &tenant_id);
    if let Some(t) = &title {
        output::kv("Title", t);
    }
    if !tag_vec.is_empty() {
        output::kv("Tags", &tag_vec.join(", "));
    }
    let preview = if content.len() > 80 {
        format!("{}...", &content[..77])
    } else {
        content
    };
    output::kv("Content", &preview);
    if result.was_duplicate {
        output::warning("Duplicate entry (already queued)");
    }

    Ok(())
}

// ─── Qdrant scroll response types ──────────────────────────────────────────

#[derive(Deserialize)]
struct ScrollResponse {
    result: ScrollResult,
}

#[derive(Deserialize)]
struct ScrollResult {
    points: Vec<QdrantPoint>,
}

#[derive(Deserialize)]
struct QdrantPoint {
    #[allow(dead_code)]
    id: serde_json::Value,
    payload: Option<serde_json::Value>,
}

// ─── Display types ─────────────────────────────────────────────────────────

#[derive(Tabled)]
struct ScratchRow {
    #[tabled(rename = "Title")]
    title: String,
    #[tabled(rename = "Tenant")]
    tenant_id: String,
    #[tabled(rename = "Tags")]
    tags: String,
    #[tabled(rename = "Created")]
    created_at: String,
}

impl ColumnHints for ScratchRow {
    fn content_columns() -> &'static [usize] { &[0] }
}

#[derive(Tabled)]
struct ScratchRowVerbose {
    #[tabled(rename = "Title")]
    title: String,
    #[tabled(rename = "Tenant")]
    tenant_id: String,
    #[tabled(rename = "Tags")]
    tags: String,
    #[tabled(rename = "Content")]
    content: String,
    #[tabled(rename = "Created")]
    created_at: String,
}

impl ColumnHints for ScratchRowVerbose {
    fn content_columns() -> &'static [usize] { &[0, 3] }
}

#[derive(Serialize)]
struct ScratchJson {
    title: String,
    content: String,
    tenant_id: String,
    tags: Vec<String>,
    source_type: String,
    created_at: String,
}

fn payload_str(payload: &serde_json::Value, key: &str) -> String {
    payload.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn payload_tags(payload: &serde_json::Value) -> Vec<String> {
    payload.get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .or_else(|| {
            payload.get("tags")
                .and_then(|v| v.as_str())
                .map(|s| s.split(',').map(String::from).collect())
        })
        .unwrap_or_default()
}

// ─── List implementation ───────────────────────────────────────────────────

fn qdrant_url() -> String {
    std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string())
}

fn build_qdrant_client() -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10));

    if let Ok(api_key) = std::env::var("QDRANT_API_KEY") {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "api-key",
            reqwest::header::HeaderValue::from_str(&api_key)
                .context("Invalid QDRANT_API_KEY")?,
        );
        builder = builder.default_headers(headers);
    }

    builder.build().context("Failed to create HTTP client")
}

async fn list_entries(
    project: Option<String>,
    limit: usize,
    verbose: bool,
    format: &str,
) -> Result<()> {
    let client = build_qdrant_client()?;
    let collection = wqm_common::constants::COLLECTION_SCRATCHPAD;
    let url = format!("{}/collections/{}/points/scroll", qdrant_url(), collection);

    let mut body = serde_json::json!({
        "limit": limit,
        "with_payload": true,
    });

    // Optional tenant_id filter
    if let Some(ref proj) = project {
        let tenant_id = resolve_tenant_id(Some(proj))?;
        body["filter"] = serde_json::json!({
            "must": [{
                "key": "tenant_id",
                "match": { "value": tenant_id }
            }]
        });
    }

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Failed to connect to Qdrant")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if status.as_u16() == 404 {
            output::info("Scratchpad collection does not exist yet. No entries stored.");
            return Ok(());
        }
        anyhow::bail!("Qdrant scroll failed ({}): {}", status, text);
    }

    let scroll: ScrollResponse = response
        .json()
        .await
        .context("Failed to parse Qdrant scroll response")?;

    let points = &scroll.result.points;

    if points.is_empty() {
        output::info("No scratchpad entries found.");
        return Ok(());
    }

    // JSON output
    if format == "json" {
        let entries: Vec<ScratchJson> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| ScratchJson {
                title: payload_str(payload, "title"),
                content: payload_str(payload, "content"),
                tenant_id: payload_str(payload, "tenant_id"),
                tags: payload_tags(payload),
                source_type: payload_str(payload, "source_type"),
                created_at: payload_str(payload, "created_at"),
            })
            .collect();
        output::print_json(&entries);
        return Ok(());
    }

    // Table output
    output::section("Scratchpad Entries");
    output::kv("Total", &points.len().to_string());
    if let Some(p) = &project {
        output::kv("Filter", p);
    }
    output::separator();

    if verbose {
        let rows: Vec<ScratchRowVerbose> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| ScratchRowVerbose {
                title: payload_str(payload, "title"),
                tenant_id: payload_str(payload, "tenant_id"),
                tags: payload_tags(payload).join(", "),
                content: payload_str(payload, "content"),
                created_at: payload_str(payload, "created_at"),
            })
            .collect();
        output::print_table_auto(&rows);
    } else {
        let rows: Vec<ScratchRow> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| ScratchRow {
                title: payload_str(payload, "title"),
                tenant_id: payload_str(payload, "tenant_id"),
                tags: payload_tags(payload).join(", "),
                created_at: payload_str(payload, "created_at"),
            })
            .collect();
        output::print_table_auto(&rows);
    }

    Ok(())
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn resolve_tenant_id(project: Option<&str>) -> Result<String> {
    match project {
        None => Ok("_global_".to_string()),
        Some(p) => {
            let path = std::path::Path::new(p);
            if path.exists() {
                Ok(wqm_common::project_id::calculate_tenant_id(path))
            } else {
                // Assume it's a project ID directly
                Ok(p.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_tenant_id_default() {
        assert_eq!(resolve_tenant_id(None).unwrap(), "_global_");
    }

    #[test]
    fn test_resolve_tenant_id_direct_id() {
        assert_eq!(
            resolve_tenant_id(Some("proj_abc123")).unwrap(),
            "proj_abc123"
        );
    }

    #[test]
    fn test_payload_str_missing() {
        let v = serde_json::json!({});
        assert_eq!(payload_str(&v, "missing"), "");
    }

    #[test]
    fn test_payload_str_present() {
        let v = serde_json::json!({"title": "hello"});
        assert_eq!(payload_str(&v, "title"), "hello");
    }

    #[test]
    fn test_payload_tags_array() {
        let v = serde_json::json!({"tags": ["a", "b"]});
        assert_eq!(payload_tags(&v), vec!["a", "b"]);
    }

    #[test]
    fn test_payload_tags_string() {
        let v = serde_json::json!({"tags": "a,b"});
        assert_eq!(payload_tags(&v), vec!["a", "b"]);
    }

    #[test]
    fn test_payload_tags_missing() {
        let v = serde_json::json!({});
        assert!(payload_tags(&v).is_empty());
    }
}
