//! Memory command - LLM rules management
//!
//! Manages LLM memory rules stored in the `memory` collection.
//! Subcommands: list, add, remove, search, scope

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use tabled::Tabled;

use wqm_common::schema::qdrant::memory as mem_schema;
use wqm_common::schema::sqlite::watch_folders as wf_schema;

use crate::config::get_database_path_checked;
use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{DeleteDocumentRequest, IngestTextRequest};
use crate::output::{self, ColumnHints, ServiceStatus};
use crate::queue::{ContentPayload as QueueContentPayload, UnifiedQueueClient};

/// Memory command arguments
#[derive(Args)]
pub struct MemoryArgs {
    #[command(subcommand)]
    command: MemoryCommand,
}

/// Memory subcommands
#[derive(Subcommand)]
enum MemoryCommand {
    /// List memory rules
    List {
        /// Show only global rules
        #[arg(long, conflicts_with = "project")]
        global: bool,

        /// Show only rules for a specific project (path or ID)
        #[arg(long, conflicts_with = "global")]
        project: Option<String>,

        /// Filter by type (preference, behavior, constraint, pattern)
        #[arg(short = 't', long)]
        rule_type: Option<String>,

        /// Show detailed information including full content
        #[arg(short, long)]
        verbose: bool,

        /// Output format: table (default) or json
        #[arg(short, long, default_value = "table")]
        format: String,
    },

    /// Add a new memory rule
    Add {
        /// Rule label (identifier for the rule)
        #[arg(long)]
        label: String,

        /// Rule content
        #[arg(long)]
        content: String,

        /// Apply to all projects (global rule)
        #[arg(long, conflicts_with = "project")]
        global: bool,

        /// Apply to specific project (path or ID)
        #[arg(long, conflicts_with = "global")]
        project: Option<String>,

        /// Rule type (preference, behavior, constraint, pattern)
        #[arg(short = 't', long, default_value = "preference")]
        rule_type: String,

        /// Priority (1-10, higher = more important)
        #[arg(short, long, default_value = "5")]
        priority: u32,
    },

    /// Remove a memory rule
    Remove {
        /// Rule label to remove
        #[arg(long)]
        label: String,

        /// Remove from global scope
        #[arg(long, conflicts_with = "project")]
        global: bool,

        /// Remove from specific project (path or ID)
        #[arg(long, conflicts_with = "global")]
        project: Option<String>,
    },

    /// Search memory rules
    Search {
        /// Search query
        query: String,

        /// Search only global rules
        #[arg(long, conflicts_with = "project")]
        global: bool,

        /// Search only rules for a specific project
        #[arg(long, conflicts_with = "global")]
        project: Option<String>,

        /// Maximum results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Inject memory rules into Claude Code context (SessionStart hook)
    #[command(hide = true)]
    Inject,

    /// Manage rule scopes (list available scopes, show scope hierarchy)
    Scope {
        /// List all available scopes
        #[arg(long)]
        list: bool,

        /// Show rules for a specific scope
        #[arg(long)]
        show: Option<String>,

        /// Show verbose scope information
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Execute memory command
pub async fn execute(args: MemoryArgs) -> Result<()> {
    match args.command {
        MemoryCommand::List {
            global,
            project,
            rule_type,
            verbose,
            format,
        } => {
            let scope = resolve_scope(global, project);
            list_rules(scope, rule_type, verbose, &format).await
        }
        MemoryCommand::Add {
            label,
            content,
            global,
            project,
            rule_type,
            priority,
        } => {
            let scope = resolve_scope(global, project);
            add_rule(&label, &content, &rule_type, &scope, priority).await
        }
        MemoryCommand::Remove {
            label,
            global,
            project,
        } => {
            let scope = resolve_scope(global, project);
            remove_rule(&label, &scope).await
        }
        MemoryCommand::Search {
            query,
            global,
            project,
            limit,
        } => {
            let scope = resolve_scope(global, project);
            search_rules(&query, scope, limit).await
        }
        MemoryCommand::Inject => inject_rules().await,
        MemoryCommand::Scope {
            list,
            show,
            verbose,
        } => manage_scopes(list, show, verbose).await,
    }
}

/// Resolve scope from --global / --project flags
fn resolve_scope(global: bool, project: Option<String>) -> Option<String> {
    if global {
        Some("global".to_string())
    } else {
        project.map(|p| format!("project:{}", p))
    }
}

// ─── Qdrant REST helpers ────────────────────────────────────────────────────

/// Get Qdrant URL from environment or default
fn qdrant_url() -> String {
    std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| wqm_common::constants::DEFAULT_QDRANT_URL.to_string())
}

/// Build a reqwest client with optional Qdrant API key header
fn build_qdrant_client() -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        headers.insert(
            "api-key",
            reqwest::header::HeaderValue::from_str(&key)
                .context("Invalid QDRANT_API_KEY value")?,
        );
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP client")
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

/// Compact table row for default display
#[derive(Tabled)]
struct MemoryRuleRow {
    #[tabled(rename = "Label")]
    label: String,
    #[tabled(rename = "Title")]
    title: String,
    #[tabled(rename = "Scope")]
    scope: String,
    #[tabled(rename = "Priority")]
    priority: String,
    #[tabled(rename = "Created")]
    created_at: String,
}

impl ColumnHints for MemoryRuleRow {
    // Title(1) is content
    fn content_columns() -> &'static [usize] { &[1] }
}

/// Verbose table row with content
#[derive(Tabled)]
struct MemoryRuleRowVerbose {
    #[tabled(rename = "Label")]
    label: String,
    #[tabled(rename = "Title")]
    title: String,
    #[tabled(rename = "Scope")]
    scope: String,
    #[tabled(rename = "Priority")]
    priority: String,
    #[tabled(rename = "Tags")]
    tags: String,
    #[tabled(rename = "Content")]
    content: String,
    #[tabled(rename = "Created")]
    created_at: String,
}

impl ColumnHints for MemoryRuleRowVerbose {
    // Title(1), Content(5) are content
    fn content_columns() -> &'static [usize] { &[1, 5] }
}

/// Full rule data for JSON output
#[derive(Serialize)]
struct MemoryRuleJson {
    label: String,
    title: String,
    content: String,
    scope: String,
    project_id: Option<String>,
    source_type: String,
    priority: Option<u32>,
    tags: Vec<String>,
    created_at: String,
    updated_at: String,
}

/// Helper to extract a string from a JSON payload field
fn payload_str(payload: &serde_json::Value, key: &str) -> String {
    payload.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Helper to extract an optional u32 from a JSON payload field
fn payload_u32(payload: &serde_json::Value, key: &str) -> Option<u32> {
    payload.get(key).and_then(|v| {
        v.as_u64().map(|n| n as u32)
    })
}

/// Format a title with optional project name for project-scoped rules.
///
/// For rules with scope "project", appends project info in parenthesis
/// after the title text. Non-verbose shows just the project name;
/// verbose includes the tenant_id. Uses same-line format so tabled's
/// word wrapping handles layout naturally.
fn format_title_with_project(
    payload: &serde_json::Value,
    project_names: &HashMap<String, String>,
    verbose: bool,
) -> String {
    let title = payload_str(payload, mem_schema::TITLE.name);
    let scope = payload_str(payload, mem_schema::SCOPE.name);
    if scope == "project" {
        if let Some(pid) = payload.get(mem_schema::PROJECT_ID.name).and_then(|v| v.as_str()) {
            let name = project_names.get(pid);
            return match (name, verbose) {
                (Some(name), false) => format!("{} (project: {})", title, name),
                (Some(name), true) => format!("{} (project: {} / {})", title, name, pid),
                (None, _) => format!("{} (project: {})", title, pid),
            };
        }
    }
    title
}

/// Build a tenant_id → project name mapping from watch_folders.
///
/// Extracts the last path component as the project name. Returns an
/// empty map if the database is unavailable.
fn load_project_names() -> HashMap<String, String> {
    let mut map = HashMap::new();
    let db_path = match get_database_path_checked() {
        Ok(p) => p,
        Err(_) => return map,
    };
    let conn = match rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        Ok(c) => c,
        Err(_) => return map,
    };
    let sql = format!(
        "SELECT {}, {} FROM {} WHERE {} = 'projects'",
        wf_schema::TENANT_ID.name,
        wf_schema::PATH.name,
        wf_schema::TABLE.name,
        wf_schema::COLLECTION.name,
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return map,
    };
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
        ))
    });
    if let Ok(rows) = rows {
        for row in rows.flatten() {
            let (tenant_id, path) = row;
            let name = Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            map.insert(tenant_id, name);
        }
    }
    map
}

/// Normalize comma-separated values for table display.
///
/// Ensures a space after each comma so `keep_words()` can wrap at
/// word boundaries instead of breaking mid-word.
fn normalize_commas(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 10);
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        result.push(c);
        if c == ',' {
            if chars.peek() != Some(&' ') {
                result.push(' ');
            }
        }
    }
    result
}

// ─── List implementation ───────────────────────────────────────────────────

async fn list_rules(
    scope: Option<String>,
    _rule_type: Option<String>,
    verbose: bool,
    format: &str,
) -> Result<()> {
    let client = build_qdrant_client()?;
    let collection = wqm_common::constants::COLLECTION_MEMORY;
    let url = format!("{}/collections/{}/points/scroll", qdrant_url(), collection);

    // Build scroll request with optional scope filter
    let mut body = serde_json::json!({
        "limit": 100,
        "with_payload": true,
    });

    if let Some(ref scope_str) = scope {
        let filter = build_scope_filter(scope_str);
        body["filter"] = filter;
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
            output::info("Memory collection does not exist yet. No rules stored.");
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
        output::info("No memory rules found.");
        return Ok(());
    }

    // JSON output
    if format == "json" {
        let rules: Vec<MemoryRuleJson> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| MemoryRuleJson {
                label: payload_str(payload, mem_schema::LABEL.name),
                title: payload_str(payload, mem_schema::TITLE.name),
                content: payload_str(payload, mem_schema::CONTENT.name),
                scope: payload_str(payload, mem_schema::SCOPE.name),
                project_id: payload.get(mem_schema::PROJECT_ID.name).and_then(|v| v.as_str()).map(String::from),
                source_type: payload_str(payload, mem_schema::SOURCE_TYPE.name),
                priority: payload_u32(payload, mem_schema::PRIORITY.name),
                tags: payload.get(mem_schema::TAGS.name)
                    .and_then(|v| v.as_str())
                    .map(|s| s.split(',').map(String::from).collect())
                    .unwrap_or_default(),
                created_at: payload_str(payload, mem_schema::CREATED_AT.name),
                updated_at: payload_str(payload, mem_schema::UPDATED_AT.name),
            })
            .collect();
        output::print_json(&rules);
        return Ok(());
    }

    // Table output
    output::section("Memory Rules");
    output::kv("Total", &points.len().to_string());
    if let Some(s) = &scope {
        output::kv("Filter", s);
    }
    output::separator();

    let project_names = load_project_names();

    if verbose {
        // Verbose columns: Label(0), Title(1), Scope(2), Priority(3),
        //                   Tags(4), Content(5), Created(6)
        // Content columns: Title(1), Content(5)
        let rows: Vec<MemoryRuleRowVerbose> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| MemoryRuleRowVerbose {
                label: payload_str(payload, mem_schema::LABEL.name),
                title: format_title_with_project(payload, &project_names, true),
                scope: payload_str(payload, mem_schema::SCOPE.name),
                priority: payload_u32(payload, mem_schema::PRIORITY.name)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                tags: normalize_commas(&payload_str(payload, mem_schema::TAGS.name)),
                content: payload_str(payload, mem_schema::CONTENT.name),
                created_at: output::format_date(&payload_str(payload, mem_schema::CREATED_AT.name)),
            })
            .collect();
        output::print_table_auto(&rows);
    } else {
        // Columns: Label(0), Title(1), Scope(2), Priority(3), Created(4)
        // Content columns: Title(1)
        let rows: Vec<MemoryRuleRow> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| MemoryRuleRow {
                label: payload_str(payload, mem_schema::LABEL.name),
                title: format_title_with_project(payload, &project_names, false),
                scope: payload_str(payload, mem_schema::SCOPE.name),
                priority: payload_u32(payload, mem_schema::PRIORITY.name)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                created_at: output::format_date(&payload_str(payload, mem_schema::CREATED_AT.name)),
            })
            .collect();
        output::print_table_auto(&rows);
    }

    Ok(())
}

/// Build a Qdrant filter from scope string
fn build_scope_filter(scope_str: &str) -> serde_json::Value {
    let mut must = Vec::new();

    if scope_str == "global" {
        must.push(serde_json::json!({
            "key": mem_schema::SCOPE.name,
            "match": { "value": "global" }
        }));
    } else if let Some(project_id) = scope_str.strip_prefix("project:") {
        must.push(serde_json::json!({
            "key": mem_schema::SCOPE.name,
            "match": { "value": "project" }
        }));
        must.push(serde_json::json!({
            "key": mem_schema::PROJECT_ID.name,
            "match": { "value": project_id }
        }));
    }

    serde_json::json!({ "must": must })
}

// ─── Add rule ──────────────────────────────────────────────────────────────

async fn add_rule(label: &str, content: &str, rule_type: &str, scope: &Option<String>, priority: u32) -> Result<()> {
    output::section("Add Memory Rule");

    let scope_str = scope.as_deref().unwrap_or("global");

    output::kv("Label", label);
    output::kv("Content", content);
    output::kv("Type", rule_type);
    output::kv("Scope", scope_str);
    output::kv("Priority", &priority.to_string());
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("label".to_string(), label.to_string());
            metadata.insert("rule_type".to_string(), rule_type.to_string());
            metadata.insert("scope".to_string(), scope_str.to_string());
            metadata.insert("priority".to_string(), priority.to_string());
            metadata.insert("enabled".to_string(), "true".to_string());

            let request = IngestTextRequest {
                content: content.to_string(),
                collection_basename: "memory".to_string(),
                tenant_id: String::new(),
                document_id: Some(label.to_string()), // Use label as document ID
                metadata,
                chunk_text: false, // Don't chunk rules
            };

            match client.document().ingest_text(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    if result.success {
                        output::success("Memory rule added");
                        output::kv("Label", label);
                        output::kv("Rule ID", &result.document_id);
                    } else {
                        output::error(format!("Failed to add rule: {}", result.error_message));
                        // Try unified queue fallback
                        try_queue_fallback_memory(label, content, rule_type, scope_str, priority)?;
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to add rule via daemon: {}", e));
                    // Try unified queue fallback
                    try_queue_fallback_memory(label, content, rule_type, scope_str, priority)?;
                }
            }
        }
        Err(_) => {
            // Daemon not running - use unified queue fallback (Task 37.12)
            output::warning("Daemon not running, using unified queue fallback");
            try_queue_fallback_memory(label, content, rule_type, scope_str, priority)?;
        }
    }

    Ok(())
}

/// Fallback to unified queue when daemon is unavailable (Task 37.12)
fn try_queue_fallback_memory(
    label: &str,
    content: &str,
    rule_type: &str,
    scope: &str,
    priority: u32,
) -> Result<()> {
    output::info("Enqueueing memory rule to unified_queue for later processing...");

    match UnifiedQueueClient::connect() {
        Ok(queue_client) => {
            // Create content payload with memory rule metadata in the content
            let full_content = format!(
                "MEMORY_RULE\nlabel:{}\ntype:{}\nscope:{}\npriority:{}\n---\n{}",
                label, rule_type, scope, priority, content
            );

            let payload = QueueContentPayload {
                content: full_content,
                source_type: "cli_memory".to_string(),
                main_tag: Some(format!("memory_{}", rule_type)),
                full_tag: Some(format!("memory_{}_{}", rule_type, scope)),
            };

            // Memory rules use "memory" collection
            match queue_client.enqueue_content(
                "_global", // Memory is global
                "memory",
                &payload,
                0, // Queue priority is dynamic (computed at dequeue time)
                "main", // Memory rules are branch-agnostic
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::warning("Memory rule already queued (duplicate)");
                        output::kv("Idempotency Key", &result.idempotency_key);
                    } else {
                        output::success("Memory rule queued for processing");
                        output::kv("Label", label);
                        output::kv("Queue ID", &result.queue_id);
                        output::kv("Status", "pending");
                        output::kv("Fallback Mode", "unified_queue");
                    }
                    output::separator();
                    output::info("The rule will be added when the daemon starts.");
                    output::info("Check status with: wqm status queue");
                }
                Err(e) => {
                    output::error(format!("Failed to enqueue memory rule: {}", e));
                }
            }
        }
        Err(e) => {
            output::error(format!("Failed to connect to queue database: {}", e));
            output::info("Ensure the workspace-qdrant directory exists.");
        }
    }

    Ok(())
}

// ─── Remove rule ───────────────────────────────────────────────────────────

async fn remove_rule(label: &str, scope: &Option<String>) -> Result<()> {
    output::section("Remove Memory Rule");

    let scope_str = scope.as_deref().unwrap_or("all");

    output::kv("Label", label);
    output::kv("Scope", scope_str);
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            // Use label as document_id since we store rules with label as ID
            let request = DeleteDocumentRequest {
                document_id: label.to_string(),
                collection_name: "memory".to_string(),
            };

            match client.document().delete_document(request).await {
                Ok(_) => {
                    output::success("Memory rule removed");
                    output::kv("Label", label);
                }
                Err(e) => {
                    output::error(format!("Failed to remove rule: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}

// ─── Search rules ──────────────────────────────────────────────────────────

async fn search_rules(query: &str, scope: Option<String>, limit: usize) -> Result<()> {
    output::section("Search Memory Rules");

    output::kv("Query", query);
    if let Some(s) = &scope {
        output::kv("Scope", s);
    } else {
        output::kv("Scope", "all");
    }
    output::kv("Limit", &limit.to_string());
    output::separator();

    output::info("Memory search via MCP:");
    output::info("  mcp__workspace_qdrant__memory(");
    output::info("    action=\"search\",");
    output::info(&format!("    query=\"{}\",", query));
    if let Some(s) = &scope {
        if s == "global" {
            output::info("    scope=\"global\",");
        } else if s.starts_with("project:") {
            output::info(&format!(
                "    project=\"{}\",",
                s.strip_prefix("project:").unwrap_or("")
            ));
        }
    }
    output::info(&format!("    limit={}", limit));
    output::info("  )");

    Ok(())
}

// ─── Scope management ─────────────────────────────────────────────────────

async fn manage_scopes(list: bool, show: Option<String>, verbose: bool) -> Result<()> {
    output::section("Memory Rule Scopes");

    if list || show.is_none() {
        // List available scopes
        output::info("Available scope types:");
        output::separator();

        output::kv("global", "Rules apply to all projects");
        output::kv("project:<id>", "Rules apply to a specific project");
        output::kv("branch:<name>", "Rules apply to a specific branch");
        output::separator();

        output::info("Scope hierarchy (highest to lowest priority):");
        output::info("  1. branch:* - Branch-specific rules");
        output::info("  2. project:* - Project-specific rules");
        output::info("  3. global - Global rules");
        output::separator();

        // Try to get projects from daemon to show active scopes
        match DaemonClient::connect_default().await {
            Ok(mut client) => {
                output::status_line("Daemon", ServiceStatus::Healthy);
                output::separator();

                output::info("Active project scopes:");
                match client.system().get_status(()).await {
                    Ok(response) => {
                        let status = response.into_inner();
                        if status.active_projects.is_empty() {
                            output::info("  (no active projects)");
                        } else {
                            for project_id in &status.active_projects {
                                output::kv("  project", project_id);
                            }
                        }
                    }
                    Err(e) => {
                        output::warning(format!("Could not get projects: {}", e));
                    }
                }
                let _ = verbose; // Verbose flag reserved for future detailed output
            }
            Err(_) => {
                output::status_line("Daemon", ServiceStatus::Unhealthy);
                output::warning("Cannot list active scopes without daemon");
            }
        }
    }

    if let Some(scope_name) = show {
        output::separator();
        output::kv("Showing scope", &scope_name);
        output::separator();

        // Show rules for the specified scope
        output::info("Rules in this scope:");
        output::info("  (Query daemon for scope-specific rules)");
        output::separator();

        output::info("MCP command to list rules for this scope:");
        if scope_name == "global" {
            output::info("  mcp__workspace_qdrant__memory(action=\"list\", scope=\"global\")");
        } else if scope_name.starts_with("project:") {
            let project = scope_name.strip_prefix("project:").unwrap_or(&scope_name);
            output::info(&format!(
                "  mcp__workspace_qdrant__memory(action=\"list\", project=\"{}\")",
                project
            ));
        } else {
            output::info(&format!(
                "  mcp__workspace_qdrant__memory(action=\"list\", scope=\"{}\")",
                scope_name
            ));
        }
    }

    Ok(())
}

// ─── Inject (SessionStart hook) ────────────────────────────────────────────

/// Fetch memory rules from Qdrant via scroll API with a scope filter.
/// Returns payload values for matching points. Empty vec on any failure.
async fn fetch_rules_by_scope(
    client: &reqwest::Client,
    base_url: &str,
    filter: serde_json::Value,
) -> Vec<serde_json::Value> {
    let url = format!(
        "{}/collections/{}/points/scroll",
        base_url,
        wqm_common::constants::COLLECTION_MEMORY,
    );
    let body = serde_json::json!({
        "limit": 100,
        "with_payload": true,
        "filter": filter,
    });

    let resp = match client.post(&url).json(&body).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => return Vec::new(),
    };

    let scroll: ScrollResponse = match resp.json().await {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    scroll
        .result
        .points
        .into_iter()
        .filter_map(|p| p.payload)
        .collect()
}

/// Format fetched rules into the output block for Claude Code context injection.
fn format_inject_output(
    global_rules: &[serde_json::Value],
    project_rules: &[serde_json::Value],
    project_name: Option<&str>,
) -> String {
    if global_rules.is_empty() && project_rules.is_empty() {
        return String::new();
    }

    let mut out = String::from("<workspace-qdrant-memory>\n");

    if !global_rules.is_empty() {
        out.push_str("## Global Rules\n");
        for payload in global_rules {
            let label = payload_str(payload, mem_schema::LABEL.name);
            let content = payload_str(payload, mem_schema::CONTENT.name);
            out.push_str(&format!("- **{}**: {}\n", label, content));
        }
    }

    if !project_rules.is_empty() {
        if !global_rules.is_empty() {
            out.push('\n');
        }
        let header = match project_name {
            Some(name) => format!("## Project Rules ({})\n", name),
            None => "## Project Rules\n".to_string(),
        };
        out.push_str(&header);
        for payload in project_rules {
            let label = payload_str(payload, mem_schema::LABEL.name);
            let content = payload_str(payload, mem_schema::CONTENT.name);
            out.push_str(&format!("- **{}**: {}\n", label, content));
        }
    }

    out.push_str("</workspace-qdrant-memory>");
    out
}

/// SessionStart hook entry point.
///
/// Reads JSON from stdin (contains `cwd`), resolves to a project,
/// fetches global + project rules from Qdrant, prints formatted output.
/// Always exits 0 — failures produce no output.
async fn inject_rules() -> Result<()> {
    use std::io::Read;

    // Initialize tracing to stderr (visible in Claude Code verbose mode)
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .try_init();

    // Read stdin (non-blocking: if empty/closed, we get "")
    let mut raw_input = String::new();
    let _ = std::io::stdin().read_to_string(&mut raw_input);

    tracing::info!("inject input: {}", raw_input.trim());

    // Parse JSON and extract cwd
    let cwd = match serde_json::from_str::<serde_json::Value>(&raw_input) {
        Ok(v) => v
            .get("cwd")
            .and_then(|c| c.as_str())
            .map(String::from),
        Err(_) => None,
    };

    let cwd = match cwd {
        Some(c) => c,
        None => return Ok(()), // No valid cwd → exit silently
    };

    let cwd_path = std::path::PathBuf::from(&cwd);

    // Resolve project from watch_folders
    let db_path = match crate::config::get_database_path_checked() {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };

    let project_info =
        wqm_common::project_id::resolve_path_to_project(&db_path, &cwd_path);

    // Build Qdrant client
    let client = match build_qdrant_client() {
        Ok(c) => c,
        Err(_) => return Ok(()),
    };
    let base_url = qdrant_url();

    // Fetch global rules
    let global_filter = build_scope_filter("global");
    let global_rules = fetch_rules_by_scope(&client, &base_url, global_filter).await;

    // Fetch project rules if project resolved
    let (project_rules, project_name) = match &project_info {
        Some((tenant_id, path)) => {
            let filter = build_scope_filter(&format!("project:{}", tenant_id));
            let rules = fetch_rules_by_scope(&client, &base_url, filter).await;
            let name = Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string());
            (rules, name)
        }
        None => (Vec::new(), None),
    };

    // Format and output
    let output = format_inject_output(
        &global_rules,
        &project_rules,
        project_name.as_deref(),
    );

    if !output.is_empty() {
        tracing::info!("inject output: {}", output);
        print!("{}", output);
    }

    Ok(())
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_scope_global() {
        let scope = resolve_scope(true, None);
        assert_eq!(scope, Some("global".to_string()));
    }

    #[test]
    fn test_resolve_scope_project() {
        let scope = resolve_scope(false, Some("abc123".to_string()));
        assert_eq!(scope, Some("project:abc123".to_string()));
    }

    #[test]
    fn test_resolve_scope_none() {
        let scope = resolve_scope(false, None);
        assert_eq!(scope, None);
    }

    #[test]
    fn test_build_scope_filter_global() {
        let filter = build_scope_filter("global");
        let must = filter["must"].as_array().unwrap();
        assert_eq!(must.len(), 1);
        assert_eq!(must[0]["key"], "scope");
        assert_eq!(must[0]["match"]["value"], "global");
    }

    #[test]
    fn test_build_scope_filter_project() {
        let filter = build_scope_filter("project:abc123");
        let must = filter["must"].as_array().unwrap();
        assert_eq!(must.len(), 2);
        assert_eq!(must[0]["key"], "scope");
        assert_eq!(must[0]["match"]["value"], "project");
        assert_eq!(must[1]["key"], "project_id");
        assert_eq!(must[1]["match"]["value"], "abc123");
    }

    #[test]
    fn test_payload_str() {
        let payload = serde_json::json!({
            "label": "test-label",
            "missing": null,
        });
        assert_eq!(payload_str(&payload, "label"), "test-label");
        assert_eq!(payload_str(&payload, "nonexistent"), "");
        assert_eq!(payload_str(&payload, "missing"), "");
    }

    #[test]
    fn test_payload_u32() {
        let payload = serde_json::json!({
            "priority": 8,
            "zero": 0,
        });
        assert_eq!(payload_u32(&payload, "priority"), Some(8));
        assert_eq!(payload_u32(&payload, "zero"), Some(0));
        assert_eq!(payload_u32(&payload, "missing"), None);
    }

    #[test]
    fn test_scroll_response_deserialization() {
        let json = r#"{
            "result": {
                "points": [
                    {
                        "id": "abc-123",
                        "payload": {
                            "content": "test rule",
                            "label": "test-label",
                            "scope": "global",
                            "priority": 5,
                            "source_type": "memory_rule",
                            "created_at": "2026-02-12T10:00:00.000Z"
                        }
                    }
                ],
                "next_page_offset": null
            },
            "status": "ok",
            "time": 0.001
        }"#;

        let response: ScrollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.result.points.len(), 1);

        let payload = response.result.points[0].payload.as_ref().unwrap();
        assert_eq!(payload_str(payload, "label"), "test-label");
        assert_eq!(payload_str(payload, "content"), "test rule");
        assert_eq!(payload_u32(payload, "priority"), Some(5));
    }

    #[test]
    fn test_memory_rule_json_serialization() {
        let rule = MemoryRuleJson {
            label: "test".to_string(),
            title: "Test Rule".to_string(),
            content: "test content".to_string(),
            scope: "global".to_string(),
            project_id: None,
            source_type: "memory_rule".to_string(),
            priority: Some(5),
            tags: vec!["tag1".to_string(), "tag2".to_string()],
            created_at: "2026-02-12T10:00:00.000Z".to_string(),
            updated_at: "2026-02-12T10:00:00.000Z".to_string(),
        };
        let json = serde_json::to_string(&rule).unwrap();
        assert!(json.contains("\"label\":\"test\""));
        assert!(json.contains("\"priority\":5"));
        assert!(json.contains("tag1"));
    }

    #[test]
    fn test_normalize_commas_adds_spaces() {
        assert_eq!(normalize_commas("a,b,c"), "a, b, c");
        assert_eq!(normalize_commas("one,two,three"), "one, two, three");
    }

    #[test]
    fn test_normalize_commas_preserves_existing_spaces() {
        assert_eq!(normalize_commas("a, b, c"), "a, b, c");
        assert_eq!(normalize_commas("a,b, c"), "a, b, c");
    }

    #[test]
    fn test_normalize_commas_empty() {
        assert_eq!(normalize_commas(""), "");
        assert_eq!(normalize_commas("no-commas"), "no-commas");
    }

    #[test]
    fn test_format_title_global_scope() {
        let payload = serde_json::json!({
            "title": "Some Rule",
            "scope": "global",
        });
        let names = HashMap::new();
        assert_eq!(format_title_with_project(&payload, &names, false), "Some Rule");
        assert_eq!(format_title_with_project(&payload, &names, true), "Some Rule");
    }

    #[test]
    fn test_format_title_project_scope_with_name() {
        let payload = serde_json::json!({
            "title": "My Rule",
            "scope": "project",
            "project_id": "abc123",
        });
        let mut names = HashMap::new();
        names.insert("abc123".to_string(), "my-project".to_string());

        assert_eq!(
            format_title_with_project(&payload, &names, false),
            "My Rule (project: my-project)"
        );
        assert_eq!(
            format_title_with_project(&payload, &names, true),
            "My Rule (project: my-project / abc123)"
        );
    }

    #[test]
    fn test_format_title_project_scope_unknown_id() {
        let payload = serde_json::json!({
            "title": "My Rule",
            "scope": "project",
            "project_id": "unknown999",
        });
        let names = HashMap::new();
        // Falls back to showing just the tenant_id
        assert_eq!(
            format_title_with_project(&payload, &names, false),
            "My Rule (project: unknown999)"
        );
        assert_eq!(
            format_title_with_project(&payload, &names, true),
            "My Rule (project: unknown999)"
        );
    }

    // ─── inject format tests ────────────────────────────────────────────

    #[test]
    fn test_inject_format_output() {
        let global = vec![serde_json::json!({
            "label": "always-test",
            "content": "Always run tests before committing",
        })];
        let project = vec![serde_json::json!({
            "label": "use-tokio",
            "content": "Use tokio for async runtime",
        })];

        let output = format_inject_output(&global, &project, Some("my-project"));
        assert!(output.starts_with("<workspace-qdrant-memory>"));
        assert!(output.ends_with("</workspace-qdrant-memory>"));
        assert!(output.contains("## Global Rules"));
        assert!(output.contains("- **always-test**: Always run tests before committing"));
        assert!(output.contains("## Project Rules (my-project)"));
        assert!(output.contains("- **use-tokio**: Use tokio for async runtime"));
    }

    #[test]
    fn test_inject_format_global_only() {
        let global = vec![serde_json::json!({
            "label": "be-concise",
            "content": "Keep responses short",
        })];
        let project: Vec<serde_json::Value> = vec![];

        let output = format_inject_output(&global, &project, None);
        assert!(output.contains("## Global Rules"));
        assert!(!output.contains("## Project Rules"));
        assert!(output.contains("- **be-concise**: Keep responses short"));
    }

    #[test]
    fn test_inject_format_empty() {
        let global: Vec<serde_json::Value> = vec![];
        let project: Vec<serde_json::Value> = vec![];

        let output = format_inject_output(&global, &project, None);
        assert!(output.is_empty());
    }
}
