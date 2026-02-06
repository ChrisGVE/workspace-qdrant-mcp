//! Memory command - LLM rules management
//!
//! Manages LLM memory rules stored in the `memory` collection.
//! Subcommands: list, add, remove, search, scope

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{DeleteDocumentRequest, IngestTextRequest};
use crate::output::{self, ServiceStatus};
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

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
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
        } => {
            let scope = resolve_scope(global, project);
            list_rules(scope, rule_type, verbose).await
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

async fn list_rules(
    scope: Option<String>,
    rule_type: Option<String>,
    verbose: bool,
) -> Result<()> {
    output::section("Memory Rules");

    if let Some(s) = &scope {
        output::kv("Scope", s);
    } else {
        output::kv("Scope", "all (no filter)");
    }
    if let Some(t) = &rule_type {
        output::kv("Type Filter", t);
    }
    output::separator();

    // Memory rules are stored in canonical `memory` collection
    output::info("Memory rules stored in memory collection.");
    output::info("Query via MCP:");
    output::info("  mcp__workspace_qdrant__memory(action=\"list\")");
    output::separator();

    output::info("Direct Qdrant query:");
    output::info("  curl 'http://localhost:6333/collections/memory/points/scroll' \\");
    output::info("    -H 'Content-Type: application/json' \\");
    output::info("    -d '{\"limit\": 100}'");

    if verbose {
        output::separator();
        output::info("Rule types: preference, behavior, constraint, pattern");
        output::info("Scopes: global (--global), project (--project <path>)");
    }

    Ok(())
}

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
