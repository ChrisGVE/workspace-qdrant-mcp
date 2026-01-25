//! Memory command - LLM rules management
//!
//! Phase 2 MEDIUM priority command for LLM memory rules.
//! Subcommands: list, add, remove, search
//!
//! Memory rules are stored in the `memory` collection and guide LLM behavior.

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::IngestTextRequest;
use crate::output;

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
        /// Filter by scope (global, project, language)
        #[arg(short, long)]
        scope: Option<String>,

        /// Filter by type (preference, behavior, constraint, pattern)
        #[arg(short = 't', long)]
        rule_type: Option<String>,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Add a new memory rule
    Add {
        /// Rule content
        content: String,

        /// Rule type (preference, behavior, constraint, pattern)
        #[arg(short = 't', long, default_value = "preference")]
        rule_type: String,

        /// Rule scope (global, project, language)
        #[arg(short, long, default_value = "global")]
        scope: String,

        /// Priority (1-10, higher = more important)
        #[arg(short, long, default_value = "5")]
        priority: u32,
    },

    /// Remove a memory rule
    Remove {
        /// Rule ID to remove
        rule_id: String,
    },

    /// Search memory rules
    Search {
        /// Search query
        query: String,

        /// Maximum results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },
}

/// Execute memory command
pub async fn execute(args: MemoryArgs) -> Result<()> {
    match args.command {
        MemoryCommand::List {
            scope,
            rule_type,
            verbose,
        } => list_rules(scope, rule_type, verbose).await,
        MemoryCommand::Add {
            content,
            rule_type,
            scope,
            priority,
        } => add_rule(&content, &rule_type, &scope, priority).await,
        MemoryCommand::Remove { rule_id } => remove_rule(&rule_id).await,
        MemoryCommand::Search { query, limit } => search_rules(&query, limit).await,
    }
}

async fn list_rules(
    scope: Option<String>,
    rule_type: Option<String>,
    verbose: bool,
) -> Result<()> {
    output::section("Memory Rules");

    if let Some(s) = &scope {
        output::kv("Scope Filter", s);
    }
    if let Some(t) = &rule_type {
        output::kv("Type Filter", t);
    }
    output::separator();

    // Memory rules are stored in canonical `memory` collection
    output::info("Memory rules stored in memory collection.");
    output::info("Query via MCP:");
    output::info("  mcp__workspace_qdrant__search(scope=\"memory\", query=\"*\")");
    output::separator();

    output::info("Direct Qdrant query:");
    let mut filter = String::new();
    if let Some(s) = scope {
        filter.push_str(&format!("scope={}", s));
    }
    if let Some(t) = rule_type {
        if !filter.is_empty() {
            filter.push_str("&");
        }
        filter.push_str(&format!("rule_type={}", t));
    }

    output::info("  curl 'http://localhost:6333/collections/memory/points/scroll' \\");
    output::info("    -H 'Content-Type: application/json' \\");
    output::info("    -d '{\"limit\": 100}'");

    if verbose {
        output::separator();
        output::info("Rule types: preference, behavior, constraint, pattern");
        output::info("Scopes: global, project, language");
    }

    Ok(())
}

async fn add_rule(content: &str, rule_type: &str, scope: &str, priority: u32) -> Result<()> {
    output::section("Add Memory Rule");

    output::kv("Content", content);
    output::kv("Type", rule_type);
    output::kv("Scope", scope);
    output::kv("Priority", &priority.to_string());
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("rule_type".to_string(), rule_type.to_string());
            metadata.insert("scope".to_string(), scope.to_string());
            metadata.insert("priority".to_string(), priority.to_string());
            metadata.insert("enabled".to_string(), "true".to_string());

            let request = IngestTextRequest {
                content: content.to_string(),
                collection_basename: "memory".to_string(),
                tenant_id: String::new(),
                document_id: None,
                metadata,
                chunk_text: false, // Don't chunk rules
            };

            match client.document().ingest_text(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    if result.success {
                        output::success("Memory rule added");
                        output::kv("Rule ID", &result.document_id);
                    } else {
                        output::error(format!("Failed to add rule: {}", result.error_message));
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to add rule: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}

async fn remove_rule(rule_id: &str) -> Result<()> {
    output::section("Remove Memory Rule");

    output::kv("Rule ID", rule_id);
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = crate::grpc::proto::DeleteTextRequest {
                document_id: rule_id.to_string(),
                collection_name: "memory".to_string(),
            };

            match client.document().delete_text(request).await {
                Ok(_) => {
                    output::success("Memory rule removed");
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

async fn search_rules(query: &str, limit: usize) -> Result<()> {
    output::section("Search Memory Rules");

    output::kv("Query", query);
    output::kv("Limit", &limit.to_string());
    output::separator();

    output::info("Memory search via MCP:");
    output::info("  mcp__workspace_qdrant__search(");
    output::info(&format!("    query=\"{}\",", query));
    output::info("    scope=\"memory\",");
    output::info(&format!("    limit={}", limit));
    output::info("  )");

    Ok(())
}
