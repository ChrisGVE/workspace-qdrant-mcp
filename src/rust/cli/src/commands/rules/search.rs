//! Search rules subcommand
//!
//! Displays MCP search instructions for querying rules.

use anyhow::Result;

use crate::output;

/// Show instructions for searching rules via MCP.
pub async fn search_rules(query: &str, scope: Option<String>, limit: usize) -> Result<()> {
    output::section("Search Rules");

    output::kv("Query", query);
    if let Some(s) = &scope {
        output::kv("Scope", s);
    } else {
        output::kv("Scope", "all");
    }
    output::kv("Limit", limit.to_string());
    output::separator();

    output::info("Rules search via MCP:");
    output::info("  mcp__workspace_qdrant__rules(");
    output::info("    action=\"search\",");
    output::info(format!("    query=\"{}\",", query));
    if let Some(s) = &scope {
        if s == "global" {
            output::info("    scope=\"global\",");
        } else if s.starts_with("project:") {
            output::info(format!(
                "    project=\"{}\",",
                s.strip_prefix("project:").unwrap_or("")
            ));
        }
    }
    output::info(format!("    limit={}", limit));
    output::info("  )");

    Ok(())
}
