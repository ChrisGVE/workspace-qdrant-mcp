//! Inject rules subcommand (SessionStart hook)
//!
//! Reads JSON from stdin, resolves the working directory to a project,
//! fetches global + project rules from Qdrant, and prints formatted
//! output for Claude Code context injection. Always exits 0.

use std::path::Path;

use anyhow::Result;

use wqm_common::schema::qdrant::rules as rules_schema;

use super::helpers::{
    build_qdrant_client, build_scope_filter, payload_str, qdrant_url, ScrollResponse,
};

/// SessionStart hook entry point.
///
/// Reads JSON from stdin (contains `cwd`), resolves to a project,
/// fetches global + project rules from Qdrant, prints formatted output.
/// Always exits 0 -- failures produce no output.
pub async fn inject_rules() -> Result<()> {
    use std::io::{IsTerminal, Read};

    // When run interactively from a terminal, exit silently.
    if std::io::stdin().is_terminal() {
        return Ok(());
    }

    // Initialize tracing to stderr (visible in Claude Code verbose mode)
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .try_init();

    // Read piped stdin
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
        None => return Ok(()), // No valid cwd -> exit silently
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

/// Fetch rules from Qdrant via scroll API with a scope filter.
/// Returns payload values for matching points. Empty vec on any failure.
async fn fetch_rules_by_scope(
    client: &reqwest::Client,
    base_url: &str,
    filter: serde_json::Value,
) -> Vec<serde_json::Value> {
    let url = format!(
        "{}/collections/{}/points/scroll",
        base_url,
        wqm_common::constants::COLLECTION_RULES,
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

    let mut out = String::from("<workspace-qdrant-rules>\n");

    if !global_rules.is_empty() {
        out.push_str("## Global Rules\n");
        for payload in global_rules {
            let label = payload_str(payload, rules_schema::LABEL.name);
            let content = payload_str(payload, rules_schema::CONTENT.name);
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
            let label = payload_str(payload, rules_schema::LABEL.name);
            let content = payload_str(payload, rules_schema::CONTENT.name);
            out.push_str(&format!("- **{}**: {}\n", label, content));
        }
    }

    out.push_str("</workspace-qdrant-rules>");
    out
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(output.starts_with("<workspace-qdrant-rules>"));
        assert!(output.ends_with("</workspace-qdrant-rules>"));
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
