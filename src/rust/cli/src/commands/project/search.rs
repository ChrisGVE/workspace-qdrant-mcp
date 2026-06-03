//! Project search subcommand
//!
//! Text search via daemon's TextSearchService. Searches across all indexed
//! files in the current project.

use anyhow::Result;
use tabled::Tabled;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::TextSearchRequest;
use crate::output::style::home_to_tilde;
use crate::output::{self, ColumnHints};

use super::resolver::resolve_project_id_or_cwd;

/// Search result row for table display
#[derive(Tabled)]
struct SearchResultRow {
    #[tabled(rename = "File")]
    file: String,
    #[tabled(rename = "Line")]
    line: String,
    #[tabled(rename = "Content")]
    content: String,
}

impl ColumnHints for SearchResultRow {
    fn content_columns() -> &'static [usize] {
        &[2] // Content column
    }
}

/// Execute project text search via daemon gRPC.
pub async fn search_project(
    query: &str,
    regex: bool,
    case_sensitive: bool,
    path_glob: Option<String>,
    limit: usize,
    context_lines: u32,
) -> Result<()> {
    let tenant_id = resolve_project_id_or_cwd(None)?;

    let mut client = ensure_daemon_available().await?;

    let request = TextSearchRequest {
        pattern: query.to_string(),
        regex,
        case_sensitive,
        tenant_id: Some(tenant_id.clone()),
        branch: None,
        path_glob,
        path_prefix: None,
        context_lines: context_lines as i32,
        max_results: limit as i32,
    };

    let response = client.text_search_client().search(request).await?.into_inner();

    if response.matches.is_empty() {
        output::info("No matches found.");
        return Ok(());
    }

    output::section("Search Results");
    output::kv("Query", query);
    output::kv("Matches", response.total_matches.to_string());
    if response.truncated {
        output::warning(format!(
            "Results truncated to {} (use -n to increase)",
            limit
        ));
    }
    output::separator();

    let rows: Vec<SearchResultRow> = response
        .matches
        .iter()
        .map(|m| SearchResultRow {
            file: home_to_tilde(&m.file_path),
            line: m.line_number.to_string(),
            content: m.content.trim().to_string(),
        })
        .collect();

    output::print_table_auto(&rows);
    output::summary(format!(
        "{} match(es) in {:.0}ms",
        response.total_matches, response.query_time_ms
    ));

    Ok(())
}
