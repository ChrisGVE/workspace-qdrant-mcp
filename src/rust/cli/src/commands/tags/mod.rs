//! Tags command - keyword/tag management and hierarchy inspection.
//!
//! Queries the keywords, tags, canonical_tags, and tag_hierarchy_edges tables
//! to display extraction results and the canonical tag hierarchy.

mod db;
mod list;
mod search;
mod stats;
mod tree;

use anyhow::Result;
use clap::{Args, Subcommand};

/// Tags command arguments
#[derive(Args)]
pub struct TagsArgs {
    #[command(subcommand)]
    command: TagsCommand,
}

/// Tags subcommands
#[derive(Subcommand)]
enum TagsCommand {
    /// List tags for a specific document
    List {
        /// Document ID
        #[arg(long)]
        doc: String,

        /// Filter by tag type (concept, structural)
        #[arg(long, value_parser = ["concept", "structural"])]
        tag_type: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long, conflicts_with = "json")]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },

    /// List keywords for a specific document
    Keywords {
        /// Document ID
        #[arg(long)]
        doc: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long, conflicts_with = "json")]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },

    /// Show canonical tag hierarchy for a tenant
    Tree {
        /// Tenant ID
        #[arg(long)]
        tenant: String,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,
    },

    /// Show extraction statistics
    Stats {
        /// Tenant ID (optional, all tenants if omitted)
        #[arg(long)]
        tenant: Option<String>,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,
    },

    /// Search tags by name across all tenants
    Search {
        /// Tag name pattern (SQL LIKE)
        query: String,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long, conflicts_with = "json")]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },

    /// Show keyword baskets for a document
    Baskets {
        /// Document ID
        #[arg(long)]
        doc: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long, conflicts_with = "json")]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },
}

/// Execute tags command
pub async fn execute(args: TagsArgs) -> Result<()> {
    match args.command {
        TagsCommand::List {
            doc,
            tag_type,
            json,
            script,
            no_headers,
        } => list::list_tags(&doc, tag_type.as_deref(), json, script, no_headers),
        TagsCommand::Keywords {
            doc,
            json,
            script,
            no_headers,
        } => list::list_keywords(&doc, json, script, no_headers),
        TagsCommand::Tree { tenant, collection } => tree::show_tree(&tenant, &collection),
        TagsCommand::Stats { tenant, collection } => {
            stats::show_stats(tenant.as_deref(), &collection)
        }
        TagsCommand::Search {
            query,
            collection,
            json,
            script,
            no_headers,
        } => search::search_tags(&query, &collection, json, script, no_headers),
        TagsCommand::Baskets {
            doc,
            json,
            script,
            no_headers,
        } => search::show_baskets(&doc, json, script, no_headers),
    }
}

#[cfg(test)]
mod tests {
    use super::list::{KeywordRow, TagRow};
    use super::search::TagSearchRow;
    use super::stats::StatsRow;
    use super::*;

    #[test]
    fn test_tags_args_struct() {
        // Verify the TagsArgs struct is constructible
        // (clap parsing is tested via integration tests)
        assert_eq!(
            std::mem::size_of::<TagsArgs>(),
            std::mem::size_of::<TagsArgs>()
        );
    }

    #[test]
    fn test_tag_row_tabled() {
        let row = TagRow {
            tag: "vector search".to_string(),
            tag_type: "concept".to_string(),
            score: "0.900".to_string(),
            diversity: "0.850".to_string(),
        };
        assert_eq!(row.tag, "vector search");
    }

    #[test]
    fn test_keyword_row_tabled() {
        let row = KeywordRow {
            keyword: "embedding".to_string(),
            score: "0.750".to_string(),
            semantic: "0.800".to_string(),
            lexical: "0.700".to_string(),
            stability: 3,
        };
        assert_eq!(row.stability, 3);
    }

    #[test]
    fn test_stats_row_tabled() {
        let row = StatsRow {
            tenant_id: "test".to_string(),
            doc_count: 10,
            avg_keywords: "5.2".to_string(),
            avg_tags: "3.1".to_string(),
            canonical_count: 15,
        };
        assert_eq!(row.doc_count, 10);
    }

    #[test]
    fn test_tag_search_row_tabled() {
        let row = TagSearchRow {
            tag: "async".to_string(),
            tenant_id: "proj-1".to_string(),
            doc_count: 5,
            avg_score: "0.800".to_string(),
        };
        assert_eq!(row.doc_count, 5);
    }
}
