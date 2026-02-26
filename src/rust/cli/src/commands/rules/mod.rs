//! Rules command - behavioral rules management
//!
//! Manages behavioral rules stored in the `rules` collection.
//! Subcommands: list, add, remove, search, scope, inject

mod add;
pub mod helpers;
mod inject;
mod list;
mod remove;
mod scope;
mod search;

use anyhow::Result;
use clap::{Args, Subcommand};

/// Rules command arguments
#[derive(Args)]
pub struct RulesArgs {
    #[command(subcommand)]
    command: RulesCommand,
}

/// Rules subcommands
#[derive(Subcommand)]
enum RulesCommand {
    /// List behavioral rules
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

    /// Add a new rule
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

    /// Remove a rule
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

    /// Search rules
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

    /// Inject rules into Claude Code context (SessionStart hook)
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

/// Execute rules command
pub async fn execute(args: RulesArgs) -> Result<()> {
    match args.command {
        RulesCommand::List {
            global,
            project,
            rule_type,
            verbose,
            format,
        } => {
            let scope = resolve_scope(global, project);
            list::list_rules(scope, rule_type, verbose, &format).await
        }
        RulesCommand::Add {
            label,
            content,
            global,
            project,
            rule_type,
            priority,
        } => {
            let scope = resolve_scope(global, project);
            add::add_rule(&label, &content, &rule_type, &scope, priority).await
        }
        RulesCommand::Remove {
            label,
            global,
            project,
        } => {
            let scope = resolve_scope(global, project);
            remove::remove_rule(&label, &scope).await
        }
        RulesCommand::Search {
            query,
            global,
            project,
            limit,
        } => {
            let scope = resolve_scope(global, project);
            search::search_rules(&query, scope, limit).await
        }
        RulesCommand::Inject => inject::inject_rules().await,
        RulesCommand::Scope {
            list,
            show,
            verbose,
        } => scope::manage_scopes(list, show, verbose).await,
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
}
