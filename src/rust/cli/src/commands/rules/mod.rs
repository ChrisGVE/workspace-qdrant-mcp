//! Rules command - behavioral rules management
//!
//! Manages behavioral rules stored in the `rules` collection.
//! Subcommands: list, add, remove, search, scope, inject

mod add;
pub mod helpers;
mod info;
mod inject;
mod list;
mod manage;
mod remove;
mod search;

use anyhow::Result;
use clap::{Args, Subcommand};
use wqm_common::constants::TENANT_GLOBAL;

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
    #[command(
        long_about = "Display all behavioral rules, optionally filtered by scope (global or \
            per-project) and rule type. Rules guide AI assistant behavior and persist \
            across sessions.",
        after_long_help = "Examples:\n  \
            wqm rules list                              List all rules\n  \
            wqm rules list --global                     List global rules only\n  \
            wqm rules list --project .                  List rules for current project\n  \
            wqm rules list -t constraint                Filter by rule type\n  \
            wqm rules list --verbose                    Show full rule content\n  \
            wqm rules list --format json                Output as JSON\n  \
            wqm rules list --script --no-headers        Machine-readable output"
    )]
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

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long)]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },

    /// Add a new rule
    #[command(
        long_about = "Create a new behavioral rule with a label, content, scope, and type. \
            Rules must be scoped to either global (all projects) or a specific project. \
            The label serves as a unique identifier within its scope.",
        after_long_help = "Examples:\n  \
            wqm rules add -l no-emoji -c 'Never use emojis in code' --global\n  \
            wqm rules add -l test-first -c 'Write tests before implementation' -p .\n  \
            wqm rules add -l max-lines -c 'Functions must be under 80 lines' -g -t constraint"
    )]
    Add {
        /// Rule label (identifier for the rule)
        #[arg(short = 'l', long)]
        label: String,

        /// Rule content
        #[arg(short = 'c', long)]
        content: String,

        /// Apply to all projects (global rule) — must specify either --global or --project
        #[arg(
            short = 'g',
            long,
            conflicts_with = "project",
            required_unless_present = "project"
        )]
        global: bool,

        /// Apply to specific project (path or ID) — must specify either --global or --project
        #[arg(
            short = 'p',
            long,
            conflicts_with = "global",
            required_unless_present = "global"
        )]
        project: Option<String>,

        /// Rule type (preference, behavior, constraint, pattern)
        #[arg(short = 't', long, default_value = "preference")]
        rule_type: String,
    },

    /// Remove a rule
    #[command(
        long_about = "Delete a behavioral rule by its label. Specify the scope (global or project) \
            to identify which rule to remove. If neither --global nor --project is given, \
            searches all scopes.",
        after_long_help = "Examples:\n  \
            wqm rules remove --label no-emoji --global       Remove a global rule\n  \
            wqm rules remove --label test-first --project .  Remove a project rule"
    )]
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

        /// Skip the typed "Delete <label>" confirmation
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Reassign a rule to another scope (global <-> project)
    #[command(
        long_about = "Move a rule, identified by label, to a different scope: a specific \
            project (path or tenant id via --to) or the global scope when --to is omitted. \
            Queued as add-under-new-scope + remove-under-old-scope.",
        after_long_help = "Examples:\n  \
            wqm rules reassign --label no-emoji --to .   Move to current project\n  \
            wqm rules reassign --label no-emoji          Move to global scope"
    )]
    Reassign {
        /// Rule label to reassign
        #[arg(long)]
        label: String,

        /// Target project (path or tenant id; omit for global)
        #[arg(long)]
        to: Option<String>,
    },

    /// Update (amend) a rule's content or title
    #[command(
        long_about = "Amend a rule in place, identified by label. Content and/or title can \
            change; label, scope, priority, and tags are preserved.",
        after_long_help = "Examples:\n  \
            wqm rules update --label no-emoji --content 'New rule text'\n  \
            wqm rules update --label no-emoji --title 'No emoji anywhere'"
    )]
    Update {
        /// Rule label to update
        #[arg(long)]
        label: String,

        /// New content
        #[arg(long)]
        content: Option<String>,

        /// New title
        #[arg(long)]
        title: Option<String>,
    },

    /// Show detailed information about a specific rule
    #[command(
        long_about = "Display full details for a specific rule identified by its label, \
            including content, scope, type, and creation timestamp.",
        after_long_help = "Examples:\n  \
            wqm rules info no-emoji                     Show rule details\n  \
            wqm rules info no-emoji --json              Output as JSON"
    )]
    Info {
        /// Rule label to look up
        label: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Search rules
    #[command(
        long_about = "Semantic search across rule content. Returns rules whose content matches \
            the query, ranked by relevance. Optionally filter to global or project scope.",
        after_long_help = "Examples:\n  \
            wqm rules search 'testing'                  Search all rules\n  \
            wqm rules search 'style' --global           Search global rules only\n  \
            wqm rules search 'error handling' -n 5      Limit to 5 results"
    )]
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
            script,
            no_headers,
        } => {
            let scope = resolve_scope(global, project);
            list::list_rules(scope, rule_type, verbose, &format, script, no_headers).await
        }
        RulesCommand::Add {
            label,
            content,
            global,
            project,
            rule_type,
        } => {
            let scope = resolve_scope(global, project);
            add::add_rule(&label, &content, &rule_type, &scope).await
        }
        RulesCommand::Remove {
            label,
            global,
            project,
            yes,
        } => {
            let scope = resolve_scope(global, project);
            remove::remove_rule(&label, &scope, yes).await
        }
        RulesCommand::Reassign { label, to } => manage::reassign_rule(&label, to).await,
        RulesCommand::Update {
            label,
            content,
            title,
        } => manage::update_rule(&label, content, title).await,
        RulesCommand::Info { label, json } => info::rule_info(&label, json).await,
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
    }
}

/// Resolve scope from --global / --project flags
fn resolve_scope(global: bool, project: Option<String>) -> Option<String> {
    if global {
        Some(TENANT_GLOBAL.to_string())
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
