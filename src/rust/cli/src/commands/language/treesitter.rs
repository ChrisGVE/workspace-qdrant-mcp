//! `language ts-install`, `language ts-remove`, and `language ts-search` subcommands
//!
//! The tree-sitter grammar engine (download / compile / dlopen + on-disk cache)
//! lives in the daemon (WI-e1/e2, #82). These commands drive it over the
//! `LanguageService` gRPC surface via the shared `wqm_client::DaemonClient`;
//! registry display data (repos, quality, patterns) is read locally from
//! `wqm-common` with no `workspace-qdrant-core` link.

use anyhow::{anyhow, Result};
use colored::{ColoredString, Colorize};

use crate::output;

use super::helpers::{
    find_language, load_definitions, try_grammar_cached_and_status, try_grammar_status,
};

/// Map a daemon `GrammarStatus` string to the CLI's coloured status label.
fn grammar_status_label(status: &str) -> ColoredString {
    match status {
        "loaded" => "Loaded".green(),
        "cached" => "Cached".blue(),
        "needs_download" => "Available".yellow(),
        "incompatible_version" => "Incompat".red(),
        "not_available" => "N/A".dimmed(),
        other => other.dimmed(),
    }
}

/// Install a Tree-sitter grammar via the daemon's LanguageService.
pub async fn ts_install(language: &str, force: bool) -> Result<()> {
    output::section(format!("Installing Tree-sitter Grammar: {}", language));

    let mut client = crate::grpc::ensure_daemon_available().await?;

    // Already cached? (skip unless --force) — mirror the prior local guard.
    if !force {
        if let Ok(q) = client.query_language(language.to_string()).await {
            if q.found && matches!(q.grammar_status.as_str(), "cached" | "loaded") {
                output::warning("Grammar already cached. Use --force to reinstall.");
                return Ok(());
            }
        }
    }

    output::info("Downloading grammar...");
    match client.install_grammar(language.to_string(), force).await {
        Ok(resp) => {
            output::success("Grammar installed successfully");
            output::kv("Status", &resp.status);
        }
        Err(status) => {
            return Err(anyhow!("Failed to install grammar: {}", status.message()));
        }
    }

    Ok(())
}

/// Remove a Tree-sitter grammar (or all) via the daemon's LanguageService.
pub async fn ts_remove(language: &str) -> Result<()> {
    output::section(format!("Removing Tree-sitter Grammar: {}", language));

    let mut client = crate::grpc::ensure_daemon_available().await?;

    if language == "all" {
        let listed = client
            .list_grammars()
            .await
            .map_err(|s| anyhow!("Failed to list grammars: {}", s.message()))?;
        let mut count = 0usize;
        for lang in listed.cached {
            if let Ok(resp) = client.remove_grammar(lang).await {
                if resp.removed {
                    count += 1;
                }
            }
        }
        output::success(format!("Removed {} grammar(s) from cache", count));
    } else {
        match client.remove_grammar(language.to_string()).await {
            Ok(resp) if resp.removed => output::success("Grammar removed from cache"),
            Ok(_) => output::warning("Grammar not found in cache"),
            Err(s) => return Err(anyhow!("Failed to remove grammar: {}", s.message())),
        }
    }

    Ok(())
}

/// List Tree-sitter grammars with cache and registry status.
pub async fn ts_list(show_all: bool) -> Result<()> {
    output::section("Tree-sitter Grammars");

    // Best-effort: live cache status when the daemon is up; registry-only otherwise.
    let (cached, status_map) = try_grammar_cached_and_status().await;

    let mut defs = load_definitions();
    defs.sort_by_key(|d| d.language.to_lowercase());

    println!(
        "  {:<16} {:<42} {:<10} {}",
        "Language".bold(),
        "Repository".bold(),
        "Status".bold(),
        "Patterns".bold()
    );
    println!("  {}", "─".repeat(78).dimmed());

    let mut shown = 0;
    let mut cached_count = 0;

    for def in &defs {
        if !def.has_grammar() {
            continue;
        }

        let lang_id = def.id();
        let is_cached = cached.contains(&lang_id);

        if !show_all && !is_cached {
            continue;
        }

        let status = status_map
            .get(&lang_id)
            .map(String::as_str)
            .unwrap_or("not_available");
        let status_str = grammar_status_label(status);

        let repo = def
            .grammar
            .sources
            .first()
            .map(|s| s.repo.as_str())
            .unwrap_or("—");

        let has_patterns = if def.has_semantic_patterns() {
            "✓".green().to_string()
        } else {
            "—".dimmed().to_string()
        };

        println!(
            "  {:<16} {:<42} {:<10} {}",
            def.language, repo, status_str, has_patterns
        );

        shown += 1;
        if is_cached {
            cached_count += 1;
        }
    }

    println!();
    if show_all {
        output::kv("Cached", format!("{cached_count}/{shown}"));
    } else {
        output::kv("Cached", format!("{cached_count}"));
        output::info("Use --all to show available but uncached grammars.");
    }

    Ok(())
}

/// Search available Tree-sitter grammars for a language.
pub async fn ts_search(language: &str) -> Result<()> {
    use wqm_common::language_registry::types::GrammarQuality;

    output::section(format!("Grammar Search: {}", language));

    let def = match find_language(language) {
        Some(d) => d,
        None => {
            output::warning(format!("Unknown language: {language}"));
            output::info("Use 'wqm language list' to see all known languages.");
            return Ok(());
        }
    };

    let status = try_grammar_status(language).await;

    output::kv("Language", &def.language);
    output::kv("Status", &status);
    output::separator();

    if def.grammar.sources.is_empty() {
        output::info("No grammar sources found in registry.");
        return Ok(());
    }

    println!(
        "  {:<45} {:<10} {}",
        "Repository".bold(),
        "Quality".bold(),
        "Origin".bold()
    );
    println!("  {}", "─".repeat(70).dimmed());

    for src in &def.grammar.sources {
        let quality = format!("{:?}", src.quality);
        let quality_colored = match src.quality {
            GrammarQuality::Curated => quality.green(),
            GrammarQuality::Official => quality.blue(),
            GrammarQuality::Community => quality.yellow(),
        };
        let origin = src.origin.as_deref().unwrap_or("unknown");
        println!("  {:<45} {:<10} {}", src.repo, quality_colored, origin);
    }

    if def.grammar.has_cpp_scanner {
        println!();
        output::info("Note: This grammar requires a C++ compiler for its scanner.");
    }

    println!();
    output::info(format!("Install with: wqm language ts-install {language}"));

    Ok(())
}
