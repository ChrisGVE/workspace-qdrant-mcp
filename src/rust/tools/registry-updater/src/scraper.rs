//! Upstream data source scraper.
//!
//! Orchestrates fetching from all upstream providers in tier order.
//! Handles retries, rate limiting, and error aggregation.

use anyhow::{Context, Result};
use workspace_qdrant_core::language_registry::types::{GrammarEntry, LanguageEntry, LspEntry};

/// Aggregated scraped data from all upstream sources.
#[derive(Debug, Default)]
pub struct ScrapedData {
    /// Language identity entries (from Linguist).
    pub languages: Vec<LanguageEntry>,
    /// Grammar entries from all grammar providers.
    pub grammars: Vec<GrammarEntry>,
    /// LSP server entries from all LSP providers.
    pub lsp_servers: Vec<LspEntry>,
    /// Errors encountered during scraping (non-fatal).
    pub warnings: Vec<String>,
}

/// All available source names for filtering.
const ALL_SOURCES: &[&str] = &[
    "linguist",
    "ts-grammars",
    "nvim-treesitter",
    "mason",
    "microsoft-lsp",
    "langserver-org",
    "ts-wiki",
];

/// Check if a source is enabled by the filter.
fn is_enabled(source: &str, filter: Option<&[String]>) -> bool {
    match filter {
        None => true,
        Some(list) => list.iter().any(|s| s == source),
    }
}

/// Scrape all upstream sources, filtered by the optional source list.
pub async fn scrape_all(
    source_filter: Option<&[String]>,
    github_token: Option<&str>,
) -> Result<ScrapedData> {
    let mut data = ScrapedData::default();

    // ── LANGUAGES (backbone) ────────────────────────────────────────
    if is_enabled("linguist", source_filter) {
        match scrape_linguist().await {
            Ok(languages) => {
                tracing::info!(count = languages.len(), "Linguist: fetched languages");
                data.languages = languages;
            }
            Err(e) => {
                let msg = format!("Linguist scrape failed: {e}");
                tracing::warn!("{msg}");
                data.warnings.push(msg);
            }
        }
    }

    // ── GRAMMARS (tier 1-3) ─────────────────────────────────────────
    // Tier 1: tree-sitter-grammars org (curated)
    if is_enabled("ts-grammars", source_filter) {
        match scrape_ts_grammars_org(github_token).await {
            Ok(grammars) => {
                tracing::info!(count = grammars.len(), "ts-grammars: fetched grammars");
                data.grammars.extend(grammars);
            }
            Err(e) => {
                let msg = format!("ts-grammars scrape failed: {e}");
                tracing::warn!("{msg}");
                data.warnings.push(msg);
            }
        }
    }

    // Tier 2: tree-sitter wiki List-of-parsers
    if is_enabled("ts-wiki", source_filter) {
        match scrape_ts_wiki().await {
            Ok(grammars) => {
                tracing::info!(count = grammars.len(), "ts-wiki: fetched grammars");
                data.grammars.extend(grammars);
            }
            Err(e) => {
                let msg = format!("ts-wiki scrape failed: {e}");
                tracing::warn!("{msg}");
                data.warnings.push(msg);
            }
        }
    }

    // Tier 3: nvim-treesitter lockfile
    if is_enabled("nvim-treesitter", source_filter) {
        match scrape_nvim_treesitter().await {
            Ok(grammars) => {
                tracing::info!(count = grammars.len(), "nvim-treesitter: fetched grammars");
                data.grammars.extend(grammars);
            }
            Err(e) => {
                let msg = format!("nvim-treesitter scrape failed: {e}");
                tracing::warn!("{msg}");
                data.warnings.push(msg);
            }
        }
    }

    // ── LSP SERVERS (tier 1-3) ──────────────────────────────────────
    // Tier 1a: Microsoft LSP implementors
    if is_enabled("microsoft-lsp", source_filter) {
        match scrape_microsoft_lsp().await {
            Ok(servers) => {
                tracing::info!(count = servers.len(), "microsoft-lsp: fetched LSP servers");
                data.lsp_servers.extend(servers);
            }
            Err(e) => {
                let msg = format!("microsoft-lsp scrape failed: {e}");
                tracing::warn!("{msg}");
                data.warnings.push(msg);
            }
        }
    }

    // Tier 1b: langserver.org
    if is_enabled("langserver-org", source_filter) {
        match scrape_langserver_org().await {
            Ok(servers) => {
                tracing::info!(count = servers.len(), "langserver-org: fetched LSP servers");
                data.lsp_servers.extend(servers);
            }
            Err(e) => {
                let msg = format!("langserver-org scrape failed: {e}");
                tracing::warn!("{msg}");
                data.warnings.push(msg);
            }
        }
    }

    // Tier 3: mason-registry
    if is_enabled("mason", source_filter) {
        match scrape_mason().await {
            Ok(servers) => {
                tracing::info!(count = servers.len(), "mason: fetched LSP servers");
                data.lsp_servers.extend(servers);
            }
            Err(e) => {
                let msg = format!("mason scrape failed: {e}");
                tracing::warn!("{msg}");
                data.warnings.push(msg);
            }
        }
    }

    if data.languages.is_empty() && data.grammars.is_empty() && data.lsp_servers.is_empty() {
        anyhow::bail!("All sources failed — no data scraped");
    }

    Ok(data)
}

/// Scrape language identity from GitHub Linguist.
async fn scrape_linguist() -> Result<Vec<LanguageEntry>> {
    use workspace_qdrant_core::language_registry::providers::linguist::LinguistProvider;
    let provider = LinguistProvider::new();
    provider
        .fetch_languages()
        .await
        .context("Linguist fetch_languages")
}

/// Scrape curated grammars from tree-sitter-grammars org.
async fn scrape_ts_grammars_org(github_token: Option<&str>) -> Result<Vec<GrammarEntry>> {
    use workspace_qdrant_core::language_registry::providers::ts_grammars_org::TreeSitterGrammarsOrgProvider;
    // TODO: Pass github_token to provider for authenticated requests
    let _ = github_token;
    let provider = TreeSitterGrammarsOrgProvider::new();
    provider
        .fetch_grammars()
        .await
        .context("ts-grammars fetch_grammars")
}

/// Scrape grammar repos from tree-sitter wiki.
async fn scrape_ts_wiki() -> Result<Vec<GrammarEntry>> {
    use workspace_qdrant_core::language_registry::types::GrammarQuality;

    let url = "https://raw.githubusercontent.com/tree-sitter/tree-sitter/master/docs/src/assets/parsers.json";
    let client = reqwest::Client::builder()
        .user_agent("workspace-qdrant-mcp/registry-updater")
        .build()?;

    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        // Fallback: try the wiki page as markdown
        return scrape_ts_wiki_markdown().await;
    }

    let body = response.text().await?;

    // The parsers.json is an array of objects with "language" and "url" fields
    #[derive(serde::Deserialize)]
    struct WikiParser {
        #[serde(default)]
        language: Option<String>,
        #[serde(default)]
        url: Option<String>,
    }

    let parsers: Vec<WikiParser> = serde_json::from_str(&body).unwrap_or_default();
    let mut entries = Vec::new();

    for parser in parsers {
        let (language, url) = match (parser.language, parser.url) {
            (Some(l), Some(u)) => (l, u),
            _ => continue,
        };

        // Extract owner/repo from GitHub URL
        let repo = extract_github_repo(&url).unwrap_or(url);
        let lang_id = language.to_lowercase().replace(' ', "-");

        entries.push(GrammarEntry {
            language: lang_id,
            repo,
            quality: GrammarQuality::Community,
            has_cpp_scanner: false,
            src_subdir: None,
            symbol_name: None,
            archive_branch: None,
        });
    }

    Ok(entries)
}

/// Fallback: scrape the tree-sitter wiki markdown page.
async fn scrape_ts_wiki_markdown() -> Result<Vec<GrammarEntry>> {
    use workspace_qdrant_core::language_registry::types::GrammarQuality;

    let url = "https://raw.githubusercontent.com/wiki/tree-sitter/tree-sitter/List-of-parsers.md";
    let client = reqwest::Client::builder()
        .user_agent("workspace-qdrant-mcp/registry-updater")
        .build()?;

    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        anyhow::bail!("Failed to fetch tree-sitter wiki: {}", response.status());
    }

    let body = response.text().await?;
    let mut entries = Vec::new();

    // Parse markdown table rows: | Language | Repository |
    let link_re = regex::Regex::new(r"\[([^\]]+)\]\((https://github\.com/[^\)]+)\)").unwrap();

    for line in body.lines() {
        if !line.starts_with('|') {
            continue;
        }
        // Skip header and separator rows
        if line.contains("---") {
            continue;
        }

        // Extract GitHub links from the line
        for cap in link_re.captures_iter(line) {
            let url = &cap[2];
            if url.contains("tree-sitter") {
                let repo = extract_github_repo(url).unwrap_or_else(|| url.to_string());
                // Infer language from repo name
                let language = repo
                    .split('/')
                    .last()
                    .unwrap_or("")
                    .strip_prefix("tree-sitter-")
                    .unwrap_or("")
                    .to_string();

                if !language.is_empty() {
                    entries.push(GrammarEntry {
                        language,
                        repo,
                        quality: GrammarQuality::Community,
                        has_cpp_scanner: false,
                        src_subdir: None,
                        symbol_name: None,
                        archive_branch: None,
                    });
                }
            }
        }
    }

    Ok(entries)
}

/// Scrape grammar repos from nvim-treesitter lockfile.
async fn scrape_nvim_treesitter() -> Result<Vec<GrammarEntry>> {
    use workspace_qdrant_core::language_registry::providers::nvim_treesitter::NvimTreesitterProvider;
    let provider = NvimTreesitterProvider::new();
    provider
        .fetch_grammars()
        .await
        .context("nvim-treesitter fetch_grammars")
}

/// Scrape LSP servers from Microsoft's LSP implementors list.
async fn scrape_microsoft_lsp() -> Result<Vec<LspEntry>> {
    use workspace_qdrant_core::language_registry::types::{InstallMethod, LspServerEntry};

    // Microsoft maintains a JSON-based listing at the LSP specification site
    let url = "https://raw.githubusercontent.com/nicktomlin/lsp-servers/main/data/lsp-servers.json";
    let client = reqwest::Client::builder()
        .user_agent("workspace-qdrant-mcp/registry-updater")
        .build()?;

    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        // Fallback: try the official Microsoft page
        return scrape_microsoft_lsp_html().await;
    }

    let body = response.text().await?;

    #[derive(serde::Deserialize)]
    struct MsLspServer {
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        languages: Option<Vec<String>>,
        #[serde(default)]
        url: Option<String>,
    }

    let servers: Vec<MsLspServer> = serde_json::from_str(&body).unwrap_or_default();
    let mut entries = Vec::new();

    for server in servers {
        let name = match server.name {
            Some(n) => n,
            None => continue,
        };
        let languages = server.languages.unwrap_or_default();
        let repo_url = server.url.unwrap_or_default();

        for language in &languages {
            let lang_id = language.to_lowercase().replace(' ', "-");
            entries.push(LspEntry {
                language: lang_id,
                server: LspServerEntry {
                    name: name.clone(),
                    binary: name.to_lowercase().replace(' ', "-"),
                    args: Vec::new(),
                    priority: 30,
                    install_methods: vec![InstallMethod {
                        manager: "see".to_string(),
                        command: format!("See: {repo_url}"),
                    }],
                },
            });
        }
    }

    Ok(entries)
}

/// Fallback: scrape Microsoft LSP implementors HTML page.
async fn scrape_microsoft_lsp_html() -> Result<Vec<LspEntry>> {
    // This is a placeholder — HTML parsing is complex and fragile.
    // Return empty for now; mason-registry provides adequate LSP coverage.
    tracing::info!("Microsoft LSP HTML fallback not implemented, skipping");
    Ok(Vec::new())
}

/// Scrape LSP servers from langserver.org.
async fn scrape_langserver_org() -> Result<Vec<LspEntry>> {
    use workspace_qdrant_core::language_registry::types::{InstallMethod, LspServerEntry};

    // langserver.org data is available via their GitHub repo
    let url = "https://raw.githubusercontent.com/nicktomlin/lsp-servers/main/data/langservers.json";
    let client = reqwest::Client::builder()
        .user_agent("workspace-qdrant-mcp/registry-updater")
        .build()?;

    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        tracing::info!("langserver.org data not available, skipping");
        return Ok(Vec::new());
    }

    let body = response.text().await?;

    #[derive(serde::Deserialize)]
    struct LangServer {
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        language: Option<String>,
        #[serde(default)]
        url: Option<String>,
    }

    let servers: Vec<LangServer> = serde_json::from_str(&body).unwrap_or_default();
    let mut entries = Vec::new();

    for server in servers {
        let (name, language) = match (server.name, server.language) {
            (Some(n), Some(l)) => (n, l),
            _ => continue,
        };

        let lang_id = language.to_lowercase().replace(' ', "-");
        let repo_url = server.url.unwrap_or_default();

        entries.push(LspEntry {
            language: lang_id,
            server: LspServerEntry {
                name: name.clone(),
                binary: name.to_lowercase().replace(' ', "-"),
                args: Vec::new(),
                priority: 35,
                install_methods: vec![InstallMethod {
                    manager: "see".to_string(),
                    command: format!("See: {repo_url}"),
                }],
            },
        });
    }

    Ok(entries)
}

/// Scrape LSP servers from mason-registry.
async fn scrape_mason() -> Result<Vec<LspEntry>> {
    use workspace_qdrant_core::language_registry::providers::mason::MasonProvider;
    let provider = MasonProvider::new();
    provider
        .fetch_lsp_servers()
        .await
        .context("mason fetch_lsp_servers")
}

/// Extract "owner/repo" from a GitHub URL.
fn extract_github_repo(url: &str) -> Option<String> {
    let url = url.trim_end_matches('/');
    let url = url.strip_suffix(".git").unwrap_or(url);

    // Handle https://github.com/owner/repo
    if let Some(rest) = url.strip_prefix("https://github.com/") {
        let parts: Vec<&str> = rest.splitn(3, '/').collect();
        if parts.len() >= 2 {
            return Some(format!("{}/{}", parts[0], parts[1]));
        }
    }

    None
}

/// Get the list of all available source names.
pub fn available_sources() -> &'static [&'static str] {
    ALL_SOURCES
}

// ── Trait import for provider methods ──────────────────────────────
use workspace_qdrant_core::language_registry::provider::LanguageSourceProvider;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_github_repo() {
        assert_eq!(
            extract_github_repo("https://github.com/tree-sitter/tree-sitter-rust"),
            Some("tree-sitter/tree-sitter-rust".to_string())
        );
        assert_eq!(
            extract_github_repo("https://github.com/tree-sitter/tree-sitter-rust.git"),
            Some("tree-sitter/tree-sitter-rust".to_string())
        );
        assert_eq!(
            extract_github_repo("https://github.com/tree-sitter/tree-sitter-rust/"),
            Some("tree-sitter/tree-sitter-rust".to_string())
        );
        assert_eq!(extract_github_repo("not-a-url"), None);
    }

    #[test]
    fn test_is_enabled() {
        assert!(is_enabled("linguist", None));
        assert!(is_enabled(
            "linguist",
            Some(&["linguist".to_string(), "mason".to_string()])
        ));
        assert!(!is_enabled("linguist", Some(&["mason".to_string()])));
    }

    #[test]
    fn test_available_sources() {
        let sources = available_sources();
        assert!(sources.contains(&"linguist"));
        assert!(sources.contains(&"mason"));
        assert!(sources.contains(&"ts-wiki"));
    }
}
