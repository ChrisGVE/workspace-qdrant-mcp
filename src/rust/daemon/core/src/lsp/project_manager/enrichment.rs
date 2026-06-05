//! LSP enrichment: chunk enrichment, references, type info, and helpers.

use std::path::Path;

use super::{
    EnrichmentStatus, LanguageServerManager, LspEnrichment, ProjectLanguageKey, ProjectLspResult,
    Reference, ResolvedImport, TypeInfo,
};
use crate::lsp::Language;

/// Compute the UTF-16 column (LSP position encoding) of `symbol` within
/// `line_text`, or 0 if the symbol is empty or not present.
///
/// LSP positions count UTF-16 code units, so the column is the UTF-16 length
/// of the text preceding the symbol's first occurrence. For ASCII identifiers
/// this equals the byte offset. Kept as a free, pure function so it can be
/// unit-tested without process or filesystem setup.
pub(crate) fn symbol_column_in_line(line_text: &str, symbol: &str) -> u32 {
    if symbol.is_empty() {
        return 0;
    }
    match line_text.find(symbol) {
        Some(byte_idx) => line_text[..byte_idx].encode_utf16().count() as u32,
        None => 0,
    }
}

impl LanguageServerManager {
    /// Check if an LSP server is ready for a given file in a project.
    ///
    /// Returns true if there is a running server instance that can handle
    /// the file's language for the given project.
    pub async fn is_server_ready_for_file(&self, project_id: &str, file: &Path) -> bool {
        let file_language = file
            .extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension);

        let Some(language) = file_language else {
            return false;
        };

        // A server process must exist for this project+language…
        let exists = {
            let instances = self.instances.read().await;
            instances
                .iter()
                .any(|(k, _)| k.project_id == project_id && k.language == language)
        };
        if !exists {
            return false;
        }

        // …and it must be past its warm-up grace. Firing enrichment at a server
        // still building its initial index makes the first request time out;
        // instead we defer (the chunk is marked pending and re-enriched later by
        // the metadata-uplift pass). A missing ready-at entry (legacy/restored
        // state) is treated as ready for backward compatibility.
        let ready_at = self.ready_at.read().await;
        ready_at
            .iter()
            .find(|(k, _)| k.project_id == project_id && k.language == language)
            .map(|(_, &t)| std::time::Instant::now() >= t)
            .unwrap_or(true)
    }

    /// Enrich a semantic chunk with LSP data
    ///
    /// This is the main entry point for queue processor integration.
    /// Returns enrichment data or gracefully degrades if LSP unavailable.
    ///
    /// Emits a `lsp.enrich_chunk` tracing span carrying the real project,
    /// file, resolved language, and final enrichment status — exported to
    /// OTLP/Tempo when the daemon runs with tracing enabled.
    #[tracing::instrument(
        name = "lsp.enrich_chunk",
        skip_all,
        fields(
            project_id = %project_id,
            file = %file.display(),
            line = start_line,
            language = tracing::field::Empty,
            enrichment_status = tracing::field::Empty,
        )
    )]
    pub async fn enrich_chunk(
        &self,
        project_id: &str,
        file: &Path,
        symbol_name: &str,
        start_line: u32,
        _end_line: u32,
        _is_project_active: bool,
    ) -> LspEnrichment {
        let span = tracing::Span::current();
        let language = file
            .extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension)
            .map(|l| l.identifier().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        span.record("language", language.as_str());

        {
            let mut metrics = self.metrics.write().await;
            metrics.total_enrichment_queries += 1;
        }

        let result = if !self.is_server_ready_for_file(project_id, file).await {
            self.skipped_enrichment(file).await
        } else {
            self.perform_enrichment(project_id, file, start_line, symbol_name)
                .await
        };

        span.record("enrichment_status", result.enrichment_status.as_str());
        result
    }

    /// Return a skipped enrichment when the server is not ready
    async fn skipped_enrichment(&self, file: &Path) -> LspEnrichment {
        let language = file
            .extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension)
            .map(|l| l.identifier().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        tracing::debug!(
            file = %file.display(),
            language = %language,
            "LSP server not ready, skipping enrichment"
        );

        {
            let mut metrics = self.metrics.write().await;
            metrics.skipped_enrichments += 1;
        }

        LspEnrichment {
            references: Vec::new(),
            type_info: None,
            resolved_imports: Vec::new(),
            definition: None,
            enrichment_status: EnrichmentStatus::Skipped,
            error_message: Some(format!("LSP server not ready for {}", language)),
        }
    }

    /// Execute all enrichment queries and determine the composite status
    async fn perform_enrichment(
        &self,
        project_id: &str,
        file: &Path,
        start_line: u32,
        symbol_name: &str,
    ) -> LspEnrichment {
        // Touch last_enrichment_at for idle eviction tracking
        self.touch_enrichment_time(project_id, file).await;

        // Resolve the symbol's column on its declaration line. Querying at
        // column 0 (the previous behavior) points at the start of the line —
        // usually a keyword like `pub`/`def` — so the server returns no
        // references or hover even when the symbol is fully indexed.
        let column = match tokio::fs::read_to_string(file).await {
            Ok(content) => content
                .lines()
                .nth(start_line as usize)
                .map(|line| symbol_column_in_line(line, symbol_name))
                .unwrap_or(0),
            Err(_) => 0,
        };

        let mut errors: Vec<String> = Vec::new();
        let mut successes = 0;

        let references = match self.get_references(file, start_line, column).await {
            Ok(refs) => {
                successes += 1;
                refs
            }
            Err(e) => {
                tracing::debug!(project_id, file = %file.display(), error = %e, "get_references failed");
                errors.push(format!("get_references: {}", e));
                Vec::new()
            }
        };

        let type_info = match self.get_type_info(file, start_line, column).await {
            Ok(info) => {
                successes += 1;
                info
            }
            Err(e) => {
                tracing::debug!(project_id, file = %file.display(), error = %e, "get_type_info failed");
                errors.push(format!("get_type_info: {}", e));
                None
            }
        };

        let resolved_imports = match self.resolve_imports(file).await {
            Ok(imports) => {
                successes += 1;
                imports
            }
            Err(e) => {
                tracing::debug!(project_id, file = %file.display(), error = %e, "resolve_imports failed");
                errors.push(format!("resolve_imports: {}", e));
                Vec::new()
            }
        };

        let (status, error_message) = Self::determine_enrichment_status(
            successes,
            &errors,
            &references,
            &type_info,
            &resolved_imports,
        );

        {
            let mut metrics = self.metrics.write().await;
            match status {
                EnrichmentStatus::Success => metrics.successful_enrichments += 1,
                EnrichmentStatus::Partial => metrics.partial_enrichments += 1,
                EnrichmentStatus::Failed => metrics.failed_enrichments += 1,
                EnrichmentStatus::Skipped => metrics.skipped_enrichments += 1,
            }
        }

        LspEnrichment {
            references,
            type_info,
            resolved_imports,
            definition: None,
            enrichment_status: status,
            error_message,
        }
    }

    /// Determine the enrichment status from query results
    fn determine_enrichment_status(
        successes: usize,
        errors: &[String],
        references: &[Reference],
        type_info: &Option<TypeInfo>,
        resolved_imports: &[ResolvedImport],
    ) -> (EnrichmentStatus, Option<String>) {
        let has_data =
            !references.is_empty() || type_info.is_some() || !resolved_imports.is_empty();
        let all_failed = successes == 0;
        let some_failed = !errors.is_empty() && successes > 0;

        if all_failed {
            (
                EnrichmentStatus::Failed,
                Some(format!("All queries failed: {}", errors.join("; "))),
            )
        } else if has_data {
            (EnrichmentStatus::Success, None)
        } else if some_failed {
            (
                EnrichmentStatus::Partial,
                Some(format!("Some queries failed: {}", errors.join("; "))),
            )
        } else {
            (EnrichmentStatus::Success, None)
        }
    }

    /// Get references for a symbol at a specific position
    #[tracing::instrument(
        name = "lsp.references",
        skip_all,
        fields(file = %file.display(), line, column)
    )]
    pub async fn get_references(
        &self,
        file: &Path,
        line: u32,
        column: u32,
    ) -> ProjectLspResult<Vec<Reference>> {
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_references_queries += 1;
        }

        let cache_key = format!("refs:{}:{}:{}", file.display(), line, column);
        if let Some(cached) = self.check_cache(&cache_key).await {
            return Ok(cached.references.clone());
        }

        let (server_key, server_instance) = self.find_server_for_file(file).await;

        let Some(instance) = server_instance else {
            tracing::debug!(file = %file.display(), "No LSP server available for file");
            return Ok(Vec::new());
        };

        let params = serde_json::json!({
            "textDocument": { "uri": Self::file_to_uri(file) },
            "position": { "line": line, "character": column },
            "context": { "includeDeclaration": true }
        });

        let inst = instance.lock().await;
        let rpc_client = inst.rpc_client();
        drop(inst);

        let response = match rpc_client
            .send_request("textDocument/references", params)
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                let error_msg = e.to_string();
                tracing::debug!(file = %file.display(), error = %error_msg, "Failed to get references");
                if let Some(ref key) = server_key {
                    self.handle_potential_crash(key, &error_msg).await;
                }
                return Ok(Vec::new());
            }
        };

        let references: Vec<Reference> = response
            .result
            .as_ref()
            .and_then(|r| r.as_array())
            .map(|locs| {
                locs.iter()
                    .filter_map(|loc| Self::parse_location(loc))
                    .collect()
            })
            .unwrap_or_default();

        tracing::debug!(file = %file.display(), line, column, count = references.len(), "Got references");

        if !references.is_empty() {
            self.store_in_cache(
                cache_key,
                LspEnrichment {
                    references: references.clone(),
                    type_info: None,
                    resolved_imports: Vec::new(),
                    definition: None,
                    enrichment_status: EnrichmentStatus::Success,
                    error_message: None,
                },
            )
            .await;
        }

        Ok(references)
    }

    /// Get type information for a symbol at a specific position
    #[tracing::instrument(
        name = "lsp.type_info",
        skip_all,
        fields(file = %file.display(), line, column)
    )]
    pub async fn get_type_info(
        &self,
        file: &Path,
        line: u32,
        column: u32,
    ) -> ProjectLspResult<Option<TypeInfo>> {
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_type_info_queries += 1;
        }

        let cache_key = format!("type:{}:{}:{}", file.display(), line, column);
        if let Some(cached) = self.check_cache(&cache_key).await {
            return Ok(cached.type_info.clone());
        }

        let (server_key, server_instance) = self.find_server_for_file(file).await;

        let Some(instance) = server_instance else {
            tracing::debug!(file = %file.display(), "No LSP server available for file");
            return Ok(None);
        };

        let params = serde_json::json!({
            "textDocument": { "uri": Self::file_to_uri(file) },
            "position": { "line": line, "character": column }
        });

        let inst = instance.lock().await;
        let rpc_client = inst.rpc_client();
        drop(inst);

        let response = match rpc_client.send_request("textDocument/hover", params).await {
            Ok(resp) => resp,
            Err(e) => {
                let error_msg = e.to_string();
                tracing::debug!(file = %file.display(), error = %error_msg, "Failed to get hover info");
                if let Some(ref key) = server_key {
                    self.handle_potential_crash(key, &error_msg).await;
                }
                return Ok(None);
            }
        };

        let type_info = response
            .result
            .as_ref()
            .and_then(|r| Self::parse_hover_response(r));

        tracing::debug!(file = %file.display(), line, column, has_type_info = type_info.is_some(), "Got type info");

        if type_info.is_some() {
            self.store_in_cache(
                cache_key,
                LspEnrichment {
                    references: Vec::new(),
                    type_info: type_info.clone(),
                    resolved_imports: Vec::new(),
                    definition: None,
                    enrichment_status: EnrichmentStatus::Success,
                    error_message: None,
                },
            )
            .await;
        }

        Ok(type_info)
    }

    // --- Helper methods ---

    /// Update `last_enrichment_at` for the server handling this file
    async fn touch_enrichment_time(&self, project_id: &str, file: &Path) {
        let file_language = file
            .extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension);

        if let Some(language) = file_language {
            let key = ProjectLanguageKey::new(project_id, language);
            let mut servers = self.servers.write().await;
            if let Some(state) = servers.get_mut(&key) {
                state.last_enrichment_at = Some(std::time::Instant::now());
            }
        }
    }

    /// Convert file path to LSP URI
    pub(crate) fn file_to_uri(file: &Path) -> String {
        format!("file://{}", file.display())
    }

    /// Parse LSP Location response into Reference
    fn parse_location(location: &serde_json::Value) -> Option<Reference> {
        let uri = location.get("uri")?.as_str()?;
        let range = location.get("range")?;
        let start = range.get("start")?;

        let file = uri.strip_prefix("file://").unwrap_or(uri);

        Some(Reference {
            file: file.to_string(),
            line: start.get("line")?.as_u64()? as u32,
            column: start.get("character")?.as_u64()? as u32,
            end_line: range
                .get("end")
                .and_then(|e| e.get("line"))
                .and_then(|l| l.as_u64())
                .map(|l| l as u32),
            end_column: range
                .get("end")
                .and_then(|e| e.get("character"))
                .and_then(|c| c.as_u64())
                .map(|c| c as u32),
        })
    }

    /// Parse hover response into TypeInfo
    fn parse_hover_response(hover: &serde_json::Value) -> Option<TypeInfo> {
        let contents = hover.get("contents")?;

        let type_signature = if contents.is_object() {
            contents.get("value")?.as_str()?.to_string()
        } else if contents.is_string() {
            contents.as_str()?.to_string()
        } else if contents.is_array() {
            contents
                .as_array()?
                .iter()
                .filter_map(|c| {
                    if c.is_string() {
                        c.as_str().map(|s| s.to_string())
                    } else {
                        c.get("value")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            return None;
        };

        let kind = if type_signature.contains("fn ") || type_signature.contains("function") {
            "function"
        } else if type_signature.contains("struct ") || type_signature.contains("class") {
            "class"
        } else if type_signature.contains("trait ") || type_signature.contains("interface") {
            "interface"
        } else if type_signature.contains("type ") {
            "type"
        } else if type_signature.contains("const ") || type_signature.contains("let ") {
            "variable"
        } else {
            "unknown"
        };

        Some(TypeInfo {
            type_signature,
            documentation: None,
            kind: kind.to_string(),
            container: None,
        })
    }

    /// Check the cache for an enrichment entry and track hit/miss
    pub(crate) async fn check_cache(&self, cache_key: &str) -> Option<LspEnrichment> {
        // `LruCache::get` takes `&mut self` (it updates recency), so lock the
        // Mutex, clone the hit, and release before touching `metrics` — never
        // hold the cache lock across the metrics `.await`.
        let hit = {
            let mut cache = self.cache.lock().await;
            cache.get(cache_key).cloned()
        };
        let mut metrics = self.metrics.write().await;
        if hit.is_some() {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
        hit
    }

    /// Store an enrichment entry in the cache (LRU; evicts the least-recently
    /// used entry past capacity).
    pub(crate) async fn store_in_cache(&self, key: String, enrichment: LspEnrichment) {
        let mut cache = self.cache.lock().await;
        cache.put(key, enrichment);
    }

    /// Find a server instance matching the file's language
    pub(crate) async fn find_server_for_file(
        &self,
        file: &Path,
    ) -> (
        Option<ProjectLanguageKey>,
        Option<std::sync::Arc<tokio::sync::Mutex<super::ServerInstance>>>,
    ) {
        let instances = self.instances.read().await;
        let file_language = file
            .extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension);

        if let Some(ref language) = file_language {
            instances
                .iter()
                .find(|(k, _)| k.language == *language)
                .map(|(k, v)| (k.clone(), v.clone()))
                .unzip()
        } else {
            (None, None)
        }
    }
}
