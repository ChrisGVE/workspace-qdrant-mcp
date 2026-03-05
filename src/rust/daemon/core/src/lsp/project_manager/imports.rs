//! Import resolution: extract and resolve import statements via LSP.

use std::path::Path;

use super::{
    EnrichmentStatus, LanguageServerManager, LspEnrichment, ProjectLanguageKey, ProjectLspResult,
    ResolvedImport,
};
use crate::lsp::Language;

impl LanguageServerManager {
    /// Resolve imports in a file
    pub async fn resolve_imports(&self, file: &Path) -> ProjectLspResult<Vec<ResolvedImport>> {
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_import_queries += 1;
        }

        let cache_key = format!("imports:{}", file.display());
        if let Some(cached) = self.check_cache(&cache_key).await {
            return Ok(cached.resolved_imports.clone());
        }

        let instances = self.instances.read().await;
        let file_language = file
            .extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension);

        let Some(language) = file_language else {
            return Ok(Vec::new());
        };

        let (server_key, server_instance) = instances
            .iter()
            .find(|(k, _)| k.language == language)
            .map(|(k, v)| (k.clone(), v.clone()))
            .unzip();

        drop(instances);

        let content = match tokio::fs::read_to_string(file).await {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!(file = %file.display(), error = %e, "Failed to read file");
                return Ok(Vec::new());
            }
        };

        let import_names = Self::extract_imports(&content, &language);
        if import_names.is_empty() {
            return Ok(Vec::new());
        }

        tracing::debug!(file = %file.display(), imports_found = import_names.len(), "Extracted imports");

        let resolved_imports = self
            .resolve_import_list(file, &content, &import_names, server_instance, &server_key)
            .await;

        tracing::debug!(
            file = %file.display(),
            resolved = resolved_imports.iter().filter(|i| i.resolved).count(),
            total = resolved_imports.len(),
            "Import resolution complete"
        );

        if !resolved_imports.is_empty() {
            self.store_in_cache(
                cache_key,
                LspEnrichment {
                    references: Vec::new(),
                    type_info: None,
                    resolved_imports: resolved_imports.clone(),
                    definition: None,
                    enrichment_status: EnrichmentStatus::Success,
                    error_message: None,
                },
            )
            .await;
        }

        Ok(resolved_imports)
    }

    /// Resolve a list of imports via LSP or return them as unresolved
    async fn resolve_import_list(
        &self,
        file: &Path,
        content: &str,
        import_names: &[String],
        server_instance: Option<std::sync::Arc<tokio::sync::Mutex<super::ServerInstance>>>,
        server_key: &Option<ProjectLanguageKey>,
    ) -> Vec<ResolvedImport> {
        let mut resolved_imports = Vec::new();
        let crash_detected = false;

        if let Some(instance) = server_instance {
            let inst = instance.lock().await;
            let rpc_client = inst.rpc_client();
            drop(inst);

            'outer: for (line_idx, line) in content.lines().enumerate() {
                for import_name in import_names {
                    if line.contains(import_name) {
                        let column = line.find(import_name).unwrap_or(0) as u32;

                        let params = serde_json::json!({
                            "textDocument": { "uri": Self::file_to_uri(file) },
                            "position": { "line": line_idx as u32, "character": column }
                        });

                        match rpc_client
                            .send_request("textDocument/definition", params)
                            .await
                        {
                            Ok(response) => {
                                resolved_imports.push(Self::parse_definition_response(
                                    import_name,
                                    response.result.as_ref(),
                                ));
                            }
                            Err(e) => {
                                let error_msg = e.to_string();
                                tracing::debug!(import = import_name, error = %error_msg, "Failed to resolve import");

                                if !crash_detected {
                                    if let Some(ref key) = server_key {
                                        if self.handle_potential_crash(key, &error_msg).await {
                                            Self::add_remaining_unresolved(
                                                import_names,
                                                &mut resolved_imports,
                                            );
                                            break 'outer;
                                        }
                                    }
                                }

                                resolved_imports.push(ResolvedImport {
                                    import_name: import_name.clone(),
                                    target_file: None,
                                    target_symbol: None,
                                    is_stdlib: false,
                                    resolved: false,
                                });
                            }
                        }

                        break; // Only resolve once per import name
                    }
                }
            }
        } else {
            for import_name in import_names {
                resolved_imports.push(ResolvedImport {
                    import_name: import_name.clone(),
                    target_file: None,
                    target_symbol: None,
                    is_stdlib: false,
                    resolved: false,
                });
            }
        }

        resolved_imports
    }

    /// Add remaining unresolved imports after a server crash
    fn add_remaining_unresolved(
        import_names: &[String],
        resolved_imports: &mut Vec<ResolvedImport>,
    ) {
        for remaining in import_names {
            if !resolved_imports.iter().any(|r| &r.import_name == remaining) {
                resolved_imports.push(ResolvedImport {
                    import_name: remaining.clone(),
                    target_file: None,
                    target_symbol: None,
                    is_stdlib: false,
                    resolved: false,
                });
            }
        }
    }

    /// Parse a definition location into a `ResolvedImport`
    fn parse_definition_response(
        import_name: &str,
        definition: Option<&serde_json::Value>,
    ) -> ResolvedImport {
        let (target_file, resolved) = if let Some(def) = definition {
            let location = if def.is_array() {
                def.as_array().and_then(|arr| arr.first())
            } else {
                Some(def)
            };

            if let Some(loc) = location {
                let uri = loc.get("uri").and_then(|u| u.as_str());
                let target = uri.map(|u| u.strip_prefix("file://").unwrap_or(u).to_string());
                (target, uri.is_some())
            } else {
                (None, false)
            }
        } else {
            (None, false)
        };

        let is_stdlib = target_file
            .as_ref()
            .map(|p| {
                p.contains("/site-packages/")
                    || p.contains("/.rustup/")
                    || p.contains("/lib/rustlib/")
                    || p.contains("/node_modules/@types/")
                    || p.contains("/usr/lib/")
                    || p.contains("/Library/Developer/")
            })
            .unwrap_or(false);

        ResolvedImport {
            import_name: import_name.to_string(),
            target_file,
            target_symbol: None,
            is_stdlib,
            resolved,
        }
    }

    /// Extract import statements from file content (basic pattern matching)
    pub(crate) fn extract_imports(content: &str, language: &Language) -> Vec<String> {
        let mut imports = Vec::new();

        let import_patterns: Vec<(&str, usize)> = match language {
            Language::Python => vec![(r"^import\s+(\S+)", 1), (r"^from\s+(\S+)\s+import", 1)],
            Language::Rust => vec![(r"^use\s+([^;]+)", 1)],
            Language::TypeScript | Language::JavaScript => vec![
                (r#"import\s+.*\s+from\s+['"]([^'"]+)['"]"#, 1),
                (r#"require\s*\(\s*['"]([^'"]+)['"]"#, 1),
            ],
            Language::Go => vec![
                (r#"import\s+["']([^"']+)["']"#, 1),
                (r#"^\s*"([^"]+)"$"#, 1),
            ],
            _ => vec![],
        };

        for line in content.lines() {
            for (pattern, group) in &import_patterns {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if let Some(captures) = re.captures(line) {
                        if let Some(import) = captures.get(*group) {
                            imports.push(import.as_str().to_string());
                        }
                    }
                }
            }
        }

        imports
    }
}
