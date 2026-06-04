//! gRPC handler implementations for LanguageService (WI-e1, #82).
//!
//! Handlers delegate the grammar engine to `GrammarManager` and the security
//! gate + registry queries to `workspace_qdrant_core::language_registry`. No
//! download / compile / dlopen happens until the request passes
//! [`validate_install_request`].

use tonic::{Request, Response, Status};
use tracing::{info, warn};
use workspace_qdrant_core::language_registry::{
    find_language, registry_summary, validate_install_request, validate_language_name,
    GrammarInstallError, RegistryProvider,
};
use workspace_qdrant_core::tree_sitter::GrammarStatus;

use crate::proto::language_service_server::LanguageService;
use crate::proto::{
    GrammarEntryProto, InstallGrammarRequest, InstallGrammarResponse, ListGrammarsResponse,
    QueryLanguageRequest, QueryLanguageResponse, RefreshLanguageRegistryResponse,
    RemoveGrammarRequest, RemoveGrammarResponse,
};

use super::service_impl::LanguageServiceImpl;

/// Stringify a [`GrammarStatus`] for the wire (mirrors the CLI's status labels).
fn status_str(status: &GrammarStatus) -> &'static str {
    match status {
        GrammarStatus::Loaded => "loaded",
        GrammarStatus::Cached => "cached",
        GrammarStatus::NeedsDownload => "needs_download",
        GrammarStatus::IncompatibleVersion => "incompatible_version",
        GrammarStatus::NotAvailable => "not_available",
    }
}

/// Map a security-gate error to the appropriate gRPC status code.
fn map_install_err(e: GrammarInstallError) -> Status {
    match e {
        GrammarInstallError::InvalidName(_)
        | GrammarInstallError::UnknownLanguage(_)
        | GrammarInstallError::NoGrammarSource(_) => Status::invalid_argument(e.to_string()),
        GrammarInstallError::InsecureSource(_)
        | GrammarInstallError::ChecksumVerificationDisabled => {
            Status::failed_precondition(e.to_string())
        }
        GrammarInstallError::Registry(_) => Status::internal(e.to_string()),
    }
}

#[tonic::async_trait]
impl LanguageService for LanguageServiceImpl {
    #[tracing::instrument(skip_all, fields(method = "LanguageService.install_grammar"))]
    async fn install_grammar(
        &self,
        request: Request<InstallGrammarRequest>,
    ) -> Result<Response<InstallGrammarResponse>, Status> {
        let req = request.into_inner();
        let mut mgr = self.grammar_manager.lock().await;

        // Read the pinned source config so the gate can enforce https + checksums.
        let (base_url, verify) = {
            let cfg = mgr.config();
            (cfg.download_base_url.clone(), cfg.verify_checksums)
        };

        // SECURITY GATE — runs BEFORE any download / compile / dlopen.
        validate_install_request(&req.language, &base_url, verify).map_err(|e| {
            warn!(language = %req.language, error = %e, "InstallGrammar rejected by security gate");
            map_install_err(e)
        })?;

        if req.force {
            // Best-effort clear so the next load re-downloads + re-verifies.
            if let Err(e) = mgr.clear_cache(&req.language) {
                warn!(language = %req.language, error = %e, "force clear_cache failed");
            }
        }

        match mgr.get_grammar(&req.language).await {
            Ok(_) => {
                let status = mgr.grammar_status(&req.language);
                info!(language = %req.language, status = status_str(&status), "grammar installed");
                Ok(Response::new(InstallGrammarResponse {
                    language: req.language.clone(),
                    status: status_str(&status).to_string(),
                    installed: true,
                    message: format!("grammar for '{}' ready", req.language),
                }))
            }
            Err(e) => Err(Status::internal(format!(
                "grammar install failed for '{}': {e}",
                req.language
            ))),
        }
    }

    #[tracing::instrument(skip_all, fields(method = "LanguageService.remove_grammar"))]
    async fn remove_grammar(
        &self,
        request: Request<RemoveGrammarRequest>,
    ) -> Result<Response<RemoveGrammarResponse>, Status> {
        let req = request.into_inner();
        // Validate the name shape before constructing any cache path.
        validate_language_name(&req.language).map_err(map_install_err)?;

        let mgr = self.grammar_manager.lock().await;
        let removed = mgr
            .clear_cache(&req.language)
            .map_err(|e| Status::internal(format!("remove failed for '{}': {e}", req.language)))?;
        Ok(Response::new(RemoveGrammarResponse {
            language: req.language,
            removed,
        }))
    }

    #[tracing::instrument(skip_all, fields(method = "LanguageService.list_grammars"))]
    async fn list_grammars(
        &self,
        _request: Request<()>,
    ) -> Result<Response<ListGrammarsResponse>, Status> {
        let mgr = self.grammar_manager.lock().await;
        let cached = mgr.cached_languages().unwrap_or_default();

        let provider = RegistryProvider::new()
            .map_err(|e| Status::internal(format!("registry error: {e}")))?;
        let known: Vec<GrammarEntryProto> = provider
            .definitions()
            .iter()
            .filter(|d| d.has_grammar())
            .map(|d| {
                let id = d.id();
                let status = mgr.grammar_status(&id);
                GrammarEntryProto {
                    language: id,
                    status: status_str(&status).to_string(),
                }
            })
            .collect();

        Ok(Response::new(ListGrammarsResponse { cached, known }))
    }

    #[tracing::instrument(skip_all, fields(method = "LanguageService.query_language"))]
    async fn query_language(
        &self,
        request: Request<QueryLanguageRequest>,
    ) -> Result<Response<QueryLanguageResponse>, Status> {
        let req = request.into_inner();
        let def = find_language(&req.language).map_err(map_install_err)?;
        match def {
            Some(def) => {
                let mgr = self.grammar_manager.lock().await;
                let status = mgr.grammar_status(&def.id());
                Ok(Response::new(QueryLanguageResponse {
                    found: true,
                    language: def.language.clone(),
                    grammar_status: status_str(&status).to_string(),
                    has_grammar: def.has_grammar(),
                    has_lsp: def.has_lsp(),
                    has_semantic_patterns: def.has_semantic_patterns(),
                    extensions: def.extensions.clone(),
                    aliases: def.aliases.clone(),
                }))
            }
            None => Ok(Response::new(QueryLanguageResponse {
                found: false,
                language: req.language,
                ..Default::default()
            })),
        }
    }

    #[tracing::instrument(skip_all, fields(method = "LanguageService.refresh_language_registry"))]
    async fn refresh_language_registry(
        &self,
        _request: Request<()>,
    ) -> Result<Response<RefreshLanguageRegistryResponse>, Status> {
        let s = registry_summary().map_err(map_install_err)?;
        Ok(Response::new(RefreshLanguageRegistryResponse {
            total: s.total,
            with_grammars: s.with_grammars,
            with_lsp: s.with_lsp,
            with_semantic_patterns: s.with_semantic_patterns,
        }))
    }
}
