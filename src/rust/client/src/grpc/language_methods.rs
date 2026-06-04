//! `LanguageService` RPC wrappers (WI-e2, #82).
//!
//! Thin typed wrappers over the daemon's grammar/language-registry service so
//! the CLI can drive `wqm language …` without linking `workspace-qdrant-core`.

use tonic::Status;

use super::client::DaemonClient;
use crate::workspace_daemon::{
    InstallGrammarRequest, InstallGrammarResponse, ListGrammarsResponse, QueryLanguageRequest,
    QueryLanguageResponse, RefreshLanguageRegistryResponse, RemoveGrammarRequest,
    RemoveGrammarResponse,
};

impl DaemonClient {
    /// Install (download + compile + load) a grammar for `language`.
    ///
    /// The daemon validates the language against the registry allowlist, rejects
    /// path/URL metacharacters, requires an https pinned source, and verifies the
    /// checksum before any compile/dlopen — invalid requests return `Err(Status)`
    /// with `InvalidArgument` / `FailedPrecondition`.
    pub async fn install_grammar(
        &mut self,
        language: String,
        force: bool,
    ) -> Result<InstallGrammarResponse, Status> {
        let client = self.language.clone();
        self.call("installGrammar", None, || {
            let mut c = client.clone();
            let req = InstallGrammarRequest {
                language: language.clone(),
                force,
            };
            async move {
                c.install_grammar(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Remove the cached grammar (and metadata) for `language`.
    pub async fn remove_grammar(
        &mut self,
        language: String,
    ) -> Result<RemoveGrammarResponse, Status> {
        let client = self.language.clone();
        self.call("removeGrammar", None, || {
            let mut c = client.clone();
            let req = RemoveGrammarRequest {
                language: language.clone(),
            };
            async move {
                c.remove_grammar(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// List cached + known grammars.
    pub async fn list_grammars(&mut self) -> Result<ListGrammarsResponse, Status> {
        let client = self.language.clone();
        self.call("listGrammars", None, || {
            let mut c = client.clone();
            async move {
                c.list_grammars(tonic::Request::new(()))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Query a single language's definition + grammar status.
    pub async fn query_language(
        &mut self,
        language: String,
    ) -> Result<QueryLanguageResponse, Status> {
        let client = self.language.clone();
        self.call("queryLanguage", None, || {
            let mut c = client.clone();
            let req = QueryLanguageRequest {
                language: language.clone(),
            };
            async move {
                c.query_language(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Refresh the language registry and return summary counts.
    pub async fn refresh_language_registry(
        &mut self,
    ) -> Result<RefreshLanguageRegistryResponse, Status> {
        let client = self.language.clone();
        self.call("refreshLanguageRegistry", None, || {
            let mut c = client.clone();
            async move {
                c.refresh_language_registry(tonic::Request::new(()))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }
}
