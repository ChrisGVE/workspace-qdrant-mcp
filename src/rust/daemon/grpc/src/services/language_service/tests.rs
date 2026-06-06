//! Hermetic handler tests for LanguageService (WI-e1, #82).
//!
//! These exercise the handlers that need no network: query / refresh / list /
//! remove round-trips, plus the install security-gate rejections (which fail
//! BEFORE any download). The pure security gate is unit-tested in
//! `workspace_qdrant_core::language_registry::grammar_install`.

use std::sync::Arc;

use tokio::sync::Mutex;
use tonic::{Code, Request};
use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::tree_sitter::create_grammar_manager;

use crate::proto::language_service_server::LanguageService;
use crate::proto::{InstallGrammarRequest, QueryLanguageRequest, RemoveGrammarRequest};

use super::service_impl::LanguageServiceImpl;

/// Build a service over a throwaway grammar cache dir (no writes happen in these
/// tests, but keep it off the real cache regardless).
fn service_with(verify_checksums: bool) -> LanguageServiceImpl {
    let cache_dir = std::env::temp_dir().join("wqm-langservice-test-cache");
    let config = GrammarConfig {
        cache_dir,
        verify_checksums,
        ..GrammarConfig::default()
    };
    let manager = Arc::new(Mutex::new(create_grammar_manager(config)));
    LanguageServiceImpl::new(manager)
}

fn default_service() -> LanguageServiceImpl {
    service_with(GrammarConfig::default().verify_checksums)
}

#[tokio::test]
async fn query_known_language_reports_registry_facts() {
    let svc = default_service();
    let resp = svc
        .query_language(Request::new(QueryLanguageRequest {
            language: "rust".to_string(),
        }))
        .await
        .expect("query ok")
        .into_inner();
    assert!(resp.found);
    assert_eq!(resp.language.to_lowercase(), "rust");
    assert!(resp.has_grammar);
    assert!(!resp.extensions.is_empty(), "rust has extensions");
    assert!(!resp.grammar_status.is_empty());
}

#[tokio::test]
async fn query_unknown_language_is_not_found() {
    let svc = default_service();
    let resp = svc
        .query_language(Request::new(QueryLanguageRequest {
            language: "definitelynotalanguage".to_string(),
        }))
        .await
        .expect("query ok")
        .into_inner();
    assert!(!resp.found);
}

#[tokio::test]
async fn refresh_reports_sane_counts() {
    let svc = default_service();
    let resp = svc
        .refresh_language_registry(Request::new(()))
        .await
        .expect("refresh ok")
        .into_inner();
    assert!(resp.total >= 40, "expected ~44 bundled languages");
    assert!(resp.with_grammars > 0);
    assert!(resp.with_grammars <= resp.total);
    assert!(resp.with_semantic_patterns <= resp.total);
}

#[tokio::test]
async fn list_grammars_returns_known_set() {
    let svc = default_service();
    let resp = svc
        .list_grammars(Request::new(()))
        .await
        .expect("list ok")
        .into_inner();
    assert!(!resp.known.is_empty(), "known grammars must be non-empty");
    assert!(resp.known.iter().any(|g| g.language == "rust"));
}

#[tokio::test]
async fn remove_uncached_grammar_is_noop_ok() {
    let svc = default_service();
    let resp = svc
        .remove_grammar(Request::new(RemoveGrammarRequest {
            language: "rust".to_string(),
        }))
        .await
        .expect("remove ok")
        .into_inner();
    // Nothing cached in the throwaway dir → removed=false, no error.
    assert!(!resp.removed);
}

#[tokio::test]
async fn install_rejects_path_traversal_name_invalid_argument() {
    let svc = default_service();
    let err = svc
        .install_grammar(Request::new(InstallGrammarRequest {
            language: "../evil".to_string(),
            force: false,
        }))
        .await
        .expect_err("must reject");
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[tokio::test]
async fn install_rejects_out_of_allowlist_name_invalid_argument() {
    let svc = default_service();
    let err = svc
        .install_grammar(Request::new(InstallGrammarRequest {
            language: "notalanguage".to_string(),
            force: false,
        }))
        .await
        .expect_err("must reject");
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[tokio::test]
async fn install_rejects_checksum_verification_disabled_failed_precondition() {
    // A known language but checksum verification is configured off → fail
    // closed before any download.
    let svc = service_with(false);
    let err = svc
        .install_grammar(Request::new(InstallGrammarRequest {
            language: "rust".to_string(),
            force: false,
        }))
        .await
        .expect_err("must reject");
    assert_eq!(err.code(), Code::FailedPrecondition);
}
