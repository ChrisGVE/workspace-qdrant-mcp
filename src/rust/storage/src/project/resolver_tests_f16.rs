//! Tests for `ProjectRegistry::resolve_by_handle` (AC-F16.6).
//!
//! File: `wqm-storage/src/project/resolver_tests_f16.rs`
//! Context: included as a sub-module from `resolver_tests.rs` to keep each
//!   file under the 500-line codesize budget.

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;
use tempfile::NamedTempFile;
use wqm_common::handle::{HandleResolveError, ResolveAction, Resolved};

use super::{create_schema, create_writable_pool, seed_state_db};
use crate::project::resolver::ProjectRegistry;

// ---------------------------------------------------------------------------
// resolve_by_handle integration tests (AC-F16.6)
// ---------------------------------------------------------------------------

/// Seed a state.db with named projects via `seed_state_db`, then open a
/// read-only `ProjectRegistry` against it.
async fn open_registry_with_projects(
    pool: &SqlitePool,
    projects: &[(&str, &str, &str, &str, &str)],
) {
    seed_state_db(pool, projects).await;
}

// Exact project name → tenant_id via resolve_by_handle (round-trip).
#[tokio::test]
async fn t_f16_resolve_by_handle_exact_match_returns_tenant_id() {
    let tmp = NamedTempFile::new().expect("tempfile");

    {
        let w_pool = create_writable_pool(tmp.path()).await;
        open_registry_with_projects(
            &w_pool,
            &[
                (
                    "mathlex",
                    "tid-mathlex",
                    "/d/ml/store.db",
                    "/loc/ml",
                    "main",
                ),
                (
                    "mathlex-eval",
                    "tid-mathlex-eval",
                    "/d/mle/store.db",
                    "/loc/mle",
                    "main",
                ),
            ],
        )
        .await;
        w_pool.close().await;
    }

    let registry = ProjectRegistry::open(tmp.path()).await.expect("open");

    // Exact match: "mathlex" must NOT pick "mathlex-eval".
    let result = registry
        .resolve_by_handle("mathlex", ResolveAction::Read)
        .await
        .expect("resolve ok");

    match result {
        Resolved::Exact(c) => {
            assert_eq!(
                c.key, "tid-mathlex",
                "exact match must return the correct tenant_id"
            );
            assert_eq!(c.handle, "mathlex");
        }
        other => panic!("expected Exact, got {other:?}"),
    }
}

// Case-insensitive exact: "MathLex" resolves to "mathlex" (tid-mathlex).
#[tokio::test]
async fn t_f16_resolve_by_handle_case_insensitive() {
    let tmp = NamedTempFile::new().expect("tempfile");

    {
        let w_pool = create_writable_pool(tmp.path()).await;
        open_registry_with_projects(
            &w_pool,
            &[(
                "mathlex",
                "tid-mathlex",
                "/d/ml/store.db",
                "/loc/ml",
                "main",
            )],
        )
        .await;
        w_pool.close().await;
    }

    let registry = ProjectRegistry::open(tmp.path()).await.expect("open");
    let result = registry
        .resolve_by_handle("MathLex", ResolveAction::Read)
        .await
        .expect("resolve ok");

    match result {
        Resolved::Exact(c) => assert_eq!(c.key, "tid-mathlex"),
        other => panic!("expected Exact, got {other:?}"),
    }
}

// Fuzzy typo resolves under Read when single candidate clears threshold.
#[tokio::test]
async fn t_f16_resolve_by_handle_fuzzy_typo_read() {
    let tmp = NamedTempFile::new().expect("tempfile");

    {
        let w_pool = create_writable_pool(tmp.path()).await;
        open_registry_with_projects(
            &w_pool,
            &[(
                "mathlex",
                "tid-mathlex",
                "/d/ml/store.db",
                "/loc/ml",
                "main",
            )],
        )
        .await;
        w_pool.close().await;
    }

    let registry = ProjectRegistry::open(tmp.path()).await.expect("open");
    // "mathlx" is a 1-char deletion — should clear the 0.92 threshold.
    let result = registry
        .resolve_by_handle("mathlx", ResolveAction::Read)
        .await
        .expect("fuzzy resolve ok");

    match result {
        Resolved::Exact(c) => assert_eq!(c.key, "tid-mathlex"),
        other => panic!("expected Exact for near-match Read, got {other:?}"),
    }
}

// Completely unrelated name → NotFound.
#[tokio::test]
async fn t_f16_resolve_by_handle_not_found() {
    let tmp = NamedTempFile::new().expect("tempfile");

    {
        let w_pool = create_writable_pool(tmp.path()).await;
        open_registry_with_projects(
            &w_pool,
            &[(
                "mathlex",
                "tid-mathlex",
                "/d/ml/store.db",
                "/loc/ml",
                "main",
            )],
        )
        .await;
        w_pool.close().await;
    }

    let registry = ProjectRegistry::open(tmp.path()).await.expect("open");
    let err = registry
        .resolve_by_handle("zzzzzzz", ResolveAction::Read)
        .await
        .expect_err("should not resolve");

    matches!(err, HandleResolveError::NotFound { .. });
}

// Write tier on a near-miss returns BestGuess (not a silent resolve).
#[tokio::test]
async fn t_f16_resolve_by_handle_write_near_miss_best_guess() {
    let tmp = NamedTempFile::new().expect("tempfile");

    {
        let w_pool = create_writable_pool(tmp.path()).await;
        open_registry_with_projects(
            &w_pool,
            &[(
                "mathlex",
                "tid-mathlex",
                "/d/ml/store.db",
                "/loc/ml",
                "main",
            )],
        )
        .await;
        w_pool.close().await;
    }

    let registry = ProjectRegistry::open(tmp.path()).await.expect("open");
    let result = registry
        .resolve_by_handle("mathlx", ResolveAction::Write)
        .await
        .expect("resolve ok");

    match result {
        Resolved::BestGuess { best, .. } => {
            assert_eq!(
                best.key, "tid-mathlex",
                "BestGuess.best must carry the correct tenant_id key"
            );
        }
        Resolved::Exact(_) => {
            panic!("Write tier on non-exact input must return BestGuess, not Exact")
        }
    }
}
