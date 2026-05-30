//! search_tests_m3_m4_m5 overflow — exact_mode_without_project_id guard.
//!
//! Included from `search_tests_m3_m4_m5.rs` via
//! `#[path = "search_tests_m3_m4_m5_part2.rs"] mod part2;`.

use std::sync::{Arc, Mutex};

use crate::proto::{TextSearchRequest, TextSearchResponse};
use crate::tools::search::exact::{search_exact, ExactSearchDaemon};
use crate::tools::search::options::SearchOptions;
use crate::tools::search::types::SearchScope;

use super::super::opts_hybrid;

// ---------------------------------------------------------------------------
// #3 continued: exact search without project_id returns unresolved
// ---------------------------------------------------------------------------

/// Exact daemon stub that records the tenant_id sent in the request.
struct TenantCapturingExactDaemon2 {
    captured_tenant: Arc<Mutex<Option<String>>>,
}

impl ExactSearchDaemon for TenantCapturingExactDaemon2 {
    async fn text_search(
        &mut self,
        request: TextSearchRequest,
    ) -> Result<TextSearchResponse, tonic::Status> {
        *self.captured_tenant.lock().unwrap() = request.tenant_id.clone();
        Ok(TextSearchResponse {
            matches: vec![],
            total_matches: 0,
            truncated: false,
            query_time_ms: 0,
        })
    }
}

#[tokio::test]
async fn exact_mode_without_project_id_returns_unresolved() {
    // When opts.project_id is None AND scope=Project, exact mode must return an
    // unresolved response (not attempt RPC).  This verifies the guard is intact.
    let captured = Arc::new(Mutex::new(None));
    let mut daemon = TenantCapturingExactDaemon2 {
        captured_tenant: captured.clone(),
    };

    let opts = SearchOptions {
        exact: true,
        project_id: None, // absent — unresolved
        scope: SearchScope::Project,
        ..opts_hybrid("fn main", 10)
    };

    let resp = search_exact(&mut daemon, &opts).await;

    // No RPC should fire (daemon not called).
    assert!(
        captured.lock().unwrap().is_none(),
        "no RPC must fire when tenant is unresolved"
    );
    assert_eq!(
        resp.status.as_deref(),
        Some("uncertain"),
        "unresolved exact mode must set status='uncertain'"
    );
}
