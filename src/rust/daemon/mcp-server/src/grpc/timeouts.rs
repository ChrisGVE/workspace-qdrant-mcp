//! Per-method gRPC timeout resolution.
//!
//! Mirrors `DaemonClientBase.getMethodTimeout` from
//! `src/typescript/mcp-server/src/clients/daemon-client/connection.ts` lines 159-165.
//!
//! # Timeout policy
//!
//! Resolution order (matches TS exactly):
//! 1. `override_timeout` — caller-supplied one-shot ceiling.
//! 2. `method_name` is exactly `"search"` → 10 s (2× default).
//! 3. Default 5 s.
//!
//! The TS `getMethodTimeout` uses an exact equality check
//! (`methodName === 'search'`), NOT a substring match — so wire names that
//! merely *contain* "search" (e.g. `resolveSearchScope`) stay at the 5 s
//! default.  Matching that precisely is load-bearing for timeout parity.
//!
//! # Why `tokio::time::timeout`, not tonic deadline
//!
//! The TypeScript implementation uses `Promise.race` (see `grpcUnaryWithTimeout`
//! in connection.ts): the timer fires and rejects the outer promise, but the
//! underlying gRPC call is **abandoned in place** — no cancellation signal is
//! sent to the server and **no `grpc-timeout` header** is added to the request.
//!
//! Tonic's `.timeout(d)` on a `Channel` takes a different path: it sets a
//! deadline on the request metadata (`grpc-timeout` header), which the server
//! honours to cancel its own work, and the client also cancels the in-flight
//! request future.
//!
//! To preserve the same client-side-abandon semantics (AC-DC4), all callers
//! wrap their `tonic` futures with `tokio::time::timeout(...)` rather than
//! using tonic's deadline propagation.

use std::time::Duration;

/// Default per-call timeout: 5 seconds (matches TS `DEFAULT_TIMEOUT_MS = 5000`).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Timeout for search operations: 10 seconds (2× default, matches TS `search` override).
pub const SEARCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Resolve the effective timeout for a gRPC call.
///
/// # Arguments
/// * `method_name` – the gRPC method (or operation) name, e.g. `"search"`, `"health"`.
/// * `override_timeout` – caller-supplied one-shot override; takes highest priority.
///
/// # Returns
/// The resolved [`Duration`] to pass to `tokio::time::timeout`.
pub fn resolve_timeout(method_name: &str, override_timeout: Option<Duration>) -> Duration {
    if let Some(d) = override_timeout {
        return d;
    }
    // Exact match only — mirrors TS `methodName === 'search'`.  Substring
    // matches (e.g. "resolveSearchScope") must NOT be promoted to 10 s.
    if method_name == "search" {
        return SEARCH_TIMEOUT;
    }
    DEFAULT_TIMEOUT
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── override takes precedence ────────────────────────────────────────────

    #[test]
    fn override_beats_default() {
        let d = resolve_timeout("health", Some(Duration::from_secs(3)));
        assert_eq!(d, Duration::from_secs(3));
    }

    #[test]
    fn override_beats_search_method() {
        let d = resolve_timeout("search", Some(Duration::from_secs(1)));
        assert_eq!(d, Duration::from_secs(1));
    }

    // ── only the exact wire name "search" gets 10 s ──────────────────────────

    #[test]
    fn exact_search_method_name() {
        // TS: methodName === 'search' → timeoutMs * 2  (5000 * 2 = 10000 ms)
        let d = resolve_timeout("search", None);
        assert_eq!(d, Duration::from_secs(10));
    }

    #[test]
    fn resolve_search_scope_stays_default() {
        // Regression (parity): TS uses exact equality, so "resolveSearchScope"
        // — though it contains "search" — resolves to the 5 s default, NOT 10 s.
        let d = resolve_timeout("resolveSearchScope", None);
        assert_eq!(d, DEFAULT_TIMEOUT);
    }

    #[test]
    fn camelcase_text_search_string_is_not_the_wire_name() {
        // The text-search RPC's wire name is "search" (10 s); the literal string
        // "textSearch" is not "search", so it gets the default.
        let d = resolve_timeout("textSearch", None);
        assert_eq!(d, DEFAULT_TIMEOUT);
    }

    #[test]
    fn search_is_case_sensitive() {
        // TS `=== 'search'` is case-sensitive: "Search" is not a match.
        assert_eq!(resolve_timeout("Search", None), DEFAULT_TIMEOUT);
        assert_eq!(
            resolve_timeout("AdvancedSEARCHQuery", None),
            DEFAULT_TIMEOUT
        );
    }

    // ── non-search methods get 5 s ───────────────────────────────────────────

    #[test]
    fn health_gets_default() {
        let d = resolve_timeout("health", None);
        assert_eq!(d, DEFAULT_TIMEOUT);
    }

    #[test]
    fn ingest_text_gets_default() {
        let d = resolve_timeout("ingestText", None);
        assert_eq!(d, DEFAULT_TIMEOUT);
    }

    #[test]
    fn register_project_gets_default() {
        let d = resolve_timeout("registerProject", None);
        assert_eq!(d, DEFAULT_TIMEOUT);
    }

    #[test]
    fn get_status_gets_default() {
        let d = resolve_timeout("getStatus", None);
        assert_eq!(d, DEFAULT_TIMEOUT);
    }

    #[test]
    fn empty_method_name_gets_default() {
        let d = resolve_timeout("", None);
        assert_eq!(d, DEFAULT_TIMEOUT);
    }

    // ── constant sanity checks ───────────────────────────────────────────────

    #[test]
    fn default_timeout_is_5s() {
        assert_eq!(DEFAULT_TIMEOUT, Duration::from_secs(5));
    }

    #[test]
    fn search_timeout_is_10s() {
        assert_eq!(SEARCH_TIMEOUT, Duration::from_secs(10));
    }

    #[test]
    fn search_timeout_is_double_default() {
        assert_eq!(SEARCH_TIMEOUT, DEFAULT_TIMEOUT * 2);
    }
}
