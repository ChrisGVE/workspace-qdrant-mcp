# Code Audit Response — Round 1

All 12 findings were confirmed against the TS source and addressed.
No findings were false positives.  No findings were architecturally deferred.

## Dispositions

| # | Severity | Finding | Disposition |
|---|----------|---------|-------------|
| 1 | HIGH | Retrieve tool missing session project_id fallback | **FIXED** |
| 2 | HIGH | search_collection returns Err on leg failure | **FIXED** |
| 3 | HIGH | Exact search missing session project_id fallback | **FIXED** |
| 4 | MEDIUM | SearchInput strict deserialization (missing query / unknown mode/scope) | **FIXED** |
| 5 | MEDIUM | URL scheme case-sensitive comparison | **FIXED** |
| 8 | MEDIUM | Whitespace-only libraryName becomes empty-string tenant | **FIXED** |
| 9 | MEDIUM | HTTP routing accepts mcp_path wildcard subpaths | **FIXED** |
| 10 | MEDIUM | Heartbeat failure does not flip session.daemon_connected | **FIXED** |
| 12 | MEDIUM | rules.duplicationThreshold config not threaded | **FIXED** |
| 6 | LOW | List format whitelist not enforced | **FIXED** |
| 7 | LOW | Grep timer starts before missing-pattern check | **FIXED** |
| 11 | LOW | Functions > 80 lines (serve_http, run_search_pipeline) | **FIXED** |

## Detail

### #1 — Retrieve session project_id fallback (HIGH) — FIXED
- **TS evidence**: `retrieve.ts:124` — `projectId ?? (await this.resolveProjectId())`
- **Fix**: Added `session_project_id: Option<&str>` parameter to `retrieve_tool`; uses it as fallback when `input.project_id` is absent for projects/scratchpad collections.  Threaded from `dispatch.rs` via `ctx.session.project_id`.
- **Test**: `retrieve_uses_session_project_id_when_input_project_id_absent` + `retrieve_refuses_when_no_project_id_and_no_session`
- **Commit**: `fix(mcp-server): thread session_project_id fallback into retrieve_tool`

### #2 — search_collection leg failure behavior (HIGH) — FIXED
- **TS evidence**: `search-qdrant.ts:99-113` `searchDense`, `search-qdrant.ts:117-144` `searchSparse` each have own try/catch returning `[]` on failure. `searchCollection` (line 149-158) simply combines, never throws. `searchAllCollections` (search-helpers.ts:242-252) sets `uncertain` only when `searchCollection` itself throws.
- **Fix**: `search_collection` return type changed from `Result<Vec<TaggedResult>, String>` to `Vec<TaggedResult>` — leg errors silently swallowed.  `run_search_pipeline` no longer tracks `failed_collections`; `status` stays `None` after a normal pipeline run.
- **Test**: `m3_leg_failure_keeps_partial_results_status_ok` + `m3_dense_leg_failure_silent_sparse_results_kept` + `m3_all_legs_succeed_no_uncertain_status`
- **Commit**: `fix(mcp-server): swallow per-leg search failures matching TS parity`

### #3 — Exact search session project_id fallback (HIGH) — FIXED
- **TS evidence**: `search-exact.ts` calls `this.resolveProjectId()` as the fallback; `search_tool` resolved the session fallback in `resolve_project_id` but only passed it to `run_search_pipeline`, not to `search_exact`.
- **Fix**: Write resolved `project_id` (opts.project_id ?? session.project_id) back into `opts` before branching to `search_exact`.
- **Test**: `exact_mode_with_project_id_in_opts_sends_tenant` + `exact_mode_without_project_id_returns_unresolved`
- **Commit**: `fix(mcp-server): fold session_project_id into opts before exact search`

### #4 — SearchInput permissive parsing (MEDIUM) — FIXED
- **TS evidence**: `tool-builders/search.ts:130` — `query: (args?.['query'] as string) ?? ''`; `extractScopeOptions` lines 37-41 only sets `mode`/`scope` for recognized strings.
- **Fix**: Replaced serde deserialization in `parse_args` with manual extraction matching TS behavior: `query` defaults to `""` when absent; `mode`/`scope` silently dropped for unrecognized values.
- **Test**: `parse_args_missing_query_defaults_to_empty_string` + `parse_args_unrecognized_mode_silently_dropped` + `parse_args_unrecognized_scope_silently_dropped` + `parse_args_known_mode_and_scope_parsed_correctly`
- **Commit**: `fix(mcp-server): make SearchInput parsing permissive matching TS parity`

### #5 — URL scheme case-sensitivity (MEDIUM) — FIXED
- **TS evidence**: `store-handlers.ts:33-38` — `new URL(trimmed)` normalizes protocol to lowercase; `parsed.protocol` for `"HTTP://x"` is `'http:'`, so it is accepted.
- **Fix**: `validate_url` lowercases `scheme_raw` via `to_ascii_lowercase()` before the `http`/`https` check.  Error messages use the lowercase form.
- **Test**: `validate_url_uppercase_http_accepted`, `validate_url_uppercase_https_accepted`, `validate_url_mixed_case_scheme_accepted`, `validate_url_uppercase_non_http_rejected_with_lowercase_message`
- **Commit**: `fix(mcp-server): normalize URL scheme to lowercase before http/https check`

### #6 — List format whitelist (LOW) — FIXED
- **TS evidence**: `tool-builders/list.ts:31` — `if (format === 'tree' || format === 'summary' || format === 'flat') options.format = format;`  Anything else is NOT set.
- **Fix**: Added `.filter(|&f| matches!(f, "tree" | "summary" | "flat"))` in `ListInput::from_args` so invalid values are dropped.
- **Test**: `list_input_from_args_accepts_valid_formats` + `list_input_from_args_drops_invalid_format`
- **Commit**: `fix(mcp-server): whitelist list format values matching TS parity`

### #7 — Grep latency timer placement (LOW) — FIXED
- **TS evidence**: `grep.ts:132` — `if (!pattern) return grepError('Search pattern is required', 0);` with literal `0`, BEFORE `const startTime = Date.now()` at line 134.
- **Fix**: Moved pattern check before `let start = Instant::now()`; missing-pattern error uses literal `0` instead of `elapsed_ms(start)`.
- **Test**: `empty_pattern_latency_is_zero`
- **Commit**: `fix(mcp-server): return latency_ms=0 for missing-pattern grep error`

### #8 — Whitespace-only libraryName tenant fallback (MEDIUM) — FIXED
- **TS evidence**: `store-handlers.ts:92` — `const tenantId = libraryName?.trim() || sessionState.projectId || TENANT_GLOBAL;`  An empty/whitespace-only trim result is falsy in JS → falls back.
- **Fix**: `url_scratchpad.rs` now chains `.filter(|s| !s.is_empty())` after `.map(str::trim)` so trimmed-empty libraryName is treated as absent.
- **Test**: `url_whitespace_library_name_falls_back_to_session_project_id` + `url_whitespace_library_name_falls_back_to_global_when_no_session`
- **Commit**: `fix(mcp-server): treat whitespace-only libraryName as absent in URL store`

### #9 — HTTP routing wildcard subpath (MEDIUM) — FIXED
- **TS evidence**: `mcp-http-server.ts:192` — `if (urlPath !== mcpPath)` exact match only.
- **Architecture note**: rmcp uses `Mcp-Session-Id` headers (not URL sub-paths) for session routing; the `starts_with(mcp_path + "/")` clause was not required.
- **Fix**: Removed `|| !path_no_query.starts_with(&format!("{mcp_path}/"))` from the 404 check.
- **Test**: `mcp_subpath_returns_404` + `mcp_exact_path_passes_auth_returns_200` + `mcp_subpath_with_query_returns_404`
- **Commit**: `fix(mcp-server): require exact mcp_path match, drop wildcard subpath`

### #10 — Heartbeat failure state (MEDIUM) — FIXED
- **TS evidence**: `session-lifecycle.ts:254-258` — heartbeat catch sets `sessionState.daemonConnected = false` and calls `logDaemonStatus(false, { reason: 'heartbeat_failed' })`.
- **Fix**: `fire_heartbeat` signature changed to `&mut SessionState`; on heartbeat RPC error, sets `session.daemon_connected = false`.
- **Test**: `fire_heartbeat_skips_when_daemon_disconnected` + `fire_heartbeat_skips_when_project_id_absent`
- **Commit**: `fix(mcp-server): flip daemon_connected=false on heartbeat failure`

### #11 — Functions > 80 lines (LOW) — FIXED
- `serve_http` was 128 lines → split into `build_mcp_service` (57 lines) + `bind_and_serve` (43 lines) + `serve_http` coordinator (49 lines).
- `run_search_pipeline` was 147 lines → split into `embed_and_expand` + `search_all_collections` + `finalize_results` + `enrich_results` + `build_response` + `run_search_pipeline` coordinator (57 lines). All helpers < 80 lines.
- **Commit**: `refactor(mcp-server): split oversized functions for <80 line compliance`

### #12 — rules.duplicationThreshold config (MEDIUM) — FIXED
- **TS evidence**: `server-factory.ts:52` — `config.rules?.duplicationThreshold` wired into `RulesTool` constructor; `rules.ts:61` — `this.duplicationThreshold = config.duplicationThreshold ?? DEFAULT_DUPLICATION_THRESHOLD`.
- **Fix**: Added `WQM_RULES_DEDUP_THRESHOLD` env var in `apply_env_overrides` (range-validated: 0 < t <= 1) → sets `config.rules.duplication_threshold`. Threaded through `ToolsHandler.rules_dup_threshold` → `DispatchContext.rules_dup_threshold` → `rules_tool` parameter → `find_similar_rules` call.
- **Test**: Env var tests in `env_overrides.rs`; `add_rule_config_threshold_blocks_when_score_above_threshold` + `add_rule_config_threshold_allows_when_score_below_threshold`
- **Commit**: `feat(mcp-server): thread rules.duplicationThreshold config override`
