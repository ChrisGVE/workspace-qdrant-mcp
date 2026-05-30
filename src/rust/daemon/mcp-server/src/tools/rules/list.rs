//! `list_rules` operation and `find_similar_rules` dup-check for the rules tool.
//!
//! ## list_rules — Qdrant-first (FIX 1)
//!
//! Primary path: `RulesQdrant::scroll_rules` (matches TS `qdrantClient.scroll`
//! in rules-list.ts:119-122).
//!
//! Fallback on Qdrant error:
//! - mirror rows exist → `{success:true, …, message:"Found N rule(s) from local mirror
//!   (Qdrant unavailable)"}` (rules-list.ts:95)
//! - mirror null/throws → `{success:false, …, message:"Failed to list rules: <msg>"}` (rules-list.ts:130-133)
//!
//! ## find_similar_rules — embed + search (FIX 2)
//!
//! 1. embed content via `RulesDaemon::embed_text`; empty embedding → return `[]`
//! 2. Qdrant search with `score_threshold = duplication_threshold`, `limit = 5`,
//!    filter = `buildDuplicateScopeFilter` (rules.ts:140, :166-185)
//! 3. round `score` to 3 decimals: `(score * 1000.0).round() / 1000.0` (rules.ts:153)
//! 4. any error → return `[]` (allow add to proceed)

use wqm_common::constants::TENANT_GLOBAL;

use super::helpers::mirror_to_rule_item;
use super::traits::{RulesDaemon, RulesQdrant, RulesReader};
use super::types::{RuleItem, RulesInput, RulesResponse};

use crate::tools::envelope::ok_text;
use rmcp::model::CallToolResult;

/// DEFAULT_DUPLICATION_THRESHOLD — mirrors `DEFAULT_DUPLICATION_THRESHOLD = 0.7`
/// in `src/typescript/mcp-server/src/tools/rules.ts:32`.
pub const DEFAULT_DUPLICATION_THRESHOLD: f64 = 0.7;

// ─────────────────────────────────────────────────────────────────────────────
// List — Qdrant-first, mirror fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Payload field constants — mirrors `FIELD_*` from `native-bridge.ts`.
const FIELD_CONTENT: &str = "content";
const FIELD_PROJECT_ID: &str = "project_id";
const FIELD_TITLE: &str = "title";

/// Build a Qdrant filter for scope-based list.
///
/// Mirrors `buildListFilter` in rules-list.ts:14-28.
fn build_list_filter(
    scope: &str,
    project_id: Option<&str>,
) -> Option<qdrant_client::qdrant::Filter> {
    use qdrant_client::qdrant::{Condition, Filter};

    let mut must: Vec<Condition> = Vec::new();

    if scope == TENANT_GLOBAL {
        must.push(Condition::matches("scope", TENANT_GLOBAL.to_string()));
    } else if scope == "project" {
        if let Some(pid) = project_id {
            must.push(Condition::matches("scope", "project".to_string()));
            must.push(Condition::matches(FIELD_PROJECT_ID, pid.to_string()));
        }
    }

    if must.is_empty() {
        None
    } else {
        Some(Filter::must(must))
    }
}

/// Map a `QdrantRetrievedPoint` payload to a `RuleItem`.
///
/// Mirrors `pointToRule` in rules-list.ts:31-55.
fn point_to_rule_item(point: &crate::qdrant::client::QdrantRetrievedPoint) -> RuleItem {
    let payload = &point.payload;

    let content = payload
        .get(FIELD_CONTENT)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let scope = payload
        .get("scope")
        .and_then(|v| v.as_str())
        .unwrap_or(TENANT_GLOBAL)
        .to_string();

    let label = payload
        .get("label")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let project_id = payload
        .get(FIELD_PROJECT_ID)
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let title = payload
        .get(FIELD_TITLE)
        .and_then(|v| v.as_str())
        .map(str::to_string);

    // Guard: TS `if (tagsStr)` (rules-list.ts:46-47) — empty-string tags field
    // yields NO tags. An empty string split yields `[""]` without this guard.
    let tags = payload
        .get("tags")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.split(',').map(str::to_string).collect::<Vec<_>>());

    let priority = payload.get("priority").and_then(|v| v.as_i64());

    let created_at = payload
        .get("created_at")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let updated_at = payload
        .get("updated_at")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    RuleItem {
        id: point.id.clone(),
        content,
        scope,
        label,
        project_id,
        title,
        tags,
        priority,
        created_at,
        updated_at,
        similarity: None,
    }
}

/// List rules: query Qdrant first, fall back to SQLite mirror.
///
/// Mirrors `listRules` in rules-list.ts:103-135.
pub async fn list_rules<R, Q>(
    input: &RulesInput,
    reader: &R,
    qdrant: &Q,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    R: RulesReader,
    Q: RulesQdrant,
{
    // Resolve project_id for project-scope list — rules-list.ts:111-114
    let resolved_pid: Option<String> = if input.scope == "project" {
        input
            .project_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| session_project_id.map(str::to_string))
    } else {
        None
    };

    let filter = build_list_filter(input.scope.as_str(), resolved_pid.as_deref());

    // PRIMARY: scroll Qdrant — rules-list.ts:119-122
    match qdrant.scroll_rules(filter, input.limit as u32).await {
        Ok(points) => {
            let rules: Vec<RuleItem> = points.iter().map(point_to_rule_item).collect();
            let count = rules.len();
            ok_text(&RulesResponse {
                success: true,
                action: "list".to_string(),
                label: None,
                rules: Some(rules),
                similar_rules: None,
                message: Some(format!("Found {count} rule(s)")),
                fallback_mode: None,
                queue_id: None,
            })
        }
        Err(err) => {
            // FALLBACK: SQLite mirror — rules-list.ts:126-133
            let scope_str: Option<&str> = Some(input.scope.as_str());
            let tenant_str: Option<&str> = resolved_pid.as_deref();
            let rows = reader.list_from_mirror(scope_str, tenant_str, input.limit);

            if !rows.is_empty() {
                let rules: Vec<RuleItem> = rows.iter().map(mirror_to_rule_item).collect();
                let count = rules.len();
                ok_text(&RulesResponse {
                    success: true,
                    action: "list".to_string(),
                    label: None,
                    rules: Some(rules),
                    similar_rules: None,
                    message: Some(format!(
                        "Found {count} rule(s) from local mirror (Qdrant unavailable)"
                    )),
                    fallback_mode: None,
                    queue_id: None,
                })
            } else {
                ok_text(&RulesResponse {
                    success: false,
                    action: "list".to_string(),
                    label: None,
                    rules: Some(vec![]),
                    similar_rules: None,
                    message: Some(format!("Failed to list rules: {err}")),
                    fallback_mode: None,
                    queue_id: None,
                })
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// find_similar_rules — embed + Qdrant search for dup-check (add action)
// ─────────────────────────────────────────────────────────────────────────────

/// Build the Qdrant filter for duplicate-scope detection.
///
/// Mirrors `buildDuplicateScopeFilter` in rules.ts:166-185.
///
/// - project scope + project_id → must[should[project_id=X, scope=global]]
/// - project scope + no project_id → None (broaden rather than block)
/// - global scope → must[scope=global]
pub fn build_duplicate_scope_filter(
    scope: &str,
    project_id: Option<&str>,
) -> Option<qdrant_client::qdrant::Filter> {
    use qdrant_client::qdrant::{Condition, Filter};

    if scope == "project" {
        let pid = project_id?;
        // must[ should[ project_id=pid, scope=global ] ]
        let should_inner = Filter::should(vec![
            Condition::matches(FIELD_PROJECT_ID, pid.to_string()),
            Condition::matches("scope", "global".to_string()),
        ]);
        return Some(Filter::must(vec![Condition::from(should_inner)]));
    }
    // global add — only match other global rules
    Some(Filter::must(vec![Condition::matches(
        "scope",
        "global".to_string(),
    )]))
}

/// Search for rules similar to `content` using embedding similarity.
///
/// Mirrors `findSimilarRules` in rules.ts:119-158.
///
/// Returns `Vec<RuleItem>` with `similarity` set (3-decimal rounded).
/// Returns empty vec on any embed/search error (allow add to proceed).
pub async fn find_similar_rules<D, Q>(
    daemon: &mut D,
    qdrant: &Q,
    content: &str,
    scope: &str,
    project_id: Option<&str>,
    duplication_threshold: f64,
) -> Vec<RuleItem>
where
    D: RulesDaemon,
    Q: RulesQdrant,
{
    // 1. embed — empty embedding → skip dup-check (rules.ts:126)
    let embedding = daemon.embed_text(content.to_string()).await;
    if embedding.is_empty() {
        return Vec::new();
    }

    // 2. build scope filter — rules.ts:140-141
    let filter = build_duplicate_scope_filter(scope, project_id);

    // 3. search — rules.ts:128-142
    let threshold = duplication_threshold as f32;
    let results = match qdrant.search_rules(embedding, 5, threshold, filter).await {
        Ok(pts) => pts,
        Err(_) => return Vec::new(), // any error → allow add (rules.ts:155-158)
    };

    // 4. filter by threshold + map to RuleItem with similarity rounded to 3 decimals
    results
        .into_iter()
        .filter(|pt| pt.score >= duplication_threshold)
        .map(|pt| {
            let payload = &pt.payload;
            let content_str = payload
                .get(FIELD_CONTENT)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let scope_str = payload
                .get("scope")
                .and_then(|v| v.as_str())
                .unwrap_or(TENANT_GLOBAL)
                .to_string();
            let label = payload
                .get("label")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let title = payload
                .get(FIELD_TITLE)
                .and_then(|v| v.as_str())
                .map(str::to_string);
            // round to 3 decimals: (score * 1000.0).round() / 1000.0 — rules.ts:153
            let similarity = (pt.score * 1000.0).round() / 1000.0;
            // TS `findSimilarRules` (rules.ts:147-154) maps hits to exactly
            // { id, content, scope, label, title, similarity } — no projectId.
            RuleItem {
                id: pt.id.clone(),
                content: content_str,
                scope: scope_str,
                label,
                project_id: None,
                title,
                tags: None,
                priority: None,
                created_at: None,
                updated_at: None,
                similarity: Some(similarity),
            }
        })
        .collect()
}
