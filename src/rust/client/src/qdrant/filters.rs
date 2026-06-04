//! Qdrant filter construction for MCP search operations.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/search-filters.ts` exactly.
//! Each TypeScript helper function has a direct Rust counterpart; the
//! `FilterParams` struct maps to the TS `FilterParams` interface.
//!
//! Filter anatomy (matches TS `buildFilter`):
//!   - `must`      — all conditions that MUST match (logical AND)
//!   - `must_not`  — conditions that MUST NOT match (logical NOT)
//!   - `should`    — at least one must match within a nested group (logical OR)

use qdrant_client::qdrant::{Condition, Filter};
use wqm_common::constants::{
    field, COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_SCRATCHPAD,
};

// ── Public parameter type ─────────────────────────────────────────────────────

/// Typed parameters for building a Qdrant filter.
///
/// Maps 1-to-1 to the TypeScript `FilterParams` interface in `search-types.ts`.
#[derive(Debug, Clone, Default)]
pub struct FilterParams {
    /// Collection being filtered (drives library-specific conditions).
    pub collection: String,
    /// Search scope: "project", "group", or "all".
    pub scope: String,
    /// Tenant ID for project-scope filtering.
    pub project_id: Option<String>,
    /// Multiple tenant IDs for group-scope filtering.
    pub group_tenant_ids: Option<Vec<String>>,
    /// Git branch filter (`None` or `"*"` skips the branch condition).
    pub branch: Option<String>,
    /// File extension discriminator.
    pub file_type: Option<String>,
    /// Library name filter (libraries collection only).
    pub library_name: Option<String>,
    /// Library path prefix filter (libraries collection only).
    pub library_path: Option<String>,
    /// Single tag: matched against both `concept_tags` and `tags`.
    pub tag: Option<String>,
    /// Multiple tags (OR logic): matched against both `concept_tags` and `tags`.
    pub tags: Option<Vec<String>>,
    /// Path glob pattern; only the deterministic prefix is used.
    pub path_glob: Option<String>,
    /// Component ID prefix filter.
    pub component: Option<String>,
    /// Instance-aware base_point IDs (projects collection only).
    pub base_points: Option<Vec<String>>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Extract the deterministic path prefix from a glob pattern.
///
/// Returns everything before the first glob metacharacter (`* ? [ {`),
/// trimmed to the last path separator for a clean directory prefix.
///
/// Examples
/// ```
/// # use wqm_client::qdrant::filters::extract_glob_prefix;
/// assert_eq!(extract_glob_prefix("src/**/*.rs"), "src/");
/// assert_eq!(extract_glob_prefix("**/*.rs"),     "");
/// assert_eq!(extract_glob_prefix("src/main.rs"), "src/main.rs");
/// assert_eq!(extract_glob_prefix(""),            "");
/// ```
pub fn extract_glob_prefix(glob: &str) -> &str {
    let meta_pos = glob.find(|c| matches!(c, '*' | '?' | '[' | '{'));
    match meta_pos {
        None => glob, // no metachar → whole string is literal
        Some(pos) => {
            let before = &glob[..pos];
            match before.rfind('/') {
                Some(slash_pos) => &glob[..=slash_pos],
                None => "",
            }
        }
    }
}

/// Build a `qdrant_client::qdrant::Filter` from [`FilterParams`].
///
/// Returns `None` when no conditions are needed (caller should omit the
/// filter argument entirely so Qdrant does an unfiltered scan).
///
/// When `scope == "group"` but `group_tenant_ids` is empty or absent,
/// the tenant filter is silently omitted (returns `None` for the project
/// condition). This avoids a panic on a code path reachable at runtime.
/// Full group-scope handling via `resolveSearchScope` is deferred.
pub fn build_filter(params: &FilterParams) -> Option<Filter> {
    let must = build_must_conditions(params);
    let must_not = build_must_not_conditions(params);

    if must.is_empty() && must_not.is_empty() {
        return None;
    }

    let filter = Filter {
        must,
        must_not,
        should: vec![],
        min_should: None,
    };
    Some(filter)
}

/// Determine which collections to search based on scope.
///
/// Mirrors `determineCollections` in `search-filters.ts`.
pub fn determine_collections(
    explicit_collection: Option<&str>,
    scope: &str,
    include_libraries: bool,
) -> Vec<String> {
    if let Some(c) = explicit_collection {
        return vec![c.to_string()];
    }
    match scope {
        "project" | "group" => {
            if include_libraries {
                vec![
                    COLLECTION_PROJECTS.to_string(),
                    COLLECTION_LIBRARIES.to_string(),
                ]
            } else {
                vec![COLLECTION_PROJECTS.to_string()]
            }
        }
        "all" => vec![
            COLLECTION_PROJECTS.to_string(),
            COLLECTION_LIBRARIES.to_string(),
            COLLECTION_SCRATCHPAD.to_string(),
        ],
        _ => vec![COLLECTION_PROJECTS.to_string()],
    }
}

// ── Must / must_not condition builders ───────────────────────────────────────

fn build_must_conditions(params: &FilterParams) -> Vec<Condition> {
    let mut conditions: Vec<Condition> = Vec::new();

    if let Some(c) = build_project_condition(params) {
        conditions.push(c);
    }
    if let Some(c) = build_base_point_condition(params) {
        conditions.push(c);
    }
    if let Some(c) = build_branch_condition(params) {
        conditions.push(c);
    }
    if let Some(c) = build_file_type_condition(params) {
        conditions.push(c);
    }
    if let Some(c) = build_library_name_condition(params) {
        conditions.push(c);
    }
    if let Some(c) = build_library_path_condition(params) {
        conditions.push(c);
    }

    conditions.extend(build_tag_conditions(params));

    if let Some(c) = build_component_condition(params) {
        conditions.push(c);
    }
    if let Some(c) = build_path_glob_condition(params) {
        conditions.push(c);
    }

    conditions
}

fn build_must_not_conditions(params: &FilterParams) -> Vec<Condition> {
    // Mirror TS: must_not [deleted=true] applies only to the libraries collection.
    if params.collection != COLLECTION_LIBRARIES {
        return vec![];
    }
    vec![Condition::matches(field::DELETED, true)]
}

// ── Individual condition builders (mirror TS private functions) ───────────────

/// Mirrors `buildProjectCondition` in `search-filters.ts`.
///
/// PANIC FIX (S3 partial): when group scope reaches here with empty/missing
/// tenant ids, return `None` (skip the tenant filter) instead of panicking.
/// A panic is never acceptable in production. The TS equivalent throws, but
/// Rust must not abort via `panic!` on a code path reachable at runtime.
/// Full group-scope resolution via `resolveSearchScope` is deferred — see
/// DEFERRED comment in `flow.rs`.
fn build_project_condition(params: &FilterParams) -> Option<Condition> {
    if params.scope == "group" {
        // DEFERRED (task 30 follow-up, GitHub #81): full group/all scope via resolveSearchScope,
        // including relevance decay (applyRelevanceDecay) and base_points/basePointsDegraded.
        // Until resolveSearchScope is wired, group_tenant_ids must be pre-populated by the
        // caller. Empty/None is handled gracefully (returns None → no tenant filter).
        let ids = params
            .group_tenant_ids
            .as_deref()
            .filter(|ids| !ids.is_empty())?; // returns None gracefully when empty/missing
        return Some(Condition::matches(field::TENANT_ID, ids.to_vec()));
    }
    if params.scope != "project" {
        return None;
    }
    let project_id = params.project_id.as_deref()?;
    Some(Condition::matches(field::TENANT_ID, project_id.to_string()))
}

/// Mirrors `buildBasePointCondition` in `search-filters.ts`.
fn build_base_point_condition(params: &FilterParams) -> Option<Condition> {
    let points = params.base_points.as_deref().filter(|p| !p.is_empty())?;
    Some(Condition::matches(field::BASE_POINT, points.to_vec()))
}

/// Mirrors `buildBranchCondition` in `search-filters.ts`.
fn build_branch_condition(params: &FilterParams) -> Option<Condition> {
    let branch = params.branch.as_deref()?;
    if branch == "*" {
        return None;
    }
    // TS uses FIELD_BRANCHES (the array field) — use `field::BRANCHES` not `field::BRANCH`.
    Some(Condition::matches(field::BRANCHES, branch.to_string()))
}

/// Mirrors `buildFileTypeCondition` in `search-filters.ts`.
fn build_file_type_condition(params: &FilterParams) -> Option<Condition> {
    let file_type = params.file_type.as_deref()?;
    Some(Condition::matches(field::FILE_TYPE, file_type.to_string()))
}

/// Mirrors `buildLibraryNameCondition` in `search-filters.ts`.
fn build_library_name_condition(params: &FilterParams) -> Option<Condition> {
    if params.collection != COLLECTION_LIBRARIES {
        return None;
    }
    let name = params.library_name.as_deref()?;
    Some(Condition::matches(field::LIBRARY_NAME, name.to_string()))
}

/// Mirrors `buildLibraryPathCondition` in `search-filters.ts` (text match).
fn build_library_path_condition(params: &FilterParams) -> Option<Condition> {
    if params.collection != COLLECTION_LIBRARIES {
        return None;
    }
    let path = params.library_path.as_deref()?;
    Some(Condition::matches_text(field::LIBRARY_PATH, path))
}

/// Mirrors `buildTagConditions` in `search-filters.ts` (returns 0–2 conditions).
///
/// Each condition is a `should` group over both `concept_tags` and `tags` fields.
fn build_tag_conditions(params: &FilterParams) -> Vec<Condition> {
    let mut conditions: Vec<Condition> = Vec::new();

    // Single tag: `should [concept_tags=tag, tags=tag]`
    if let Some(ref tag) = params.tag {
        let should_conditions = vec![
            Condition::matches(field::CONCEPT_TAGS, tag.clone()),
            Condition::matches(field::TAGS, tag.clone()),
        ];
        conditions.push(Condition::from(Filter::should(should_conditions)));
    }

    // Multiple tags: `should [concept_tags matches any, tags matches any]`
    if let Some(ref tags) = params.tags {
        if !tags.is_empty() {
            let should_conditions: Vec<Condition> = tags
                .iter()
                .flat_map(|t| {
                    [
                        Condition::matches(field::CONCEPT_TAGS, t.clone()),
                        Condition::matches(field::TAGS, t.clone()),
                    ]
                })
                .collect();
            conditions.push(Condition::from(Filter::should(should_conditions)));
        }
    }

    conditions
}

/// Mirrors `buildComponentCondition` in `search-filters.ts`.
///
/// Matches `component_id` by exact value OR by text prefix (`component.`).
fn build_component_condition(params: &FilterParams) -> Option<Condition> {
    let component = params.component.as_deref()?;
    let dot_prefix = format!("{component}.");
    let should_conditions = vec![
        Condition::matches("component_id", component.to_string()),
        Condition::matches_text("component_id", dot_prefix),
    ];
    Some(Condition::from(Filter::should(should_conditions)))
}

/// Mirrors `buildPathGlobCondition` in `search-filters.ts`.
///
/// Extracts the deterministic prefix from the glob and matches it as a
/// `text` (prefix/substring) condition on `file_path`.  Returns `None`
/// when the prefix is empty (e.g. `**/*.rs`).
fn build_path_glob_condition(params: &FilterParams) -> Option<Condition> {
    let glob = params.path_glob.as_deref()?;
    let prefix = extract_glob_prefix(glob);
    if prefix.is_empty() {
        return None;
    }
    Some(Condition::matches_text(field::FILE_PATH, prefix))
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "filters_tests.rs"]
mod tests;
