//! `rules` MCP tool handler.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/rules.ts`,
//! `rules-mutations.ts`, `rules-list.ts`, and `rules-mutation-helpers.ts`.
//!
//! # Action dispatch (rules.ts:65-100)
//!
//! | action    | Rust fn        | TS fn            |
//! |-----------|----------------|------------------|
//! | `"add"`   | `add_rule`     | `addRule`        |
//! | `"update"`| `update_rule`  | `updateRule`     |
//! | `"remove"`| `remove_rule`  | `removeRule`     |
//! | `"list"`  | `list_rules`   | `listRules`      |
//!
//! # Parity fixes implemented in this module
//!
//! FIX 1 — list queries Qdrant first, falls back to SQLite mirror.
//! FIX 2 — add runs findSimilarRules duplicate detection before persisting.
//!
//! # DEFAULT_DUPLICATION_THRESHOLD
//!
//! `0.7` — mirrors `DEFAULT_DUPLICATION_THRESHOLD` in rules.ts:32.
//! Sourced from `list::DEFAULT_DUPLICATION_THRESHOLD` (single source of truth).

mod helpers;
mod list;
mod mutations;
mod traits;
mod types;

// ── Public re-exports (used by tools/mod.rs and tests) ───────────────────────
pub use traits::{RulesDaemon, RulesQdrant, RulesReader};
pub use types::{RuleItem, RulesInput, RulesResponse};

use rmcp::model::CallToolResult;

use crate::tools::envelope::error_text;
use list::{list_rules, DEFAULT_DUPLICATION_THRESHOLD};
use mutations::{add_rule, remove_rule, update_rule};

// ─────────────────────────────────────────────────────────────────────────────
// Tool entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Execute the `rules` tool.
///
/// Dispatches by `action`.  An invalid `action` returns `error_text` matching
/// the TS dispatcher throw semantics (tool-dispatcher.ts:106-109).
pub async fn rules_tool<D, R, Q>(
    input: RulesInput,
    daemon: &mut D,
    reader: &R,
    qdrant: &Q,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: RulesDaemon,
    R: RulesReader,
    Q: RulesQdrant,
{
    // Read duplication threshold from config if present, else default.
    // (In the test harness the constant is always DEFAULT_DUPLICATION_THRESHOLD.
    //  Production wiring reads from ServerConfig in the call-site dispatcher.)
    let dup_threshold = DEFAULT_DUPLICATION_THRESHOLD;

    match input.action.as_str() {
        "add" => add_rule(input, daemon, qdrant, session_project_id, dup_threshold).await,
        "update" => update_rule(input, daemon, qdrant, session_project_id).await,
        "remove" => remove_rule(input, daemon, qdrant, session_project_id).await,
        "list" => list_rules(&input, reader, qdrant, session_project_id).await,
        other => error_text(&format!("Unknown action: {other}")),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../rules_tests.rs"]
mod tests;

#[cfg(test)]
mod parity_tests;
