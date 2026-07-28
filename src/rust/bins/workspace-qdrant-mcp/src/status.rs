//! The `status` tool's `data` block (MCP-SURFACE.md §3.5).
//!
//! Nine keys, all of them always present. The skeleton can populate three of them
//! honestly and must say `null` for the rest -- and *saying null* is the design's
//! own convention for an inapplicable value ("present and null, like every other
//! inapplicable value on this surface, never absent"), not a shortcut taken here.
//!
//! The alternative -- reporting `{"files_tracked": 0}` for an index subsystem this
//! build does not contain -- would be indistinguishable from an empty index, and
//! `P04-GT001-WO011`'s acceptance says the answer must be "not a crash and not a
//! lie". Zero would be the lie.

use serde_json::{json, Value};
use wqm_client::{DaemonReport, DaemonState, DaemonStatus};
use wqm_common::names::Negotiated;

/// Build the `status` response body.
pub fn build(report: &DaemonReport, handshake: &Negotiated) -> Value {
    json!({
        "server": {
            "name": env!("CARGO_BIN_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
            "protocol": handshake.served,
        },
        "daemon": daemon_block(report),
        "index": index_block(report),
        "handshake": {
            "protocol_requested": handshake.requested,
            "protocol_served": handshake.served,
            // This build writes nothing to stderr during a session, so nothing
            // can have been dropped. It is reported rather than omitted because
            // the key's job is to make the count visible when it is not zero.
            "stderr_dropped_bytes": 0,
        },
        // No embedding provider, no Qdrant client and no project detection exist
        // in this build; each arrives with its own slice (N17, N35's collection
        // reads, N18/N7). Null is the honest report -- see the module note.
        "embedding": Value::Null,
        "collections": Value::Null,
        "project": Value::Null,
        "capabilities": capabilities(),
        // Non-null only when the caller passed `schemas`, which this build does
        // not yet serve (§2.9's on-demand channel).
        "schemas": Value::Null,
    })
}

/// `daemon` is the one block that is always answerable, because the whole point
/// of `status` is that it answers when the daemon does not (§4.3: `status` never
/// raises `backend_unavailable`).
fn daemon_block(report: &DaemonReport) -> Value {
    match report {
        DaemonReport::Reachable(DaemonStatus {
            state,
            detail,
            since_unix_seconds,
            ..
        }) => json!({
            "reachable": true,
            "state": state_name(*state),
            "detail": if detail.is_empty() { Value::Null } else { json!(detail) },
            "since": since_unix_seconds.map(Value::from).unwrap_or(Value::Null),
        }),
        DaemonReport::Unreachable { reason, address } => json!({
            "reachable": false,
            "state": "unreachable",
            // The same identifier telemetry records (§4.5 rule 1): one vocabulary
            // for the agent and for the metric.
            "detail": reason.as_str(),
            // Nothing is known about when it went away -- this client only knows
            // that nothing answered just now.
            "since": Value::Null,
            "address": address,
        }),
    }
}

/// The daemon's state as the surface spells it. The full vocabulary is owed by
/// CR-010; these are the members that can be asserted today.
fn state_name(state: DaemonState) -> Value {
    match state {
        DaemonState::Ok => json!("ok"),
        // A state this build does not recognise is reported as unrecognised
        // rather than flattened into "ok" -- guessing a health claim is the
        // failure class this whole tool exists to end.
        DaemonState::Unrecognized(code) => json!(format!("unrecognized:{code}")),
    }
}

/// `index` mirrors the daemon's own answer: absent there, null here.
fn index_block(report: &DaemonReport) -> Value {
    match report {
        DaemonReport::Reachable(DaemonStatus { index: Some(i), .. }) => json!({
            "files_tracked": i.files_tracked,
            "queue_pending": i.queue_pending,
            "complete": i.complete,
            "lag_seconds": i.lag_seconds,
        }),
        _ => Value::Null,
    }
}

/// What this build can actually do (§5.4's capability manifest).
///
/// # Every value here is read from the code that enforces it
///
/// §5.4's law is that "every conforming build executes every member of `core`",
/// asserted by the conformance suite. Until that suite exists, the cheapest way
/// not to publish a `core` the build does not execute is to publish the constants
/// the executor and planner are already built from: [`crate::mcp::SERVED_TOOLS`],
/// [`crate::query::SERVED_SOURCES`], and `wqm_search`'s `MAX_LIMIT`. A manifest
/// transcribed beside the code is a second rendering, and §7.3a's restatement rule
/// is the standing warning about exactly that.
///
/// # Why `core` is this narrow
///
/// One mode, one field, one operator, one shape. `from_optional` is **false**
/// because resolving "the project containing your working directory" needs project
/// detection this build does not have, and a default scope that cannot be resolved
/// is the silent-wrong-answer class. Everything absent from `core` is refused by
/// name with this manifest's key attached, so an agent that reads this never meets
/// a surprise and an agent that does not gets a correction rather than a zero.
///
/// # Why `empty_diagnosis` is `off`
///
/// §4.2's procedure re-runs the plan with its predicates dropped and counts what
/// is left. This build's only predicate is the retrieval text itself -- dropping it
/// leaves no query for a text index to run -- and N41's sealed read face
/// (`query`/`available`) exposes no count primitive to run it with. §4.2 provides
/// for exactly this: declare `off` and let the agent read the declaration rather
/// than guess. Filed as a finding against the contract rather than worked around
/// silently (`SCAFFOLD.md` §7).
fn capabilities() -> Value {
    json!({
        "tools": crate::mcp::SERVED_TOOLS,
        "grammar_version": "1",
        "core": {
            "modes": ["text"],
            // The objects addressable at the served sources' granularity. `note`
            // is what a scratchpad row is; `document` and `rule` are the other
            // document-level objects the planner admits.
            "objects": ["note", "document", "rule"],
            "sources": served_sources(),
            "fields": ["q"],
            "ops": ["MATCH"],
            "order_by": [],
            "shapes": ["compact"],
            "from_optional": false,
        },
        "gated": {
            "path_match": Value::Null,
            "graph_source": false,
            "graph_predicates": [],
            "subquery": false,
            "aggregate": [],
            "join": "deferred",
            "vector_operator": false,
            "objects_extra": [],
            "export_formats": [],
        },
        "empty_diagnosis": "off",
        "max_limit": wqm_search::executor::MAX_LIMIT,
        // No byte budget is enforced by this build, and a number here would be a
        // promise about truncation behaviour that nothing implements.
        "response_byte_budget": Value::Null,
        // The on-demand schema channel (§2.9) is not served, so the enumeration
        // that makes it discoverable is empty rather than listing names that
        // cannot be fetched -- §5.4 calls a listed-but-unfetchable name a
        // conformance failure.
        "schemas": [],
    })
}

/// The sources `query` serves, from the constant the planner's scope check uses.
fn served_sources() -> Vec<&'static str> {
    crate::query::SERVED_SOURCES
        .iter()
        .map(|c| c.name())
        .collect()
}
