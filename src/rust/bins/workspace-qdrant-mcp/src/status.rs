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

/// What this build can actually do.
///
/// §5.4's full capability manifest -- the `core`/`gated` clause partition and the
/// twenty-two schema names -- arrives with the surface slice. What must be true
/// *now* is that this block never over-claims: it lists the one tool this build
/// serves, and declares the clause vocabularies empty because it executes none.
fn capabilities() -> Value {
    json!({
        "tools": ["status"],
        "core": [],
        "gated": [],
        "schemas": [],
    })
}
