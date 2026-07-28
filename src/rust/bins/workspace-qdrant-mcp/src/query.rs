//! The `query` tool -- the read leg of the walking-skeleton slice
//! (`P04-GT001-WO013`).
//!
//! # The path this file completes
//!
//! S2 bin -> N56 planner -> N38 executor -> N41's `DerivedIndex` read leg (the
//! FTS5 concrete) -> §3.1's envelope. The planner and executor live in
//! `wqm-search`; this module is the composition root that wires them to a concrete
//! index and a profile table, and the shape that turns an [`Execution`] into
//! §3.3's `data`.
//!
//! # Reads are in-process, and that is the architecture's own decision
//!
//! Nothing here crosses the daemon. ARCH rev15 §9.1 decision (B) embeds the N38
//! read pipeline in the client directly over `wqm-store`; a client links no facade.
//! So `query` answers with the daemon down, for the same structural reason
//! `status` does -- not as a fallback.
//!
//! # Three fields are `null` on purpose, and each has a different reason
//!
//! - **`score`** -- the index orders by FTS5's own rank, but `Hit` carries no
//!   score and this build has exactly one leg. A number an agent cannot compare
//!   against anything is what AS-F013/F014 measured; §3.3 answers it directly
//!   ("`rank` is always present so an agent that only needs ordering never touches
//!   `score`"), and `scoring.scale:"none"` is the declared form of not having one.
//! - **`page.matched`** -- null only when the matched set is larger than the page,
//!   because then it was not counted. See [`wqm_search::executor`].
//! - **`digest`** -- computable exactly when the whole matched set was seen. r02's
//!   A1-S5 corrected a digest defined over the returned *page*: page-dependent, so
//!   never comparable to the corpus it exists to compare against. Rather than
//!   repeat that, this build publishes the digest when it is the real thing and
//!   `null` when it is not.

use std::path::Path;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use wqm_common::envelope::{AppliedDefault, ErrorCode, ToolError};
use wqm_common::names::Collection;
use wqm_common::plan::Plan;
use wqm_search::executor::{Execution, MAX_LIMIT};
use wqm_search::QueryError;
use wqm_store::{DerivedIndex, Fts5Index, Hit};

use crate::profiles::DeploymentProfiles;

/// The caller scope this build serves.
///
/// N56's contract is that a plan may narrow within the caller's scope and never
/// widen it (`CONTRACTS.md`:2343), which makes "what this build serves" a scope
/// question rather than a profile one. The store is a single SQLite file holding
/// the scratchpad SoT; a query naming any other collection is refused by the
/// planner instead of running against a table that holds nothing for it, because
/// an empty result there would be indistinguishable from a true absence.
///
/// Declared here, in the composition root, and reported by `status` from this same
/// constant -- so the refusal and the declaration cannot disagree.
pub const SERVED_SOURCES: [Collection; 1] = [Collection::Scratchpad];

/// A successful call: the `data` block and the defaults it applied.
pub struct Answer {
    /// §3.3's `data`.
    pub data: Value,
    /// The rows §3.1 owes `defaults_applied`.
    pub defaults: Vec<AppliedDefault>,
}

/// A refusal, in the level §4.1 carries it at.
pub enum Refusal {
    /// `invalid_argument` -- rides the JSON-RPC error, never an envelope.
    Protocol {
        /// One sentence.
        message: String,
        /// §4.1's `data` member.
        details: Value,
    },
    /// Every other code -- a tool-level failure inside the envelope.
    Tool(Box<ToolError>),
}

/// Run one `query` call.
pub fn call(store: Option<&Path>, arguments: &Value) -> Result<Answer, Refusal> {
    let (q, limit) = arguments_of(arguments)?;
    let store = store.ok_or_else(missing_store)?;

    let index = Fts5Index::open_path(store).map_err(|e| unreadable_store(store, &e))?;
    // N41's availability probe, asked of the concrete and converted into the
    // planner's vocabulary here -- `wqm-search` never has to know which adapter
    // answers which leg method.
    let available = if index.available() {
        vec![wqm_common::plan::LegMethod::Trigram]
    } else {
        Vec::new()
    };

    let execution = wqm_search::search(
        &q,
        limit,
        &SERVED_SOURCES,
        &DeploymentProfiles,
        &index,
        &available,
    )
    .map_err(refusal_for)?;

    Ok(Answer {
        defaults: defaults_of(&execution),
        data: data_of(&execution),
    })
}

/// Read `q` and `limit`, refusing anything this build's schema does not declare.
///
/// The schema is `additionalProperties:false` (§2.1), so an unknown key is an
/// error rather than something ignored -- an ignored parameter is a request that
/// silently did something other than what was asked.
fn arguments_of(arguments: &Value) -> Result<(String, Option<u32>), Refusal> {
    let object = arguments.as_object().cloned().unwrap_or_default();
    for key in object.keys() {
        if key != "q" && key != "limit" {
            return Err(unknown_parameter(key));
        }
    }

    let q = object
        .get("q")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| missing_q(&object))?
        .to_string();

    let limit = match object.get("limit") {
        None => None,
        Some(Value::Number(n)) if n.is_u64() => Some(n.as_u64().unwrap_or(0) as u32),
        Some(other) => return Err(bad_limit(other)),
    };
    Ok((q, limit))
}

/// §3.3's `data`.
fn data_of(execution: &Execution) -> Value {
    let results: Vec<Value> = execution
        .hits
        .iter()
        .enumerate()
        .map(|(i, hit)| result_of(hit, i, execution))
        .collect();

    json!({
        "results": results,
        "page": {
            "returned": results.len(),
            "matched": execution.matched,
            "matched_exact": execution.matched_exact,
            "has_more": execution.has_more,
            // Pagination is not executed by this build, so the cursor an agent
            // would pass back does not exist. Emitting one it cannot honour is
            // exactly the r02 defect where the emitted key and the accepted
            // parameter disagreed.
            "cursor": Value::Null,
        },
        "plan": plan_of(&execution.plan),
        "sources_searched": [execution.collection.deployed_name()],
        "scoring": {"scale": "none", "higher_is_better": Value::Null, "comparable": "not_applicable"},
        "shape": "compact",
        "digest": digest_of(execution),
    })
}

/// One §3.2 Result: seven keys, always the same seven.
fn result_of(hit: &Hit, offset: usize, execution: &Execution) -> Value {
    let level = level_of(execution.collection);
    json!({
        "object": execution.plan.object,
        // N40 owns the addressable-unit identifier scheme, and this build has
        // neither a tenant segment nor unit ids -- so the id is composed from what
        // the read leg genuinely knows (collection, keep, unit level) rather than
        // padded out to N40's full shape with invented segments. Debt: SCAFFOLD §7.
        "id": format!("wqm://{}/{}/{}", execution.collection.deployed_name(), hit.keep_id, level),
        "score": Value::Null,
        "rank": offset + 1,
        "location": location_of(hit, execution, level),
        "text": hit.content,
        // `compact` is the only shape this build serves; `extended`/`verbose`
        // carry metadata no subsystem here produces.
        "meta": Value::Null,
    })
}

/// `location` has a fixed key set with `null` for inapplicable (§3.2), so an agent
/// writes one accessor and never branches on which tool answered.
fn location_of(hit: &Hit, execution: &Execution, level: &'static str) -> Value {
    json!({
        // The DEPLOYED name: what was actually read. Reporting the logical name
        // while running in parallel would hide the one fact the -v2 knob exists
        // to keep visible.
        "collection": execution.collection.deployed_name(),
        "project": Value::Null,
        "branch": hit.branch_id,
        "library": Value::Null,
        "path": Value::Null,
        "line": Value::Null,
        "span": Value::Null,
        "unit": {"level": level, "id": hit.keep_id},
        // Nothing in this build records when a note was indexed; the fixture
        // writes the SoT directly. A timestamp invented here would be the
        // staleness signal AS-F021 exists to make trustworthy, made untrustworthy.
        "indexed_at": Value::Null,
    })
}

/// The unit level a collection's results are addressable at, read from the
/// profile rather than matched on here so the ladder keeps one owner
/// (`CONTRACTS.md`:1089-1090).
fn level_of(collection: Collection) -> &'static str {
    use wqm_common::profile::CollectionProfiles;
    DeploymentProfiles.profile(collection).granularity.as_str()
}

/// The executed plan (§3.4), serialized from the shared types.
fn plan_of(plan: &Plan) -> Value {
    serde_json::to_value(plan).unwrap_or(Value::Null)
}

/// A digest of the FULL matched set, or `null` when the set was not fully seen.
fn digest_of(execution: &Execution) -> Value {
    if !execution.matched_exact {
        return Value::Null;
    }
    let mut hasher = Sha256::new();
    for hit in &execution.hits {
        hasher.update(hit.keep_id.as_bytes());
        hasher.update([0]);
        hasher.update(hit.content.as_bytes());
        hasher.update([0]);
    }
    let full = format!("{:x}", hasher.finalize());
    // §3.3's own rendering is a 16-hex-digit prefix -- ~30 bytes on the wire,
    // which is the cost the design priced. The full digest is not more useful
    // here and every result-bearing response pays for it.
    json!(format!("sha256:{}", &full[..DIGEST_HEX_DIGITS]))
}

/// How much of the SHA-256 §3.3's rendering carries.
const DIGEST_HEX_DIGITS: usize = 16;

/// L-2: every default this call applied, named back to the caller.
fn defaults_of(execution: &Execution) -> Vec<AppliedDefault> {
    let mut rows = vec![AppliedDefault {
        parameter: "shape",
        value: json!("compact"),
        reason: "each result carries its id, location and matching text and nothing else".into(),
        override_with: "this build serves `compact` only".into(),
    }];
    if execution.limit_defaulted {
        rows.push(AppliedDefault {
            parameter: "limit",
            value: json!(execution.plan.limit),
            reason: "you receive the highest-ranked results, not every match".into(),
            override_with: format!("add `LIMIT n` to `q`, up to {MAX_LIMIT}"),
        });
    }
    if execution.mode_was_alias {
        rows.push(AppliedDefault {
            parameter: "mode",
            value: json!("text"),
            reason: "`EXACT` is a deprecated alias of `TEXT` and is echoed normalized".into(),
            override_with: "write `SELECT TEXT …`".into(),
        });
    }
    rows
}

/// Map a query failure onto the level and code §4 carries it at.
fn refusal_for(error: QueryError) -> Refusal {
    match error {
        QueryError::Conflict { parameter, clause } => Refusal::Protocol {
            message: format!("`{parameter}` and a `{clause}` clause both set the same bound"),
            details: json!({
                "code": "invalid_argument",
                "details": {"parameter": parameter, "clause": clause},
                "retryable": false,
            }),
        },
        QueryError::Parse {
            position,
            ref message,
            ref expected,
            ref suggestion,
        } => Refusal::Tool(Box::new(ToolError {
            code: ErrorCode::GrammarParse,
            message: message.clone(),
            details: json!({
                "position": position,
                "expected": expected,
                "suggestion": suggestion,
            }),
            retryable: false,
        })),
        QueryError::Unsupported {
            capability_key,
            ref message,
            ref suggestion,
        } => Refusal::Tool(Box::new(ToolError {
            code: ErrorCode::GrammarUnsupported,
            message: message.clone(),
            details: json!({"capability_key": capability_key, "suggestion": suggestion}),
            retryable: false,
        })),
        QueryError::UnknownReference {
            kind,
            ref name,
            ref known,
            ref message,
        } => Refusal::Tool(Box::new(ToolError {
            code: ErrorCode::UnknownReference,
            message: message.clone(),
            details: json!({"kind": kind, "name": name, "known": known}),
            retryable: false,
        })),
        QueryError::Backend(ref e) => Refusal::Tool(Box::new(ToolError {
            code: ErrorCode::BackendUnavailable,
            message: e.to_string(),
            details: json!({"leg": "trigram"}),
            retryable: true,
        })),
    }
}

fn missing_store() -> Refusal {
    Refusal::Tool(Box::new(ToolError {
        code: ErrorCode::BackendUnavailable,
        message: "this server was started without a store to read; pass `--store <PATH>`".into(),
        details: json!({"argument": "--store"}),
        // Retryable in the sense §4.3 means it: the same call succeeds once the
        // backend is there, and nothing about the request needs changing.
        retryable: true,
    }))
}

fn unreadable_store(path: &Path, error: &wqm_store::StoreError) -> Refusal {
    Refusal::Tool(Box::new(ToolError {
        code: ErrorCode::BackendUnavailable,
        message: format!(
            "the store at {} could not be opened: {error}",
            path.display()
        ),
        details: json!({"argument": "--store"}),
        retryable: true,
    }))
}

fn unknown_parameter(key: &str) -> Refusal {
    Refusal::Protocol {
        message: format!("`{key}` is not a parameter this build's `query` accepts"),
        details: json!({
            "code": "invalid_argument",
            "details": {"parameter": key, "expected": ["q", "limit"]},
            "retryable": false,
        }),
    }
}

fn missing_q(object: &Map<String, Value>) -> Refusal {
    Refusal::Protocol {
        message: "`q` is required and must be a non-empty string".into(),
        details: json!({
            "code": "invalid_argument",
            "details": {"parameter": "q", "got": object.get("q")},
            "retryable": false,
        }),
    }
}

fn bad_limit(got: &Value) -> Refusal {
    Refusal::Protocol {
        message: "`limit` must be a positive whole number".into(),
        details: json!({
            "code": "invalid_argument",
            "details": {"parameter": "limit", "got": got, "maximum": MAX_LIMIT},
            "retryable": false,
        }),
    }
}
