//! N38 -- the executor. It PULLS the plan, runs it, and reports what it ran.
//!
//! # One leg, and the counting trick that keeps `page` honest
//!
//! §3.3 makes `page` the single truncation vocabulary and `matched_exact` its
//! honesty valve: the engine must never *invent* a total, and `matched:null,
//! matched_exact:false` is a different statement from a number. N41's read face
//! offers `query`/`available` and no count, so a separate count query is not
//! available through the sealed port.
//!
//! What *is* available is asking for one row more than the caller wanted. If the
//! index returns at most `limit`, the whole matched set was seen and the count is
//! exact; if it returns `limit + 1`, more exist and the total is unknown -- which
//! is reported as unknown. One extra row buys an exact count in the common case
//! and an honest `null` in the rest, and it borrows no surface the measurement did
//! not license.
//!
//! # What this executor does not do
//!
//! No fusion (the planner refuses multi-leg plans -- see [`crate::planner`]), no
//! rerank, no expansion, no pagination cursor, and no empty-result diagnosis.
//! §4.2's procedure re-runs the plan with its predicates dropped and counts; this
//! build's only predicate is the retrieval text itself, and dropping it leaves no
//! query for a text index to run. So the build declares
//! `capabilities.empty_diagnosis: "off"` -- §4.2's own provision for a build that
//! cannot afford the procedure -- rather than emitting a `corpus_empty` or
//! `filter_emptied` notice it did not earn.

use wqm_common::names::Collection;
use wqm_common::plan::{LegMethod, Plan};
use wqm_common::profile::CollectionProfiles;
use wqm_store::{DerivedIndex, Hit};

use crate::grammar::{parse, ParsedQuery};
use crate::planner::{plan, DEFAULT_LIMIT};
use crate::QueryError;

/// §5.4's `max_limit`. Declared in the manifest and enforced here, so the
/// declaration and the refusal are the same number.
pub const MAX_LIMIT: u32 = 200;

/// One executed query: the plan that ran, what it found, and the disclosures the
/// envelope owes the caller.
#[derive(Debug)]
pub struct Execution {
    /// The plan, for the response's `plan` block.
    pub plan: Plan,
    /// The hits, already cut to the plan's limit.
    pub hits: Vec<Hit>,
    /// The collection that was read.
    pub collection: Collection,
    /// The true total, when the executor could see it exactly.
    pub matched: Option<u32>,
    /// Whether [`Execution::matched`] is a counted total rather than unknown.
    pub matched_exact: bool,
    /// Whether results exist beyond the ones returned.
    pub has_more: bool,
    /// Whether the caller spelled the mode `EXACT` and got `text` back.
    pub mode_was_alias: bool,
    /// Whether `limit` was defaulted rather than supplied.
    pub limit_defaulted: bool,
}

/// Run one query end to end: parse, plan, execute.
///
/// `available` is the leg set the caller proved with N41's `available()` probe.
/// Passing it in rather than probing here keeps the "which concrete exists"
/// question with the composition root that built the concrete.
pub fn search(
    q: &str,
    limit_param: Option<u32>,
    scope: &[Collection],
    profiles: &dyn CollectionProfiles,
    index: &dyn DerivedIndex,
    available: &[LegMethod],
) -> Result<Execution, QueryError> {
    let parsed = reconcile_limit(parse(q)?, limit_param)?;
    let limit_defaulted = parsed.limit.is_none();
    let plan = plan(&parsed, scope, profiles, available)?;
    check_limit(plan.limit)?;

    let text = parsed
        .match_text
        .as_deref()
        .ok_or(QueryError::Unsupported {
            capability_key: "fields",
            message: "this build retrieves by text only, so a `WHERE q MATCH '…'` \
                  predicate is required"
                .into(),
            suggestion: None,
        })?;

    // One row beyond the caller's bound -- see the module note.
    let probe_limit = plan.limit as usize + 1;
    let mut hits = index.query(parsed.source, text, probe_limit)?;

    let has_more = hits.len() > plan.limit as usize;
    hits.truncate(plan.limit as usize);
    let matched = (!has_more).then_some(hits.len() as u32);

    Ok(Execution {
        plan,
        hits,
        collection: parsed.source,
        matched,
        matched_exact: !has_more,
        has_more,
        mode_was_alias: parsed.mode_was_alias,
        limit_defaulted,
    })
}

/// §2.2: supplying the `limit` parameter AND a `LIMIT` clause is an error, "not a
/// silent override". Two callers disagreeing about the bound is a question only
/// the caller can settle.
fn reconcile_limit(
    parsed: ParsedQuery,
    limit_param: Option<u32>,
) -> Result<ParsedQuery, QueryError> {
    match (limit_param, parsed.limit) {
        (Some(_), Some(_)) => Err(QueryError::Conflict {
            parameter: "limit",
            clause: "LIMIT",
        }),
        (Some(value), None) => Ok(ParsedQuery {
            limit: Some(value),
            ..parsed
        }),
        _ => Ok(parsed),
    }
}

fn check_limit(limit: u32) -> Result<(), QueryError> {
    if limit == 0 {
        return Err(QueryError::Unsupported {
            capability_key: "max_limit",
            message: "a limit of zero asks for no results; the accepted range is 1 \
                      to 200"
                .into(),
            suggestion: Some(format!("LIMIT {DEFAULT_LIMIT}")),
        });
    }
    if limit > MAX_LIMIT {
        return Err(QueryError::Unsupported {
            capability_key: "max_limit",
            message: format!("the largest result bound this surface serves is {MAX_LIMIT}"),
            suggestion: Some(format!("LIMIT {MAX_LIMIT}")),
        });
    }
    Ok(())
}
