//! The executor against a real derived index -- specifically, the honesty of
//! `page`.
//!
//! §3.3 makes `matched_exact` the valve that stops the engine inventing a total:
//! `matched:null, matched_exact:false` is a *different statement* from a number,
//! and the current v0.1 surface makes exactly that mistake (returning `total: 2`
//! for a request that matched 215). The executor buys the distinction with one
//! extra row, and this file is where the two sides of it are proven.

use wqm_common::names::{Collection, WriteTarget};
use wqm_common::plan::LegMethod;
use wqm_common::profile::{CollectionProfile, CollectionProfiles, UnitLevel};
use wqm_search::{search, QueryError};
use wqm_store::{DerivedIndex, Fts5Index};
use wqm_test_harness::scratchpad_fixture::{in_memory, seed, SeedNote};

const BRANCH: &str = "branch-none-placeholder";
const SCOPE: [Collection; 1] = [Collection::Scratchpad];

/// The sealed scratchpad profile.
struct Profiles;

impl CollectionProfiles for Profiles {
    fn profile(&self, _collection: Collection) -> CollectionProfile {
        CollectionProfile {
            is_searchable: true,
            granularity: UnitLevel::Document,
            grep_eligible: false,
        }
    }
}

/// `count` notes, each carrying the same searchable term.
fn index_of(count: usize) -> Fts5Index {
    let mut conn = in_memory().expect("schema installs");
    let contents: Vec<String> = (0..count)
        .map(|i| format!("note {i} mentions the seam it crosses"))
        .collect();
    let keeps: Vec<String> = (0..count).map(|i| format!("keep-{i}")).collect();
    let notes: Vec<SeedNote<'_>> = (0..count)
        .map(|i| SeedNote {
            keep_id: &keeps[i],
            branch_id: BRANCH,
            content: &contents[i],
        })
        .collect();
    seed(&mut conn, &WriteTarget::of(Collection::Scratchpad), &notes).expect("seeds");
    Fts5Index::open(conn)
}

fn query(limit_clause: &str) -> String {
    format!(
        "SELECT TEXT note FROM {} WHERE q MATCH 'seam'{limit_clause}",
        Collection::Scratchpad.name()
    )
}

fn available(index: &Fts5Index) -> Vec<LegMethod> {
    if index.available() {
        vec![LegMethod::Trigram]
    } else {
        Vec::new()
    }
}

/// The whole matched set fits inside the page, so the total is COUNTED.
#[test]
fn a_set_smaller_than_the_page_is_counted_exactly() {
    let index = index_of(3);
    let execution = search(
        &query(""),
        None,
        &SCOPE,
        &Profiles,
        &index,
        &available(&index),
    )
    .expect("the query runs");

    assert_eq!(execution.hits.len(), 3);
    assert_eq!(execution.matched, Some(3));
    assert!(execution.matched_exact);
    assert!(!execution.has_more);
}

/// More exists than was returned, so the total is UNKNOWN and says so. The
/// alternative -- reporting the page size as the total -- is the defect.
#[test]
fn a_set_larger_than_the_page_reports_unknown_rather_than_the_page_size() {
    let index = index_of(5);
    let execution = search(
        &query(" LIMIT 2"),
        None,
        &SCOPE,
        &Profiles,
        &index,
        &available(&index),
    )
    .expect("the query runs");

    assert_eq!(
        execution.hits.len(),
        2,
        "the caller's bound is honoured exactly"
    );
    assert!(execution.has_more);
    assert!(!execution.matched_exact);
    assert_eq!(
        execution.matched, None,
        "an uncounted total is null, never the page size"
    );
}

/// A true absence: the set is empty and the emptiness is counted, so the response
/// can say zero rather than shrug.
#[test]
fn a_query_that_matches_nothing_counts_zero_exactly() {
    let index = index_of(2);
    let text = query("").replace("seam", "nothingmatchesthis");
    let execution =
        search(&text, None, &SCOPE, &Profiles, &index, &available(&index)).expect("runs");
    assert!(execution.hits.is_empty());
    assert_eq!(execution.matched, Some(0));
    assert!(execution.matched_exact);
}

/// §2.2: the `limit` parameter and a `LIMIT` clause together are an error, "not a
/// silent override" -- two callers disagreeing is a question only the caller can
/// settle.
#[test]
fn a_limit_parameter_and_a_limit_clause_together_are_refused() {
    let index = index_of(1);
    let error = search(
        &query(" LIMIT 2"),
        Some(5),
        &SCOPE,
        &Profiles,
        &index,
        &available(&index),
    )
    .expect_err("the two bounds disagree");
    assert!(matches!(error, QueryError::Conflict { .. }));
}

/// The store answers a query whose retrieval leg it has; when the concrete is
/// absent the planner refuses rather than the executor failing mid-flight.
#[test]
fn an_unavailable_leg_is_refused_before_execution() {
    let index = index_of(1);
    let error =
        search(&query(""), None, &SCOPE, &Profiles, &index, &[]).expect_err("no concrete, no plan");
    assert!(matches!(error, QueryError::Unsupported { .. }));
}
