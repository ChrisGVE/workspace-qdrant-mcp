//! wqm-search -- N56 the planner and N38 the executor (`CONTRACTS.md` S2.17).
//!
//! # The split, and which way the call goes
//!
//! N38 **pulls**: it calls `N56.plan(query, scope)` when no explicit plan was
//! supplied, then runs the plan -- legs, fusion, envelope (:1043-1058). The call
//! direction is always N38 -> N56, never the reverse, and the module layout says
//! so: [`executor`] depends on [`planner`], and [`planner`] knows nothing about
//! execution.
//!
//! # What is here at `P04-GT001-WO013`, and what is absent rather than stubbed
//!
//! The narrowed slice (`SCAFFOLD.md` §7.1/§7.2) is a **single FTS5 leg, no fusion,
//! no embedding**. Three things therefore do not appear in this crate at all:
//!
//! - **Fusion.** N4 takes `legs: Vec<RankedList>` and needs N17 dense scores
//!   (`CONTRACTS.md`:1012-1014). One leg never reaches it, so the planner refuses
//!   to emit a multi-leg plan rather than emitting one it cannot execute.
//! - **Embedding.** A keyword query is not embedded. `SEMANTIC` mode is refused by
//!   name, with the leg set this build has, instead of being silently served by
//!   the text leg -- substituting one retrieval regime for another and calling it
//!   the same query is the failure class this surface exists to end.
//! - **Grep.** N35's `grep_eligible` is false for scratchpad (`CONTRACTS.md`:565-566),
//!   so the regex leg is gated off *before* availability is even consulted.
//!
//! # The two errors an agent can act on
//!
//! §1.6's honest cost of a free-text `q` is that validation is server-side, and its
//! mandatory mitigation is that a parse failure carries position, the expected-token
//! set, and a suggestion where one is derivable. [`QueryError::Parse`] carries all
//! three, so the second call is a correction rather than a guess.

use wqm_common::names::Collection;

pub mod executor;
pub mod grammar;
pub mod planner;

pub use executor::{search, Execution};
pub use grammar::{parse, ParsedQuery};
pub use planner::plan;

/// What a query can fail with, in the shapes MCP-SURFACE.md §4.3 carries.
///
/// Each variant maps to exactly one sealed error code, and the mapping lives with
/// the surface that emits it rather than here -- this crate states what went
/// wrong, the transport states it in the protocol's terms.
#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    /// `q` did not parse (`grammar_parse`).
    #[error("{message} at position {position}")]
    Parse {
        /// Byte offset into `q` where the parser stopped.
        position: usize,
        /// One sentence, in the caller's terms.
        message: String,
        /// What the parser would have accepted here.
        expected: Vec<&'static str>,
        /// A corrected query, when one is derivable.
        suggestion: Option<String>,
    },

    /// `q` parsed and names a clause this build does not execute
    /// (`grammar_unsupported`).
    ///
    /// `capability_key` is the manifest key an agent reads to discover the limit
    /// ahead of time (§5.4) -- so the error and the declaration are the same
    /// vocabulary, and an agent that read the manifest never gets here.
    #[error("{message}")]
    Unsupported {
        /// The §5.4 manifest key that declares this limit.
        capability_key: &'static str,
        /// One sentence naming what was asked for and what this build does.
        message: String,
        /// A query this build would execute, when one is derivable.
        suggestion: Option<String>,
    },

    /// An addressing argument names something that does not exist
    /// (`unknown_reference`).
    #[error("{message}")]
    UnknownReference {
        /// What kind of thing was addressed.
        kind: &'static str,
        /// The name that did not resolve.
        name: String,
        /// What does exist, so the correction needs no second call.
        known: Vec<&'static str>,
        /// One sentence.
        message: String,
    },

    /// A tool parameter and a `q` clause both set the same thing (§2.2 makes this
    /// an error rather than a silent override) -- `invalid_argument`.
    #[error("`{parameter}` was supplied together with a {clause} clause in `q`")]
    Conflict {
        /// The tool parameter.
        parameter: &'static str,
        /// The clause that also sets it.
        clause: &'static str,
    },

    /// The derived index could not answer (`backend_unavailable`).
    #[error("the derived index could not answer: {0}")]
    Backend(#[from] wqm_store::StoreError),
}

/// Every canonical collection's name, for the "what does exist" list an
/// `unknown_reference` carries. Sourced from N8, never re-spelled.
pub(crate) fn known_collections() -> Vec<&'static str> {
    Collection::ALL.iter().map(|c| c.name()).collect()
}
