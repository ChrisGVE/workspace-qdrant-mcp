//! wqm-store -- N41's derived-index READ leg (ARCH rev15 §9.1).
//!
//! # What this crate is, and what it deliberately is not
//!
//! N41 is a port family (`CONTRACTS.md`:1114-1125): the PORT is the `DerivedIndex`
//! query + lifecycle trait, and the ADAPTERS are FTS5, grep, Qdrant-read and
//! KG-CTE, one file per concrete. Two boundaries shape what may live here.
//!
//! **SQLite is not one of the adapters.** It is the SoT that every derived index
//! rebuilds *from*; listing it as a concrete would contradict the frozen storage
//! lock. So this crate opens SQLite, but it does so as the thing being indexed.
//!
//! **The trait bifurcates across the store seam.** The read leg -- [`DerivedIndex::query`]
//! and [`DerivedIndex::available`] -- is implemented here and is client-linked.
//! The write leg `rebuild_from_sot` is homed SOLELY in the S1-only
//! `wqm-store-write` rebuild driver, and is therefore **absent from this trait**
//! rather than present-and-unimplemented. That absence is what makes the S1-only
//! rebuild boundary a build-time fact instead of a runtime convention: a client
//! linking this crate has no rebuild method to call, so it cannot be tempted into
//! one.
//!
//! # Scope at `P04-GT001-WO012`
//!
//! One concrete, FTS5. `SCAFFOLD.md` §7.1 measured the write half at 24 borrowed
//! GT002/GT003 surfaces, which tripped CHARTER §5A.1 and moved it to `P04-GT002`;
//! §7.2 measured what remains at four. The single-leg read is the whole of it --
//! no fusion (N4/RRF takes a *vector* of legs and needs N17 dense scores, so one
//! leg never reaches it), and no embedding (a keyword query is not embedded).

use wqm_common::names::Collection;

pub mod schema;

mod fts5;

pub use fts5::Fts5Index;

/// What can go wrong reading a derived index.
///
/// N9 owns the product's error taxonomy and arrives with its own slice; this is
/// the local minimum the read leg needs, not a claim on that vocabulary.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The underlying SQLite call failed.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// The index was queried while unavailable. Callers that consult
    /// [`DerivedIndex::available`] first will not see this; it exists so that
    /// skipping the probe degrades to an error rather than to a wrong answer.
    #[error(
        "the {index} derived index is not available in this build; \
         `available()` reports false and the query was not run"
    )]
    Unavailable {
        /// Which concrete refused.
        index: &'static str,
    },
}

/// One hit from a derived index.
///
/// `keep_id` and `branch_id` are carried as opaque strings ON PURPOSE. N3 owns
/// the identity vocabulary -- `keep_id`, `branch_id`, and the fixed-width
/// `BRANCH_NONE_ID` sentinel that libraries/scratchpad/rules collapse to
/// (`CONTRACTS.md`:604-608, which explicitly forbids an empty string and, under
/// DP-ID4, non-fixed-width keys). Minting those values here would be deciding N3's
/// constants as a side effect of building a read path. The store round-trips them
/// and interprets neither; the debt is recorded in `SCAFFOLD.md` §7.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    /// The keep this row belongs to, opaque to this crate.
    pub keep_id: String,
    /// The branch membership value, opaque to this crate.
    pub branch_id: String,
    /// The indexed text.
    pub content: String,
}

/// N41's read leg: query a derived index, and say honestly whether it is there.
///
/// `available()` is the graceful-degrade probe (DP-6.3): it narrows the contract
/// rather than letting a missing backend surface as a query failure. The
/// selection strategy that consumes it -- degrade-to-available across the four
/// concretes -- belongs to the slice that has more than one concrete to choose
/// between.
pub trait DerivedIndex {
    /// The concrete's name, for diagnostics and for the degrade decision.
    fn name(&self) -> &'static str;

    /// Whether this index can answer right now. Probed, never assumed: the FTS5
    /// concrete asks SQLite whether the module is compiled in rather than
    /// trusting a Cargo feature flag.
    fn available(&self) -> bool;

    /// Full-text query within one collection, most-relevant first.
    ///
    /// The collection is taken as a [`Collection`] rather than a string so the
    /// deployed name is derived at the boundary instead of being passed in --
    /// a caller cannot address production by spelling it.
    fn query(
        &self,
        collection: Collection,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Hit>, StoreError>;
}
