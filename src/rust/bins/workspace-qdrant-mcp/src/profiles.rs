//! N35's collection-profile table -- the GLUE side, in the composition root.
//!
//! # Why the table is in the binary and not in the kernel
//!
//! N35 has two faces (`CONTRACTS.md`:481-489): the injected READ trait a kernel
//! crate consults, and the registry that owns the data. The registry's home is
//! `wqm-conventions`, and **no kernel crate may link `wqm-conventions`** (:585-586,
//! enforced by `ci/guard_link_closure.py`). `wqm-conventions` does not exist in
//! this workspace yet -- it arrives with `P04-GT002` -- so the table lives where a
//! composition root's wiring always lives: in the binary that composes.
//!
//! That is a placement decision, not a shortcut. The planner takes
//! `&dyn CollectionProfiles` and cannot name this type; moving the table to
//! `wqm-conventions` later changes this file and nothing in `wqm-search`. If the
//! table had been written *inside* the planner, the injection would have been
//! decoration and the move would be a rewrite.

use wqm_common::names::Collection;
use wqm_common::profile::{CollectionProfile, CollectionProfiles, UnitLevel};

/// The four canonical collections' profiles, at the axes this build reads.
pub struct DeploymentProfiles;

impl CollectionProfiles for DeploymentProfiles {
    /// Sourced from the sealed contract rather than chosen here:
    ///
    /// - **`grep_eligible`** -- `CONTRACTS.md`:565-566 carries GT002 §B's rule
    ///   verbatim, "no grep on library/scratchpad/rules", so `projects` is the one
    ///   true row. This build reads only the scratchpad, where the axis is false,
    ///   and the planner refuses a regex plan on it (`wqm_search::planner`).
    /// - **`granularity`** -- `CONTRACTS.md`:1089-1090 enumerates N40's unit ladder
    ///   from this axis and states that rules and scratchpad are document-level
    ///   only.
    /// - **`is_searchable`** -- all four collections are searchable (:564 gives all
    ///   four live vectors).
    ///
    /// The other six axes are absent from [`CollectionProfile`] entirely, so no row
    /// here guesses one.
    fn profile(&self, collection: Collection) -> CollectionProfile {
        match collection {
            Collection::Projects => CollectionProfile {
                is_searchable: true,
                granularity: UnitLevel::Chunk,
                grep_eligible: true,
            },
            Collection::Libraries => CollectionProfile {
                is_searchable: true,
                granularity: UnitLevel::Chunk,
                grep_eligible: false,
            },
            Collection::Rules | Collection::Scratchpad => CollectionProfile {
                is_searchable: true,
                granularity: UnitLevel::Document,
                grep_eligible: false,
            },
        }
    }
}
