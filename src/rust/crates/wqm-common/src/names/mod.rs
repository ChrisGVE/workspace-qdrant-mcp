//! N8 -- the canonical name registry (ARCH rev15 nexus N8; P02-GT005 CONTRACTS N8).
//!
//! N8 is the single owner of the literal identifier *strings* the whole workspace
//! keys on -- collection names, environment-variable keys, and the operation-class
//! and consumer vocabularies the N51 access-capability registry grants over. It is
//! a pure symbol table: compile-time constants and `const` accessors, no runtime
//! logic. N35 (`P04-GT055`) owns the behavioural *profile* keyed by a
//! [`Collection`]; N8 owns only the string that key is spelled with.
//!
//! # Single-producer rule (FP-2)
//!
//! Every literal here is spelled in exactly one place; no other crate re-spells it.
//! `src/rust/ci/guard_name_registry.py` greps the workspace `src/` trees for a
//! stray re-spelling of any guarded literal outside this module and fails CI on a
//! hit. Renaming a name is therefore a one-site edit.
//!
//! # Grows per work order
//!
//! `P04-GT001` seeds the vocabulary its early consumers (N9/N35/N51/N7) key on: the
//! four canonical collections, the five environment keys, and the N51
//! operation-class and consumer sets. Later slices that introduce new literals
//! (N24 service/RPC names at `P04-GT011`, N1 table/column names, N35 payload keys
//! at `P04-GT055`) add them here under the same single-producer rule rather than
//! spelling them inline.
//!
//! # The deployment namespace
//!
//! Collection names are ALSO where the `-v2` parallel-deployment suffix must enter
//! (`PROJECT_LOGISTICS.md`: one knob, not N literals; v0.2 writes only to `-v2`
//! collections). The `deployment` module owns that knob: [`DEPLOYMENT_SUFFIX`] is
//! spelled once, `Collection::deployed_name` applies it, and [`WriteTarget`] is
//! the structural refusal -- a write path typed on it cannot address a collection
//! outside this deployment (`P04-GT001-WO008`/`WO009`).

mod access;
mod collections;
mod deployment;
mod env;
mod protocol;

pub use access::{Consumer, OpClass};
pub use collections::{collection_name, Collection, RESERVED_IMAGES_COLLECTION};
pub use deployment::{
    dir_base, WriteGuardError, WriteTarget, DEPLOYMENT_DIR, DEPLOYMENT_SUFFIX,
    IS_PARALLEL_DEPLOYMENT, SERVICE_LABEL,
};
pub use env::EnvVar;
pub use protocol::{negotiate, Negotiated, PREFERRED_PROTOCOL, SUPPORTED_PROTOCOLS};
