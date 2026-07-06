//! N8 -- the canonical name registry (ARCH rev08 nexus N8; GT001 contracts doc N8).
//!
//! N8 is the single owner of the literal identifier *strings* the whole workspace
//! keys on -- collection names, environment-variable keys, and the operation-class
//! and consumer vocabularies the N51 access-capability registry grants over. It is
//! a pure symbol table: compile-time constants and `const` accessors, no runtime
//! logic. N35 (F-05) owns the behavioural *profile* keyed by a [`Collection`]; N8
//! owns only the string that key is spelled with.
//!
//! # Single-producer rule (FP-2)
//!
//! Every literal here is spelled in exactly one place; no other crate re-spells it.
//! `src/rust/ci/guard_name_registry.py` greps the workspace `src/` trees for a
//! stray re-spelling of any guarded literal outside this module and fails CI on a
//! hit (PRD F-01 AC2). Renaming a name is therefore a one-site edit.
//!
//! # Grows per phase
//!
//! Phase 0 seeds the vocabulary the Phase-0 consumers (N9/N35/N51/N7) key on: the
//! four canonical collections, the five environment keys, and the N51
//! operation-class and consumer sets. Later features that introduce new literals
//! (N24 service/RPC names, N1 table/column names, N35 payload keys) add them here
//! under the same single-producer rule rather than spelling them inline.

mod access;
mod collections;
mod env;

pub use access::{Consumer, OpClass};
pub use collections::{collection_name, Collection, RESERVED_IMAGES_COLLECTION};
pub use env::EnvVar;
