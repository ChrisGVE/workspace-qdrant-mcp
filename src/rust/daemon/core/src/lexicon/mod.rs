//! Dynamic Lexicon Manager (Task 17).
//!
//! Manages per-collection BM25 vocabulary with SQLite persistence.
//! Provides document frequency lookup for the keyword extraction pipeline
//! and corpus statistics for IDF weighting.
//!
//! On startup, loads persisted vocabulary from `sparse_vocabulary` and
//! `corpus_statistics` tables. During processing, accumulates new terms
//! in memory and periodically flushes to SQLite.

mod manager;
mod operations;

#[cfg(test)]
mod tests;

pub use manager::LexiconManager;
