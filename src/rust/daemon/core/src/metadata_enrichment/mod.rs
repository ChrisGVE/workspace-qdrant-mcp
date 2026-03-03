//! Metadata Enrichment Module
//!
//! Implements metadata enrichment rules for different collection types according to
//! architectural decisions in Task 375. Enriches document metadata based on collection
//! type: PROJECT, LIBRARY, USER, or RULES.
//!
//! Collection Type Detection:
//! - PROJECT: _{project_id} where project_id is 12-char hex hash
//! - LIBRARY: _{library_name} where library_name is alphanumeric with hyphens
//! - USER: {basename}-{type} format
//! - RULES: exact match "rules" (also accepts legacy "memory")
//!
//! Metadata Enrichment Rules:
//! - PROJECT: project_id, branch, file_type, extension, is_test
//! - USER: project_id only (no branch)
//! - LIBRARY: library_name (no project_id or branch)
//! - RULES: global metadata only (no project_id or branch)

mod collection_type;
mod enrichment;

#[cfg(test)]
mod tests;

pub use collection_type::CollectionType;
pub use enrichment::enrich_metadata;
