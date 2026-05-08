//! Point operations
//!
//! Insert, delete, update, count, and existence check operations on Qdrant
//! points. Scroll operations are in the sibling `scroll` module.
//!
//! # Module structure
//!
//! - [`upsert`]: Single and batch point insertion
//! - [`delete`]: Delete by filter, tenant, IDs, document ID, or payload field
//! - [`update`]: Sparse vector updates and payload field updates
//! - [`count`]: Point counting and existence checks

mod count;
mod delete;
mod update;
mod upsert;
