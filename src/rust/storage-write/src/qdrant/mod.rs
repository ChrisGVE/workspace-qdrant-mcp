//! Mutating Qdrant access for the write crate.
//!
//! Location: `wqm-storage-write/src/qdrant/`. Logical context: the EXCLUSIVE
//! home of Qdrant mutation. Every `upsert_points`/`delete_points`/payload-write/
//! `create_collection` call lives here and nowhere else; Guard 3 asserts none of
//! these symbols are reachable in the read-only `mcp-server`/`wqm-cli` binaries.
//! Neighbors: `wqm-storage::qdrant::QdrantReadClient` (the read sibling).

pub mod membership;
pub mod write_client;

pub use write_client::QdrantWriteClient;
