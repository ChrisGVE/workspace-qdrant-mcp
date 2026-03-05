mod computation;
/// Automated project affinity grouping via embeddings.
///
/// Computes aggregate embeddings per project (mean of all chunk embeddings),
/// stores them as binary blobs in SQLite, then groups projects whose pairwise
/// cosine similarity exceeds a threshold into `project_groups` entries with
/// `group_type = "affinity"`.
///
/// Group labels are derived from zero-shot taxonomy classification
/// (see `tagging::tier2`).
mod config;
mod grouper;
mod schema_sql;
mod storage;

// Re-export all public items so callers see the same paths as before.
pub use computation::{build_affinity_groups, compute_pairwise_affinities, ProjectAffinity};
pub use config::AffinityConfig;
pub use grouper::{AffinityGroupInfo, AffinityGrouper};
pub use schema_sql::{CREATE_AFFINITY_LABELS_SQL, CREATE_PROJECT_EMBEDDINGS_SQL};
pub use storage::{
    delete_project_embedding, load_affinity_label, load_all_project_embeddings,
    load_project_embedding, store_project_embedding,
};

// Re-export private helpers and sibling modules needed by the test suite.
// These are gated to test builds to avoid polluting the production API.
#[cfg(test)]
pub(crate) use crate::grouping::schema;
#[cfg(test)]
pub(crate) use computation::{
    affinity_group_id, compute_group_mean_similarity, group_mean_embedding,
};
#[cfg(test)]
pub(crate) use sqlx::SqlitePool;
#[cfg(test)]
pub(crate) use storage::{blob_to_embedding, embedding_to_blob, store_affinity_label};

#[cfg(test)]
#[path = "../affinity_tests.rs"]
mod tests;
