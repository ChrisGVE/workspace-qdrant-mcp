//! Project grouping subsystem.
//!
//! Consolidates all project grouping strategies:
//! - **schema**: Core `project_groups` table operations
//! - **affinity**: Embedding-based similarity grouping
//! - **dependency**: Shared dependency (Jaccard) grouping
//! - **workspace**: Workspace membership (Cargo, npm, Go) grouping
//! - **git_org**: Git organization/user grouping

pub mod affinity;
pub mod dependency;
pub mod git_org;
pub mod schema;
pub mod workspace;

// Re-export schema types (formerly project_groups_schema)
pub use schema::{
    add_to_group, get_group_members, list_tenant_groups, remove_from_group,
    CREATE_PROJECT_GROUPS_INDEXES_SQL, CREATE_PROJECT_GROUPS_SQL,
};

// Re-export affinity types (formerly affinity_grouper)
pub use affinity::{
    build_affinity_groups, compute_pairwise_affinities, delete_project_embedding,
    load_affinity_label, load_all_project_embeddings, load_project_embedding,
    store_project_embedding, AffinityConfig, AffinityGroupInfo, AffinityGrouper, ProjectAffinity,
    CREATE_AFFINITY_LABELS_SQL, CREATE_PROJECT_EMBEDDINGS_SQL,
};

// Re-export dependency types (formerly dependency_grouper)
pub use dependency::{
    compute_dependency_groups, is_dependency_file, jaccard_similarity, load_all_dependency_sets,
    parse_dependencies, store_dependencies, CREATE_PROJECT_DEPENDENCIES_INDEXES_SQL,
    CREATE_PROJECT_DEPENDENCIES_SQL,
};

// Re-export workspace types (formerly workspace_grouper)
pub use workspace::{
    compute_workspace_groups, detect_cargo_workspace, detect_go_workspace, detect_npm_workspace,
    update_project_workspace_group, WorkspaceInfo,
};

// Re-export git_org types (formerly git_org_grouper)
pub use git_org::{
    compute_git_org_groups, extract_git_org, org_to_group_id, update_project_org_group,
};
