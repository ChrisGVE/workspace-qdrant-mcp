//! Project grouping subsystem.
//!
//! Consolidates all project grouping strategies:
//! - **schema**: Core `project_groups` table operations
//! - **affinity**: Embedding-based similarity grouping
//! - **dependency**: Shared dependency (Jaccard) grouping
//! - **workspace**: Workspace membership (Cargo, npm, Go) grouping
//! - **git_org**: Git organization/user grouping

pub mod schema;
pub mod affinity;
pub mod dependency;
pub mod workspace;
pub mod git_org;

// Re-export schema types (formerly project_groups_schema)
pub use schema::{
    CREATE_PROJECT_GROUPS_SQL, CREATE_PROJECT_GROUPS_INDEXES_SQL,
    get_group_members, add_to_group, remove_from_group, list_tenant_groups,
};

// Re-export affinity types (formerly affinity_grouper)
pub use affinity::{
    AffinityConfig, AffinityGrouper, AffinityGroupInfo, ProjectAffinity,
    CREATE_PROJECT_EMBEDDINGS_SQL, CREATE_AFFINITY_LABELS_SQL,
    store_project_embedding, load_project_embedding, load_all_project_embeddings,
    delete_project_embedding, compute_pairwise_affinities, build_affinity_groups,
    load_affinity_label,
};

// Re-export dependency types (formerly dependency_grouper)
pub use dependency::{
    CREATE_PROJECT_DEPENDENCIES_SQL, CREATE_PROJECT_DEPENDENCIES_INDEXES_SQL,
    is_dependency_file, parse_dependencies, jaccard_similarity,
    store_dependencies, load_all_dependency_sets, compute_dependency_groups,
};

// Re-export workspace types (formerly workspace_grouper)
pub use workspace::{
    WorkspaceInfo,
    detect_cargo_workspace, detect_npm_workspace, detect_go_workspace,
    compute_workspace_groups, update_project_workspace_group,
};

// Re-export git_org types (formerly git_org_grouper)
pub use git_org::{
    extract_git_org, org_to_group_id,
    compute_git_org_groups, update_project_org_group,
};
