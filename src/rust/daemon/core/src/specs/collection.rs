//! Canonical definition of the 4 Qdrant collections.
//!
//! Every collection name, tenant field, and creation config is defined here
//! in one place. Strategy code and collection-ensure helpers reference this
//! enum instead of ad-hoc string constants.

use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_RULES, COLLECTION_PROJECTS, COLLECTION_SCRATCHPAD,
};

use crate::storage::MultiTenantConfig;

/// The 4 canonical Qdrant collections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Collection {
    /// Code and documents from all projects, isolated by `project_id`.
    Projects,
    /// Library documentation, isolated by `library_name`.
    Libraries,
    /// Behavioral rules (global).
    Rules,
    /// Persistent LLM scratch space, isolated by `tenant_id`.
    Scratchpad,
}

impl Collection {
    /// The Qdrant collection name (matches `wqm_common::constants::COLLECTION_*`).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Projects => COLLECTION_PROJECTS,
            Self::Libraries => COLLECTION_LIBRARIES,
            Self::Rules => COLLECTION_RULES,
            Self::Scratchpad => COLLECTION_SCRATCHPAD,
        }
    }

    /// The `MultiTenantConfig` used when creating this collection.
    ///
    /// Currently returns default for all collections — this is the single
    /// point to customize per-collection settings in the future.
    pub fn creation_config(&self) -> MultiTenantConfig {
        MultiTenantConfig::default()
    }

    /// Parse a collection name string into the enum.
    ///
    /// Returns `None` for non-canonical names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            COLLECTION_PROJECTS => Some(Self::Projects),
            COLLECTION_LIBRARIES => Some(Self::Libraries),
            COLLECTION_RULES => Some(Self::Rules),
            COLLECTION_SCRATCHPAD => Some(Self::Scratchpad),
            _ => None,
        }
    }

    /// The payload field used to isolate tenants within this collection.
    pub fn tenant_field(&self) -> &'static str {
        match self {
            Self::Projects => "project_id",
            Self::Libraries => "library_name",
            Self::Rules => "tenant_id",
            Self::Scratchpad => "tenant_id",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_names() {
        assert_eq!(Collection::Projects.name(), "projects");
        assert_eq!(Collection::Libraries.name(), "libraries");
        assert_eq!(Collection::Rules.name(), "rules");
        assert_eq!(Collection::Scratchpad.name(), "scratchpad");
    }

    #[test]
    fn test_from_name_valid() {
        assert_eq!(Collection::from_name("projects"), Some(Collection::Projects));
        assert_eq!(Collection::from_name("libraries"), Some(Collection::Libraries));
        assert_eq!(Collection::from_name("rules"), Some(Collection::Rules));
        assert_eq!(Collection::from_name("scratchpad"), Some(Collection::Scratchpad));
    }

    #[test]
    fn test_from_name_invalid() {
        assert_eq!(Collection::from_name("unknown"), None);
        assert_eq!(Collection::from_name(""), None);
        assert_eq!(Collection::from_name("Projects"), None); // case-sensitive
    }

    #[test]
    fn test_creation_config_defaults() {
        for collection in [
            Collection::Projects,
            Collection::Libraries,
            Collection::Rules,
            Collection::Scratchpad,
        ] {
            let config = collection.creation_config();
            assert_eq!(config.vector_size, 384, "{:?} vector_size", collection);
            assert_eq!(config.hnsw_m, 16, "{:?} hnsw_m", collection);
            assert_eq!(config.hnsw_ef_construct, 100, "{:?} hnsw_ef_construct", collection);
        }
    }

    #[test]
    fn test_tenant_field_per_collection() {
        assert_eq!(Collection::Projects.tenant_field(), "project_id");
        assert_eq!(Collection::Libraries.tenant_field(), "library_name");
        assert_eq!(Collection::Rules.tenant_field(), "tenant_id");
        assert_eq!(Collection::Scratchpad.tenant_field(), "tenant_id");
    }
}
