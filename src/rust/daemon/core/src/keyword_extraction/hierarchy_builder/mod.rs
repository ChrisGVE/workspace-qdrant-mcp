//! Nightly canonical tag hierarchy rebuild.
//!
//! Periodically rebuilds the canonical tag graph per tenant by:
//! 1. Collecting all concept tags from the `tags` table
//! 2. Embedding tag phrases for vector similarity
//! 3. Running `canonical_tags::build_hierarchy()` (dedup → cluster)
//! 4. Writing results to `canonical_tags` and `tag_hierarchy_edges` tables
//!
//! Triggers:
//! - Scheduled nightly at 2 AM local time
//! - Manual via CLI: `wqm tags rebuild --tenant <id>`

mod builder;
mod scheduler;
mod types;

pub use builder::HierarchyBuilder;
pub use types::{FullRebuildResult, HierarchyError, HierarchyRebuildConfig, RebuildResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HierarchyRebuildConfig::default();
        assert_eq!(config.min_tags_threshold, 10);
        assert_eq!(config.collection, "projects");
        assert!((config.canonical.merge_threshold - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_rebuild_result_fields() {
        let result = RebuildResult {
            tenant_id: "test-tenant".to_string(),
            tags_collected: 50,
            level3_count: 30,
            level2_count: 10,
            level1_count: 3,
            edges_created: 40,
        };
        assert_eq!(result.tenant_id, "test-tenant");
        assert_eq!(result.tags_collected, 50);
    }

    #[test]
    fn test_full_rebuild_result_defaults() {
        let result = FullRebuildResult {
            tenants_processed: 0,
            tenants_skipped: 0,
            total_canonical_tags: 0,
            total_edges: 0,
            tenant_results: Vec::new(),
        };
        assert!(result.tenant_results.is_empty());
    }

    #[test]
    fn test_delay_until_next_2am_is_positive() {
        let delay = scheduler::delay_until_next_2am();
        assert!(delay.as_secs() > 0, "Delay should be positive");
        // Should never be more than 24 hours
        assert!(
            delay.as_secs() < 86400 + 3600, // 25 hours max (DST edge)
            "Delay should be less than ~25 hours, got {}s",
            delay.as_secs()
        );
    }

    #[test]
    fn test_delay_is_reasonable() {
        let delay = scheduler::delay_until_next_2am();
        // At minimum a few seconds, at maximum ~24h
        assert!(delay.as_secs() >= 1);
        assert!(delay.as_secs() <= 90000); // 25 hours
    }

    #[test]
    fn test_hierarchy_error_display() {
        let err = HierarchyError::Database(sqlx::Error::RowNotFound);
        assert!(err.to_string().contains("Database error"));
    }
}
