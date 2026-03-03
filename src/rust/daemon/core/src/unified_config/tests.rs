//! Tests for unified configuration

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::config::DaemonConfig;
    use crate::unified_config::types::ConfigFormat;
    use crate::unified_config::expand_env_vars;
    use crate::unified_config::UnifiedConfigManager;
    use wqm_common::env_expand::expand_path_env_vars;

    #[test]
    fn test_config_format_detection() {
        assert_eq!(ConfigFormat::from_path("config.yaml"), ConfigFormat::Yaml);
        assert_eq!(ConfigFormat::from_path("config.yml"), ConfigFormat::Yaml);
        assert_eq!(ConfigFormat::from_path("config"), ConfigFormat::Yaml);
        assert_eq!(ConfigFormat::from_path("config.toml"), ConfigFormat::Yaml);
        assert_eq!(ConfigFormat::from_path("config.json"), ConfigFormat::Yaml);
    }

    #[test]
    fn test_no_project_local_in_search_paths() {
        let config_manager = UnifiedConfigManager::new(None::<PathBuf>);
        let paths = config_manager.get_unified_search_paths();

        // No path should reference project-local config files
        for p in &paths {
            let s = p.to_string_lossy();
            assert!(
                !s.ends_with(".workspace-qdrant.yaml")
                    && !s.ends_with(".workspace-qdrant.yml"),
                "Should not search project-local config: {:?}",
                p
            );
        }
    }

    #[test]
    fn test_default_config_creation() {
        let config = DaemonConfig::default();
        assert_eq!(config.max_concurrent_tasks, Some(4));
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn test_config_validation() {
        let config_manager = UnifiedConfigManager::new(None::<PathBuf>);

        let valid_config = DaemonConfig::default();
        assert!(config_manager.validate_config(&valid_config).is_ok());

        let mut invalid_config = DaemonConfig::default();
        invalid_config.chunk_size = 0;
        assert!(config_manager.validate_config(&invalid_config).is_err());

        invalid_config = DaemonConfig::default();
        invalid_config.max_concurrent_tasks = Some(0);
        assert!(config_manager.validate_config(&invalid_config).is_err());

        invalid_config = DaemonConfig::default();
        invalid_config.log_level = "invalid".to_string();
        assert!(config_manager.validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_expand_env_vars() {
        std::env::set_var("WQM_TEST_VAR", "/test/path");

        assert_eq!(expand_env_vars("${WQM_TEST_VAR}/cache"), "/test/path/cache");
        assert_eq!(expand_env_vars("$WQM_TEST_VAR/cache"), "/test/path/cache");

        let result = expand_env_vars("$WQM_NONEXISTENT_VAR/path");
        assert!(result.contains("WQM_NONEXISTENT_VAR"));

        assert_eq!(expand_env_vars("/static/path"), "/static/path");

        std::env::set_var("WQM_TEST_VAR2", "subdir");
        assert_eq!(
            expand_env_vars("${WQM_TEST_VAR}/${WQM_TEST_VAR2}"),
            "/test/path/subdir"
        );

        std::env::remove_var("WQM_TEST_VAR");
        std::env::remove_var("WQM_TEST_VAR2");
    }

    #[test]
    fn test_expand_path_env_vars() {
        std::env::set_var("WQM_TEST_HOME", "/home/testuser");

        let path = PathBuf::from("${WQM_TEST_HOME}/.cache/models");
        let expanded = expand_path_env_vars(Some(&path));
        assert!(expanded.is_some());
        assert_eq!(
            expanded.unwrap(),
            PathBuf::from("/home/testuser/.cache/models")
        );

        let expanded_none = expand_path_env_vars(None);
        assert!(expanded_none.is_none());

        std::env::remove_var("WQM_TEST_HOME");
    }

    #[test]
    fn test_expand_config_paths() {
        let config_manager = UnifiedConfigManager::new(None::<PathBuf>);

        std::env::set_var("WQM_TEST_DIR", "/expanded/dir");

        let mut config = DaemonConfig::default();
        config.log_file = Some(PathBuf::from("${WQM_TEST_DIR}/daemon.log"));
        config.project_path = Some(PathBuf::from("$WQM_TEST_DIR/project"));
        config.embedding.model_cache_dir =
            Some(PathBuf::from("${WQM_TEST_DIR}/models"));

        let expanded = config_manager.expand_config_paths(config);

        assert_eq!(
            expanded.log_file.unwrap(),
            PathBuf::from("/expanded/dir/daemon.log")
        );
        assert_eq!(
            expanded.project_path.unwrap(),
            PathBuf::from("/expanded/dir/project")
        );
        assert_eq!(
            expanded.embedding.model_cache_dir.unwrap(),
            PathBuf::from("/expanded/dir/models")
        );

        std::env::remove_var("WQM_TEST_DIR");
    }
}
