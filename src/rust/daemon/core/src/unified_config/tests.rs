//! Tests for unified configuration

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::config::DaemonConfig;
    use crate::unified_config::expand_env_vars;
    use crate::unified_config::types::ConfigFormat;
    use crate::unified_config::UnifiedConfigManager;
    use wqm_common::env_expand::expand_path_env_vars;

    /// F-051: DaemonConfig must survive a full YAML round-trip without losing fields.
    ///
    /// `DaemonConfig` uses `#[serde(skip_serializing_if = "...")]` on several optional
    /// fields, so serialising a default instance and deserialising it back must leave all
    /// representative fields intact.
    #[test]
    fn test_daemon_config_yaml_round_trip() {
        let original = DaemonConfig::default();

        // Serialise to YAML — this is what `wqm config generate` produces.
        let yaml =
            serde_yaml_ng::to_string(&original).expect("DaemonConfig must serialise to YAML");

        // Deserialise back — this exercises the serde round-trip path.
        let restored: DaemonConfig =
            serde_yaml_ng::from_str(&yaml).expect("serialised YAML must deserialise back");

        // Assert representative fields from every top-level section survive intact.
        assert_eq!(restored.chunk_size, original.chunk_size, "chunk_size");
        assert_eq!(
            restored.max_concurrent_tasks, original.max_concurrent_tasks,
            "max_concurrent_tasks"
        );
        assert_eq!(restored.log_level, original.log_level, "log_level");
        assert_eq!(
            restored.enable_preemption, original.enable_preemption,
            "enable_preemption"
        );

        // embedding section
        assert_eq!(
            restored.embedding.cache_max_entries, original.embedding.cache_max_entries,
            "embedding.cache_max_entries"
        );
        assert_eq!(
            restored.embedding.model, original.embedding.model,
            "embedding.model"
        );

        // gRPC endpoint
        assert_eq!(
            restored.daemon_endpoint.grpc_port, original.daemon_endpoint.grpc_port,
            "daemon_endpoint.grpc_port"
        );

        // git section
        assert_eq!(
            restored.git.enable_branch_detection, original.git.enable_branch_detection,
            "git.enable_branch_detection"
        );

        // resource limits
        assert_eq!(
            restored.resource_limits.nice_level, original.resource_limits.nice_level,
            "resource_limits.nice_level"
        );

        // url_ingestion
        assert_eq!(
            restored.url_ingestion.max_body_bytes, original.url_ingestion.max_body_bytes,
            "url_ingestion.max_body_bytes"
        );
        assert_eq!(
            restored.url_ingestion.max_redirects, original.url_ingestion.max_redirects,
            "url_ingestion.max_redirects"
        );
    }

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
                !s.ends_with(".workspace-qdrant.yaml") && !s.ends_with(".workspace-qdrant.yml"),
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
        config.embedding.model_cache_dir = Some(PathBuf::from("${WQM_TEST_DIR}/models"));

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

    // ── T3.17 / T3.19 / T3.20 — mount-map end-to-end through load_config ─

    /// T3.17: a config file with an invalid mount entry causes the
    /// daemon's load_config (the same path startup.rs uses) to fail with
    /// an error that names the `mounts:` section.
    #[test]
    fn test_load_config_rejects_invalid_mount_relative_host() {
        use crate::unified_config::types::UnifiedConfigError;

        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmp.path().join("config.yaml");
        // Start from defaults so the only failure mode is mount validation.
        let mut cfg = crate::config::DaemonConfig::default();
        cfg.mounts = vec![wqm_common::yaml_defaults::YamlMountEntry {
            host: "relative/host".to_string(),
            container: "/mnt/x".to_string(),
        }];
        let yaml = serde_yaml_ng::to_string(&cfg).expect("serialise");
        std::fs::write(&cfg_path, yaml).expect("write");

        let manager = UnifiedConfigManager::new(None::<PathBuf>);
        let result = manager.load_config(Some(&cfg_path));
        let err = result.expect_err("invalid mount must reject load_config");
        match err {
            UnifiedConfigError::ValidationError(msg) => {
                assert!(
                    msg.contains("mounts:"),
                    "expected 'mounts:' prefix in validation error '{msg}'"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    /// T3.19: explicit `mounts: []` round-trips through load_config; the
    /// resulting MountMap is the identity map.
    #[test]
    fn test_load_config_accepts_empty_mounts() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmp.path().join("config.yaml");

        let mut cfg = crate::config::DaemonConfig::default();
        cfg.mounts.clear();
        std::fs::write(
            &cfg_path,
            serde_yaml_ng::to_string(&cfg).expect("serialise"),
        )
        .expect("write");

        let manager = UnifiedConfigManager::new(None::<PathBuf>);
        let loaded = manager
            .load_config(Some(&cfg_path))
            .expect("empty mounts must load");
        assert!(loaded.mounts.is_empty());
        let map = loaded.build_mount_map().expect("identity always builds");
        assert!(map.is_identity());
    }

    /// T3.19: a valid two-entry mount map survives a full load_config cycle.
    #[test]
    fn test_load_config_round_trip_with_valid_mounts() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmp.path().join("config.yaml");

        let mut cfg = crate::config::DaemonConfig::default();
        cfg.mounts = vec![
            wqm_common::yaml_defaults::YamlMountEntry {
                host: "/Users/chris/dev".to_string(),
                container: "/Users/chris/dev".to_string(),
            },
            wqm_common::yaml_defaults::YamlMountEntry {
                host: "/Volumes/External/books".to_string(),
                container: "/mnt/external-books".to_string(),
            },
        ];
        std::fs::write(
            &cfg_path,
            serde_yaml_ng::to_string(&cfg).expect("serialise"),
        )
        .expect("write");

        let manager = UnifiedConfigManager::new(None::<PathBuf>);
        let loaded = manager
            .load_config(Some(&cfg_path))
            .expect("valid mounts must load");
        assert_eq!(loaded.mounts.len(), 2);
        let map = loaded.build_mount_map().expect("build_mount_map");
        assert_eq!(map.len(), 2);
        assert!(!map.is_identity());
    }
}
