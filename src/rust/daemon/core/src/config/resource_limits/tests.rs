#[cfg(test)]
mod tests {
    use crate::config::resource_limits::ResourceLimitsConfig;

    #[test]
    fn test_resource_limits_config_defaults() {
        let config = ResourceLimitsConfig::default();
        assert_eq!(config.nice_level, 10);
        assert_eq!(
            config.max_concurrent_embeddings, 0,
            "default should be 0 (auto-detect)"
        );
        assert_eq!(config.max_memory_percent, 70);
        assert_eq!(
            config.onnx_intra_threads, 0,
            "default should be 0 (auto-detect)"
        );
    }

    #[test]
    fn test_resource_limits_auto_detection() {
        let mut config = ResourceLimitsConfig::default();
        assert_eq!(config.max_concurrent_embeddings, 0);
        assert_eq!(config.onnx_intra_threads, 0);

        config.resolve_auto_values();

        // After resolution, values should be non-zero
        assert!(
            config.max_concurrent_embeddings >= 1,
            "auto-detected embeddings should be >= 1"
        );
        assert!(
            config.max_concurrent_embeddings <= 8,
            "auto-detected embeddings should be <= 8"
        );
        assert_eq!(
            config.onnx_intra_threads, 2,
            "onnx_intra_threads always resolves to 2"
        );

        // Validation should pass after resolution
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_resource_limits_explicit_values_not_overridden() {
        let mut config = ResourceLimitsConfig::default();
        config.max_concurrent_embeddings = 3;
        config.onnx_intra_threads = 4;

        config.resolve_auto_values();

        // Explicitly set values should not be changed
        assert_eq!(config.max_concurrent_embeddings, 3);
        assert_eq!(config.onnx_intra_threads, 4);
    }

    #[test]
    fn test_resource_limits_config_validation() {
        let mut config = ResourceLimitsConfig::default();
        config.resolve_auto_values(); // Must resolve before validation

        // Valid settings after resolution
        assert!(config.validate().is_ok());

        // Invalid nice_level (too low)
        config.nice_level = -21;
        assert!(config.validate().is_err());
        // Invalid nice_level (too high)
        config.nice_level = 20;
        assert!(config.validate().is_err());
        config.nice_level = 10;

        // Invalid max_concurrent_embeddings (zero = unresolved auto-detect)
        config.max_concurrent_embeddings = 0;
        assert!(config.validate().is_err());
        // Invalid max_concurrent_embeddings (too high)
        config.max_concurrent_embeddings = 9;
        assert!(config.validate().is_err());
        config.max_concurrent_embeddings = 2;

        // Invalid max_memory_percent (too low)
        config.max_memory_percent = 19;
        assert!(config.validate().is_err());
        // Invalid max_memory_percent (too high)
        config.max_memory_percent = 96;
        assert!(config.validate().is_err());
        config.max_memory_percent = 70;

        // Invalid onnx_intra_threads (zero = unresolved)
        config.onnx_intra_threads = 0;
        assert!(config.validate().is_err());
        // Invalid onnx_intra_threads (too high)
        config.onnx_intra_threads = 17;
        assert!(config.validate().is_err());
        config.onnx_intra_threads = 2;

        // Valid again
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_resource_limits_config_boundary_values() {
        let mut config = ResourceLimitsConfig::default();
        config.resolve_auto_values(); // Must resolve before validation

        config.nice_level = -20;
        assert!(config.validate().is_ok());
        config.nice_level = 19;
        assert!(config.validate().is_ok());

        config.max_concurrent_embeddings = 1;
        assert!(config.validate().is_ok());
        config.max_concurrent_embeddings = 8;
        assert!(config.validate().is_ok());

        config.onnx_intra_threads = 1;
        assert!(config.validate().is_ok());
        config.onnx_intra_threads = 16;
        assert!(config.validate().is_ok());

        config.max_memory_percent = 20;
        assert!(config.validate().is_ok());
        config.max_memory_percent = 95;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_resource_limits_env_overrides() {
        let mut config = ResourceLimitsConfig::default();

        // Set environment variables
        std::env::set_var("WQM_RESOURCE_NICE_LEVEL", "5");
        std::env::set_var("WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS", "4");
        std::env::set_var("WQM_RESOURCE_MAX_MEMORY_PERCENT", "80");

        config.apply_env_overrides();

        assert_eq!(config.nice_level, 5);
        assert_eq!(config.max_concurrent_embeddings, 4);
        assert_eq!(config.max_memory_percent, 80);

        // Clean up
        std::env::remove_var("WQM_RESOURCE_NICE_LEVEL");
        std::env::remove_var("WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS");
        std::env::remove_var("WQM_RESOURCE_MAX_MEMORY_PERCENT");
    }

    #[test]
    fn test_resource_limits_serialization() {
        let config = ResourceLimitsConfig {
            nice_level: 5,
            max_concurrent_embeddings: 4,
            max_memory_percent: 80,
            onnx_intra_threads: 2,
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"nice_level\":5"));
        assert!(json.contains("\"max_memory_percent\":80"));

        let deserialized: ResourceLimitsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nice_level, 5);
        assert_eq!(deserialized.max_memory_percent, 80);
        // Idle config should have defaults
        assert_eq!(deserialized.idle_threshold_secs, 120);
        assert_eq!(deserialized.idle_confirmation_secs, 300);
        assert_eq!(deserialized.ramp_up_step_secs, 120);
        assert_eq!(deserialized.ramp_down_step_secs, 300);
        assert_eq!(deserialized.burst_hold_secs, 600);
        assert!((deserialized.burst_concurrency_multiplier - 2.0).abs() < f64::EPSILON);
    }
}
