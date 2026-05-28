//! Validation logic for [`ResourceLimitsConfig`].
use super::ResourceLimitsConfig;

impl ResourceLimitsConfig {
    /// Validate configuration settings.
    ///
    /// Must be called AFTER `resolve_auto_values()` -- 0 is not valid post-resolution.
    pub fn validate(&self) -> Result<(), String> {
        if self.nice_level < -20 || self.nice_level > 19 {
            return Err("nice_level must be between -20 and 19".to_string());
        }
        if self.max_concurrent_embeddings == 0 || self.max_concurrent_embeddings > 8 {
            return Err("max_concurrent_embeddings must be between 1 and 8 \
                 (0 should have been auto-resolved)"
                .to_string());
        }
        if self.max_memory_percent < 20 || self.max_memory_percent > 95 {
            return Err("max_memory_percent must be between 20 and 95".to_string());
        }
        if self.onnx_intra_threads == 0 || self.onnx_intra_threads > 16 {
            return Err("onnx_intra_threads must be between 1 and 16 \
                 (0 should have been auto-resolved)"
                .to_string());
        }
        match self.linux_idle_source.as_str() {
            "none" | "proc" => {}
            other => {
                return Err(format!(
                    "linux_idle_source must be one of: none, proc (got: {other})"
                ));
            }
        }
        if self.linux_idle_load_threshold <= 0.0 || self.linux_idle_load_threshold > 10.0 {
            return Err("linux_idle_load_threshold must be > 0.0 and <= 10.0".to_string());
        }
        Ok(())
    }
}
