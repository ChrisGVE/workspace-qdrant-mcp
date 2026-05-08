//! Platform-specific detection and auto-resolution of resource limits.
use super::ResourceLimitsConfig;

impl ResourceLimitsConfig {
    /// Resolve auto-detected values (0 = auto) based on hardware.
    ///
    /// Call this after loading config and applying env overrides, before validation.
    /// Values that are already non-zero (explicitly set) are left unchanged.
    ///
    /// Formula:
    ///   - `max_concurrent_embeddings`: max(1, physical_cores / 4)
    ///   - `onnx_intra_threads`: 2 (all-MiniLM-L6-v2 is too small to benefit from more)
    pub fn resolve_auto_values(&mut self) {
        let physical_cores = detect_physical_cores();

        if self.max_concurrent_embeddings == 0 {
            self.max_concurrent_embeddings = std::cmp::max(1, physical_cores / 4);
            tracing::info!(
                "Auto-detected max_concurrent_embeddings = {} (physical_cores={}, formula: cores/4)",
                self.max_concurrent_embeddings,
                physical_cores
            );
        }

        if self.onnx_intra_threads == 0 {
            // 2 is optimal for the small all-MiniLM-L6-v2 model regardless of core count.
            // The ONNX graph is narrow enough that additional threads yield diminishing returns.
            self.onnx_intra_threads = 2;
            tracing::info!("Auto-detected onnx_intra_threads = 2 (optimal for all-MiniLM-L6-v2)");
        }

        tracing::info!(
            "Embedding resource budget: {} workers x {} threads/worker = {} total threads (physical_cores={})",
            self.max_concurrent_embeddings,
            self.onnx_intra_threads,
            self.max_concurrent_embeddings * self.onnx_intra_threads,
            physical_cores
        );
    }
}

/// Detect the number of physical CPU cores on the current machine.
///
/// Uses `sysinfo::System::physical_core_count()` with a fallback to
/// `std::thread::available_parallelism()` (which returns logical cores).
/// Returns 4 as a safe fallback if both methods fail.
pub fn detect_physical_cores() -> usize {
    use sysinfo::System;

    let sys = System::new();
    if let Some(physical) = sys.physical_core_count() {
        return physical;
    }

    // Fallback: logical cores (includes hyperthreading)
    if let Ok(logical) = std::thread::available_parallelism() {
        return logical.get();
    }

    // Ultimate fallback
    4
}
