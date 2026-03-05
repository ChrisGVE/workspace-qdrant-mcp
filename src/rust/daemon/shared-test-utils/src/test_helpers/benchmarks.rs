//! Performance benchmarking and memory tracking helpers for tests

use std::time::{Duration, Instant};

use crate::TestResult;

/// Performance benchmarking helper
pub struct PerformanceBenchmark {
    start_time: Instant,
    operation_name: String,
}

impl PerformanceBenchmark {
    /// Start a new benchmark
    pub fn start(operation_name: &str) -> Self {
        Self {
            start_time: Instant::now(),
            operation_name: operation_name.to_string(),
        }
    }

    /// End the benchmark and return duration
    pub fn end(self) -> Duration {
        let duration = self.start_time.elapsed();
        tracing::info!(
            "Performance benchmark '{}' completed in {:?}",
            self.operation_name,
            duration
        );
        duration
    }

    /// End the benchmark and assert it completed within expected time
    pub fn end_with_assertion(self, max_duration: Duration) -> TestResult<Duration> {
        let operation_name = self.operation_name.clone();
        let duration = self.end();
        if duration > max_duration {
            return Err(format!(
                "Performance benchmark '{}' took {:?}, expected <= {:?}",
                operation_name, duration, max_duration
            )
            .into());
        }
        Ok(duration)
    }
}

/// Memory usage tracking helper
pub struct MemoryTracker {
    initial_memory: Option<usize>,
    operation_name: String,
}

impl MemoryTracker {
    /// Start memory tracking
    pub fn start(operation_name: &str) -> Self {
        // Note: Getting actual memory usage would require platform-specific code
        // For now, this is a placeholder that could be extended
        Self {
            initial_memory: Self::get_memory_usage(),
            operation_name: operation_name.to_string(),
        }
    }

    /// End memory tracking and log difference
    pub fn end(self) -> Option<isize> {
        if let (Some(initial), Some(final_memory)) = (self.initial_memory, Self::get_memory_usage())
        {
            let diff = final_memory as isize - initial as isize;
            tracing::info!(
                "Memory usage for '{}': initial={}, final={}, diff={}",
                self.operation_name,
                initial,
                final_memory,
                diff
            );
            Some(diff)
        } else {
            tracing::warn!(
                "Memory tracking not available for '{}'",
                self.operation_name
            );
            None
        }
    }

    /// Get current process RSS (Resident Set Size) in bytes
    fn get_memory_usage() -> Option<usize> {
        Self::get_rss_bytes()
    }

    #[cfg(target_os = "macos")]
    fn get_rss_bytes() -> Option<usize> {
        use std::mem;
        // Use mach task_info to get resident memory
        unsafe {
            let mut info: libc::mach_task_basic_info_data_t = mem::zeroed();
            let mut count = (mem::size_of::<libc::mach_task_basic_info_data_t>()
                / mem::size_of::<libc::natural_t>())
                as libc::mach_msg_type_number_t;
            let result = libc::task_info(
                #[allow(deprecated)]
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as libc::task_info_t,
                &mut count,
            );
            if result == libc::KERN_SUCCESS {
                Some(info.resident_size as usize)
            } else {
                None
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn get_rss_bytes() -> Option<usize> {
        // Read /proc/self/statm for RSS in pages
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let rss_pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size > 0 {
            Some(rss_pages * page_size as usize)
        } else {
            None
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn get_rss_bytes() -> Option<usize> {
        tracing::warn!("Memory tracking not supported on this platform");
        None
    }
}
