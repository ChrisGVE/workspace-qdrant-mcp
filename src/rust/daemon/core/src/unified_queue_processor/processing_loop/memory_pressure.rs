//! Memory pressure checks for the unified queue processor.
//!
//! Contains process RSS checks and system memory pressure checks used to gate
//! processing when memory is constrained.

use crate::unified_queue_processor::UnifiedQueueProcessor;

impl UnifiedQueueProcessor {
    /// Default maximum RSS in megabytes before pausing processing.
    /// Acts as a safety valve against memory leaks in the processing pipeline.
    pub(super) const DEFAULT_MAX_RSS_MB: u64 = 2048; // 2 GB

    /// Check if the daemon should pause processing due to memory constraints.
    ///
    /// Two independent checks:
    /// 1. **Process RSS** — pauses if this process exceeds `DEFAULT_MAX_RSS_MB`.
    ///    This is the primary safety valve on macOS where OS-level pressure
    ///    reporting is delayed by the memory compressor.
    /// 2. **System memory pressure** — pauses if OS reports low available memory.
    pub(super) async fn check_memory_pressure(max_memory_percent: u8) -> bool {
        // Check process RSS first — this is the reliable safety valve
        if Self::check_process_rss() {
            return true;
        }

        #[cfg(target_os = "macos")]
        {
            Self::check_memory_pressure_macos(max_memory_percent)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self::check_memory_pressure_generic(max_memory_percent)
        }
    }

    /// Check if this process's current RSS exceeds the safety limit.
    pub(super) fn check_process_rss() -> bool {
        let rss_mb = Self::current_rss_mb();
        rss_mb > Self::DEFAULT_MAX_RSS_MB
    }

    /// Get the current RSS of this process in megabytes.
    pub(super) fn current_rss_mb() -> u64 {
        #[cfg(target_os = "macos")]
        {
            // Use mach task_info for current (not peak) RSS on macOS
            Self::current_rss_mb_macos()
        }
        #[cfg(target_os = "linux")]
        {
            // Read /proc/self/statm for current RSS on Linux
            Self::current_rss_mb_linux()
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            0
        }
    }

    #[cfg(target_os = "macos")]
    fn current_rss_mb_macos() -> u64 {
        #[repr(C)]
        struct TaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],   // time_value_t
            system_time: [u32; 2], // time_value_t
            policy: i32,
            suspend_count: i32,
        }

        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(task: u32, flavor: u32, info: *mut TaskBasicInfo, count: *mut u32) -> i32;
        }

        const MACH_TASK_BASIC_INFO: u32 = 20;
        const MACH_TASK_BASIC_INFO_COUNT: u32 =
            (std::mem::size_of::<TaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;

        unsafe {
            let mut info: TaskBasicInfo = std::mem::zeroed();
            let mut count = MACH_TASK_BASIC_INFO_COUNT;
            let ret = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info as *mut _,
                &mut count,
            );
            if ret != 0 {
                return 0;
            }
            info.resident_size / (1024 * 1024)
        }
    }

    #[cfg(target_os = "linux")]
    fn current_rss_mb_linux() -> u64 {
        // /proc/self/statm fields: size resident shared text lib data dt (in pages)
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss_pages.parse::<u64>() {
                    return pages * 4096 / (1024 * 1024); // Assume 4K pages
                }
            }
        }
        0
    }

    /// macOS-specific memory pressure check using `kern.memorystatus_level`.
    ///
    /// This sysctl returns the kernel's own "percent available" metric (0-100),
    /// matching the value reported by the `memory_pressure` CLI tool. It accounts
    /// for the compressor, purgeable pages, and unified buffer cache.
    #[cfg(target_os = "macos")]
    fn check_memory_pressure_macos(max_memory_percent: u8) -> bool {
        let mut level: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        let name = b"kern.memorystatus_level\0";
        let ret = unsafe {
            libc::sysctlbyname(
                name.as_ptr() as *const libc::c_char,
                &mut level as *mut i32 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret != 0 || level < 0 {
            // Fallback to generic if sysctl fails
            return Self::check_memory_pressure_sysinfo(max_memory_percent);
        }
        let available_percent = level as u8;
        let min_available = 100u8.saturating_sub(max_memory_percent);
        available_percent < min_available
    }

    /// Generic memory pressure check for non-macOS platforms using sysinfo.
    #[cfg(not(target_os = "macos"))]
    fn check_memory_pressure_generic(max_memory_percent: u8) -> bool {
        Self::check_memory_pressure_sysinfo(max_memory_percent)
    }

    /// Sysinfo-based memory pressure check (cross-platform fallback).
    ///
    /// Uses a thread-local `System` instance to avoid allocating a new sysinfo
    /// object on every poll cycle (~500ms). `System::new()` is expensive and
    /// leaks memory if called repeatedly without reuse.
    fn check_memory_pressure_sysinfo(max_memory_percent: u8) -> bool {
        use std::cell::RefCell;
        use sysinfo::System;
        thread_local! {
            static SYS: RefCell<System> = RefCell::new(System::new());
        }
        SYS.with(|sys| {
            let mut sys = sys.borrow_mut();
            sys.refresh_memory();
            let total = sys.total_memory();
            if total == 0 {
                return false;
            }
            let available = sys.available_memory();
            let available_percent = (available as f64 / total as f64 * 100.0) as u8;
            let min_available = 100u8.saturating_sub(max_memory_percent);
            available_percent < min_available
        })
    }
}
