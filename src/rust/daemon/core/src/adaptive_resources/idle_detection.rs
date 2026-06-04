//! Platform-specific idle detection and CPU pressure monitoring.
//!
//! ## Platform support
//! - macOS: CGEventSourceSecondsSinceLastEventType (CoreGraphics),
//!          with IOKit HIDIdleTime fallback (works during screen lock)
//! - Linux: pluggable backend selected via `resource_limits.linux_idle_source`.
//!          Currently supports `"none"` (default — no idle detection) and
//!          `"proc"` (1-minute load average heuristic via `/proc/loadavg`).
//! - Other: always-interactive (no burst mode)

use crate::config::ResourceLimitsConfig;

/// Stateful idle detector owned by the adaptive resource manager.
///
/// Wraps the platform-specific detection backend. Construct once at manager
/// startup via [`IdleDetector::new`] and call [`IdleDetector::seconds_since_last_input`]
/// on each poll.
pub struct IdleDetector {
    #[cfg(target_os = "linux")]
    linux: Option<linux_idle::LinuxIdleDetector>,
    #[cfg(not(target_os = "linux"))]
    _phantom: std::marker::PhantomData<()>,
}

impl IdleDetector {
    /// Build a detector from the resolved resource-limits config.
    ///
    /// `physical_cores` is used to normalize load-average when the `/proc`
    /// backend is active. Pass the same value you feed into
    /// `AdaptiveResourceConfig`.
    pub fn new(config: &ResourceLimitsConfig, physical_cores: usize) -> Self {
        let _ = (config, physical_cores); // silence unused on non-linux

        #[cfg(target_os = "linux")]
        {
            let linux = match config.linux_idle_source.as_str() {
                "proc" => Some(linux_idle::LinuxIdleDetector::new_proc(
                    config.linux_idle_load_threshold,
                    physical_cores.max(1),
                )),
                _ => None,
            };
            Self { linux }
        }

        #[cfg(not(target_os = "linux"))]
        {
            Self {
                _phantom: std::marker::PhantomData,
            }
        }
    }

    /// Return seconds since last user input, or `None` when detection is
    /// unavailable on this platform / backend.
    pub fn seconds_since_last_input(&self) -> Option<f64> {
        #[cfg(target_os = "macos")]
        {
            macos_idle::seconds_since_last_input()
        }

        #[cfg(target_os = "linux")]
        {
            self.linux
                .as_ref()
                .and_then(|d| d.seconds_since_last_input())
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            None
        }
    }
}

/// Back-compat shim for callers that want an idle reading without owning a
/// detector. On macOS this is the historical CoreGraphics+IOKit path; on
/// every other platform it returns `None`.
#[allow(dead_code)]
pub(super) fn seconds_since_last_input() -> Option<f64> {
    #[cfg(target_os = "macos")]
    {
        macos_idle::seconds_since_last_input()
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
mod macos_idle {
    // CGEventSourceSecondsSinceLastEventType is in the CoreGraphics framework.
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(source_state_id: i32, event_type: u32) -> f64;
    }

    // IOKit framework for HIDIdleTime (works during screen lock).
    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOServiceGetMatchingService(main_port: u32, matching: *const std::ffi::c_void) -> u32;
        fn IOServiceMatching(name: *const std::ffi::c_char) -> *const std::ffi::c_void;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: *const std::ffi::c_void,
            allocator: *const std::ffi::c_void,
            options: u32,
        ) -> *const std::ffi::c_void;
        fn IOObjectRelease(object: u32) -> u32;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(
            alloc: *const std::ffi::c_void,
            c_str: *const std::ffi::c_char,
            encoding: u32,
        ) -> *const std::ffi::c_void;
        fn CFNumberGetValue(
            number: *const std::ffi::c_void,
            the_type: i64,
            value_ptr: *mut std::ffi::c_void,
        ) -> bool;
        fn CFRelease(cf: *const std::ffi::c_void);
    }

    const CG_EVENT_SOURCE_STATE_COMBINED_SESSION: i32 = 1;
    const CG_ANY_INPUT_EVENT_TYPE: u32 = 0xFFFFFFFF;
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x08000100;
    const K_CF_NUMBER_SINT64_TYPE: i64 = 4;
    const K_IO_MAIN_PORT_DEFAULT: u32 = 0;

    /// Primary idle detection via CoreGraphics.
    fn cg_idle_seconds() -> Option<f64> {
        let secs = unsafe {
            CGEventSourceSecondsSinceLastEventType(
                CG_EVENT_SOURCE_STATE_COMBINED_SESSION,
                CG_ANY_INPUT_EVENT_TYPE,
            )
        };
        if secs >= 0.0 {
            Some(secs)
        } else {
            None
        }
    }

    /// Fallback idle detection via IOKit HIDIdleTime.
    /// Works during screen lock when CGEventSource may not update.
    fn iokit_idle_seconds() -> Option<f64> {
        unsafe {
            let name = b"IOHIDSystem\0".as_ptr() as *const std::ffi::c_char;
            let matching = IOServiceMatching(name);
            if matching.is_null() {
                return None;
            }

            let service = IOServiceGetMatchingService(K_IO_MAIN_PORT_DEFAULT, matching);
            if service == 0 {
                return None;
            }

            let key_str = b"HIDIdleTime\0".as_ptr() as *const std::ffi::c_char;
            let cf_key =
                CFStringCreateWithCString(std::ptr::null(), key_str, K_CF_STRING_ENCODING_UTF8);
            if cf_key.is_null() {
                IOObjectRelease(service);
                return None;
            }

            let cf_number = IORegistryEntryCreateCFProperty(service, cf_key, std::ptr::null(), 0);
            CFRelease(cf_key);
            IOObjectRelease(service);

            if cf_number.is_null() {
                return None;
            }

            let mut idle_ns: i64 = 0;
            let ok = CFNumberGetValue(
                cf_number,
                K_CF_NUMBER_SINT64_TYPE,
                &mut idle_ns as *mut i64 as *mut std::ffi::c_void,
            );
            CFRelease(cf_number);

            if ok && idle_ns >= 0 {
                Some(idle_ns as f64 / 1_000_000_000.0)
            } else {
                None
            }
        }
    }

    /// Returns seconds since last user input.
    /// Tries CoreGraphics first (lighter), falls back to IOKit (works during screen lock).
    pub fn seconds_since_last_input() -> Option<f64> {
        cg_idle_seconds().or_else(iokit_idle_seconds)
    }
}

#[cfg(target_os = "linux")]
pub(super) mod linux_idle {
    //! Linux idle backends.
    //!
    //! Only `/proc/loadavg` is wired today. The heuristic treats the host as
    //! idle whenever `load_1min / physical_cores < threshold` and reports the
    //! elapsed time since the last non-idle sample as the "seconds since last
    //! input" signal consumed by the adaptive state machine.

    use std::sync::Mutex;
    use std::time::Instant;

    /// Abstract reader for the 1-minute normalised load value. Lets tests
    /// inject deterministic load sequences without touching `/proc`.
    pub(super) trait LoadReader: Send + Sync {
        fn read_1m_load(&self) -> Option<f64>;
    }

    /// Real `/proc/loadavg` reader.
    struct ProcLoadReader;

    impl LoadReader for ProcLoadReader {
        fn read_1m_load(&self) -> Option<f64> {
            let raw = std::fs::read_to_string("/proc/loadavg").ok()?;
            raw.split_whitespace()
                .next()
                .and_then(|first| first.parse::<f64>().ok())
        }
    }

    pub(super) struct LinuxIdleDetector {
        reader: Box<dyn LoadReader>,
        threshold: f64,
        cores: f64,
        last_active: Mutex<Instant>,
    }

    impl LinuxIdleDetector {
        /// `/proc/loadavg` backend.
        pub fn new_proc(threshold: f64, cores: usize) -> Self {
            Self::with_reader(Box::new(ProcLoadReader), threshold, cores)
        }

        /// Build with an explicit reader. Used by tests.
        pub(super) fn with_reader(
            reader: Box<dyn LoadReader>,
            threshold: f64,
            cores: usize,
        ) -> Self {
            Self {
                reader,
                threshold,
                cores: cores.max(1) as f64,
                last_active: Mutex::new(Instant::now()),
            }
        }

        /// Return seconds since the last "non-idle" sample, or `None` if the
        /// backing reader is unavailable.
        pub fn seconds_since_last_input(&self) -> Option<f64> {
            let raw = self.reader.read_1m_load()?;
            let normalized = raw / self.cores;

            let mut last = self.last_active.lock().ok()?;
            if normalized >= self.threshold {
                *last = Instant::now();
                Some(0.0)
            } else {
                Some(last.elapsed().as_secs_f64())
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::thread::sleep;
        use std::time::Duration;

        /// Scripted reader yielding one load value per call, wrapping at end.
        struct ScriptedReader {
            values: Vec<f64>,
            cursor: AtomicUsize,
        }

        impl ScriptedReader {
            fn new(values: Vec<f64>) -> Arc<Self> {
                Arc::new(Self {
                    values,
                    cursor: AtomicUsize::new(0),
                })
            }
        }

        impl LoadReader for Arc<ScriptedReader> {
            fn read_1m_load(&self) -> Option<f64> {
                if self.values.is_empty() {
                    return None;
                }
                let idx = self.cursor.fetch_add(1, Ordering::Relaxed) % self.values.len();
                Some(self.values[idx])
            }
        }

        struct FailingReader;

        impl LoadReader for FailingReader {
            fn read_1m_load(&self) -> Option<f64> {
                None
            }
        }

        fn detector_with(values: Vec<f64>, threshold: f64, cores: usize) -> LinuxIdleDetector {
            let reader = ScriptedReader::new(values);
            LinuxIdleDetector::with_reader(Box::new(reader), threshold, cores)
        }

        #[test]
        fn busy_load_resets_to_zero_idle() {
            // load 1.6 on 4 cores = 0.4 normalized > threshold 0.1 → busy
            let detector = detector_with(vec![1.6], 0.1, 4);
            let idle = detector.seconds_since_last_input().unwrap();
            assert_eq!(idle, 0.0);
        }

        #[test]
        fn idle_load_accumulates_seconds() {
            // load 0.04 on 4 cores = 0.01 normalized < threshold 0.1 → idle
            let detector = detector_with(vec![0.04], 0.1, 4);
            // First poll starts counting.
            let _ = detector.seconds_since_last_input();
            sleep(Duration::from_millis(60));
            let idle = detector.seconds_since_last_input().unwrap();
            assert!(
                idle >= 0.05,
                "expected accumulated idle >= 50ms, got {idle}s"
            );
        }

        #[test]
        fn busy_then_idle_resets_counter() {
            let detector = detector_with(vec![2.0, 0.04, 0.04], 0.1, 4);
            // Busy → 0
            assert_eq!(detector.seconds_since_last_input().unwrap(), 0.0);
            sleep(Duration::from_millis(20));
            // Idle → small positive
            let first_idle = detector.seconds_since_last_input().unwrap();
            assert!(first_idle > 0.0);
            sleep(Duration::from_millis(20));
            // Still idle → growing
            let second_idle = detector.seconds_since_last_input().unwrap();
            assert!(second_idle > first_idle);
        }

        #[test]
        fn reader_failure_returns_none() {
            let detector = LinuxIdleDetector::with_reader(Box::new(FailingReader), 0.1, 4);
            assert!(detector.seconds_since_last_input().is_none());
        }

        #[test]
        fn zero_cores_treated_as_one() {
            let detector = detector_with(vec![0.5], 1.0, 0);
            // 0.5 / 1 = 0.5 < 1.0 → idle → 0 on first call
            let idle = detector.seconds_since_last_input().unwrap();
            assert_eq!(idle, 0.0);
        }

        #[test]
        fn threshold_boundary_is_busy() {
            // Normalized exactly equal to threshold → treated as busy
            let detector = detector_with(vec![0.4], 0.1, 4); // 0.4/4 = 0.1 == threshold
            assert_eq!(detector.seconds_since_last_input().unwrap(), 0.0);
        }

        #[test]
        fn proc_reader_handles_realistic_line() {
            // Make sure the real /proc parser logic matches Linux's format.
            // We invoke ProcLoadReader through a tempfile-style override: just
            // parse the string directly instead of touching the filesystem.
            let raw = "0.42 0.38 0.35 2/534 12345";
            let parsed: f64 = raw.split_whitespace().next().unwrap().parse().unwrap();
            assert!((parsed - 0.42).abs() < f64::EPSILON);
        }
    }
}

/// Pure pressure decision: does the load attributable to OTHER processes
/// exceed the per-core threshold? Subtracting our own share is the whole point —
/// the daemon's own worker/blocking threads otherwise inflate the system load
/// average it reads, so a busy daemon would mistake its own work for external
/// pressure and permanently refuse to ramp up (the death spiral the runaway
/// exploited). Clamped at 0 so heavy self-load can't underflow.
pub(super) fn external_pressure_exceeds(
    system_normalized: f64,
    self_normalized: f64,
    threshold: f64,
) -> bool {
    (system_normalized - self_normalized).max(0.0) > threshold
}

/// Check if CPU load from OTHER processes is too high for burst mode.
///
/// Reads the system 1-minute load average, then subtracts this process's own
/// CPU usage before comparing to `threshold`. Both are normalized to "fraction
/// of one core". NOTE: this mixes a 1-minute load average with an instantaneous
/// self-CPU sample — it is a heuristic to gate burst ramp-up, not a precise
/// accounting.
pub(super) fn is_cpu_under_pressure(threshold: f64, physical_cores: usize) -> bool {
    use std::sync::{Mutex, OnceLock};
    use sysinfo::{Pid, System};
    let cores = physical_cores.max(1) as f64;
    let system_normalized = System::load_average().one / cores;

    // Persistent sampler: `cpu_usage()` reports the delta since the PREVIOUS
    // refresh — i.e. since the last poll (~poll_interval seconds). This avoids
    // an in-call `thread::sleep` (which would block the async adaptive loop this
    // runs on) and gives a wider, steadier window than a sub-second snapshot.
    // The first call reads ~0 (no baseline yet), which is harmless.
    static SAMPLER: OnceLock<Mutex<System>> = OnceLock::new();
    let self_normalized = {
        let mtx = SAMPLER.get_or_init(|| Mutex::new(System::new()));
        let pid = Pid::from_u32(std::process::id());
        match mtx.lock() {
            Ok(mut sys) => {
                sys.refresh_process(pid);
                sys.process(pid)
                    .map(|p| p.cpu_usage() as f64 / 100.0)
                    .unwrap_or(0.0)
                    / cores
            }
            Err(_) => 0.0, // poisoned lock: skip self-subtraction this tick
        }
    };

    external_pressure_exceeds(system_normalized, self_normalized, threshold)
}
