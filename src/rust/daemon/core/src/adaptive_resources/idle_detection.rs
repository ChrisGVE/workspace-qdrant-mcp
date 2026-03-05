//! Platform-specific idle detection and CPU pressure monitoring.
//!
//! ## Platform support
//! - macOS: CGEventSourceSecondsSinceLastEventType (CoreGraphics),
//!          with IOKit HIDIdleTime fallback (works during screen lock)
//! - Linux: falls back to always-interactive (no burst mode)
//! - Other: always-interactive (no burst mode)

/// Returns seconds since last user input event, or None if unavailable.
pub(super) fn seconds_since_last_input() -> Option<f64> {
    #[cfg(target_os = "macos")]
    {
        macos_idle::seconds_since_last_input()
    }

    #[cfg(not(target_os = "macos"))]
    {
        None // No idle detection on this platform
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

/// Check if CPU load is too high for burst mode.
pub(super) fn is_cpu_under_pressure(threshold: f64, physical_cores: usize) -> bool {
    use sysinfo::System;
    let load = System::load_average();
    let normalized = load.one / physical_cores as f64;
    normalized > threshold
}
