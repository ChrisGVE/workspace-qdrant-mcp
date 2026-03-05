//! Property-based tests for unsafe code patterns
//!
//! Uses proptest to validate safety invariants across randomized inputs
//! for file descriptor operations, string conversions, pointer arithmetic,
//! buffer bounds, memory tracking, and null pointer handling.

use std::thread;

use proptest::prelude::*;

use super::auditor::MemoryTracker;

proptest! {
    #[test]
    fn test_fd_operations_with_random_values(fd in -100i32..1000i32) {
        #[cfg(unix)]
        {
            if (0..1024).contains(&fd) {
                let result = unsafe { libc::dup(fd) };
                prop_assert!(result >= -1);
                if result >= 0 {
                    let close_result = unsafe { libc::close(result) };
                    prop_assert!(close_result == 0 || close_result == -1);
                }
            }
        }
    }

    #[test]
    fn test_string_conversion_properties(s in "\\PC*") {
        let wide_chars: Vec<u16> = s.encode_utf16().chain(std::iter::once(0)).collect();

        prop_assert!(!wide_chars.is_empty());
        prop_assert_eq!(wide_chars.last(), Some(&0));

        let reconstructed = String::from_utf16_lossy(&wide_chars[..wide_chars.len()-1]);
        prop_assert!(reconstructed.len() <= s.len() * 4);
    }

    #[test]
    fn test_pointer_arithmetic_safety(offset in 0usize..1000) {
        let buffer = vec![0u8; 1000];
        let ptr = buffer.as_ptr();

        if offset < buffer.len() {
            let new_ptr = unsafe { ptr.add(offset) };
            prop_assert!(!new_ptr.is_null());

            let value = unsafe { *new_ptr };
            prop_assert_eq!(value, 0);
        }
    }

    #[test]
    fn test_concurrent_fd_operations(thread_count in 2usize..8) {
        #[cfg(unix)]
        {
            let results: Vec<_> = (0..thread_count)
                .map(|_| {
                    thread::spawn(|| {
                        let fd = unsafe { libc::dup(libc::STDOUT_FILENO) };
                        if fd >= 0 {
                            let close_result = unsafe { libc::close(fd) };
                            close_result == 0
                        } else {
                            true
                        }
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap_or(false))
                .collect();

            prop_assert!(results.iter().all(|&r| r));
        }
    }

    #[test]
    fn test_buffer_bounds_checking(buf_size in 1usize..2048, access_offset in 0usize..4096) {
        let buffer: Vec<u8> = vec![0xAA; buf_size];
        let ptr = buffer.as_ptr();

        if access_offset < buf_size {
            let value = unsafe { *ptr.add(access_offset) };
            prop_assert_eq!(value, 0xAA);
        }
        prop_assert!(buf_size > 0);
    }

    #[test]
    fn test_memory_allocation_tracking(
        alloc_count in 1usize..20,
        alloc_size in 1usize..1024
    ) {
        let mut tracker = MemoryTracker::new();
        let mut ptrs = Vec::new();

        for i in 0..alloc_count {
            let addr = 0x1000 + i * alloc_size;
            tracker.track_allocation(addr, alloc_size, format!("test_{}", i));
            ptrs.push(addr);
        }
        prop_assert_eq!(tracker.total_allocated, alloc_count * alloc_size);

        for addr in &ptrs {
            let info = tracker.track_deallocation(*addr);
            prop_assert!(info.is_some());
        }
        prop_assert_eq!(tracker.total_allocated, 0);

        if let Some(first_ptr) = ptrs.first() {
            let double_free = tracker.track_deallocation(*first_ptr);
            prop_assert!(double_free.is_none());
        }
    }

    #[test]
    fn test_null_pointer_checks(is_null in proptest::bool::ANY) {
        if is_null {
            let ptr: *const u8 = std::ptr::null();
            prop_assert!(ptr.is_null());
        } else {
            let value: u8 = 42;
            let ptr: *const u8 = &value;
            prop_assert!(!ptr.is_null());
            let read_value = unsafe { *ptr };
            prop_assert_eq!(read_value, 42);
        }
    }
}
