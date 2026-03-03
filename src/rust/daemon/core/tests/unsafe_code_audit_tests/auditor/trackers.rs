//! Helper tracker types for unsafe operation monitoring
//!
//! Provides `MemoryTracker` for allocation/deallocation accounting and
//! `ConcurrencyTracker` for recording shared-memory access patterns.

use std::collections::HashMap;
use std::thread;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Memory tracker
// ---------------------------------------------------------------------------

/// Memory tracking for unsafe operations
#[derive(Debug)]
pub(crate) struct MemoryTracker {
    allocations: HashMap<usize, AllocationInfo>,
    pub(crate) total_allocated: usize,
}

#[derive(Debug, Clone)]
struct AllocationInfo {
    size: usize,
    _timestamp: Instant,
    _location: String,
}

impl MemoryTracker {
    pub(crate) fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            total_allocated: 0,
        }
    }

    pub(crate) fn track_allocation(&mut self, ptr: usize, size: usize, location: String) {
        let info = AllocationInfo {
            size,
            _timestamp: Instant::now(),
            _location: location,
        };

        self.allocations.insert(ptr, info);
        self.total_allocated += size;
    }

    pub(crate) fn track_deallocation(&mut self, ptr: usize) -> Option<()> {
        if let Some(info) = self.allocations.remove(&ptr) {
            self.total_allocated -= info.size;
            Some(())
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Concurrency tracker
// ---------------------------------------------------------------------------

/// Concurrency tracking for unsafe operations
#[derive(Debug)]
pub(crate) struct ConcurrencyTracker {
    pub(crate) active_threads: HashMap<thread::ThreadId, ThreadInfo>,
    pub(crate) shared_accesses: HashMap<usize, Vec<AccessInfo>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ThreadInfo {
    _start_time: Instant,
    _unsafe_operations: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct AccessInfo {
    _thread_id: thread::ThreadId,
    _access_time: Instant,
    _access_type: AccessType,
}

#[derive(Debug, Clone)]
pub(crate) enum AccessType {
    Read,
    _Write,
    _ReadWrite,
}

impl ConcurrencyTracker {
    pub(crate) fn new() -> Self {
        Self {
            active_threads: HashMap::new(),
            shared_accesses: HashMap::new(),
        }
    }

    pub(crate) fn register_thread(&mut self, thread_id: thread::ThreadId) {
        self.active_threads.insert(
            thread_id,
            ThreadInfo {
                _start_time: Instant::now(),
                _unsafe_operations: 0,
            },
        );
    }

    pub(crate) fn record_access(&mut self, address: usize, access_type: AccessType) {
        let thread_id = thread::current().id();
        let access = AccessInfo {
            _thread_id: thread_id,
            _access_time: Instant::now(),
            _access_type: access_type,
        };

        self.shared_accesses
            .entry(address)
            .or_insert_with(Vec::new)
            .push(access);
    }
}
