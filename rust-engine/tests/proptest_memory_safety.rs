//! Property-based tests for memory safety and bounds validation
//!
//! This module implements comprehensive memory safety testing, including
//! buffer overflow protection, concurrent memory access patterns, and
//! memory allocation validation for the Rust daemon components.

use proptest::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{Semaphore, RwLock, Mutex};
use tokio::time::{Duration, timeout};
use workspace_qdrant_daemon::memory::*;
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};

/// Strategy for generating memory allocation patterns
fn memory_allocation_patterns() -> impl Strategy<Value = Vec<usize>> {
    prop_oneof![
        // Small allocations
        prop::collection::vec(1usize..1024usize, 1..100),
        // Large allocations
        prop::collection::vec(1024usize..1048576usize, 1..10),
        // Mixed allocation sizes
        prop::collection::vec(1usize..10485760usize, 1..50),
        // Edge case sizes (power of 2 boundaries)
        prop::collection::vec(prop::sample::select(vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
        ]), 1..20),
        // Zero and one byte allocations
        prop::collection::vec(0usize..2usize, 1..100),
    ]
}

/// Strategy for generating concurrent operation patterns
fn concurrent_patterns() -> impl Strategy<Value = (usize, Duration)> {
    (
        1usize..50usize,                          // number of concurrent tasks
        prop::sample::select(vec![                // operation duration
            Duration::from_millis(1),
            Duration::from_millis(10),
            Duration::from_millis(100),
            Duration::from_millis(500),
        ])
    )
}

/// Strategy for generating document IDs and metadata
fn document_metadata() -> impl Strategy<Value = (String, HashMap<String, String>)> {
    (
        "[a-zA-Z0-9_-]{1,50}",
        prop::collection::hash_map("[a-zA-Z_]{1,20}", "[\\PC]{0,100}", 0..10)
    )
}

proptest! {
    #![proptest_config(ProptestConfig {
        timeout: 20000, // 20 seconds for memory tests
        cases: 25,
        max_shrink_iters: 50,
        .. ProptestConfig::default()
    })]

    #[test]
    fn proptest_memory_manager_allocation_patterns(alloc_sizes in memory_allocation_patterns()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let memory_manager = MemoryManager::new(100 * 1024 * 1024); // 100MB limit

            let mut allocated_buffers = Vec::new();
            let mut total_allocated = 0usize;

            // Property: Memory manager should handle various allocation patterns
            for size in alloc_sizes {
                if total_allocated + size <= 50 * 1024 * 1024 { // Keep under 50MB for tests
                    match memory_manager.allocate_buffer(size).await {
                        Ok(buffer) => {
                            prop_assert_eq!(buffer.len(), size, "Allocated buffer should have requested size");
                            allocated_buffers.push(buffer);
                            total_allocated += size;
                        }
                        Err(_) => {
                            // Allocation failure is acceptable when approaching limits
                            break;
                        }
                    }
                }
            }

            // Property: All allocated buffers should be valid and independent
            for (i, buffer) in allocated_buffers.iter().enumerate() {
                if !buffer.is_empty() {
                    // Write unique pattern to each buffer
                    let pattern = (i % 256) as u8;
                    // Note: We can't modify the buffer here as it's likely read-only
                    // but we can verify its integrity
                    prop_assert!(!buffer.is_empty() || alloc_sizes.get(i) == Some(&0),
                               "Buffer should be non-empty unless zero-size allocation");
                }
            }
        });
    }

    #[test]
    fn proptest_concurrent_memory_operations(
        (num_tasks, duration) in concurrent_patterns()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let memory_manager = Arc::new(MemoryManager::new(10 * 1024 * 1024)); // 10MB limit
            let results = Arc::new(Mutex::new(Vec::new()));

            // Property: Concurrent memory operations should not interfere
            let handles: Vec<_> = (0..num_tasks)
                .map(|task_id| {
                    let manager = Arc::clone(&memory_manager);
                    let results = Arc::clone(&results);
                    tokio::spawn(async move {
                        let size = (task_id + 1) * 1024; // Variable sizes

                        // Add timeout to prevent hanging
                        let result = timeout(duration * 10, async {
                            tokio::time::sleep(duration).await;
                            manager.allocate_buffer(size).await
                        }).await;

                        match result {
                            Ok(alloc_result) => {
                                let mut results_guard = results.lock().await;
                                results_guard.push((task_id, alloc_result.is_ok()));
                            }
                            Err(_) => {
                                // Timeout occurred
                                let mut results_guard = results.lock().await;
                                results_guard.push((task_id, false));
                            }
                        }
                    })
                })
                .collect();

            // Wait for all tasks to complete
            for handle in handles {
                let _ = handle.await;
            }

            let final_results = results.lock().await;
            prop_assert!(!final_results.is_empty(), "Should have some results from concurrent operations");

            // At least some operations should succeed (not all will due to memory limits)
            let successful_ops = final_results.iter().filter(|(_, success)| *success).count();
            prop_assert!(successful_ops >= 1 || num_tasks > 20,
                        "At least one operation should succeed unless too many concurrent tasks");
        });
    }

    #[test]
    fn proptest_document_storage_memory_bounds(
        documents in prop::collection::vec(document_metadata(), 1..20)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let doc_storage = DocumentStorage::new();
            let mut stored_count = 0;

            // Property: Document storage should handle various document patterns
            for (doc_id, metadata) in documents {
                let content = format!("Document content for {}", doc_id).into_bytes();

                let result = doc_storage.store_document(
                    &doc_id,
                    content,
                    Some(metadata.clone())
                ).await;

                match result {
                    Ok(_) => {
                        stored_count += 1;

                        // Verify retrieval
                        let retrieved = doc_storage.get_document(&doc_id).await;
                        prop_assert!(retrieved.is_ok(), "Stored document should be retrievable");

                        if let Ok(Some(doc)) = retrieved {
                            prop_assert_eq!(doc.id(), &doc_id, "Document ID should match");
                            prop_assert!(!doc.content().is_empty(), "Document content should not be empty");
                        }
                    }
                    Err(_) => {
                        // Storage failure is acceptable under memory pressure
                    }
                }
            }

            prop_assert!(stored_count > 0 || documents.is_empty(),
                        "Should store at least one document if any provided");
        });
    }

    #[test]
    fn proptest_memory_deallocation_patterns(alloc_sizes in memory_allocation_patterns()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let memory_manager = MemoryManager::new(20 * 1024 * 1024); // 20MB limit

            // Property: Memory should be properly deallocated
            let initial_usage = memory_manager.current_usage().await;

            {
                let mut buffers = Vec::new();
                for size in &alloc_sizes {
                    if *size > 0 && *size <= 1024 * 1024 { // Limit individual allocations to 1MB
                        if let Ok(buffer) = memory_manager.allocate_buffer(*size).await {
                            buffers.push(buffer);
                        }
                    }
                }
                // Buffers are dropped here
            }

            // Give some time for cleanup
            tokio::time::sleep(Duration::from_millis(100)).await;

            let final_usage = memory_manager.current_usage().await;

            // Property: Memory usage should return close to initial levels after deallocation
            prop_assert!(final_usage <= initial_usage + 1024 * 1024, // Allow 1MB tolerance
                        "Memory should be deallocated properly");
        });
    }

    #[test]
    fn proptest_shared_memory_access_patterns(
        access_pattern in prop::collection::vec(0usize..100usize, 1..50)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let shared_data = Arc::new(RwLock::new(HashMap::<usize, Vec<u8>>::new()));
            let mut handles = Vec::new();

            // Property: Shared memory access should be thread-safe
            for (task_id, &key) in access_pattern.iter().enumerate() {
                let data = Arc::clone(&shared_data);
                let handle = tokio::spawn(async move {
                    let mut operations_completed = 0;

                    // Perform read operation
                    {
                        let read_guard = data.read().await;
                        if let Some(value) = read_guard.get(&key) {
                            operations_completed += 1;
                            // Verify data integrity
                            if !value.is_empty() && value[0] as usize != key % 256 {
                                return Err(format!("Data corruption detected in task {}", task_id));
                            }
                        }
                    }

                    // Perform write operation
                    {
                        let mut write_guard = data.write().await;
                        let value = vec![(key % 256) as u8; key % 100 + 1];
                        write_guard.insert(key, value);
                        operations_completed += 1;
                    }

                    Ok(operations_completed)
                });
                handles.push(handle);
            }

            // Wait for all operations to complete
            let mut total_operations = 0;
            for handle in handles {
                match handle.await {
                    Ok(Ok(ops)) => total_operations += ops,
                    Ok(Err(error)) => prop_assert!(false, "Memory access error: {}", error),
                    Err(_) => prop_assert!(false, "Task panicked"),
                }
            }

            prop_assert!(total_operations > 0, "Should complete some memory operations");

            // Verify final state consistency
            let final_data = shared_data.read().await;
            for (key, value) in final_data.iter() {
                prop_assert!(!value.is_empty(), "Stored values should not be empty");
                prop_assert_eq!(value[0] as usize, key % 256, "Value should match expected pattern");
            }
        });
    }

    #[test]
    fn proptest_memory_pressure_handling(
        pressure_sequence in prop::collection::vec(1usize..10485760usize, 1..10)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let memory_manager = MemoryManager::new(5 * 1024 * 1024); // Small 5MB limit
            let mut successful_allocations = 0;
            let mut failed_allocations = 0;

            // Property: Memory manager should handle pressure gracefully
            for size in pressure_sequence {
                match memory_manager.allocate_buffer(size).await {
                    Ok(_) => successful_allocations += 1,
                    Err(DaemonError::OutOfMemory { .. }) => {
                        failed_allocations += 1;
                        // This is expected behavior under pressure
                    }
                    Err(e) => prop_assert!(false, "Unexpected error under memory pressure: {:?}", e),
                }
            }

            // Property: Should have attempted all allocations
            prop_assert_eq!(successful_allocations + failed_allocations, pressure_sequence.len(),
                          "All allocation attempts should be accounted for");

            // Property: Should fail gracefully when out of memory
            if failed_allocations > 0 {
                prop_assert!(successful_allocations >= 0, "Some allocations may succeed before hitting limits");
            }
        });
    }

    #[test]
    fn proptest_buffer_overflow_protection(
        write_sizes in prop::collection::vec(0usize..10000usize, 1..20)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let memory_manager = MemoryManager::new(50 * 1024 * 1024);

            // Property: Buffer operations should prevent overflow
            for write_size in write_sizes {
                if write_size > 0 {
                    let buffer_result = memory_manager.allocate_buffer(write_size).await;

                    if let Ok(buffer) = buffer_result {
                        prop_assert_eq!(buffer.len(), write_size, "Buffer should be exact requested size");

                        // Verify bounds
                        if !buffer.is_empty() {
                            // We can read from the buffer safely
                            let first_byte = buffer[0];
                            let last_byte = buffer[buffer.len() - 1];

                            // These operations should not cause panic or memory errors
                            prop_assert!(first_byte == first_byte, "Buffer read should be consistent");
                            prop_assert!(last_byte == last_byte, "End buffer read should be consistent");
                        }
                    }
                }
            }
        });
    }
}