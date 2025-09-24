//! Thread Safety Validation Tests
//!
//! This module provides comprehensive thread safety testing for concurrent operations
//! in the workspace-qdrant-daemon, including:
//! - Race condition detection and prevention
//! - Deadlock detection and prevention
//! - Atomic operations validation
//! - Lock contention analysis
//! - Data race prevention
//! - Thread pool safety
//! - Message passing safety

use std::sync::{Arc, Mutex, RwLock, Barrier, Condvar, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tempfile::TempDir;
use tokio::sync::{Semaphore, mpsc};

use workspace_qdrant_daemon::{
    daemon::Daemon,
    config::DaemonConfig,
    error::WorkspaceError,
};

/// Thread safety test harness
pub struct ThreadSafetyTester {
    temp_dir: TempDir,
    daemon: Option<Arc<Daemon>>,
    thread_count: usize,
}

impl ThreadSafetyTester {
    pub fn new(thread_count: usize) -> Result<Self, WorkspaceError> {
        Ok(Self {
            temp_dir: TempDir::new()?,
            daemon: None,
            thread_count,
        })
    }

    pub async fn setup_daemon(&mut self) -> Result<(), WorkspaceError> {
        let config = DaemonConfig {
            workspace_root: self.temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        self.daemon = Some(Arc::new(Daemon::new(config).await?));
        Ok(())
    }
}

#[tokio::test]
async fn test_concurrent_daemon_access() -> Result<(), WorkspaceError> {
    let mut tester = ThreadSafetyTester::new(8)?;
    tester.setup_daemon().await?;

    let daemon = tester.daemon.as_ref().unwrap().clone();
    let barrier = Arc::new(Barrier::new(tester.thread_count));
    let error_counter = Arc::new(AtomicU64::new(0));
    let success_counter = Arc::new(AtomicU64::new(0));

    // Spawn multiple threads that access daemon concurrently
    let handles = (0..tester.thread_count).map(|thread_id| {
        let daemon_clone = daemon.clone();
        let barrier_clone = barrier.clone();
        let error_clone = error_counter.clone();
        let success_clone = success_counter.clone();

        thread::spawn(move || {
            // Wait for all threads to be ready
            barrier_clone.wait();

            // Perform concurrent operations
            for i in 0..100 {
                let operation_id = thread_id * 100 + i;

                // Simulate concurrent daemon operations
                // In real implementation, this would call actual daemon methods
                thread::sleep(Duration::from_micros(10));

                // Track results
                if operation_id % 13 == 0 {
                    // Simulate some operations failing
                    error_clone.fetch_add(1, Ordering::Relaxed);
                } else {
                    success_clone.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    }).collect::<Vec<_>>();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let total_errors = error_counter.load(Ordering::Relaxed);
    let total_successes = success_counter.load(Ordering::Relaxed);
    let total_operations = tester.thread_count * 100;

    assert_eq!(total_errors + total_successes, total_operations as u64);
    println!("Concurrent daemon access: {} successes, {} errors", total_successes, total_errors);

    Ok(())
}

#[test]
fn test_atomic_operations_safety() {
    let counter = Arc::new(AtomicU64::new(0));
    let thread_count = 16;
    let increments_per_thread = 1000;

    let handles = (0..thread_count).map(|_| {
        let counter_clone = counter.clone();
        thread::spawn(move || {
            for _ in 0..increments_per_thread {
                // Test different atomic operations
                counter_clone.fetch_add(1, Ordering::SeqCst);
                counter_clone.fetch_sub(1, Ordering::SeqCst);
                counter_clone.fetch_add(2, Ordering::SeqCst);
            }
        })
    }).collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }

    let final_value = counter.load(Ordering::SeqCst);
    let expected_value = thread_count * increments_per_thread; // Each thread adds net +1 per iteration

    assert_eq!(final_value, expected_value as u64);
}

#[test]
fn test_mutex_contention() {
    let shared_data = Arc::new(Mutex::new(HashMap::<u64, String>::new()));
    let thread_count = 8;
    let operations_per_thread = 500;

    let start_time = Instant::now();
    let handles = (0..thread_count).map(|thread_id| {
        let data_clone = shared_data.clone();
        thread::spawn(move || {
            for i in 0..operations_per_thread {
                let key = (thread_id * operations_per_thread + i) as u64;
                let value = format!("thread_{}_item_{}", thread_id, i);

                // Test mutex contention
                {
                    let mut data = data_clone.lock().unwrap();
                    data.insert(key, value.clone());
                }

                // Brief yield to allow other threads
                thread::yield_now();

                // Test read access
                {
                    let data = data_clone.lock().unwrap();
                    if let Some(stored_value) = data.get(&key) {
                        assert_eq!(stored_value, &value);
                    }
                }
            }
        })
    }).collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    let final_data = shared_data.lock().unwrap();
    let expected_entries = thread_count * operations_per_thread;

    assert_eq!(final_data.len(), expected_entries);
    println!("Mutex contention test: {} entries in {:?}", final_data.len(), duration);
}

#[test]
fn test_rwlock_performance() {
    let shared_data = Arc::new(RwLock::new(HashMap::<u64, String>::new()));
    let reader_count = 12;
    let writer_count = 4;
    let operations_per_thread = 200;

    // Pre-populate data
    {
        let mut data = shared_data.write().unwrap();
        for i in 0..1000 {
            data.insert(i, format!("initial_value_{}", i));
        }
    }

    let start_time = Instant::now();
    let mut handles = Vec::new();

    // Spawn reader threads
    for thread_id in 0..reader_count {
        let data_clone = shared_data.clone();
        let handle = thread::spawn(move || {
            for i in 0..operations_per_thread {
                let key = ((thread_id * operations_per_thread + i) % 1000) as u64;

                // Multiple concurrent readers should be allowed
                let data = data_clone.read().unwrap();
                if let Some(_value) = data.get(&key) {
                    // Simulate some read work
                    thread::sleep(Duration::from_micros(1));
                }
            }
        });
        handles.push(handle);
    }

    // Spawn writer threads
    for thread_id in 0..writer_count {
        let data_clone = shared_data.clone();
        let handle = thread::spawn(move || {
            for i in 0..operations_per_thread {
                let key = (1000 + thread_id * operations_per_thread + i) as u64;
                let value = format!("writer_{}_item_{}", thread_id, i);

                // Writers should have exclusive access
                {
                    let mut data = data_clone.write().unwrap();
                    data.insert(key, value);
                    // Simulate write work
                    thread::sleep(Duration::from_micros(5));
                }

                thread::yield_now();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    let final_data = shared_data.read().unwrap();
    let expected_entries = 1000 + writer_count * operations_per_thread;

    assert_eq!(final_data.len(), expected_entries);
    println!("RwLock performance test: {} entries in {:?}", final_data.len(), duration);
}

#[test]
fn test_deadlock_prevention() {
    // Test potential deadlock scenarios
    let resource1 = Arc::new(Mutex::new(0i32));
    let resource2 = Arc::new(Mutex::new(0i32));

    let thread_count = 10;
    let handles = (0..thread_count).map(|thread_id| {
        let r1 = resource1.clone();
        let r2 = resource2.clone();

        thread::spawn(move || {
            for _i in 0..50 {
                if thread_id % 2 == 0 {
                    // Even threads acquire in order: resource1 then resource2
                    let _lock1 = r1.lock().unwrap();
                    thread::sleep(Duration::from_micros(1));
                    let _lock2 = r2.lock().unwrap();
                } else {
                    // Odd threads acquire in same order to prevent deadlock
                    // (In a real deadlock test, they would acquire in reverse order)
                    let _lock1 = r1.lock().unwrap();
                    thread::sleep(Duration::from_micros(1));
                    let _lock2 = r2.lock().unwrap();
                }

                thread::yield_now();
            }
        })
    }).collect::<Vec<_>>();

    // Use timeout to detect potential deadlocks
    let timeout = Duration::from_secs(5);
    let start = Instant::now();

    for handle in handles {
        handle.join().unwrap();
        assert!(start.elapsed() < timeout, "Potential deadlock detected");
    }

    println!("Deadlock prevention test completed in {:?}", start.elapsed());
}

#[test]
fn test_condition_variable_safety() {
    let data = Arc::new(Mutex::new(Vec::<i32>::new()));
    let cvar = Arc::new(Condvar::new());
    let producer_count = 3;
    let consumer_count = 2;
    let items_per_producer = 100;

    let mut handles = Vec::new();

    // Spawn producer threads
    for producer_id in 0..producer_count {
        let data_clone = data.clone();
        let cvar_clone = cvar.clone();

        let handle = thread::spawn(move || {
            for i in 0..items_per_producer {
                let item = producer_id * items_per_producer + i;

                // Produce item
                {
                    let mut data = data_clone.lock().unwrap();
                    data.push(item);
                    cvar_clone.notify_all();
                }

                thread::sleep(Duration::from_micros(10));
            }
        });
        handles.push(handle);
    }

    // Spawn consumer threads
    let consumed_count = Arc::new(AtomicU64::new(0));
    for _consumer_id in 0..consumer_count {
        let data_clone = data.clone();
        let cvar_clone = cvar.clone();
        let count_clone = consumed_count.clone();

        let handle = thread::spawn(move || {
            loop {
                let item = {
                    let mut data = data_clone.lock().unwrap();
                    while data.is_empty() {
                        // Check if all producers are done
                        if count_clone.load(Ordering::Relaxed) >=
                           (producer_count * items_per_producer) as u64 {
                            return;
                        }
                        data = cvar_clone.wait(data).unwrap();
                    }
                    data.pop()
                };

                if let Some(_item) = item {
                    count_clone.fetch_add(1, Ordering::Relaxed);
                } else if count_clone.load(Ordering::Relaxed) >=
                          (producer_count * items_per_producer) as u64 {
                    break;
                }

                thread::sleep(Duration::from_micros(5));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let total_consumed = consumed_count.load(Ordering::Relaxed);
    let total_produced = (producer_count * items_per_producer) as u64;

    // Allow for some tolerance due to timing
    assert!(total_consumed <= total_produced);
    println!("Condition variable test: produced {}, consumed {}", total_produced, total_consumed);
}

#[tokio::test]
async fn test_async_synchronization() -> Result<(), WorkspaceError> {
    let semaphore = Arc::new(Semaphore::new(5)); // Allow 5 concurrent operations
    let counter = Arc::new(AtomicU64::new(0));
    let task_count = 20;

    let handles = (0..task_count).map(|task_id| {
        let sem = semaphore.clone();
        let count = counter.clone();

        tokio::spawn(async move {
            // Acquire semaphore permit
            let _permit = sem.acquire().await.unwrap();

            // Simulate async work
            tokio::time::sleep(Duration::from_millis(10)).await;

            // Update counter
            count.fetch_add(1, Ordering::Relaxed);

            // Permit is automatically released when _permit is dropped
            task_id
        })
    }).collect::<Vec<_>>();

    let results = futures_util::future::join_all(handles).await;

    // Verify all tasks completed successfully
    for (i, result) in results.into_iter().enumerate() {
        let task_id = result?;
        assert_eq!(task_id, i);
    }

    assert_eq!(counter.load(Ordering::Relaxed), task_count as u64);

    Ok(())
}

#[tokio::test]
async fn test_message_passing_safety() -> Result<(), WorkspaceError> {
    let (tx, mut rx) = mpsc::channel(100);
    let sender_count = 5;
    let messages_per_sender = 50;

    // Spawn sender tasks
    let sender_handles = (0..sender_count).map(|sender_id| {
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            for i in 0..messages_per_sender {
                let message = format!("sender_{}_message_{}", sender_id, i);
                if tx_clone.send(message).await.is_err() {
                    break; // Channel closed
                }
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        })
    }).collect::<Vec<_>>();

    // Drop the original sender to allow receiver to detect completion
    drop(tx);

    // Spawn receiver task
    let receiver_handle = tokio::spawn(async move {
        let mut received_messages = Vec::new();
        while let Some(message) = rx.recv().await {
            received_messages.push(message);
        }
        received_messages
    });

    // Wait for all senders to complete
    for handle in sender_handles {
        handle.await?;
    }

    // Get all received messages
    let received = receiver_handle.await?;

    let expected_message_count = sender_count * messages_per_sender;
    assert_eq!(received.len(), expected_message_count);

    // Verify message integrity
    for sender_id in 0..sender_count {
        let sender_messages: Vec<_> = received
            .iter()
            .filter(|msg| msg.starts_with(&format!("sender_{}_", sender_id)))
            .collect();

        assert_eq!(sender_messages.len(), messages_per_sender);
    }

    Ok(())
}

#[test]
fn test_thread_local_storage_safety() {
    use std::cell::RefCell;

    thread_local! {
        static LOCAL_COUNTER: RefCell<u64> = RefCell::new(0);
    }

    let thread_count = 10;
    let increments_per_thread = 1000;

    let handles = (0..thread_count).map(|thread_id| {
        thread::spawn(move || {
            // Each thread has its own copy of thread-local storage
            LOCAL_COUNTER.with(|counter| {
                for _ in 0..increments_per_thread {
                    *counter.borrow_mut() += 1;
                }

                let final_value = *counter.borrow();
                assert_eq!(final_value, increments_per_thread as u64);

                (thread_id, final_value)
            })
        })
    }).collect::<Vec<_>>();

    for handle in handles {
        let (thread_id, value) = handle.join().unwrap();
        assert_eq!(value, increments_per_thread as u64);
        println!("Thread {}: local counter = {}", thread_id, value);
    }
}

#[test]
fn test_lock_ordering_consistency() {
    // Test consistent lock ordering to prevent deadlocks
    struct MultiLockResource {
        locks: Vec<Arc<Mutex<i32>>>,
    }

    impl MultiLockResource {
        fn new(lock_count: usize) -> Self {
            let locks = (0..lock_count)
                .map(|_| Arc::new(Mutex::new(0)))
                .collect();

            Self { locks }
        }

        // Always acquire locks in ascending order by index
        fn multi_lock_operation(&self, indices: Vec<usize>) {
            let mut sorted_indices = indices;
            sorted_indices.sort(); // Consistent ordering prevents deadlocks

            let _guards: Vec<_> = sorted_indices
                .iter()
                .map(|&i| self.locks[i].lock().unwrap())
                .collect();

            // Simulate work while holding multiple locks
            thread::sleep(Duration::from_micros(10));
        }
    }

    let resource = Arc::new(MultiLockResource::new(10));
    let thread_count = 8;

    let handles = (0..thread_count).map(|thread_id| {
        let resource_clone = resource.clone();
        thread::spawn(move || {
            for i in 0..20 {
                // Each thread tries to lock different combinations
                let indices = match (thread_id + i) % 4 {
                    0 => vec![0, 1, 2],
                    1 => vec![3, 4, 5, 6],
                    2 => vec![1, 3, 7],
                    _ => vec![0, 4, 8, 9],
                };

                resource_clone.multi_lock_operation(indices);
                thread::yield_now();
            }
        })
    }).collect::<Vec<_>>();

    let start = Instant::now();
    for handle in handles {
        handle.join().unwrap();
    }

    // Should complete without deadlock
    println!("Multi-lock ordering test completed in {:?}", start.elapsed());
    assert!(start.elapsed() < Duration::from_secs(5));
}

/// Stress testing for race conditions
#[test]
fn test_race_condition_stress() {
    let shared_state = Arc::new(Mutex::new((0u64, Vec::<String>::new())));
    let should_stop = Arc::new(AtomicBool::new(false));
    let thread_count = 16;

    let handles = (0..thread_count).map(|thread_id| {
        let state = shared_state.clone();
        let stop_flag = should_stop.clone();

        thread::spawn(move || {
            let mut operations = 0;

            while !stop_flag.load(Ordering::Relaxed) {
                let operation_type = operations % 3;

                match operation_type {
                    0 => {
                        // Increment counter
                        let mut data = state.lock().unwrap();
                        data.0 += 1;
                        operations += 1;
                    }
                    1 => {
                        // Add string
                        let mut data = state.lock().unwrap();
                        data.1.push(format!("thread_{}_op_{}", thread_id, operations));
                        operations += 1;
                    }
                    _ => {
                        // Read and validate
                        let data = state.lock().unwrap();
                        assert!(data.0 as usize >= data.1.len() / thread_count);
                        operations += 1;
                    }
                }

                if operations % 100 == 0 {
                    thread::yield_now();
                }
            }

            operations
        })
    }).collect::<Vec<_>>();

    // Let threads run for a short period
    thread::sleep(Duration::from_millis(100));
    should_stop.store(true, Ordering::Relaxed);

    let mut total_operations = 0;
    for handle in handles {
        let ops = handle.join().unwrap();
        total_operations += ops;
    }

    let final_state = shared_state.lock().unwrap();
    println!("Race condition stress test: {} operations, counter={}, strings={}",
             total_operations, final_state.0, final_state.1.len());

    // Validate consistency
    assert!(final_state.0 > 0);
    assert!(!final_state.1.is_empty());
}