//! Integration, cleanup, and configuration tests for file system monitoring
//!
//! Tests for document processing pipeline integration, watcher resource
//! management (cleanup, watch/unwatch cycles), and notify crate configuration
//! options including comprehensive end-to-end scenarios.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use shared_test_utils::{async_test, serial_async_test, TestResult};
use tempfile::TempDir;
use tokio::sync::mpsc;

/// Test helper for creating a notify watcher with event collection
struct TestWatcher {
    _watcher: RecommendedWatcher,
    _events: Arc<Mutex<Vec<Event>>>,
    event_rx: mpsc::UnboundedReceiver<Event>,
}

impl TestWatcher {
    fn new() -> Result<Self, notify::Error> {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let watcher = notify::recommended_watcher(move |result: Result<Event, notify::Error>| {
            if let Ok(event) = result {
                if let Ok(mut events_lock) = events_clone.lock() {
                    events_lock.push(event.clone());
                }
                let _ = event_tx.send(event);
            }
        })?;

        Ok(Self {
            _watcher: watcher,
            _events: events,
            event_rx,
        })
    }

    fn watch<P: AsRef<Path>>(
        &mut self,
        path: P,
        recursive: RecursiveMode,
    ) -> Result<(), notify::Error> {
        self._watcher.watch(path.as_ref(), recursive)
    }

    fn unwatch<P: AsRef<Path>>(&mut self, path: P) -> Result<(), notify::Error> {
        self._watcher.unwatch(path.as_ref())
    }

    async fn wait_for_events(&mut self, expected_count: usize, timeout: Duration) -> Vec<Event> {
        let mut collected_events = Vec::new();
        let start_time = Instant::now();

        while collected_events.len() < expected_count && start_time.elapsed() < timeout {
            tokio::select! {
                Some(event) = self.event_rx.recv() => {
                    collected_events.push(event);
                },
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
        }

        collected_events
    }
}

/// Test helper for creating test files
async fn create_test_file(dir: &Path, name: &str, content: &str) -> TestResult<PathBuf> {
    let file_path = dir.join(name);
    tokio::fs::write(&file_path, content).await?;
    Ok(file_path)
}

// ============================================================================
// INTEGRATION WITH DOCUMENT PROCESSING PIPELINE TESTS
// ============================================================================

serial_async_test!(test_integration_with_document_processing, {
    use workspace_qdrant_core::DocumentProcessor;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new().map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let processor = DocumentProcessor::new();

    let documents = vec![
        ("document.txt", "This is a text document for processing."),
        (
            "README.md",
            "# Markdown Document\n\nThis is markdown content.",
        ),
        (
            "script.py",
            "#!/usr/bin/env python3\nprint('Hello, world!')",
        ),
        ("config.json", r#"{"setting": "value", "enabled": true}"#),
    ];

    for (filename, content) in documents {
        let file_path = create_test_file(temp_path, filename, content).await?;

        let process_result = processor.process_file(&file_path, "test_collection").await;
        assert!(
            process_result.is_ok(),
            "Document processing should succeed for {}",
            filename
        );
    }

    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    assert!(
        !events.is_empty(),
        "Should detect file creation events for document processing"
    );

    for event in &events {
        for path in &event.paths {
            if path.is_file() {
                let process_result = processor.process_file(path, "event_test_collection").await;
                if let Err(e) = &process_result {
                    println!("Processing warning for {}: {}", path.display(), e);
                }
            }
        }
    }

    println!(
        "Integration test: {} events processed with document pipeline",
        events.len()
    );

    Ok(())
});

// ============================================================================
// WATCHER CLEANUP AND RESOURCE MANAGEMENT TESTS
// ============================================================================

async_test!(test_watcher_cleanup_and_resource_management, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    for iteration in 0..3 {
        let mut watcher = TestWatcher::new()
            .map_err(|e| format!("Failed to create watcher {}: {}", iteration, e))?;

        watcher
            .watch(temp_path, RecursiveMode::NonRecursive)
            .map_err(|e| format!("Failed to start watching {}: {}", iteration, e))?;

        let test_file = create_test_file(
            temp_path,
            &format!("cleanup_test_{}.txt", iteration),
            "cleanup content",
        )
        .await?;

        let events = watcher.wait_for_events(1, Duration::from_secs(2)).await;
        println!("Iteration {} received {} events", iteration, events.len());

        let unwatch_result = watcher.unwatch(temp_path);
        assert!(
            unwatch_result.is_ok(),
            "Should be able to unwatch directory"
        );

        tokio::fs::remove_file(&test_file).await?;
    }

    println!("Successfully created and cleaned up 3 watchers");

    Ok(())
});

async_test!(test_watch_unwatch_cycle, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new().map_err(|e| format!("Failed to create watcher: {}", e))?;

    for cycle in 0..3 {
        watcher
            .watch(temp_path, RecursiveMode::NonRecursive)
            .map_err(|e| format!("Failed to start watching cycle {}: {}", cycle, e))?;

        tokio::time::sleep(Duration::from_millis(50)).await;

        let test_file = create_test_file(
            temp_path,
            &format!("cycle_test_{}.txt", cycle),
            "cycle content",
        )
        .await?;

        let events_while_watching = watcher.wait_for_events(1, Duration::from_secs(1)).await;

        watcher
            .unwatch(temp_path)
            .map_err(|e| format!("Failed to unwatch cycle {}: {}", cycle, e))?;

        tokio::time::sleep(Duration::from_millis(50)).await;

        tokio::fs::write(&test_file, "modified while not watching").await?;

        let events_while_not_watching =
            watcher.wait_for_events(1, Duration::from_millis(500)).await;

        println!(
            "Cycle {}: {} events while watching, {} events while not watching",
            cycle,
            events_while_watching.len(),
            events_while_not_watching.len()
        );

        tokio::fs::remove_file(&test_file).await?;
    }

    Ok(())
});

// ============================================================================
// NOTIFY CRATE CONFIGURATION TESTS
// ============================================================================

async_test!(test_notify_config_options, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let configs = vec![
        ("default", Config::default()),
        (
            "with_compare_contents",
            Config::default().with_compare_contents(true),
        ),
        (
            "poll_interval_1s",
            Config::default().with_poll_interval(Duration::from_secs(1)),
        ),
    ];

    for (config_name, config) in configs {
        println!("Testing notify config: {}", config_name);

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        let mut watcher = notify::RecommendedWatcher::new(
            move |result: Result<Event, notify::Error>| {
                if let Ok(event) = result {
                    if let Ok(mut events_lock) = events_clone.lock() {
                        events_lock.push(event);
                    }
                }
            },
            config,
        )
        .map_err(|e| format!("Failed to create watcher with {}: {}", config_name, e))?;

        watcher
            .watch(temp_path, RecursiveMode::NonRecursive)
            .map_err(|e| format!("Failed to start watching with {}: {}", config_name, e))?;

        tokio::time::sleep(Duration::from_millis(100)).await;

        let test_file = create_test_file(
            temp_path,
            &format!("config_test_{}.txt", config_name.replace(' ', "_")),
            "initial",
        )
        .await?;
        tokio::time::sleep(Duration::from_millis(100)).await;

        tokio::fs::write(&test_file, "modified").await?;
        tokio::time::sleep(Duration::from_millis(200)).await;

        let collected_events = if let Ok(events_lock) = events.lock() {
            events_lock.len()
        } else {
            0
        };

        println!(
            "  {} events collected with {}",
            collected_events, config_name
        );

        let _ = watcher.unwatch(temp_path);
        tokio::fs::remove_file(&test_file).await?;
    }

    Ok(())
});

async_test!(test_comprehensive_file_system_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher =
        TestWatcher::new().map_err(|e| format!("Failed to create comprehensive watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start comprehensive watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    println!("Starting comprehensive file system monitoring test...");

    // Scenario 1: Multiple file operations
    let test_files = vec![
        ("comprehensive_1.txt", "content 1"),
        ("comprehensive_2.md", "# Markdown\nContent"),
        ("comprehensive_3.json", r#"{"test": true}"#),
    ];

    for (filename, content) in &test_files {
        let file_path = create_test_file(temp_path, filename, content).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;

        tokio::fs::write(&file_path, format!("{} - modified", content)).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Scenario 2: Directory operations
    let test_dir = temp_path.join("comprehensive_subdir");
    tokio::fs::create_dir(&test_dir).await?;
    tokio::time::sleep(Duration::from_millis(50)).await;

    let _subdir_file =
        create_test_file(&test_dir, "subdir_file.txt", "subdirectory content").await?;
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Scenario 3: File deletions
    for (filename, _) in &test_files {
        let file_path = temp_path.join(filename);
        if file_path.exists() {
            tokio::fs::remove_file(&file_path).await?;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    let all_events = watcher.wait_for_events(10, Duration::from_secs(5)).await;

    let total_events = all_events.len();
    let creation_events = all_events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::Create(_)))
        .count();
    let modification_events = all_events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::Modify(_)))
        .count();
    let removal_events = all_events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::Remove(_)))
        .count();

    println!("Comprehensive test results:");
    println!("  Total events: {}", total_events);
    println!("  Creation events: {}", creation_events);
    println!("  Modification events: {}", modification_events);
    println!("  Removal events: {}", removal_events);

    assert!(
        total_events >= 5,
        "Should capture multiple events in comprehensive test"
    );
    assert!(
        !all_events.is_empty(),
        "Comprehensive test should generate events"
    );

    let _final_test_file = create_test_file(temp_path, "final_test.txt", "final test").await?;
    let final_events = watcher.wait_for_events(1, Duration::from_secs(2)).await;
    assert!(
        !final_events.is_empty() || total_events > 15,
        "Watcher should remain responsive after comprehensive test"
    );

    Ok(())
});
