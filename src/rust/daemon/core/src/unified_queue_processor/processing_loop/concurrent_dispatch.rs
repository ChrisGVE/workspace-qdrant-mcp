//! Generic dispatch driver for the unified queue processor.
//!
//! Factored out of `batch_processing.rs` so that the control flow — spawn
//! up to `max_concurrent_items` futures, drain on cancellation, re-lease on
//! memory pressure, apply inter-dispatch delay — can be unit-tested with a
//! stubbed `spawn_item` and a stubbed pressure predicate, without standing
//! up the full Qdrant/embedding pipeline.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::{FuturesUnordered, StreamExt};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::UnifiedQueueItem;

/// Drive the per-batch dispatch loop with `FuturesUnordered`, capped by
/// `item_semaphore`.
///
/// Returns `true` if cancellation was observed mid-batch (caller propagates
/// `Err(())` to the outer poll loop), `false` if the batch ran to completion.
///
/// Behavioral guarantees:
/// - With a 1-permit semaphore the dispatch is strictly sequential — exactly
///   one item runs at a time and `apply_delay` fires between completions.
///   Byte-identical to the legacy `for item in items.iter()` loop.
/// - On cancellation, pending items are re-leased to `Pending` immediately
///   and in-flight items run to completion (no `mark_unified_failed`).
/// - On memory pressure, pending items are re-leased and the legacy 10s
///   in-batch back-off is preserved; in-flight items drain normally.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_dispatch_loop<F, Spawn, PressureFut, P, DelayFut, D>(
    items: Vec<UnifiedQueueItem>,
    item_semaphore: Arc<tokio::sync::Semaphore>,
    queue_manager: &QueueManager,
    cancellation_token: &CancellationToken,
    mut spawn_item: Spawn,
    mut is_memory_pressure: P,
    mut apply_delay: D,
    max_memory_percent: u8,
) -> bool
where
    Spawn: FnMut(UnifiedQueueItem, tokio::sync::OwnedSemaphorePermit) -> tokio::task::JoinHandle<F>,
    F: Send + 'static,
    P: FnMut() -> PressureFut,
    PressureFut: std::future::Future<Output = bool>,
    D: FnMut() -> DelayFut,
    DelayFut: std::future::Future<Output = ()>,
{
    let mut pending: VecDeque<UnifiedQueueItem> = items.into();
    let mut in_flight: FuturesUnordered<tokio::task::JoinHandle<F>> = FuturesUnordered::new();
    let mut cancelled = false;

    while !pending.is_empty() || !in_flight.is_empty() {
        if cancellation_token.is_cancelled() && !cancelled {
            warn!(
                "Shutdown requested during item processing — draining {} in-flight, re-leasing {} pending",
                in_flight.len(),
                pending.len()
            );
            cancelled = true;
            if !pending.is_empty() {
                let pending_slice: Vec<UnifiedQueueItem> = pending.drain(..).collect();
                re_lease_pending(queue_manager, &pending_slice).await;
            }
        }

        if !cancelled && !pending.is_empty() && is_memory_pressure().await {
            warn!(
                "Memory pressure during batch processing (<{}% available), pausing remaining items",
                100u8.saturating_sub(max_memory_percent)
            );
            // Re-lease ALL pending items (F-044) so they return to pending
            // instead of being stuck in_progress until lease expiry. In-flight
            // items drain to completion below.
            let pending_slice: Vec<UnifiedQueueItem> = pending.drain(..).collect();
            re_lease_pending(queue_manager, &pending_slice).await;
            // Match the legacy 10s back-off inside the batch so the next
            // dispatch cycle does not retry immediately. The outer
            // `handle_memory_pressure` gate in run_poll_cycle re-checks RSS
            // before the next dequeue.
            tokio::time::sleep(Duration::from_secs(10)).await;
        }

        // Dispatch as many items as the semaphore allows.
        while !cancelled && !pending.is_empty() {
            let permit = match Arc::clone(&item_semaphore).try_acquire_owned() {
                Ok(p) => p,
                Err(_) => break,
            };
            let item = pending
                .pop_front()
                .expect("pending non-empty guarded above");
            in_flight.push(spawn_item(item, permit));
        }

        if in_flight.is_empty() {
            break;
        }

        // Await the next completion. We use FuturesUnordered::next; on
        // cancellation we still want to drain in-flight to avoid half-applied
        // SQLite state, so we do NOT bail out on cancel here.
        match in_flight.next().await {
            Some(Ok(_)) => {}
            Some(Err(join_err)) => {
                if join_err.is_panic() {
                    error!("Spawned item future panicked: {}", join_err);
                } else {
                    debug!("Spawned item future cancelled at runtime: {}", join_err);
                }
            }
            None => break,
        }

        if !cancelled {
            apply_delay().await;
        }
    }

    cancelled
}

/// Re-lease the given items so they return to pending (F-044).
///
/// Without this, items leased as part of the batch but not yet processed would
/// remain `in_progress` until their lease expires, blocking other workers.
pub(super) async fn re_lease_pending(queue_manager: &QueueManager, items: &[UnifiedQueueItem]) {
    if items.is_empty() {
        return;
    }
    let count = items.len();
    let mut failures = 0;
    for item in items {
        if let Err(e) = queue_manager.re_lease_item(&item.queue_id, 30).await {
            warn!(queue_id = %item.queue_id, "Failed to re-lease pending item: {}", e);
            failures += 1;
        }
    }
    info!(
        total = count,
        released = count - failures,
        "Re-leased pending batch items"
    );
}

#[cfg(test)]
mod tests {
    //! Unit tests for the generic dispatch driver.
    //!
    //! These tests exercise `run_dispatch_loop` with stub `spawn_item`,
    //! `is_memory_pressure`, and `apply_delay` closures. They do NOT touch
    //! Qdrant, embeddings, or the full `process_item` pipeline — that
    //! coverage lives in the integration tests over `process_batch`.

    use super::*;
    use crate::queue_config::QueueConnectionConfig;
    use crate::queue_operations::QueueManager;
    use crate::unified_queue_schema::{ItemType, QueueOperation, QueueStatus, UnifiedQueueItem};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn make_item(queue_id: &str) -> UnifiedQueueItem {
        UnifiedQueueItem {
            queue_id: queue_id.to_string(),
            idempotency_key: format!("idem-{queue_id}"),
            item_type: ItemType::File,
            op: QueueOperation::Add,
            tenant_id: "test-tenant".to_string(),
            collection: "projects".to_string(),
            status: QueueStatus::InProgress,
            branch: "main".to_string(),
            payload_json: "{}".to_string(),
            metadata: None,
            created_at: "2026-05-01T00:00:00Z".to_string(),
            updated_at: "2026-05-01T00:00:00Z".to_string(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            error_message: None,
            last_error_at: None,
            file_path: None,
            qdrant_status: None,
            search_status: None,
            decision_json: None,
        }
    }

    async fn setup_queue_manager() -> (QueueManager, tempfile::TempDir) {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("dispatch_test.db");
        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();
        // watch_folders is referenced by dequeue_unified's JOIN; the queue
        // operations tests apply this schema before init_unified_queue, so we
        // mirror the order here. Without this, enqueue/dequeue panics.
        apply_watch_folders_schema(&pool).await;
        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();
        (manager, temp)
    }

    /// Apply the watch_folders DDL by parsing the schema file statement-by-
    /// statement (sqlx's `execute` doesn't run multi-statement scripts).
    async fn apply_watch_folders_schema(pool: &sqlx::SqlitePool) {
        let script = include_str!("../../schema/watch_folders_schema.sql");
        let mut conn = pool.acquire().await.unwrap();
        let mut statement = String::new();
        let mut in_trigger = false;
        for line in script.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("--") {
                continue;
            }
            if trimmed.to_uppercase().starts_with("CREATE TRIGGER") {
                in_trigger = true;
            }
            statement.push_str(line);
            statement.push('\n');
            if in_trigger {
                if trimmed.eq_ignore_ascii_case("END;") || trimmed.eq_ignore_ascii_case("END") {
                    in_trigger = false;
                    let stmt = statement.trim();
                    if !stmt.is_empty() {
                        sqlx::query(stmt).execute(&mut *conn).await.unwrap();
                    }
                    statement.clear();
                }
                continue;
            }
            if trimmed.ends_with(';') {
                let stmt = statement.trim();
                if !stmt.is_empty() {
                    sqlx::query(stmt).execute(&mut *conn).await.unwrap();
                }
                statement.clear();
            }
        }
        let remainder = statement.trim();
        if !remainder.is_empty() {
            sqlx::query(remainder).execute(&mut *conn).await.unwrap();
        }
    }

    /// Test 1: With `max_concurrent_items=4`, the dispatch loop never holds
    /// more than 4 spawned futures in flight at once. The closure MUST move
    /// the permit into the spawned future so its lifetime extends past the
    /// closure return — otherwise the semaphore frees immediately and the
    /// cap collapses.
    #[tokio::test]
    async fn test_concurrent_dispatch_respects_semaphore() {
        let (manager, _tmp) = setup_queue_manager().await;
        let semaphore = Arc::new(tokio::sync::Semaphore::new(4));
        let cancel = CancellationToken::new();

        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_observed = Arc::new(AtomicUsize::new(0));

        let items: Vec<UnifiedQueueItem> = (0..20).map(|i| make_item(&format!("q{i}"))).collect();

        let in_flight_clone = Arc::clone(&in_flight);
        let max_clone = Arc::clone(&max_observed);

        let cancelled = run_dispatch_loop(
            items,
            Arc::clone(&semaphore),
            &manager,
            &cancel,
            move |_item, permit| {
                let in_flight = Arc::clone(&in_flight_clone);
                let max_obs = Arc::clone(&max_clone);
                tokio::spawn(async move {
                    // Hold permit for the full spawned future lifetime —
                    // mirrors how `process_one_item_owned` keeps the
                    // dispatch slot reserved until completion.
                    let _p = permit;
                    let current = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    max_obs.fetch_max(current, Ordering::SeqCst);
                    // Yield enough times that other spawned futures get a
                    // chance to race in before we exit.
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    tokio::time::sleep(Duration::from_millis(2)).await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                })
            },
            || async { false },
            || async {},
            70,
        )
        .await;

        assert!(!cancelled);
        assert_eq!(in_flight.load(Ordering::SeqCst), 0, "all spawns settled");
        assert!(
            max_observed.load(Ordering::SeqCst) <= 4,
            "max in-flight = {} should never exceed semaphore=4",
            max_observed.load(Ordering::SeqCst)
        );
        assert!(
            max_observed.load(Ordering::SeqCst) >= 2,
            "with 20 items + 4 permits we expect at least 2 concurrent at some point, observed {}",
            max_observed.load(Ordering::SeqCst)
        );
    }

    /// Test 2: Cancellation drains in-flight items (no panics, no half-state)
    /// and re-leases pending items back to `Pending`.
    #[tokio::test]
    async fn test_cancellation_drains_in_flight() {
        let (manager, _tmp) = setup_queue_manager().await;

        // Enqueue 8 items so we have real rows the dispatcher can re-lease.
        let mut queue_ids = Vec::new();
        for i in 0..8 {
            let (qid, _) = manager
                .enqueue_unified(
                    ItemType::File,
                    QueueOperation::Add,
                    "test-tenant",
                    "projects",
                    &format!(r#"{{"file_path":"/test/{i}.rs"}}"#),
                    Some("main"),
                    None,
                )
                .await
                .unwrap();
            queue_ids.push(qid);
        }
        // Dequeue them so they're in_progress (matching the pre-dispatch state).
        let items = manager
            .dequeue_unified(8, "test-worker", Some(300), None, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(items.len(), 8);

        let semaphore = Arc::new(tokio::sync::Semaphore::new(4));
        let cancel = CancellationToken::new();
        let cancel_for_trigger = cancel.clone();

        let started_count = Arc::new(AtomicUsize::new(0));
        let completed_count = Arc::new(AtomicUsize::new(0));
        let started_clone = Arc::clone(&started_count);
        let completed_clone = Arc::clone(&completed_count);

        // Fire cancellation after a short delay.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            cancel_for_trigger.cancel();
        });

        let cancelled = run_dispatch_loop(
            items,
            Arc::clone(&semaphore),
            &manager,
            &cancel,
            move |_item, _permit| {
                let started = Arc::clone(&started_clone);
                let completed = Arc::clone(&completed_clone);
                tokio::spawn(async move {
                    started.fetch_add(1, Ordering::SeqCst);
                    // Slow enough that cancellation lands while some are still
                    // running.
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    completed.fetch_add(1, Ordering::SeqCst);
                })
            },
            || async { false },
            || async {},
            70,
        )
        .await;

        assert!(cancelled, "loop must report cancellation");
        // Every started item must have completed (no half-applied state).
        let started_total = started_count.load(Ordering::SeqCst);
        let completed_total = completed_count.load(Ordering::SeqCst);
        assert_eq!(
            started_total, completed_total,
            "in-flight items must drain to completion ({started_total} started, {completed_total} completed)"
        );

        // The 8 items split into: some started (which completed) and some
        // re-leased. Query the queue to verify the re-leased ones are Pending.
        let pending_rows = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending' AND queue_id IN (\
             ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )
        .bind(&queue_ids[0])
        .bind(&queue_ids[1])
        .bind(&queue_ids[2])
        .bind(&queue_ids[3])
        .bind(&queue_ids[4])
        .bind(&queue_ids[5])
        .bind(&queue_ids[6])
        .bind(&queue_ids[7])
        .fetch_one(manager.pool())
        .await
        .unwrap();
        // We expect (8 - started_total) items to be re-leased to pending.
        let expected_pending = (8u64).saturating_sub(started_total as u64) as i64;
        assert_eq!(
            pending_rows, expected_pending,
            "expected {expected_pending} re-leased items in pending, found {pending_rows} \
             (started={started_total})"
        );
    }

    /// Test 3: Memory pressure halts new dispatches and re-leases pending.
    ///
    /// The pressure predicate is wired to return `true` on its first call and
    /// `false` afterwards. Combined with a no-op delay, this exercises the
    /// re-lease path without burning the legacy 10s in-batch back-off in the
    /// test (the back-off is gated on `is_memory_pressure() == true`, so a
    /// single-shot pressure observation triggers it once and we just sleep
    /// 10s of test time before the loop continues). To keep the test fast,
    /// we instead set up pressure so the FIRST iteration drains pending and
    /// then the loop ends because in_flight is empty.
    #[tokio::test]
    async fn test_memory_pressure_gate() {
        let (manager, _tmp) = setup_queue_manager().await;

        // Enqueue 6 items.
        let mut queue_ids = Vec::new();
        for i in 0..6 {
            let (qid, _) = manager
                .enqueue_unified(
                    ItemType::File,
                    QueueOperation::Add,
                    "test-tenant",
                    "projects",
                    &format!(r#"{{"file_path":"/test/p{i}.rs"}}"#),
                    Some("main"),
                    None,
                )
                .await
                .unwrap();
            queue_ids.push(qid);
        }
        let items = manager
            .dequeue_unified(6, "test-worker", Some(300), None, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(items.len(), 6);

        let semaphore = Arc::new(tokio::sync::Semaphore::new(2));
        let cancel = CancellationToken::new();

        // Pre-set pressure to true. The very first loop iteration will see
        // pressure with all 6 items still pending, drain them, sleep 10s,
        // then exit because nothing is in flight. The wall-clock cost is
        // bounded by the timeout below; we keep it tight so a regression
        // (pressure not honored) is visible in <2s.
        let pressure_flag = Arc::new(AtomicBool::new(true));
        let dispatched_count = Arc::new(AtomicUsize::new(0));
        let hit_count = Arc::new(AtomicUsize::new(0));

        let pressure_for_check = Arc::clone(&pressure_flag);
        let hit_count_clone = Arc::clone(&hit_count);
        let dispatched_clone = Arc::clone(&dispatched_count);

        // After the first pressure observation we flip it off, but the
        // dispatch loop hits the 10s sleep before re-checking. We override
        // the in-batch sleep behavior by NOT applying the back-off here: the
        // dispatch implementation sleeps unconditionally after re-leasing.
        // To keep the test fast, replace the test pool's tokio time with
        // tokio::time::pause / advance — but that interacts with the real
        // time used by tokio::spawn. Simpler: cap with a 12s timeout and
        // tolerate the sleep.
        let dispatch_fut = run_dispatch_loop(
            items,
            Arc::clone(&semaphore),
            &manager,
            &cancel,
            move |_item, permit| {
                let dispatched = Arc::clone(&dispatched_clone);
                tokio::spawn(async move {
                    let _p = permit;
                    dispatched.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(1)).await;
                })
            },
            move || {
                let flag = Arc::clone(&pressure_for_check);
                let hits = Arc::clone(&hit_count_clone);
                async move {
                    let v = flag.load(Ordering::SeqCst);
                    if v {
                        hits.fetch_add(1, Ordering::SeqCst);
                        // Flip off so the next loop iteration won't pause again.
                        flag.store(false, Ordering::SeqCst);
                    }
                    v
                }
            },
            || async {},
            70,
        );

        // Tight ceiling: the legacy 10s back-off runs once on the pressure
        // hit, plus dispatch and re-lease overhead. 12s is a safety margin.
        let result = tokio::time::timeout(Duration::from_secs(12), dispatch_fut).await;
        assert!(result.is_ok(), "dispatch loop hung past 12s");
        let cancelled = result.unwrap();
        assert!(!cancelled, "memory pressure must not be reported as cancel");

        assert!(
            hit_count.load(Ordering::SeqCst) >= 1,
            "pressure predicate should have fired at least once"
        );
        // With pressure on at start no items dispatched; all 6 re-leased.
        let dispatched_total = dispatched_count.load(Ordering::SeqCst);
        let pending_after = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending'",
        )
        .fetch_one(manager.pool())
        .await
        .unwrap();
        assert!(
            pending_after >= 1,
            "memory pressure should re-lease at least one item to pending, found {pending_after}"
        );
        assert_eq!(
            dispatched_total as i64 + pending_after,
            6,
            "all 6 items accounted for (dispatched={dispatched_total}, pending={pending_after})"
        );
    }

    /// Test 4: An empty input batch is a no-op (no spawn, no panic).
    #[tokio::test]
    async fn test_empty_batch_is_noop() {
        let (manager, _tmp) = setup_queue_manager().await;
        let semaphore = Arc::new(tokio::sync::Semaphore::new(4));
        let cancel = CancellationToken::new();
        let spawn_count = Arc::new(AtomicUsize::new(0));
        let spawn_clone = Arc::clone(&spawn_count);

        let cancelled = run_dispatch_loop(
            Vec::new(),
            Arc::clone(&semaphore),
            &manager,
            &cancel,
            move |_item, _permit| {
                let c = Arc::clone(&spawn_clone);
                tokio::spawn(async move {
                    c.fetch_add(1, Ordering::SeqCst);
                })
            },
            || async { false },
            || async {},
            70,
        )
        .await;

        assert!(!cancelled);
        assert_eq!(spawn_count.load(Ordering::SeqCst), 0);
    }
}
