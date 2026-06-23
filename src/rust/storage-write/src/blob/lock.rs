//! `ContentKeyLockManager` — one async lock per `content_key` (arch §6.3).
//!
//! File: `wqm-storage-write/src/blob/lock.rs`
//! Location: `src/rust/storage-write/src/blob/` (write-crate blob layer)
//! Context: The ingest dedup ladder (`blob::dedup`) writes one blob and its Qdrant
//!   point under a per-`content_key` lock so two concurrent ingests of the SAME
//!   content_key serialize — the first creates the blob, the second finds it present
//!   and takes the membership-update path (the F04 race, AC-F6.5). This is the
//!   §8 nexus invariant: NO blob or Qdrant blob point is written outside this lock.
//!
//!   Heap bound (AC-F6.4, §14-Q4 defaults): the lock map is eviction-bounded, never
//!   monotonic. A background cleanup task evicts idle, waiter-free locks; a hard
//!   entry cap with a single global fallback lock keeps the heap bounded even during
//!   a 400k-blob onboard that presents more distinct content_keys than the cap.
//!
//!   Deadlock freedom (AC-F6.8 / IMPL-05): a file with N chunks acquires N locks. To
//!   ingest a file the caller MUST sort its content_keys lexicographically and acquire
//!   them in that order ([`ContentKeyLockManager::lock_many`]). Two files that share
//!   >=2 chunks therefore acquire the shared locks in the same order and cannot
//!   deadlock by opposite traversal.
//!
//! Neighbors: [`crate::blob::dedup`] (the sole caller — runs the ladder under the
//!   lock), [`crate::blob::ladder`] (per-chunk write cycle).

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OwnedMutexGuard};

/// Tunable bounds for [`ContentKeyLockManager`] (arch §6.3, §14-Q4 Chris defaults).
///
/// These are flow-control knobs, not correctness boundaries: the cap trades onboard
/// concurrency for a hard memory bound. The defaults are the §14-Q4 Chris-input
/// values; the cap is the single tunable if very large branches need more concurrency.
#[derive(Debug, Clone, Copy)]
pub struct LockManagerConfig {
    /// Maximum distinct per-content_key locks held in the map. Above this the global
    /// fallback lock serializes new-content_key writes instead of minting an unbounded
    /// per-key entry (PERF-R5-N2). Default: 100,000.
    pub max_entries: usize,
    /// A lock is eligible for eviction once it has been idle (no holder, no waiter)
    /// for at least this long. Default: 300 s.
    pub idle_evict: Duration,
    /// How often the background cleanup task sweeps for evictable locks. Default: 30 s.
    pub cleanup_interval: Duration,
}

impl Default for LockManagerConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            idle_evict: Duration::from_secs(300),
            cleanup_interval: Duration::from_secs(30),
        }
    }
}

/// One lock entry: the async mutex plus the bookkeeping the eviction sweep needs.
///
/// `waiters` counts threads currently holding-or-queued for the mutex (incremented
/// before `.lock()`, decremented when the guard drops). The sweep evicts ONLY entries
/// with zero waiters — a non-zero count means the lock is live and dropping it would
/// fork it into two distinct mutexes, breaking serialization. `last_release` records
/// when the last holder let go, so the sweep can apply the idle window.
struct LockEntry {
    mutex: Arc<Mutex<()>>,
    waiters: Arc<AtomicU64>,
    last_release: Arc<std::sync::Mutex<Instant>>,
}

impl LockEntry {
    fn new() -> Self {
        Self {
            mutex: Arc::new(Mutex::new(())),
            waiters: Arc::new(AtomicU64::new(0)),
            last_release: Arc::new(std::sync::Mutex::new(Instant::now())),
        }
    }
}

/// A held content_key lock. Dropping it releases the mutex and updates the entry's
/// `last_release` timestamp so the eviction sweep can age it out.
///
/// The guard owns the mutex guard (`OwnedMutexGuard`) so it is `'static` — it can be
/// held across `.await` points and returned up the call stack without borrowing the
/// manager. It also holds the `waiters` counter and decrements it on drop.
pub struct ContentKeyLock {
    _guard: OwnedMutexGuard<()>,
    waiters: Arc<AtomicU64>,
    last_release: Arc<std::sync::Mutex<Instant>>,
}

impl Drop for ContentKeyLock {
    fn drop(&mut self) {
        // Record the release time first so a concurrent sweep sees a fresh timestamp,
        // then drop the waiter count. Order matters: decrementing waiters is what makes
        // the entry eviction-eligible, so the timestamp must already be current.
        if let Ok(mut t) = self.last_release.lock() {
            *t = Instant::now();
        }
        self.waiters.fetch_sub(1, Ordering::AcqRel);
    }
}

/// In-process lock registry keyed by `content_key` (arch §6.3).
///
/// Construct with [`ContentKeyLockManager::new`]; the map is eviction-bounded by the
/// supplied [`LockManagerConfig`]. Acquire ONE lock with [`Self::lock`], or a sorted
/// set with [`Self::lock_many`] (the deadlock-free multi-chunk path).
pub struct ContentKeyLockManager {
    locks: DashMap<String, LockEntry>,
    /// Engaged when the map is at `max_entries` and a NEW content_key arrives: rather
    /// than mint an unbounded entry, the writer serializes on this single mutex
    /// (PERF-R5-N2). A hard memory bound at the cost of some onboard concurrency.
    fallback: Arc<Mutex<()>>,
    fallback_waiters: Arc<AtomicU64>,
    config: LockManagerConfig,
}

impl ContentKeyLockManager {
    /// Build a manager with the given bounds. Does not spawn the cleanup task — call
    /// [`Self::spawn_cleanup`] (or drive [`Self::evict_idle`] manually in tests).
    pub fn new(config: LockManagerConfig) -> Arc<Self> {
        Arc::new(Self {
            locks: DashMap::new(),
            fallback: Arc::new(Mutex::new(())),
            fallback_waiters: Arc::new(AtomicU64::new(0)),
            config,
        })
    }

    /// Build a manager with the [`LockManagerConfig::default`] (§14-Q4) bounds.
    pub fn with_defaults() -> Arc<Self> {
        Self::new(LockManagerConfig::default())
    }

    /// Number of distinct per-content_key locks currently in the map. Used by the cap
    /// test to assert the map never exceeds `max_entries`.
    pub fn entry_count(&self) -> usize {
        self.locks.len()
    }

    /// Acquire the lock for one `content_key`, awaiting if another task holds it.
    ///
    /// If the map is below the cap, a per-key entry is used (created on first sight).
    /// If the map is AT the cap and this key is new, the single global fallback lock is
    /// taken instead — bounding the heap (PERF-R5-N2). An already-present key always
    /// uses its own entry regardless of the cap.
    pub async fn lock(self: &Arc<Self>, content_key: &str) -> ContentKeyLock {
        // Fast path: key already has an entry — use it (cap does not apply to existing
        // keys, only to minting new ones).
        if let Some(entry) = self.locks.get(content_key) {
            let mutex = entry.mutex.clone();
            let waiters = entry.waiters.clone();
            let last_release = entry.last_release.clone();
            drop(entry); // release the DashMap shard guard before awaiting the mutex
            return Self::acquire(mutex, waiters, last_release).await;
        }

        // New key. Honor the cap: at capacity, serialize on the fallback lock rather
        // than grow the map past `max_entries`.
        if self.locks.len() >= self.config.max_entries {
            return self.acquire_fallback().await;
        }

        // Below cap: mint the entry (the `entry` API collapses a concurrent insert race
        // — two tasks racing to create the same key converge on one LockEntry).
        let dm_entry = self
            .locks
            .entry(content_key.to_string())
            .or_insert_with(LockEntry::new);
        let mutex = dm_entry.mutex.clone();
        let waiters = dm_entry.waiters.clone();
        let last_release = dm_entry.last_release.clone();
        drop(dm_entry);
        Self::acquire(mutex, waiters, last_release).await
    }

    /// Acquire locks for many content_keys at once, ALWAYS in lexicographic order
    /// (AC-F6.8 / IMPL-05). De-duplicates the input first so a file that references the
    /// same content_key in two chunks takes its lock once (re-locking an owned async
    /// mutex would self-deadlock).
    ///
    /// Returns the guards in sorted order. Hold them for the file's whole chunk loop;
    /// dropping the returned Vec releases all of them.
    pub async fn lock_many(self: &Arc<Self>, content_keys: &[String]) -> Vec<ContentKeyLock> {
        let mut sorted: Vec<&String> = content_keys.iter().collect();
        sorted.sort_unstable();
        sorted.dedup();

        let mut guards = Vec::with_capacity(sorted.len());
        for key in sorted {
            guards.push(self.lock(key).await);
        }
        guards
    }

    /// Acquire the owned mutex guard, counting this task as a waiter for the whole
    /// hold so the eviction sweep treats the entry as live.
    async fn acquire(
        mutex: Arc<Mutex<()>>,
        waiters: Arc<AtomicU64>,
        last_release: Arc<std::sync::Mutex<Instant>>,
    ) -> ContentKeyLock {
        waiters.fetch_add(1, Ordering::AcqRel);
        let guard = mutex.lock_owned().await;
        ContentKeyLock {
            _guard: guard,
            waiters,
            last_release,
        }
    }

    /// Take the single global fallback lock (the at-capacity path).
    async fn acquire_fallback(self: &Arc<Self>) -> ContentKeyLock {
        Self::acquire(
            self.fallback.clone(),
            self.fallback_waiters.clone(),
            // The fallback lock is never evicted; a throwaway timestamp satisfies the
            // shared ContentKeyLock shape without affecting any sweep.
            Arc::new(std::sync::Mutex::new(Instant::now())),
        )
        .await
    }

    /// Evict every per-key lock that has zero waiters AND has been idle longer than
    /// `idle_evict`. Returns the number evicted. This is the body the cleanup task runs
    /// on each tick; tests call it directly to assert the map shrinks.
    pub fn evict_idle(&self) -> usize {
        let now = Instant::now();
        let idle = self.config.idle_evict;
        let mut evicted = 0;
        // `retain` visits every entry; we keep only the still-live or still-warm ones.
        self.locks.retain(|_key, entry| {
            let has_waiter = entry.waiters.load(Ordering::Acquire) > 0;
            let warm = entry
                .last_release
                .lock()
                .map(|t| now.duration_since(*t) < idle)
                .unwrap_or(true);
            let keep = has_waiter || warm;
            if !keep {
                evicted += 1;
            }
            keep
        });
        evicted
    }

    /// Spawn the background cleanup task that runs [`Self::evict_idle`] every
    /// `cleanup_interval`. The task lives as long as a clone of the manager `Arc` does.
    pub fn spawn_cleanup(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let weak = Arc::downgrade(self);
        let interval = self.config.cleanup_interval;
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                match weak.upgrade() {
                    Some(mgr) => {
                        mgr.evict_idle();
                    }
                    None => break, // manager dropped — stop sweeping
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    // AC-F6.5 (F04 race): two concurrent acquisitions of the SAME content_key
    // serialize — they never hold the lock simultaneously.
    #[tokio::test]
    async fn same_content_key_serializes() {
        let mgr = ContentKeyLockManager::with_defaults();
        let in_critical = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let mgr = mgr.clone();
            let in_critical = in_critical.clone();
            let max_seen = max_seen.clone();
            handles.push(tokio::spawn(async move {
                let _g = mgr.lock("same-key").await;
                let cur = in_critical.fetch_add(1, Ordering::AcqRel) + 1;
                max_seen.fetch_max(cur, Ordering::AcqRel);
                tokio::time::sleep(Duration::from_millis(2)).await;
                in_critical.fetch_sub(1, Ordering::AcqRel);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            max_seen.load(Ordering::Acquire),
            1,
            "only one task may hold a given content_key lock at a time"
        );
    }

    // Distinct content_keys do NOT serialize against each other.
    #[tokio::test]
    async fn distinct_keys_are_independent() {
        let mgr = ContentKeyLockManager::with_defaults();
        let _a = mgr.lock("key-a").await;
        // A second key must be immediately acquirable while the first is held.
        let acquired = tokio::time::timeout(Duration::from_millis(50), mgr.lock("key-b")).await;
        assert!(
            acquired.is_ok(),
            "distinct content_keys must not block each other"
        );
    }

    // AC-F6.8 / IMPL-05 (deadlock freedom): two "files" share >=2 chunks; two threads
    // acquire them via lock_many in opposite INPUT orders. Because lock_many sorts, both
    // take the shared locks in the same order — no deadlock.
    #[tokio::test]
    async fn shared_chunks_opposite_orders_no_deadlock() {
        let mgr = ContentKeyLockManager::with_defaults();
        // File 1 lists the shared keys as [ck-a, ck-b]; file 2 lists them reversed.
        let file1 = vec!["ck-a".to_string(), "ck-b".to_string(), "ck-1".to_string()];
        let file2 = vec!["ck-b".to_string(), "ck-a".to_string(), "ck-2".to_string()];

        let run = async {
            let m1 = mgr.clone();
            let m2 = mgr.clone();
            let h1 = tokio::spawn(async move {
                for _ in 0..50 {
                    let _g = m1.lock_many(&file1).await;
                    tokio::task::yield_now().await;
                }
            });
            let h2 = tokio::spawn(async move {
                for _ in 0..50 {
                    let _g = m2.lock_many(&file2).await;
                    tokio::task::yield_now().await;
                }
            });
            h1.await.unwrap();
            h2.await.unwrap();
        };
        // If the sorted-acquisition discipline failed, the two threads would deadlock
        // and this timeout would fire.
        tokio::time::timeout(Duration::from_secs(10), run)
            .await
            .expect("sorted multi-lock acquisition must not deadlock");
    }

    // lock_many de-duplicates: a file referencing the same content_key twice acquires
    // it once (re-locking an owned async mutex would self-deadlock).
    #[tokio::test]
    async fn lock_many_dedups_repeated_key() {
        let mgr = ContentKeyLockManager::with_defaults();
        let keys = vec!["dup".to_string(), "dup".to_string(), "other".to_string()];
        let guards = tokio::time::timeout(Duration::from_millis(200), mgr.lock_many(&keys))
            .await
            .expect("repeated content_key must not self-deadlock");
        assert_eq!(guards.len(), 2, "repeated key collapses to one lock");
    }

    // AC-F6.4 (heap bound, eviction): the map does NOT grow monotonically. After
    // exercising many keys then waiting out the idle window, the sweep shrinks it.
    #[tokio::test]
    async fn map_does_not_grow_monotonically() {
        let config = LockManagerConfig {
            max_entries: 100_000,
            idle_evict: Duration::from_millis(20),
            cleanup_interval: Duration::from_secs(30),
        };
        let mgr = ContentKeyLockManager::new(config);

        for i in 0..500 {
            let _g = mgr.lock(&format!("transient-{i}")).await;
            // guard dropped at end of iteration -> entry becomes waiter-free
        }
        assert_eq!(mgr.entry_count(), 500, "all transient keys minted an entry");

        // Let the idle window elapse, then sweep.
        tokio::time::sleep(Duration::from_millis(40)).await;
        let evicted = mgr.evict_idle();
        assert_eq!(evicted, 500, "all idle, waiter-free entries are evicted");
        assert_eq!(mgr.entry_count(), 0, "map shrinks — not monotonic growth");
    }

    // A lock with a live waiter is NOT evicted even if its last_release is stale.
    #[tokio::test]
    async fn held_lock_is_not_evicted() {
        let config = LockManagerConfig {
            max_entries: 100_000,
            idle_evict: Duration::from_millis(1),
            cleanup_interval: Duration::from_secs(30),
        };
        let mgr = ContentKeyLockManager::new(config);
        let _held = mgr.lock("held").await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        let evicted = mgr.evict_idle();
        assert_eq!(
            evicted, 0,
            "an entry with a live waiter must survive the sweep"
        );
        assert_eq!(mgr.entry_count(), 1);
    }

    // AC-F6.4 / PERF-R5-N2 (cap + fallback): more distinct keys than the cap. The
    // fallback lock engages, the map never exceeds the cap, and writes still complete.
    #[tokio::test]
    async fn cap_engages_fallback_and_bounds_map() {
        let config = LockManagerConfig {
            max_entries: 4,
            idle_evict: Duration::from_secs(300),
            cleanup_interval: Duration::from_secs(30),
        };
        let mgr = ContentKeyLockManager::new(config);

        // Fill the map to the cap with held entries so eviction cannot run.
        let mut held = Vec::new();
        for i in 0..4 {
            held.push(mgr.lock(&format!("capped-{i}")).await);
        }
        assert_eq!(mgr.entry_count(), 4, "map filled to cap");

        // Drive MORE distinct keys than the cap. Each must complete via the fallback
        // lock (proven by the writes finishing) without minting a new entry.
        let completed = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for i in 0..10 {
            let mgr = mgr.clone();
            let completed = completed.clone();
            handles.push(tokio::spawn(async move {
                let _g = mgr.lock(&format!("over-cap-{i}")).await;
                completed.fetch_add(1, Ordering::AcqRel);
            }));
        }
        for h in handles {
            tokio::time::timeout(Duration::from_secs(5), h)
                .await
                .expect("over-cap write must complete under the fallback lock")
                .unwrap();
        }

        assert_eq!(
            completed.load(Ordering::Acquire),
            10,
            "all over-cap writes complete (no deadlock, no dropped write)"
        );
        assert_eq!(
            mgr.entry_count(),
            4,
            "map never exceeds the cap — over-cap keys used the fallback lock"
        );
        drop(held);
    }
}
