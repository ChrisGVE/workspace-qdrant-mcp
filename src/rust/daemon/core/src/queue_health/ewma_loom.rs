//! Loom model-checking proof for the torn-read-safe baseline restore (#133 F10,
//! DATA-03 / DOM-10 / SEC-08).
//!
//! Compiled and run ONLY under `RUSTFLAGS="--cfg wqm_loom" cargo test -p
//! workspace-qdrant-core ewma_loom`. `loom::model` exhaustively explores every
//! interleaving and weak-memory reordering of a writer running
//! [`EwmaLane::store`] against a reader running [`DualEwma::from_atomics`], which
//! a single-threaded x86 test cannot. The assertion proves the
//! `Acquire`/`Release` gate on `seeded` (read-seeded-first, publish-`seeded`-last)
//! prevents ever observing `seeded = true` with a stale `slow = 0.0` — the torn
//! restore that would silently suppress every post-restart regression verdict on
//! a weakly-ordered target (aarch64).

use loom::sync::Arc;
use loom::thread;

use crate::queue_health::ewma::{DualEwma, EwmaLane};

#[test]
fn seeded_publish_is_torn_read_safe() {
    loom::model(|| {
        let (fast_alpha, slow_alpha) = (0.3, 0.01);
        let lane = Arc::new(EwmaLane::new());

        let writer = {
            let lane = Arc::clone(&lane);
            thread::spawn(move || {
                let mut restored = DualEwma::new(fast_alpha, slow_alpha);
                restored.restore_baseline(42.0); // slow = fast = 42, seeded = true
                lane.store(&restored);
            })
        };

        let reader = {
            let lane = Arc::clone(&lane);
            thread::spawn(move || {
                let snap = DualEwma::from_atomics(&lane, fast_alpha, slow_alpha);
                // The Acquire/Release gate forbids observing the published
                // `seeded` without the matching non-zero `slow` store.
                assert!(
                    !(snap.seeded && snap.baseline() == 0.0),
                    "torn read: seeded=true with slow=0.0"
                );
            })
        };

        writer.join().unwrap();
        reader.join().unwrap();
    });
}
