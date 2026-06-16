//! Unit tests for `EwmaState` (#133 F2/§4.2, test-strategy support for item 1/12).

use super::*;
use crate::config::queue_health::QueueHealthConfig;
use crate::switchboard::ControlFanout;

fn state() -> EwmaState {
    EwmaState::new(&QueueHealthConfig::default())
}

#[test]
fn from_fanout_shares_lanes_with_fanout() {
    // DATA-07 identity: a sample fed into a fanout lane is observed by the
    // EwmaState built from that fanout — proving from_fanout cloned the SAME
    // Arc<ControlLane> the emit advances (the #133 single-source handshake).
    let cfg = QueueHealthConfig::default();
    let fanout = ControlFanout::new(&cfg);
    let s = EwmaState::from_fanout(&fanout, &cfg);

    fanout.embedder_latency.update(123.0); // first sample seeds the lane
    let snap = s.embedder_latency_snapshot();
    assert!(snap.seeded);
    assert_eq!(snap.fast, 123.0);
    assert_eq!(snap.slow, 123.0);
}

#[test]
fn construction_takes_config_alphas() {
    let cfg = QueueHealthConfig {
        fast_alpha: 0.4,
        slow_alpha: 0.02,
        ..Default::default()
    };
    let s = EwmaState::new(&cfg);
    assert_eq!(s.alphas().fast, 0.4);
    assert_eq!(s.alphas().slow, 0.02);
}

#[test]
fn lane_update_round_trips_through_snapshot() {
    let s = state();
    // Unseeded snapshot has no ratio.
    assert_eq!(s.ms_per_kb_snapshot().ratio(), None);

    s.update_ms_per_kb(10.0); // seeds both lanes to 10.0
    let snap = s.ms_per_kb_snapshot();
    assert!(snap.seeded);
    assert_eq!(snap.fast, 10.0);
    assert_eq!(snap.slow, 10.0);
    assert_eq!(snap.ratio(), Some(1.0));
}

#[test]
fn lanes_are_independent() {
    let s = state();
    s.update_embedder_latency(5.0);
    s.update_throughput(2048.0);
    assert!(s.embedder_latency_snapshot().seeded);
    assert!(s.throughput_snapshot().seeded);
    // The untouched ms/KB and DLQ lanes stay unseeded.
    assert!(!s.ms_per_kb_snapshot().seeded);
    assert!(!s.dlq_depth_snapshot().seeded);
}

#[test]
fn debounce_majority_flips_only_after_enough_agreement() {
    // Default window = 5; majority (plurality) = 3.
    let s = state();
    // Two Reds amid Greens: Green still wins (3 Green vs 2 Red).
    assert_eq!(s.observe("p", Rag::Green), Rag::Green);
    assert_eq!(s.observe("p", Rag::Green), Rag::Green);
    assert_eq!(s.observe("p", Rag::Red), Rag::Green);
    assert_eq!(s.observe("p", Rag::Green), Rag::Green);
    assert_eq!(s.observe("p", Rag::Red), Rag::Green); // window full: [G,G,R,G,R] -> 3G
                                                      // Now push Reds; once 3 of the last 5 are Red, it flips.
    assert_eq!(s.observe("p", Rag::Red), Rag::Red); // [G,R,G,R,R] -> 3R
}

#[test]
fn debounce_window_slides() {
    let s = state();
    for _ in 0..5 {
        s.observe("p", Rag::Red);
    }
    // Window is all Red.
    assert_eq!(s.observe("p", Rag::Red), Rag::Red);
    // Five Greens slide the Reds out.
    let mut last = Rag::Red;
    for _ in 0..5 {
        last = s.observe("p", Rag::Green);
    }
    assert_eq!(last, Rag::Green, "window slid to all Green");
}

#[test]
fn debounce_rings_are_per_probe() {
    let s = state();
    for _ in 0..5 {
        s.observe("a", Rag::Red);
    }
    // Probe "b" is independent and starts Green.
    assert_eq!(s.observe("b", Rag::Green), Rag::Green);
    // Probe "a" is still Red.
    assert_eq!(s.observe("a", Rag::Red), Rag::Red);
}

#[test]
fn debounce_tie_breaks_toward_severity() {
    // Even though the window is odd, three distinct RAG values can tie on count.
    // A window of 3 with one each of Red/Amber/Green ties 1-1-1 -> most severe (Red).
    let cfg = QueueHealthConfig {
        debounce_window: 3,
        ..Default::default()
    };
    let s = EwmaState::new(&cfg);
    s.observe("p", Rag::Green);
    s.observe("p", Rag::Amber);
    let debounced = s.observe("p", Rag::Red); // [G,A,R] 1-1-1 tie
    assert_eq!(debounced, Rag::Red, "ties resolve to the most severe RAG");
}

#[test]
fn drain_snapshot_round_trips_and_is_fresh() {
    let s = state();
    assert!(
        s.drain_snapshot().is_none(),
        "no snapshot before first sample"
    );
    s.set_drain_snapshot(123_456);
    let snap = s.drain_snapshot().expect("snapshot present after set");
    assert_eq!(snap.pending_bytes, 123_456);
    // Just-sampled: the staleness guard (drain probe) sees a tiny elapsed.
    assert!(
        snap.sampled_at.elapsed().as_secs() < 5,
        "freshly sampled snapshot is recent"
    );
}
