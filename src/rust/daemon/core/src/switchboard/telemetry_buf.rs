//! Telemetry buffer — a fixed-capacity, lock-free MPMC ring.
//!
//! Wraps `crossbeam_queue::ArrayQueue<MetricSample>`. Emitters `push` without
//! blocking; a single background drain task `pop`s and converts to Prometheus
//! observations. On full (drain slow, or telemetry off and the consumer idle)
//! `push` returns `false` — the caller drops the sample and bumps a drop counter
//! (arch §6a). Telemetry can shed; the control path is independent. The buffer
//! is never read on the hot path (drain is background).

use crossbeam_queue::ArrayQueue;

use super::MetricSample;

/// Fixed ring capacity. At ~1 kHz emit this absorbs ~4 s of burst with no heap
/// growth (arch §6a).
const BUFFER_CAPACITY: usize = 4096;

/// Lock-free MPMC ring of telemetry samples.
pub struct TelemetryBuffer {
    queue: ArrayQueue<MetricSample>,
}

impl TelemetryBuffer {
    pub fn new() -> Self {
        Self {
            queue: ArrayQueue::new(BUFFER_CAPACITY),
        }
    }

    /// Push a sample. Returns `true` on success, `false` if the ring is full
    /// (sample dropped by the caller, who counts the drop).
    #[inline]
    pub fn push(&self, sample: MetricSample) -> bool {
        self.queue.push(sample).is_ok()
    }

    /// Pop the oldest sample for drain, or `None` if empty.
    #[inline]
    pub fn pop(&self) -> Option<MetricSample> {
        self.queue.pop()
    }

    /// Approximate current length (lock-free).
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl Default for TelemetryBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop_roundtrip() {
        let b = TelemetryBuffer::new();
        assert!(b.push(MetricSample::QueueItemMs(7)));
        match b.pop() {
            Some(MetricSample::QueueItemMs(v)) => assert_eq!(v, 7),
            other => panic!("unexpected: {other:?}"),
        }
        assert!(b.pop().is_none());
    }

    #[test]
    fn test_push_returns_false_when_full() {
        let b = TelemetryBuffer::new();
        for _ in 0..BUFFER_CAPACITY {
            assert!(b.push(MetricSample::QueueItemMs(1)));
        }
        // Ring is full now — next push must fail.
        assert!(!b.push(MetricSample::QueueItemMs(1)));
    }
}
