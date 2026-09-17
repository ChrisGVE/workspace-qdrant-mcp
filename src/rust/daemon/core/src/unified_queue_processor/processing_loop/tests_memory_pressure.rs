//! Tests for the memory-pressure checks of the unified queue processor.
//!
//! Only the pure limit predicate is covered here: the surrounding checks read
//! the live process and the host, so their outcome is not a property of this
//! code.

#[cfg(test)]
mod tests {
    use super::super::memory_pressure::rss_exceeds_limit;

    #[test]
    fn test_rss_below_limit_does_not_pause() {
        assert!(!rss_exceeds_limit(2047, 2048));
    }

    #[test]
    fn test_rss_at_limit_does_not_pause() {
        // The limit is a ceiling the process is allowed to reach.
        assert!(!rss_exceeds_limit(2048, 2048));
    }

    #[test]
    fn test_rss_above_limit_pauses() {
        assert!(rss_exceeds_limit(2049, 2048));
    }
}
