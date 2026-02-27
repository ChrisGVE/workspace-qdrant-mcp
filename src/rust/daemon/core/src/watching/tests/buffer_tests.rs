//! Tests for PausedEventBuffer (Task 543.4-6)

use super::*;

mod paused_event_buffer_tests {
    use super::*;

    fn make_test_event(path: &str) -> FileEvent {
        FileEvent {
            path: PathBuf::from(path),
            event_kind: EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Content,
            )),
            timestamp: Instant::now(),
            system_time: SystemTime::now(),
            size: Some(100),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_paused_event_buffer_new() {
        let buffer = PausedEventBuffer::new();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.evictions(), 0);
    }

    #[test]
    fn test_paused_event_buffer_push_and_drain() {
        let mut buffer = PausedEventBuffer::new();
        buffer.push_event(make_test_event("/a.rs"));
        buffer.push_event(make_test_event("/b.rs"));
        buffer.push_event(make_test_event("/c.rs"));

        assert_eq!(buffer.len(), 3);
        assert!(!buffer.is_empty());

        let events = buffer.drain_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].path, PathBuf::from("/a.rs"));
        assert_eq!(events[2].path, PathBuf::from("/c.rs"));
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_paused_event_buffer_capacity_eviction() {
        let mut buffer = PausedEventBuffer {
            events: VecDeque::new(),
            capacity: 3, // Small capacity for testing
            evictions: 0,
        };

        buffer.push_event(make_test_event("/a.rs"));
        buffer.push_event(make_test_event("/b.rs"));
        buffer.push_event(make_test_event("/c.rs"));
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.evictions(), 0);

        // This should evict /a.rs
        buffer.push_event(make_test_event("/d.rs"));
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.evictions(), 1);

        let events = buffer.drain_events();
        assert_eq!(events[0].path, PathBuf::from("/b.rs"));
        assert_eq!(events[2].path, PathBuf::from("/d.rs"));
    }

    #[test]
    fn test_paused_event_buffer_default() {
        let buffer = PausedEventBuffer::default();
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity, PAUSED_EVENT_BUFFER_CAPACITY);
    }
}
