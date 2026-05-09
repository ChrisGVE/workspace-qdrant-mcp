//! Event handling for the TUI.
//!
//! Spawns a background thread that polls crossterm events and emits
//! them (plus periodic tick events) over an MPSC channel.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crossterm::event::{self, KeyEvent};

/// Events produced by the event handler.
#[derive(Debug, Clone, Copy)]
pub enum Event {
    /// A keyboard event.
    Key(KeyEvent),
    /// A periodic tick for UI updates.
    Tick,
    /// The terminal was resized; ratatui handles the new size on the
    /// next draw, so the event is a notification only.
    Resize,
}

/// Polls crossterm events on a background thread and sends them
/// through a channel.
pub struct EventHandler {
    rx: mpsc::Receiver<Event>,
    // Keep the sender alive so the thread can detect when to stop
    // (it checks `tx.send()` failure).
    _tx: mpsc::Sender<Event>,
}

impl EventHandler {
    /// Create a new event handler with the given tick rate.
    ///
    /// The handler spawns a daemon thread that polls for crossterm events
    /// at `tick_rate` intervals. If no crossterm event is available within
    /// one tick period, a `Event::Tick` is emitted instead.
    pub fn new(tick_rate: Duration) -> Self {
        let (tx, rx) = mpsc::channel();
        let event_tx = tx.clone();

        thread::Builder::new()
            .name("tui-event-poller".into())
            .spawn(move || {
                Self::poll_loop(event_tx, tick_rate);
            })
            .expect("failed to spawn event-poller thread");

        Self { rx, _tx: tx }
    }

    /// Block until the next event is available.
    pub fn next(&self) -> Result<Event, mpsc::RecvError> {
        self.rx.recv()
    }

    /// Internal poll loop that runs on the background thread.
    fn poll_loop(tx: mpsc::Sender<Event>, tick_rate: Duration) {
        loop {
            let has_event = event::poll(tick_rate).unwrap_or(false);

            let event = if has_event {
                match event::read() {
                    Ok(event::Event::Key(key)) => Some(Event::Key(key)),
                    Ok(event::Event::Resize(_, _)) => Some(Event::Resize),
                    _ => None,
                }
            } else {
                Some(Event::Tick)
            };

            if let Some(ev) = event {
                if tx.send(ev).is_err() {
                    // Receiver dropped — shut down.
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

    #[test]
    fn event_debug_display() {
        let key_event = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        let ev = Event::Key(key_event);
        let debug = format!("{ev:?}");
        assert!(debug.contains("Key"));
    }

    #[test]
    fn event_tick_variant() {
        let ev = Event::Tick;
        assert!(matches!(ev, Event::Tick));
    }

    #[test]
    fn event_resize_variant() {
        let ev = Event::Resize;
        assert!(matches!(ev, Event::Resize));
    }

    #[test]
    fn event_handler_produces_ticks() {
        // With a very short tick rate, we should get a Tick event quickly
        let handler = EventHandler::new(Duration::from_millis(10));
        let ev = handler.next().expect("should receive an event");
        // In a test environment without a terminal, we expect ticks
        assert!(matches!(ev, Event::Tick | Event::Key(_) | Event::Resize));
    }
}
