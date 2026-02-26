//! Compile-time-gated telemetry macros
//!
//! Each macro checks the corresponding `telemetry-lN` feature flag at compile
//! time.  When the feature is absent the macro expands to nothing, achieving
//! true zero-cost.
//!
//! # Macros
//!
//! | Macro            | Purpose                                |
//! |------------------|----------------------------------------|
//! | `tel_span!`      | Create a named `tracing` span          |
//! | `tel_event!`     | Emit a `tracing` event (log)           |
//! | `tel_gauge!`     | Record a gauge metric as an event      |
//! | `tel_counter!`   | Increment a counter as an event        |

// ---------------------------------------------------------------------------
// tel_span! -- creates a tracing span gated by telemetry level feature
// ---------------------------------------------------------------------------

/// Create a `tracing` span gated by the telemetry level feature flag.
///
/// Expands to `tracing::info_span!` (L0-L1) or `tracing::debug_span!` (L2-L4)
/// when the corresponding feature is enabled, and to a disabled span otherwise.
///
/// # Usage
///
/// ```ignore
/// let _guard = tel_span!(L0, "subsystem_init").entered();
/// let _guard = tel_span!(L1, "process_batch", batch_size = 42).entered();
/// ```
#[macro_export]
macro_rules! tel_span {
    (L0, $name:expr $(, $($field:tt)*)?) => {{
        #[cfg(feature = "telemetry-l0")]
        {
            tracing::info_span!(concat!("tel::", $name) $(, $($field)*)?)
        }
        #[cfg(not(feature = "telemetry-l0"))]
        {
            tracing::Span::none()
        }
    }};
    (L1, $name:expr $(, $($field:tt)*)?) => {{
        #[cfg(feature = "telemetry-l1")]
        {
            tracing::info_span!(concat!("tel::", $name) $(, $($field)*)?)
        }
        #[cfg(not(feature = "telemetry-l1"))]
        {
            tracing::Span::none()
        }
    }};
    (L2, $name:expr $(, $($field:tt)*)?) => {{
        #[cfg(feature = "telemetry-l2")]
        {
            tracing::debug_span!(concat!("tel::", $name) $(, $($field)*)?)
        }
        #[cfg(not(feature = "telemetry-l2"))]
        {
            tracing::Span::none()
        }
    }};
    (L3, $name:expr $(, $($field:tt)*)?) => {{
        #[cfg(feature = "telemetry-l3")]
        {
            tracing::debug_span!(concat!("tel::", $name) $(, $($field)*)?)
        }
        #[cfg(not(feature = "telemetry-l3"))]
        {
            tracing::Span::none()
        }
    }};
    (L4, $name:expr $(, $($field:tt)*)?) => {{
        #[cfg(feature = "telemetry-l4")]
        {
            tracing::trace_span!(concat!("tel::", $name) $(, $($field)*)?)
        }
        #[cfg(not(feature = "telemetry-l4"))]
        {
            tracing::Span::none()
        }
    }};
}

// ---------------------------------------------------------------------------
// tel_event! -- emits a tracing event gated by telemetry level feature
// ---------------------------------------------------------------------------

/// Emit a `tracing` event gated by the telemetry level feature flag.
///
/// Expands to `tracing::info!` (L0-L1), `tracing::debug!` (L2-L3), or
/// `tracing::trace!` (L4) when the feature is enabled, and to nothing
/// otherwise.
///
/// # Usage
///
/// ```ignore
/// tel_event!(L0, "storage module started");
/// tel_event!(L2, queue_depth = depth, "queue snapshot");
/// ```
#[macro_export]
macro_rules! tel_event {
    (L0, $($arg:tt)+) => {
        #[cfg(feature = "telemetry-l0")]
        tracing::info!(telemetry_level = "L0", $($arg)+);
    };
    (L1, $($arg:tt)+) => {
        #[cfg(feature = "telemetry-l1")]
        tracing::info!(telemetry_level = "L1", $($arg)+);
    };
    (L2, $($arg:tt)+) => {
        #[cfg(feature = "telemetry-l2")]
        tracing::debug!(telemetry_level = "L2", $($arg)+);
    };
    (L3, $($arg:tt)+) => {
        #[cfg(feature = "telemetry-l3")]
        tracing::debug!(telemetry_level = "L3", $($arg)+);
    };
    (L4, $($arg:tt)+) => {
        #[cfg(feature = "telemetry-l4")]
        tracing::trace!(telemetry_level = "L4", $($arg)+);
    };
}

// ---------------------------------------------------------------------------
// tel_gauge! -- records a named gauge value
// ---------------------------------------------------------------------------

/// Record a gauge metric as a `tracing` event.
///
/// The event carries the field `tel_gauge = name` and `value = val`.
///
/// ```ignore
/// tel_gauge!(L2, "queue_depth", queue.len());
/// ```
#[macro_export]
macro_rules! tel_gauge {
    (L0, $name:expr, $val:expr) => {
        #[cfg(feature = "telemetry-l0")]
        tracing::info!(
            telemetry_level = "L0",
            tel_gauge = $name,
            value = $val,
            "gauge"
        );
    };
    (L1, $name:expr, $val:expr) => {
        #[cfg(feature = "telemetry-l1")]
        tracing::info!(
            telemetry_level = "L1",
            tel_gauge = $name,
            value = $val,
            "gauge"
        );
    };
    (L2, $name:expr, $val:expr) => {
        #[cfg(feature = "telemetry-l2")]
        tracing::debug!(
            telemetry_level = "L2",
            tel_gauge = $name,
            value = $val,
            "gauge"
        );
    };
    (L3, $name:expr, $val:expr) => {
        #[cfg(feature = "telemetry-l3")]
        tracing::debug!(
            telemetry_level = "L3",
            tel_gauge = $name,
            value = $val,
            "gauge"
        );
    };
    (L4, $name:expr, $val:expr) => {
        #[cfg(feature = "telemetry-l4")]
        tracing::trace!(
            telemetry_level = "L4",
            tel_gauge = $name,
            value = $val,
            "gauge"
        );
    };
}

// ---------------------------------------------------------------------------
// tel_counter! -- increments a named counter
// ---------------------------------------------------------------------------

/// Increment a counter as a `tracing` event.
///
/// Emits a field `tel_counter = name` with `delta = amount`.
///
/// ```ignore
/// tel_counter!(L2, "files_processed", 1);
/// tel_counter!(L3, "bytes_read", chunk.len());
/// ```
#[macro_export]
macro_rules! tel_counter {
    (L0, $name:expr, $delta:expr) => {
        #[cfg(feature = "telemetry-l0")]
        tracing::info!(
            telemetry_level = "L0",
            tel_counter = $name,
            delta = $delta,
            "counter"
        );
    };
    (L1, $name:expr, $delta:expr) => {
        #[cfg(feature = "telemetry-l1")]
        tracing::info!(
            telemetry_level = "L1",
            tel_counter = $name,
            delta = $delta,
            "counter"
        );
    };
    (L2, $name:expr, $delta:expr) => {
        #[cfg(feature = "telemetry-l2")]
        tracing::debug!(
            telemetry_level = "L2",
            tel_counter = $name,
            delta = $delta,
            "counter"
        );
    };
    (L3, $name:expr, $delta:expr) => {
        #[cfg(feature = "telemetry-l3")]
        tracing::debug!(
            telemetry_level = "L3",
            tel_counter = $name,
            delta = $delta,
            "counter"
        );
    };
    (L4, $name:expr, $delta:expr) => {
        #[cfg(feature = "telemetry-l4")]
        tracing::trace!(
            telemetry_level = "L4",
            tel_counter = $name,
            delta = $delta,
            "counter"
        );
    };
}
