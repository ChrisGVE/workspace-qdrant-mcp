//! W3C trace-context propagation across the enqueue->dequeue queue hop (PRD B2).
//!
//! The unified queue is a temporal/async boundary: a file event is enqueued by
//! the watcher (one span) and dequeued + processed later by the queue processor
//! (a different span, possibly a different task). OpenTelemetry parent/child
//! linkage doesn't survive that hop, so we encode the producer span's context as
//! a W3C `traceparent` string (via the globally-installed `TraceContextPropagator`),
//! carry it through the queue's `metadata` column, and on the consumer side add
//! a span *link* back to the producer.
//!
//! Both helpers are no-ops when tracing is disabled (no active OTel span / an
//! invalid stored context), so they cost ~nothing on the disabled path.

use std::collections::HashMap;

/// Capture the current tracing span's OTel context as a W3C `traceparent`
/// string, for storing across an async/temporal boundary (the enqueue->dequeue
/// queue hop). Returns `None` when there is no active/valid OTel span context
/// (e.g. tracing disabled).
pub fn current_traceparent() -> Option<String> {
    use opentelemetry::global;
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    let cx = tracing::Span::current().context();
    let mut carrier = HashMap::<String, String>::new();
    global::get_text_map_propagator(|p| p.inject_context(&cx, &mut carrier));
    carrier.remove("traceparent")
}

/// Add a span LINK from the *current* span to the context encoded in a stored
/// W3C `traceparent` string (the producer side of the enqueue->dequeue hop).
/// No-op if the traceparent is malformed / yields an invalid span context.
pub fn link_current_to_traceparent(traceparent: &str) {
    use opentelemetry::global;
    use opentelemetry::trace::TraceContextExt;
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    let mut carrier = HashMap::<String, String>::new();
    carrier.insert("traceparent".to_string(), traceparent.to_string());
    let cx = global::get_text_map_propagator(|p| p.extract(&carrier));
    let span_cx = cx.span().span_context().clone();
    if span_cx.is_valid() {
        tracing::Span::current().add_link(span_cx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_traceparent_none_without_active_span() {
        // With no active OTel span context, capture yields None.
        assert!(current_traceparent().is_none());
    }

    #[test]
    fn link_to_malformed_traceparent_is_noop() {
        // Malformed / empty input must not panic and must be a no-op.
        link_current_to_traceparent("");
        link_current_to_traceparent("not-a-traceparent");
        link_current_to_traceparent("00-deadbeef");
        link_current_to_traceparent("99-0000000000000000000000000000000000-0000000000000000-00");
    }
}
