//! Custom JSON event formatter that injects OpenTelemetry `trace_id` and
//! `span_id` into every structured log record (PRD B3, task 66).
//!
//! Grafana's logs↔traces navigation (Loki ↔ Tempo) keys off `trace_id`/
//! `span_id` fields present on each log line, formatted as the lowercase hex
//! strings used by the W3C `traceparent` header (32 hex chars for the trace,
//! 16 for the span). This formatter reads those ids from the *current* span's
//! `OtelData` extension — populated by the `tracing-opentelemetry` layer at
//! span creation — so there is no per-line `Context::current()` global lookup
//! on the hot logging path.
//!
//! When no OpenTelemetry layer is installed (OTLP disabled) or the event is
//! emitted outside any span, `OtelData` is absent and the ids are omitted
//! entirely — no fabricated/zeroed ids — preserving the zero-overhead disabled
//! path and giving Grafana a clean "no trace" signal.

use std::fmt;

use chrono::{SecondsFormat, Utc};
use serde_json::{Map, Value};
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

/// JSON event formatter that augments structured log records with the active
/// span's `trace_id`/`span_id` (PRD B3). Install via
/// `fmt::layer().event_format(TraceJsonFormat)`.
#[derive(Debug, Clone, Default)]
pub struct TraceJsonFormat;

impl<S, N> FormatEvent<S, N> for TraceJsonFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let meta = event.metadata();

        let mut record = Map::new();
        record.insert(
            "timestamp".to_string(),
            Value::String(Utc::now().to_rfc3339_opts(SecondsFormat::Micros, true)),
        );
        record.insert(
            "level".to_string(),
            Value::String(meta.level().as_str().to_string()),
        );
        record.insert(
            "target".to_string(),
            Value::String(meta.target().to_string()),
        );

        let thread = std::thread::current();
        record.insert(
            "threadId".to_string(),
            Value::String(format!("{:?}", thread.id())),
        );
        if let Some(name) = thread.name() {
            record.insert("threadName".to_string(), Value::String(name.to_string()));
        }

        // Event fields (including the `message`) are collected under `fields`
        // to match the shape `tracing_subscriber`'s built-in JSON formatter
        // produces, so downstream parsers keep working.
        let mut fields = Map::new();
        event.record(&mut JsonVisitor(&mut fields));
        record.insert("fields".to_string(), Value::Object(fields));

        // Inject trace/span ids from the current span's OTel context.
        if let Some(span) = ctx.lookup_current() {
            record.insert("span".to_string(), Value::String(span.name().to_string()));
            if let Some((trace_id, span_id)) = otel_ids(&span) {
                record.insert("trace_id".to_string(), Value::String(trace_id));
                record.insert("span_id".to_string(), Value::String(span_id));
            }
        }

        let line = serde_json::to_string(&Value::Object(record)).map_err(|_| fmt::Error)?;
        writer.write_str(&line)?;
        writeln!(writer)
    }
}

/// Read the `trace_id`/`span_id` for a span from its `tracing-opentelemetry`
/// `OtelData` extension as lowercase hex (W3C `traceparent` format). Returns
/// `None` when no OTel layer populated the span (OTLP disabled) or the ids are
/// invalid. The trace id is taken from the span builder when present (root
/// spans) and otherwise inherited from the parent context (child spans); the
/// span id is always the current span's own id.
fn otel_ids<S>(span: &tracing_subscriber::registry::SpanRef<'_, S>) -> Option<(String, String)>
where
    S: for<'a> LookupSpan<'a>,
{
    use opentelemetry::trace::{SpanId, TraceId};
    use tracing_opentelemetry::OtelData;

    let ext = span.extensions();
    let data = ext.get::<OtelData>()?;

    // `OtelData::trace_id()` resolves the trace id from the span builder for
    // roots and inherits it from the parent context for child spans; both are
    // `None` when the span carries no valid OTel context.
    let trace_id = data.trace_id()?;
    let span_id = data.span_id()?;
    if trace_id == TraceId::INVALID || span_id == SpanId::INVALID {
        return None;
    }

    // `TraceId`/`SpanId` Display impls emit fixed-width lowercase hex
    // (32 / 16 chars) — exactly the W3C `traceparent` field format.
    Some((trace_id.to_string(), span_id.to_string()))
}

/// Collects `tracing` event fields into a JSON object.
struct JsonVisitor<'a>(&'a mut Map<String, Value>);

impl JsonVisitor<'_> {
    fn insert(&mut self, field: &Field, value: Value) {
        self.0.insert(field.name().to_string(), value);
    }
}

impl Visit for JsonVisitor<'_> {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.insert(field, serde_json::json!(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.insert(field, serde_json::json!(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.insert(field, serde_json::json!(value));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.insert(field, Value::Bool(value));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.insert(field, Value::String(value.to_string()));
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        self.insert(field, Value::String(value.to_string()));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.insert(field, Value::String(format!("{:?}", value)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use opentelemetry::trace::TraceContextExt;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::Registry;

    /// In-memory `MakeWriter` capturing log output for assertions.
    #[derive(Clone)]
    struct BufWriter(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for BufWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for BufWriter {
        type Writer = BufWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Build a registry subscriber with a real OTel layer (no exporter) so
    /// `OtelData` is populated, plus the B3 JSON formatter writing to `buf`.
    fn make_subscriber(
        tracer: opentelemetry_sdk::trace::SdkTracer,
        buf: Arc<Mutex<Vec<u8>>>,
    ) -> impl tracing::Subscriber + Send + Sync {
        let otel = tracing_opentelemetry::layer().with_tracer(tracer);
        let fmt_layer = tracing_subscriber::fmt::layer()
            .event_format(TraceJsonFormat)
            .with_writer(BufWriter(buf));
        Registry::default().with(otel).with(fmt_layer)
    }

    fn find_json(out: &str, needle: &str) -> Value {
        let line = out
            .lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("no log line containing {needle:?} in: {out}"));
        serde_json::from_str(line).expect("log line is valid JSON")
    }

    #[test]
    fn injects_ids_inside_span_matching_active_span() {
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let buf = Arc::new(Mutex::new(Vec::new()));
        let sub = make_subscriber(tracer, buf.clone());

        let (want_trace, want_span) = tracing::subscriber::with_default(sub, || {
            let span = tracing::info_span!("work");
            let _enter = span.enter();
            tracing::info!("inside-marker");
            let sc = span.context().span().span_context().clone();
            (sc.trace_id().to_string(), sc.span_id().to_string())
        });

        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        let v = find_json(&out, "inside-marker");

        let tid = v["trace_id"].as_str().expect("trace_id present");
        let sid = v["span_id"].as_str().expect("span_id present");

        assert_eq!(tid.len(), 32, "trace_id is 32 hex chars");
        assert_eq!(sid.len(), 16, "span_id is 16 hex chars");
        assert!(tid
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert!(sid
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(
            tid,
            "0".repeat(32),
            "trace_id is not the invalid all-zero id"
        );
        assert_ne!(
            sid,
            "0".repeat(16),
            "span_id is not the invalid all-zero id"
        );

        // IDs in the log line match the active span's own OTel context.
        assert_eq!(tid, want_trace);
        assert_eq!(sid, want_span);
    }

    #[test]
    fn omits_ids_outside_any_span() {
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let buf = Arc::new(Mutex::new(Vec::new()));
        let sub = make_subscriber(tracer, buf.clone());

        tracing::subscriber::with_default(sub, || {
            tracing::info!("outside-marker");
        });

        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        let v = find_json(&out, "outside-marker");

        assert!(v.get("trace_id").is_none(), "no fabricated trace_id");
        assert!(v.get("span_id").is_none(), "no fabricated span_id");
    }

    #[test]
    fn message_is_recorded_under_fields() {
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let buf = Arc::new(Mutex::new(Vec::new()));
        let sub = make_subscriber(tracer, buf.clone());

        tracing::subscriber::with_default(sub, || {
            tracing::info!(answer = 42, "payload-marker");
        });

        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        let v = find_json(&out, "payload-marker");

        assert_eq!(v["fields"]["message"].as_str(), Some("payload-marker"));
        assert_eq!(v["fields"]["answer"].as_i64(), Some(42));
        assert_eq!(v["level"].as_str(), Some("INFO"));
    }
}
