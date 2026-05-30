//! REAL LSP enrichment trace — NO simulation.
//!
//! Drives a real `rust-analyzer` process against a real temporary Cargo
//! project, runs the real (instrumented) enrichment path, and captures the
//! real `tracing` spans (real wall-clock durations, real recorded fields)
//! that the daemon also exports to OTLP/Tempo in production. The captured
//! spans are rendered as a trace tree.
//!
//! Ignored by default: it requires a real `rust-analyzer` on PATH and is
//! slow (language-server startup + indexing). Run explicitly:
//!
//!   cargo test -p workspace-qdrant-core --test lsp_real_trace -- --ignored --nocapture
//!
//! On a host without rust-analyzer the test exits early with a SKIP notice.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use workspace_qdrant_core::{Language, LanguageServerManager, ProjectLspConfig};

// ─── A minimal span-capturing tracing layer ──────────────────────────────

#[derive(Clone, Debug)]
struct SpanRecord {
    id: u64,
    parent: Option<u64>,
    name: String,
    target: String,
    fields: String,
    duration_us: u128,
}

#[derive(Default)]
struct Captured {
    finished: Vec<SpanRecord>,
}

struct CaptureLayer {
    store: Arc<Mutex<Captured>>,
}

/// Per-span timing stored in the registry's span extensions.
struct Timing {
    start: Instant,
}

/// Accumulated `key=value` field text for a span (updated by later
/// `Span::record` calls, e.g. the deferred `enrichment_status`).
struct FieldStore {
    text: String,
}

struct FieldVisitor<'a>(&'a mut String);

impl tracing::field::Visit for FieldVisitor<'_> {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if !self.0.is_empty() {
            self.0.push(' ');
        }
        self.0.push_str(&format!("{}={}", field.name(), value));
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if !self.0.is_empty() {
            self.0.push(' ');
        }
        self.0.push_str(&format!("{}={:?}", field.name(), value));
    }
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("span exists");
        let mut fields = String::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        let mut ext = span.extensions_mut();
        ext.insert(Timing {
            start: Instant::now(),
        });
        ext.insert(FieldStore { text: fields });
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("span exists");
        let mut ext = span.extensions_mut();
        if let Some(fs) = ext.get_mut::<FieldStore>() {
            values.record(&mut FieldVisitor(&mut fs.text));
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).expect("span exists");
        let ext = span.extensions();
        let duration_us = ext
            .get::<Timing>()
            .map(|t| t.start.elapsed().as_micros())
            .unwrap_or(0);
        let fields = ext
            .get::<FieldStore>()
            .map(|f| f.text.clone())
            .unwrap_or_default();
        let parent = span.parent().map(|p| p.id().into_u64());
        self.store.lock().unwrap().finished.push(SpanRecord {
            id: id.into_u64(),
            parent,
            name: span.name().to_string(),
            target: span.metadata().target().to_string(),
            fields,
            duration_us,
        });
    }
}

fn render_trace(spans: &[SpanRecord]) {
    use std::collections::HashSet;
    let ids: HashSet<u64> = spans.iter().map(|s| s.id).collect();

    fn print_subtree(spans: &[SpanRecord], parent: Option<u64>, depth: usize) {
        for s in spans.iter().filter(|s| s.parent == parent) {
            let ms = s.duration_us as f64 / 1000.0;
            let fields = if s.fields.is_empty() {
                String::new()
            } else {
                format!("  {{{}}}", s.fields)
            };
            eprintln!(
                "│ {}{} [{:.2} ms] ({}){}",
                "    ".repeat(depth),
                s.name,
                ms,
                s.target,
                fields
            );
            print_subtree(spans, Some(s.id), depth + 1);
        }
    }

    eprintln!("\n┌─ REAL LSP trace — captured tracing spans ─────────────────────");
    if spans.is_empty() {
        eprintln!("│ (no spans captured)");
    } else {
        // Roots: no parent, or parent span not present in this capture.
        for root in spans
            .iter()
            .filter(|s| s.parent.is_none() || !ids.contains(&s.parent.unwrap()))
        {
            let ms = root.duration_us as f64 / 1000.0;
            let fields = if root.fields.is_empty() {
                String::new()
            } else {
                format!("  {{{}}}", root.fields)
            };
            eprintln!("│ {} [{:.2} ms] ({}){}", root.name, ms, root.target, fields);
            print_subtree(spans, Some(root.id), 1);
        }
    }
    eprintln!("└───────────────────────────────────────────────────────────────\n");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a real rust-analyzer on PATH; slow (server startup + indexing)"]
async fn real_lsp_enrichment_emits_real_trace() {
    if which::which("rust-analyzer").is_err() {
        eprintln!("SKIP: rust-analyzer not found on PATH — cannot produce a real trace.");
        return;
    }

    // Install the capture layer for the duration of this test only.
    let store = Arc::new(Mutex::new(Captured::default()));
    let subscriber = tracing_subscriber::registry().with(CaptureLayer {
        store: store.clone(),
    });
    let _guard = tracing::subscriber::set_default(subscriber);

    // Build a real, minimal Cargo project rust-analyzer can index.
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"trace_demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    let lib = root.join("src").join("lib.rs");
    std::fs::write(
        &lib,
        // `add` is defined on line 0 and called twice in `compute`.
        "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\npub fn compute() -> i32 {\n    let x = add(1, 2);\n    add(x, 3)\n}\n",
    )
    .unwrap();

    let mut mgr = LanguageServerManager::new(ProjectLspConfig::default())
        .await
        .expect("manager");
    mgr.initialize().await.expect("initialize");

    let project_id = "trace-demo";
    mgr.mark_project_active(project_id).await;

    if let Err(e) = mgr.start_server(project_id, Language::Rust, root).await {
        eprintln!("SKIP: could not start rust-analyzer: {e}");
        return;
    }
    eprintln!("rust-analyzer started; warming up (indexing the workspace)…");

    // rust-analyzer indexes the workspace from disk asynchronously after the
    // initialize handshake. Poll the real `add` definition (line 0, char 7)
    // until call-site references appear or we give up. These warm-up spans are
    // discarded so the rendered trace shows only the final, real enrichment.
    for attempt in 1..=15 {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let n = mgr.get_references(&lib, 0, 7).await.unwrap_or_default().len();
        eprintln!("  warm-up attempt {attempt}: {n} reference(s)");
        if n > 0 {
            break;
        }
    }

    // Discard warm-up spans; capture only the final real enrichment below.
    store.lock().unwrap().finished.clear();

    // (1) Real enrichment via the production entry-point — emits the
    //     `lsp.enrich_chunk` parent span + child query spans.
    let enrichment = mgr
        .enrich_chunk(project_id, &lib, "add", 0, 0, true)
        .await;

    // (2) Direct reference query at the real `add` definition (line 0,
    //     char 7) to surface real call-site references.
    let refs = mgr
        .get_references(&lib, 0, 7)
        .await
        .unwrap_or_default();

    // (3) REAL call-hierarchy resolution at `compute` (line 4, char 7), which
    //     calls `add` twice — proves precise, resolved callee extraction (#3).
    let calls = mgr
        .resolved_outgoing_calls(&lib, 4, 7)
        .await
        .unwrap_or_default();

    eprintln!("\n── Real enrichment result ──────────────────────────────────────");
    eprintln!("  enrichment_status : {:?}", enrichment.enrichment_status);
    eprintln!("  references (enrich): {}", enrichment.references.len());
    eprintln!(
        "  type_info         : {:?}",
        enrichment.type_info.as_ref().map(|t| &t.type_signature)
    );
    eprintln!("  references @add   : {} call-site(s)", refs.len());
    for r in &refs {
        eprintln!("      → {}:{}:{}", r.file, r.line, r.column);
    }
    eprintln!("  outgoing calls @compute (call hierarchy): {}", calls.len());
    for c in &calls {
        eprintln!("      → {} @ {}:{}", c.name, c.file, c.line);
    }

    let _ = mgr.shutdown().await;
    drop(_guard); // ensure no more spans are routed before we read the store

    let finished = store.lock().unwrap().finished.clone();
    render_trace(&finished);

    // ── Assertions on REAL captured data (not values) ───────────────────
    let enrich = finished
        .iter()
        .find(|s| s.name == "lsp.enrich_chunk")
        .expect("real lsp.enrich_chunk span must be captured");
    assert!(
        enrich.fields.contains("language=rust"),
        "enrich span must record the real resolved language; fields = {}",
        enrich.fields
    );
    assert!(
        enrich.fields.contains("enrichment_status="),
        "enrich span must record the final status; fields = {}",
        enrich.fields
    );
    assert!(
        enrich.duration_us > 0,
        "a real span must carry a measured (non-zero) duration"
    );
}
