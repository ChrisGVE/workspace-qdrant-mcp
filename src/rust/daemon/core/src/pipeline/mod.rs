//! Chain of Responsibility: pipeline processing with explicit handler ordering.
//!
//! Pipelines process items through an ordered sequence of handlers.
//! Each handler can:
//! - `Continue` — pass to the next handler
//! - `Skip(reason)` — stop the chain early without error (e.g., filtered out)
//! - `Abort(error)` — stop the chain with an error
//!
//! # Future submodules
//! - `file_ingestion` — file processing chain
//! - `file_event` — FS event → queue chain
//! - `handlers/` — individual chain handlers

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use serde_json::Value;

/// Result of a single pipeline handler invocation.
pub enum PipelineResult {
    /// Processing should continue to the next handler.
    Continue,
    /// Skip the remaining handlers without error (e.g., file filtered out).
    Skip(String),
    /// Abort the pipeline with an error.
    Abort(Box<dyn std::error::Error + Send + Sync>),
}

/// Mutable context threaded through pipeline handlers.
///
/// Each handler reads/writes to this context, building up the processing
/// state incrementally. The final handler typically stores the results.
pub struct PipelineContext {
    /// Arbitrary key-value state shared between handlers.
    /// Handlers use well-known keys (defined as constants) to communicate.
    pub state: HashMap<String, Value>,

    /// File path being processed (if applicable).
    pub file_path: Option<PathBuf>,

    /// Accumulated text chunks from extraction/chunking.
    pub chunks: Vec<String>,

    /// Dense embeddings computed for each chunk.
    pub dense_vectors: Vec<Vec<f32>>,

    /// Sparse vectors computed for each chunk.
    pub sparse_vectors: Vec<Option<HashMap<u32, f32>>>,

    /// Point IDs assigned to each chunk.
    pub point_ids: Vec<String>,

    /// Whether the pipeline has been aborted or skipped.
    pub terminated: bool,

    /// Skip reason, if the pipeline was skipped.
    pub skip_reason: Option<String>,
}

impl PipelineContext {
    pub fn new() -> Self {
        Self {
            state: HashMap::new(),
            file_path: None,
            chunks: Vec::new(),
            dense_vectors: Vec::new(),
            sparse_vectors: Vec::new(),
            point_ids: Vec::new(),
            terminated: false,
            skip_reason: None,
        }
    }

    /// Create a context for file processing.
    pub fn for_file(path: PathBuf) -> Self {
        let mut ctx = Self::new();
        ctx.file_path = Some(path);
        ctx
    }

    /// Set a typed state value.
    pub fn set_state(&mut self, key: impl Into<String>, value: Value) {
        self.state.insert(key.into(), value);
    }

    /// Get a state value.
    pub fn get_state(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self::new()
    }
}

/// A single handler in a processing pipeline.
#[async_trait]
pub trait PipelineHandler: Send + Sync {
    /// Process the context and return a result indicating whether to continue.
    async fn handle(&self, ctx: &mut PipelineContext) -> PipelineResult;

    /// Human-readable name for logging.
    fn name(&self) -> &'static str;
}

/// An ordered pipeline of handlers.
pub struct Pipeline {
    handlers: Vec<Box<dyn PipelineHandler>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Add a handler to the end of the pipeline.
    pub fn add_handler(&mut self, handler: Box<dyn PipelineHandler>) {
        self.handlers.push(handler);
    }

    /// Run all handlers in order. Stops on `Skip` or `Abort`.
    pub async fn run(&self, ctx: &mut PipelineContext) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for handler in &self.handlers {
            tracing::debug!("Pipeline handler: {}", handler.name());
            match handler.handle(ctx).await {
                PipelineResult::Continue => continue,
                PipelineResult::Skip(reason) => {
                    tracing::debug!("Pipeline skipped at {}: {}", handler.name(), reason);
                    ctx.terminated = true;
                    ctx.skip_reason = Some(reason);
                    return Ok(());
                }
                PipelineResult::Abort(err) => {
                    tracing::error!("Pipeline aborted at {}: {}", handler.name(), err);
                    ctx.terminated = true;
                    return Err(err);
                }
            }
        }
        Ok(())
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SkipHandler {
        reason: String,
    }
    #[async_trait]
    impl PipelineHandler for SkipHandler {
        async fn handle(&self, _ctx: &mut PipelineContext) -> PipelineResult {
            PipelineResult::Skip(self.reason.clone())
        }
        fn name(&self) -> &'static str {
            "skip"
        }
    }

    struct CountHandler;
    #[async_trait]
    impl PipelineHandler for CountHandler {
        async fn handle(&self, ctx: &mut PipelineContext) -> PipelineResult {
            let count = ctx
                .get_state("count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            ctx.set_state("count", serde_json::json!(count + 1));
            PipelineResult::Continue
        }
        fn name(&self) -> &'static str {
            "count"
        }
    }

    #[tokio::test]
    async fn test_pipeline_runs_all_handlers() {
        let mut pipeline = Pipeline::new();
        pipeline.add_handler(Box::new(CountHandler));
        pipeline.add_handler(Box::new(CountHandler));
        pipeline.add_handler(Box::new(CountHandler));

        let mut ctx = PipelineContext::new();
        pipeline.run(&mut ctx).await.unwrap();

        assert_eq!(ctx.get_state("count").unwrap().as_u64().unwrap(), 3);
        assert!(!ctx.terminated);
    }

    #[tokio::test]
    async fn test_pipeline_stops_on_skip() {
        let mut pipeline = Pipeline::new();
        pipeline.add_handler(Box::new(CountHandler));
        pipeline.add_handler(Box::new(SkipHandler {
            reason: "filtered".to_string(),
        }));
        pipeline.add_handler(Box::new(CountHandler)); // should not run

        let mut ctx = PipelineContext::new();
        pipeline.run(&mut ctx).await.unwrap();

        assert_eq!(ctx.get_state("count").unwrap().as_u64().unwrap(), 1);
        assert!(ctx.terminated);
        assert_eq!(ctx.skip_reason.as_deref(), Some("filtered"));
    }

    #[tokio::test]
    async fn test_pipeline_empty() {
        let pipeline = Pipeline::new();
        let mut ctx = PipelineContext::new();
        pipeline.run(&mut ctx).await.unwrap();
        assert!(!ctx.terminated);
    }

    #[test]
    fn test_pipeline_context_for_file() {
        let ctx = PipelineContext::for_file(PathBuf::from("/tmp/test.rs"));
        assert_eq!(
            ctx.file_path.as_deref(),
            Some(std::path::Path::new("/tmp/test.rs"))
        );
    }
}
