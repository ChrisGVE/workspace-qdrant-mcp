//! Item dispatch: routes a single unified queue item to the appropriate strategy handler.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

use crate::allowed_extensions::AllowedExtensions;
use crate::config::IngestionLimitsConfig;
use crate::context::ProcessingContext;
use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::monitoring::labels::cardinality::{bounded_file_type, bounded_language};
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;
use crate::tracing_gate::{tier_enabled, TraceTier};
use crate::tree_sitter::GrammarManager;
use crate::unified_queue_schema::{ItemType, UnifiedQueueItem};
use crate::{DocumentProcessor, EmbeddingGenerator};

use crate::unified_queue_processor::config::UnifiedProcessorConfig;
use crate::unified_queue_processor::error::UnifiedProcessorResult;
use crate::unified_queue_processor::UnifiedQueueProcessor;

/// Metadata key carrying the W3C `traceparent` across the enqueue->dequeue hop
/// (PRD B2). Mirror of `queue_operations::enqueue::TRACEPARENT_METADATA_KEY`.
const TRACEPARENT_METADATA_KEY: &str = "wqm_traceparent";

impl UnifiedQueueProcessor {
    /// Process a single unified queue item based on its type.
    ///
    /// This `queue.process_item` span is the per-item ingestion *root* (the PRD
    /// "ingest.file" root for file items): the extract/chunk/embed/upsert child
    /// spans nest under it. The producer-side `watch.detect` / `queue.enqueue`
    /// spans live on the watcher task and are joined to this span via a W3C
    /// `traceparent` LINK carried through the queue's `metadata` column.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        name = "queue.process_item",
        skip_all,
        fields(
            queue_id = %item.queue_id,
            item_type = ?item.item_type,
            op = ?item.op,
            collection = %item.collection,
            tenant_id = %item.tenant_id,
            wqm.op = tracing::field::Empty,
            wqm.file_type = tracing::field::Empty,
            wqm.language = tracing::field::Empty,
        )
    )]
    pub(crate) async fn process_item(
        queue_manager: &QueueManager,
        item: &UnifiedQueueItem,
        _config: &UnifiedProcessorConfig,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        allowed_extensions: &Arc<AllowedExtensions>,
        lexicon_manager: &Arc<LexiconManager>,
        search_db: &Option<Arc<SearchDbManager>>,
        graph_store: &Option<Arc<dyn crate::graph::GraphStore>>,
        grammar_manager: &Option<Arc<RwLock<GrammarManager>>>,
        ingestion_limits: &Arc<IngestionLimitsConfig>,
        keyword_embedding_generator: &Option<Arc<EmbeddingGenerator>>,
        tier2_tagger: &Option<Arc<crate::tagging::Tier2Tagger>>,
        concept_config: &Arc<crate::config::ConceptConfig>,
        narrative_config: &Arc<crate::config::NarrativeConfig>,
    ) -> UnifiedProcessorResult<()> {
        // B2: record bounded span attributes and link back to the producer span
        // across the queue hop. All of this is gated behind the Hot tier so it
        // is a single atomic load when tracing is off.
        if tier_enabled(TraceTier::Hot) {
            Self::record_item_span_attrs(item);
        }

        debug!(
            "Processing unified item: {} (type={:?}, op={:?}, collection={})",
            item.queue_id, item.item_type, item.op, item.collection
        );

        let ctx = build_processing_context(
            queue_manager,
            document_processor,
            embedding_generator,
            storage_client,
            lsp_manager,
            embedding_semaphore,
            allowed_extensions,
            lexicon_manager,
            search_db,
            graph_store,
            grammar_manager,
            ingestion_limits,
            keyword_embedding_generator,
            tier2_tagger,
            concept_config,
            narrative_config,
        );

        match item.item_type {
            ItemType::Text => {
                crate::strategies::processing::text::TextStrategy::process_content_item(&ctx, item).await
            }
            ItemType::File => {
                crate::strategies::processing::file::FileStrategy::process_file_item(&ctx, item).await
            }
            ItemType::Folder => {
                crate::strategies::processing::folder::FolderStrategy::process_folder_item(&ctx, item).await
            }
            ItemType::Tenant => {
                crate::strategies::processing::tenant::TenantStrategy::process_tenant_item(&ctx, item).await
            }
            ItemType::Doc => {
                crate::strategies::processing::tenant::TenantStrategy::process_doc_item(&ctx, item).await
            }
            ItemType::Url => {
                crate::strategies::processing::url::UrlStrategy::process_url_item(&ctx, item).await
            }
            ItemType::Website => {
                crate::strategies::processing::website::WebsiteStrategy::process_website_item(&ctx, item).await
            }
            ItemType::Collection => {
                crate::strategies::processing::collection::CollectionStrategy::process_collection_item(&ctx, item).await
            }
        }
    }

    /// Record the bounded `wqm.*` attributes on the current `queue.process_item`
    /// span and LINK it back to the producer span via the stored traceparent.
    /// Only called when the Hot trace tier is active.
    fn record_item_span_attrs(item: &UnifiedQueueItem) {
        let span = tracing::Span::current();
        span.record("wqm.op", tracing::field::display(&item.op));

        if let Some(fp) = &item.file_path {
            let path = std::path::Path::new(fp);
            let file_type = bounded_file_type(path);
            span.record("wqm.file_type", file_type);
            // `bounded_file_type` already maps the extension to a bounded
            // language label, so reuse it for `wqm.language`.
            span.record("wqm.language", bounded_language(file_type));
        }

        if let Some(tp) = trace_link_from_metadata(item.metadata.as_deref()) {
            crate::tracing_otel::link_current_to_traceparent(&tp);
        }
    }
}

/// Build the [`ProcessingContext`] for a queue item from its collaborators.
///
/// Extracted from [`UnifiedQueueProcessor::process_item`] to keep that dispatch
/// fn within the function-size budget after adding the B2 span plumbing.
#[allow(clippy::too_many_arguments)]
fn build_processing_context(
    queue_manager: &QueueManager,
    document_processor: &Arc<DocumentProcessor>,
    embedding_generator: &Arc<EmbeddingGenerator>,
    storage_client: &Arc<StorageClient>,
    lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
    embedding_semaphore: &Arc<tokio::sync::Semaphore>,
    allowed_extensions: &Arc<AllowedExtensions>,
    lexicon_manager: &Arc<LexiconManager>,
    search_db: &Option<Arc<SearchDbManager>>,
    graph_store: &Option<Arc<dyn crate::graph::GraphStore>>,
    grammar_manager: &Option<Arc<RwLock<GrammarManager>>>,
    ingestion_limits: &Arc<IngestionLimitsConfig>,
    keyword_embedding_generator: &Option<Arc<EmbeddingGenerator>>,
    tier2_tagger: &Option<Arc<crate::tagging::Tier2Tagger>>,
    concept_config: &Arc<crate::config::ConceptConfig>,
    narrative_config: &Arc<crate::config::NarrativeConfig>,
) -> ProcessingContext {
    let mut ctx = ProcessingContext::new(
        queue_manager.pool().clone(),
        Arc::new(queue_manager.clone()),
        Arc::clone(storage_client),
        Arc::clone(embedding_generator),
        Arc::clone(document_processor),
        Arc::clone(embedding_semaphore),
        Arc::clone(lexicon_manager),
        lsp_manager.clone(),
        search_db.clone(),
        Arc::clone(allowed_extensions),
    );
    if let Some(gs) = graph_store {
        ctx = ctx.with_graph_store(Arc::clone(gs));
    }
    if let Some(gm) = grammar_manager {
        ctx = ctx.with_grammar_manager(Arc::clone(gm));
    }
    ctx = ctx.with_ingestion_limits(Arc::clone(ingestion_limits));
    if let Some(kw_gen) = keyword_embedding_generator {
        ctx = ctx.with_keyword_embedding_generator(Arc::clone(kw_gen));
    }
    ctx = ctx.with_concept_config(Arc::clone(concept_config));
    ctx = ctx.with_narrative_config(Arc::clone(narrative_config));
    if let Some(tagger) = tier2_tagger {
        ctx = ctx.with_tier2_tagger(Arc::clone(tagger));
    }
    ctx
}

/// Extract the stored W3C `traceparent` from the item's `metadata` JSON, if the
/// metadata parses to an object containing a string `wqm_traceparent` value.
fn trace_link_from_metadata(metadata: Option<&str>) -> Option<String> {
    let raw = metadata?;
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    value
        .get(TRACEPARENT_METADATA_KEY)?
        .as_str()
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TP: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    #[test]
    fn extracts_traceparent_from_object_metadata() {
        let meta = format!(r#"{{"source":"watcher","{TRACEPARENT_METADATA_KEY}":"{TP}"}}"#);
        assert_eq!(trace_link_from_metadata(Some(&meta)), Some(TP.to_string()));
    }

    #[test]
    fn missing_or_invalid_metadata_yields_none() {
        assert_eq!(trace_link_from_metadata(None), None);
        assert_eq!(trace_link_from_metadata(Some("{}")), None);
        assert_eq!(trace_link_from_metadata(Some(r#"{"k":"v"}"#)), None);
        assert_eq!(trace_link_from_metadata(Some("not json")), None);
        assert_eq!(trace_link_from_metadata(Some("[1,2,3]")), None);
        // Non-string traceparent value is ignored.
        let meta = format!(r#"{{"{TRACEPARENT_METADATA_KEY}":42}}"#);
        assert_eq!(trace_link_from_metadata(Some(&meta)), None);
    }
}
