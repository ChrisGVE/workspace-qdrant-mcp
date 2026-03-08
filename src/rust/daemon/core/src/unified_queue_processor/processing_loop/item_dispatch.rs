//! Item dispatch: routes a single unified queue item to the appropriate strategy handler.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

use crate::allowed_extensions::AllowedExtensions;
use crate::config::IngestionLimitsConfig;
use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;
use crate::tree_sitter::GrammarManager;
use crate::unified_queue_schema::{ItemType, UnifiedQueueItem};
use crate::{DocumentProcessor, EmbeddingGenerator};

use crate::unified_queue_processor::config::UnifiedProcessorConfig;
use crate::unified_queue_processor::error::UnifiedProcessorResult;
use crate::unified_queue_processor::UnifiedQueueProcessor;

impl UnifiedQueueProcessor {
    /// Process a single unified queue item based on its type
    #[allow(clippy::too_many_arguments)]
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
        graph_store: &Option<crate::graph::SharedGraphStore<crate::graph::SqliteGraphStore>>,
        grammar_manager: &Option<Arc<RwLock<GrammarManager>>>,
        ingestion_limits: &Arc<IngestionLimitsConfig>,
    ) -> UnifiedProcessorResult<()> {
        debug!(
            "Processing unified item: {} (type={:?}, op={:?}, collection={})",
            item.queue_id, item.item_type, item.op, item.collection
        );

        let mut ctx = crate::context::ProcessingContext::new(
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
            ctx = ctx.with_graph_store(gs.clone());
        }
        if let Some(gm) = grammar_manager {
            ctx = ctx.with_grammar_manager(Arc::clone(gm));
        }
        ctx = ctx.with_ingestion_limits(Arc::clone(ingestion_limits));

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
}
