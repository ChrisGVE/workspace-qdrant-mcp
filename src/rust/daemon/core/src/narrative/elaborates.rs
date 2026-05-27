/// ELABORATES back-link creation from concept nodes to detailed narrative nodes.
///
/// The extractor is intentionally a no-op: ELABORATES edges are not created
/// during single-file extraction because they require cross-document analysis
/// (comparing depth levels of nodes covering the same concept). The actual
/// edge creation happens in [`ElaboratesMaintenanceTask`] which runs during
/// idle periods and queries the graph for matching `COVERS_TOPIC` pairs.
use std::path::Path;

use async_trait::async_trait;

use crate::graph::NodeType;

use super::{NarrativeExtractionResult, NarrativeExtractor};

pub struct ElaboratesExtractor;

#[async_trait]
impl NarrativeExtractor for ElaboratesExtractor {
    fn supported_node_types(&self) -> &[NodeType] {
        &[]
    }

    async fn extract(
        &self,
        _tenant_id: &str,
        _file_path: &Path,
        _content: &str,
        _language: Option<&str>,
    ) -> NarrativeExtractionResult {
        NarrativeExtractionResult::default()
    }
}
