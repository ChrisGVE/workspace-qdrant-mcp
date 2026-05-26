/// ELABORATES back-link creation from concept nodes to detailed narrative nodes.
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
