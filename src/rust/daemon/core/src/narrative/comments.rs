/// CodeComment and Docstring extraction from source code.
use std::path::Path;

use async_trait::async_trait;

use crate::graph::NodeType;

use super::{NarrativeExtractionResult, NarrativeExtractor};

pub struct CommentExtractor;

#[async_trait]
impl NarrativeExtractor for CommentExtractor {
    fn supported_node_types(&self) -> &[NodeType] {
        &[NodeType::CodeComment, NodeType::Docstring]
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
