/// DocumentSection extraction from markdown and structured documents.
///
/// Splits documents into heading-delimited sections, creating
/// DocumentSection graph nodes for each.
use std::path::Path;

use async_trait::async_trait;

use crate::graph::NodeType;

use super::{NarrativeExtractionResult, NarrativeExtractor};

pub struct SectionExtractor;

#[async_trait]
impl NarrativeExtractor for SectionExtractor {
    fn supported_node_types(&self) -> &[NodeType] {
        &[NodeType::DocumentSection, NodeType::LibrarySection]
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
