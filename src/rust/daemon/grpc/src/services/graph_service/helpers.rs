//! Helper utilities for the GraphService.

use tonic::Status;
use workspace_qdrant_core::graph::EdgeType;

/// Parse edge_types from proto string list, returning None for "all".
pub(crate) fn parse_edge_type_filter(types: &[String]) -> Result<Option<Vec<String>>, Status> {
    if types.is_empty() {
        return Ok(None);
    }
    // Validate all types are known
    for t in types {
        if EdgeType::from_str(t).is_none() {
            return Err(Status::invalid_argument(format!(
                "unknown edge type: {}",
                t
            )));
        }
    }
    Ok(Some(types.to_vec()))
}
