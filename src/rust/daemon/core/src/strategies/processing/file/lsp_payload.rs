//! LSP enrichment payload construction.
//!
//! Serializes LSP enrichment data (references, type info, imports, definitions)
//! into Qdrant point payload fields.

use std::collections::HashMap;

use crate::lsp::{EnrichmentStatus, LspEnrichment};

/// Add LSP enrichment data to a point payload.
pub(crate) fn add_lsp_enrichment_to_payload(
    payload: &mut HashMap<String, serde_json::Value>,
    enrichment: &LspEnrichment,
) {
    // Add enrichment status (lowercase for consistent metadata filtering)
    payload.insert(
        "lsp_enrichment_status".to_string(),
        serde_json::json!(enrichment.enrichment_status.as_str()),
    );

    // Skip adding empty data for non-success status
    if enrichment.enrichment_status == EnrichmentStatus::Skipped
        || enrichment.enrichment_status == EnrichmentStatus::Failed
    {
        if let Some(error) = &enrichment.error_message {
            payload.insert(
                "lsp_enrichment_error".to_string(),
                serde_json::json!(error),
            );
        }
        return;
    }

    // Add references (limited to avoid huge payloads)
    if !enrichment.references.is_empty() {
        let refs: Vec<_> = enrichment
            .references
            .iter()
            .take(20)
            .map(|r| {
                serde_json::json!({
                    "file": r.file,
                    "line": r.line,
                    "column": r.column
                })
            })
            .collect();
        payload.insert("lsp_references".to_string(), serde_json::json!(refs));
        payload.insert(
            "lsp_references_count".to_string(),
            serde_json::json!(enrichment.references.len()),
        );
    }

    // Add type info
    if let Some(type_info) = &enrichment.type_info {
        payload.insert(
            "lsp_type_signature".to_string(),
            serde_json::json!(type_info.type_signature),
        );
        payload.insert(
            "lsp_type_kind".to_string(),
            serde_json::json!(type_info.kind),
        );
        if let Some(doc) = &type_info.documentation {
            // Truncate long docs
            let truncated = if doc.len() > 500 {
                format!("{}...", &doc[..500])
            } else {
                doc.clone()
            };
            payload.insert(
                "lsp_type_documentation".to_string(),
                serde_json::json!(truncated),
            );
        }
    }

    // Add resolved imports
    if !enrichment.resolved_imports.is_empty() {
        let imports: Vec<_> = enrichment
            .resolved_imports
            .iter()
            .map(|imp| {
                serde_json::json!({
                    "name": imp.import_name,
                    "target_file": imp.target_file,
                    "is_stdlib": imp.is_stdlib,
                    "resolved": imp.resolved
                })
            })
            .collect();
        payload.insert("lsp_imports".to_string(), serde_json::json!(imports));
    }

    // Add definition location
    if let Some(def) = &enrichment.definition {
        payload.insert(
            "lsp_definition".to_string(),
            serde_json::json!({
                "file": def.file,
                "line": def.line,
                "column": def.column
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::project_manager::{EnrichmentStatus, LspEnrichment};

    #[test]
    fn test_lsp_enrichment_status_lowercase_in_payload() {
        let mut payload = std::collections::HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);
        let status = payload
            .get("lsp_enrichment_status")
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(status, "success", "lsp_enrichment_status must be lowercase");

        let mut payload2 = std::collections::HashMap::new();
        let enrichment2 = LspEnrichment {
            enrichment_status: EnrichmentStatus::Failed,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: Some("test error".to_string()),
        };

        add_lsp_enrichment_to_payload(&mut payload2, &enrichment2);
        let status2 = payload2
            .get("lsp_enrichment_status")
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(
            status2, "failed",
            "lsp_enrichment_status must be lowercase"
        );
    }
}
