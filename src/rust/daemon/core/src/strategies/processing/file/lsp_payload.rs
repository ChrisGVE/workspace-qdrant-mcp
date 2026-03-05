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
            payload.insert("lsp_enrichment_error".to_string(), serde_json::json!(error));
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
    use crate::lsp::project_manager::{
        EnrichmentStatus, LspEnrichment, Reference, ResolvedImport, TypeInfo,
    };

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
        assert_eq!(status2, "failed", "lsp_enrichment_status must be lowercase");
    }

    #[test]
    fn test_lsp_enrichment_skipped_includes_error() {
        let mut payload = HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Skipped,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: Some("server not ready".to_string()),
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);

        assert_eq!(
            payload["lsp_enrichment_status"],
            serde_json::json!("skipped")
        );
        assert_eq!(
            payload["lsp_enrichment_error"],
            serde_json::json!("server not ready")
        );
        // No other fields should be added for skipped status
        assert!(!payload.contains_key("lsp_references"));
        assert!(!payload.contains_key("lsp_type_signature"));
    }

    #[test]
    fn test_lsp_enrichment_skipped_without_error() {
        let mut payload = HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Skipped,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);

        assert_eq!(
            payload["lsp_enrichment_status"],
            serde_json::json!("skipped")
        );
        assert!(!payload.contains_key("lsp_enrichment_error"));
    }

    #[test]
    fn test_lsp_enrichment_references_truncated_at_20() {
        let refs: Vec<Reference> = (0..30)
            .map(|i| Reference {
                file: format!("/src/file_{}.rs", i),
                line: i,
                column: 0,
                end_line: None,
                end_column: None,
            })
            .collect();

        let mut payload = HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: refs,
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);

        let lsp_refs = payload["lsp_references"].as_array().unwrap();
        assert_eq!(lsp_refs.len(), 20, "References should be capped at 20");
        // But the count should reflect the total
        assert_eq!(payload["lsp_references_count"], serde_json::json!(30));
    }

    #[test]
    fn test_lsp_enrichment_type_info_doc_truncation() {
        let long_doc = "x".repeat(600);

        let mut payload = HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: Some(TypeInfo {
                type_signature: "fn foo() -> Bar".to_string(),
                kind: "function".to_string(),
                documentation: Some(long_doc),
                container: None,
            }),
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);

        assert_eq!(
            payload["lsp_type_signature"],
            serde_json::json!("fn foo() -> Bar")
        );
        assert_eq!(payload["lsp_type_kind"], serde_json::json!("function"));

        let doc = payload["lsp_type_documentation"].as_str().unwrap();
        assert!(
            doc.ends_with("..."),
            "Long docs should be truncated with '...'"
        );
        assert_eq!(doc.len(), 503, "Truncated doc: 500 chars + '...'");
    }

    #[test]
    fn test_lsp_enrichment_type_info_short_doc_not_truncated() {
        let mut payload = HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: Some(TypeInfo {
                type_signature: "struct Foo".to_string(),
                kind: "struct".to_string(),
                documentation: Some("A simple struct.".to_string()),
                container: None,
            }),
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);

        assert_eq!(
            payload["lsp_type_documentation"],
            serde_json::json!("A simple struct.")
        );
    }

    #[test]
    fn test_lsp_enrichment_imports_serialization() {
        let mut payload = HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: None,
            resolved_imports: vec![
                ResolvedImport {
                    import_name: "std::collections::HashMap".to_string(),
                    target_file: Some("/rustlib/collections.rs".to_string()),
                    target_symbol: None,
                    is_stdlib: true,
                    resolved: true,
                },
                ResolvedImport {
                    import_name: "crate::config::Settings".to_string(),
                    target_file: Some("/project/src/config.rs".to_string()),
                    target_symbol: None,
                    is_stdlib: false,
                    resolved: true,
                },
            ],
            definition: None,
            error_message: None,
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);

        let imports = payload["lsp_imports"].as_array().unwrap();
        assert_eq!(imports.len(), 2);
        assert_eq!(
            imports[0]["name"],
            serde_json::json!("std::collections::HashMap")
        );
        assert_eq!(imports[0]["is_stdlib"], serde_json::json!(true));
        assert_eq!(imports[1]["is_stdlib"], serde_json::json!(false));
    }

    #[test]
    fn test_lsp_enrichment_definition_location() {
        let mut payload = HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: Some(Reference {
                file: "/project/src/types.rs".to_string(),
                line: 42,
                column: 4,
                end_line: None,
                end_column: None,
            }),
            error_message: None,
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);

        let def = &payload["lsp_definition"];
        assert_eq!(def["file"], serde_json::json!("/project/src/types.rs"));
        assert_eq!(def["line"], serde_json::json!(42));
        assert_eq!(def["column"], serde_json::json!(4));
    }

    #[test]
    fn test_lsp_enrichment_success_no_data() {
        let mut payload = HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);

        assert_eq!(
            payload["lsp_enrichment_status"],
            serde_json::json!("success")
        );
        // Empty collections should not be added
        assert!(!payload.contains_key("lsp_references"));
        assert!(!payload.contains_key("lsp_imports"));
        assert!(!payload.contains_key("lsp_definition"));
        assert!(!payload.contains_key("lsp_type_signature"));
    }
}
