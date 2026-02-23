/// Format-based routing configuration and metadata enrichment.
///
/// Provides configurable routing rules that determine which Qdrant collection
/// a file should be ingested into, and enriches the queue metadata with
/// `routing_reason`, `source_project_id`, and `library_name` for traceability.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::debug;

// ─── Configuration ─────────────────────────────────────────────────────

/// Routing configuration for format-based collection assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Extensions that always route to libraries (lowercase, with leading dot).
    /// Default: `.pdf`, `.epub`, `.docx`, `.doc`, `.rtf`, `.odt`, `.mobi`,
    ///          `.pptx`, `.ppt`, `.pages`, `.key`, `.odp`,
    ///          `.xlsx`, `.xls`, `.ods`, `.parquet`
    pub library_extensions: Vec<String>,

    /// Where to route `.docx` files found in project folders.
    /// `"libraries"` (default) or `"projects"`.
    pub route_docx_to: String,

    /// Where to route `.pptx` files found in project folders.
    /// `"libraries"` (default) or `"projects"`.
    pub route_pptx_to: String,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            library_extensions: vec![
                ".pdf".into(),
                ".epub".into(),
                ".docx".into(),
                ".doc".into(),
                ".rtf".into(),
                ".odt".into(),
                ".mobi".into(),
                ".pptx".into(),
                ".ppt".into(),
                ".pages".into(),
                ".key".into(),
                ".odp".into(),
                ".xlsx".into(),
                ".xls".into(),
                ".ods".into(),
                ".parquet".into(),
            ],
            route_docx_to: "libraries".into(),
            route_pptx_to: "libraries".into(),
        }
    }
}

impl RoutingConfig {
    /// Check if an extension should route to the libraries collection.
    ///
    /// Respects configurable overrides for `.docx` and `.pptx`.
    pub fn should_route_to_library(&self, extension: &str) -> bool {
        let ext = extension.to_lowercase();

        // Configurable overrides
        if ext == ".docx" || ext == ".doc" {
            return self.route_docx_to == "libraries";
        }
        if ext == ".pptx" || ext == ".ppt" {
            return self.route_pptx_to == "libraries";
        }

        self.library_extensions.contains(&ext)
    }
}

// ─── Routing metadata ──────────────────────────────────────────────────

/// Metadata attached to a format-routed queue item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingMetadata {
    /// The project tenant_id this file originated from.
    pub source_project_id: String,
    /// The reason for routing: `"format_based"`.
    pub routing_reason: String,
    /// The library_name (tenant_id in the libraries collection).
    pub library_name: String,
}

/// Generate the library name for a format-routed file.
///
/// Format: `{project_tenant_id}-refs`
///
/// This creates a dedicated library partition for reference documents
/// found inside a project folder, keeping them associated with the
/// originating project.
pub fn generate_library_name(project_tenant_id: &str) -> String {
    format!("{}-refs", project_tenant_id)
}

/// Build complete routing metadata for a file routed from projects to libraries.
pub fn build_routing_metadata(source_project_id: &str) -> RoutingMetadata {
    RoutingMetadata {
        source_project_id: source_project_id.to_string(),
        routing_reason: "format_based".to_string(),
        library_name: generate_library_name(source_project_id),
    }
}

/// Serialize routing metadata to a JSON string for queue item metadata field.
pub fn routing_metadata_json(source_project_id: &str) -> String {
    let meta = build_routing_metadata(source_project_id);
    serde_json::to_string(&meta).unwrap_or_else(|_| {
        format!(
            r#"{{"source_project_id":"{}","routing_reason":"format_based","library_name":"{}"}}"#,
            source_project_id,
            generate_library_name(source_project_id)
        )
    })
}

// ─── Routing decision ──────────────────────────────────────────────────

/// Result of a routing decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingDecision {
    /// Keep in projects collection.
    Projects,
    /// Route to libraries with metadata.
    Libraries {
        source_project_id: String,
        library_name: String,
    },
    /// Exclude (not in any allowlist — handled upstream).
    Excluded,
}

/// Determine the routing for a file in a project watch folder.
///
/// Uses the routing config to check if the file's extension should be
/// redirected to the libraries collection.
pub fn route_project_file(
    file_path: &Path,
    project_tenant_id: &str,
    config: &RoutingConfig,
) -> RoutingDecision {
    let ext = match file_path.extension().and_then(|e| e.to_str()) {
        Some(e) => format!(".{}", e.to_lowercase()),
        None => return RoutingDecision::Excluded,
    };

    if config.should_route_to_library(&ext) {
        let library_name = generate_library_name(project_tenant_id);
        debug!(
            file = %file_path.display(),
            ext = ext.as_str(),
            source_project = project_tenant_id,
            library_name = library_name.as_str(),
            "Format-based routing to libraries"
        );
        RoutingDecision::Libraries {
            source_project_id: project_tenant_id.to_string(),
            library_name,
        }
    } else {
        RoutingDecision::Projects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Config tests ───────────────────────────────────────────────────

    #[test]
    fn test_default_config_routes_pdf() {
        let config = RoutingConfig::default();
        assert!(config.should_route_to_library(".pdf"));
        assert!(config.should_route_to_library(".PDF"));
    }

    #[test]
    fn test_default_config_routes_epub() {
        let config = RoutingConfig::default();
        assert!(config.should_route_to_library(".epub"));
    }

    #[test]
    fn test_default_config_routes_docx() {
        let config = RoutingConfig::default();
        assert!(config.should_route_to_library(".docx"));
    }

    #[test]
    fn test_config_override_docx_to_projects() {
        let config = RoutingConfig {
            route_docx_to: "projects".into(),
            ..Default::default()
        };
        assert!(!config.should_route_to_library(".docx"));
        assert!(!config.should_route_to_library(".doc"));
        // PDF still routes to libraries
        assert!(config.should_route_to_library(".pdf"));
    }

    #[test]
    fn test_config_override_pptx_to_projects() {
        let config = RoutingConfig {
            route_pptx_to: "projects".into(),
            ..Default::default()
        };
        assert!(!config.should_route_to_library(".pptx"));
        assert!(!config.should_route_to_library(".ppt"));
    }

    #[test]
    fn test_source_code_not_routed() {
        let config = RoutingConfig::default();
        assert!(!config.should_route_to_library(".rs"));
        assert!(!config.should_route_to_library(".py"));
        assert!(!config.should_route_to_library(".ts"));
        assert!(!config.should_route_to_library(".md"));
    }

    // ─── Library name tests ─────────────────────────────────────────────

    #[test]
    fn test_generate_library_name() {
        assert_eq!(generate_library_name("my-project"), "my-project-refs");
        assert_eq!(
            generate_library_name("abc123def456"),
            "abc123def456-refs"
        );
    }

    // ─── Routing metadata tests ─────────────────────────────────────────

    #[test]
    fn test_build_routing_metadata() {
        let meta = build_routing_metadata("proj-a");
        assert_eq!(meta.source_project_id, "proj-a");
        assert_eq!(meta.routing_reason, "format_based");
        assert_eq!(meta.library_name, "proj-a-refs");
    }

    #[test]
    fn test_routing_metadata_json() {
        let json = routing_metadata_json("proj-x");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["source_project_id"], "proj-x");
        assert_eq!(parsed["routing_reason"], "format_based");
        assert_eq!(parsed["library_name"], "proj-x-refs");
    }

    // ─── Routing decision tests ─────────────────────────────────────────

    #[test]
    fn test_route_pdf_to_libraries() {
        let config = RoutingConfig::default();
        let decision = route_project_file(
            Path::new("/project/docs/manual.pdf"),
            "my-proj",
            &config,
        );
        assert_eq!(
            decision,
            RoutingDecision::Libraries {
                source_project_id: "my-proj".to_string(),
                library_name: "my-proj-refs".to_string(),
            }
        );
    }

    #[test]
    fn test_route_rust_stays_in_projects() {
        let config = RoutingConfig::default();
        let decision = route_project_file(
            Path::new("/project/src/main.rs"),
            "my-proj",
            &config,
        );
        assert_eq!(decision, RoutingDecision::Projects);
    }

    #[test]
    fn test_route_no_extension_excluded() {
        let config = RoutingConfig::default();
        let decision = route_project_file(
            Path::new("/project/Makefile"),
            "my-proj",
            &config,
        );
        assert_eq!(decision, RoutingDecision::Excluded);
    }

    #[test]
    fn test_route_docx_configurable() {
        // Default: docx → libraries
        let default_config = RoutingConfig::default();
        let decision = route_project_file(
            Path::new("/project/spec.docx"),
            "proj",
            &default_config,
        );
        assert!(matches!(decision, RoutingDecision::Libraries { .. }));

        // Override: docx → projects
        let override_config = RoutingConfig {
            route_docx_to: "projects".into(),
            ..Default::default()
        };
        let decision = route_project_file(
            Path::new("/project/spec.docx"),
            "proj",
            &override_config,
        );
        assert_eq!(decision, RoutingDecision::Projects);
    }

    #[test]
    fn test_route_case_insensitive() {
        let config = RoutingConfig::default();
        let decision = route_project_file(
            Path::new("/project/MANUAL.PDF"),
            "proj",
            &config,
        );
        assert!(matches!(decision, RoutingDecision::Libraries { .. }));
    }

    #[test]
    fn test_route_spreadsheets() {
        let config = RoutingConfig::default();

        let xlsx = route_project_file(
            Path::new("/project/data.xlsx"),
            "proj",
            &config,
        );
        assert!(matches!(xlsx, RoutingDecision::Libraries { .. }));

        let ods = route_project_file(
            Path::new("/project/data.ods"),
            "proj",
            &config,
        );
        assert!(matches!(ods, RoutingDecision::Libraries { .. }));
    }
}
