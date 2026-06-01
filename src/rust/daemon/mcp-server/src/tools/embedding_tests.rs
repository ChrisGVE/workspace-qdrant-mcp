//! Unit tests for the `embedding` tool.
//!
//! All tests are hermetic — no live daemon required.

use super::{
    embedding_tool, EmbeddingProviderFields, EmbeddingStatusProvider, EmbeddingToolResult,
};

// ---------------------------------------------------------------------------
// Stub daemon
// ---------------------------------------------------------------------------

struct OkProvider(EmbeddingProviderFields);
struct ErrProvider(String);

impl EmbeddingStatusProvider for OkProvider {
    async fn get_embedding_provider_status(&mut self) -> Result<EmbeddingProviderFields, String> {
        Ok(self.0.clone())
    }
}

impl EmbeddingStatusProvider for ErrProvider {
    async fn get_embedding_provider_status(&mut self) -> Result<EmbeddingProviderFields, String> {
        Err(self.0.clone())
    }
}

fn fastembed_fields() -> EmbeddingProviderFields {
    EmbeddingProviderFields {
        provider: "fastembed".to_string(),
        model: "all-MiniLM-L6-v2".to_string(),
        output_dim: 384,
        base_url: String::new(),
        probe_status: "healthy".to_string(),
        probe_message: "probe succeeded".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Helper: extract pretty-JSON text from CallToolResult
// ---------------------------------------------------------------------------

fn result_text(r: &rmcp::model::CallToolResult) -> &str {
    r.content
        .first()
        .expect("content not empty")
        .raw
        .as_text()
        .expect("text content")
        .text
        .as_str()
}

fn parse_result(r: &rmcp::model::CallToolResult) -> EmbeddingToolResult {
    serde_json::from_str(result_text(r)).expect("valid JSON")
}

// ---------------------------------------------------------------------------
// Success path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn success_sets_success_true() {
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert!(result.success);
}

#[tokio::test]
async fn success_no_is_error_flag() {
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    // In-band success: isError must be absent
    assert!(r.is_error.is_none());
}

#[tokio::test]
async fn success_provider_field_present() {
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert_eq!(result.provider.as_deref(), Some("fastembed"));
}

#[tokio::test]
async fn success_model_field_present() {
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert_eq!(result.model.as_deref(), Some("all-MiniLM-L6-v2"));
}

#[tokio::test]
async fn success_output_dim_field() {
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert_eq!(result.output_dim, Some(384));
}

#[tokio::test]
async fn success_base_url_present() {
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    // base_url is Some("") for local providers — empty string, not absent
    assert_eq!(result.base_url.as_deref(), Some(""));
}

#[tokio::test]
async fn success_probe_status_present() {
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert_eq!(result.probe_status.as_deref(), Some("healthy"));
}

#[tokio::test]
async fn success_probe_message_present() {
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert_eq!(result.probe_message.as_deref(), Some("probe succeeded"));
}

#[tokio::test]
async fn success_error_field_absent() {
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    // On success, 'error' key must be absent from JSON
    assert!(result.error.is_none());
    // Also verify directly from the raw JSON string
    assert!(!result_text(&r).contains("\"error\""));
}

// ---------------------------------------------------------------------------
// Error path (daemon down / gRPC failure)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn error_sets_success_false() {
    let mut p = ErrProvider("connection refused".to_string());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert!(!result.success);
}

#[tokio::test]
async fn error_no_is_error_flag() {
    // TS catch returns object in-band — NOT via thrown error wrapper
    let mut p = ErrProvider("connection refused".to_string());
    let r = embedding_tool(&mut p).await;
    assert!(r.is_error.is_none());
}

#[tokio::test]
async fn error_message_prefix() {
    let mut p = ErrProvider("connection refused".to_string());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    let err = result.error.expect("error field present");
    assert!(
        err.starts_with("Failed to fetch embedding provider status:"),
        "expected prefix, got: {err:?}"
    );
    assert!(err.contains("connection refused"));
}

#[tokio::test]
async fn error_provider_field_absent() {
    let mut p = ErrProvider("timeout".to_string());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert!(result.provider.is_none());
    assert!(!result_text(&r).contains("\"provider\""));
}

#[tokio::test]
async fn error_model_field_absent() {
    let mut p = ErrProvider("timeout".to_string());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert!(result.model.is_none());
}

#[tokio::test]
async fn error_output_dim_absent() {
    let mut p = ErrProvider("timeout".to_string());
    let r = embedding_tool(&mut p).await;
    let result = parse_result(&r);
    assert!(result.output_dim.is_none());
}

// ---------------------------------------------------------------------------
// Golden: exact pretty-JSON comparison
// ---------------------------------------------------------------------------

#[tokio::test]
async fn golden_success_json_field_order() {
    // Field order in the JSON must match TS EmbeddingToolResult declaration order:
    // success, provider, model, output_dim, base_url, probe_status, probe_message
    let mut p = OkProvider(fastembed_fields());
    let r = embedding_tool(&mut p).await;
    let text = result_text(&r);
    let expected = serde_json::json!({
        "success": true,
        "provider": "fastembed",
        "model": "all-MiniLM-L6-v2",
        "output_dim": 384,
        "base_url": "",
        "probe_status": "healthy",
        "probe_message": "probe succeeded"
    });
    let actual: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn golden_error_json_field_order() {
    // Error shape must have only success + error fields
    let mut p = ErrProvider("daemon offline".to_string());
    let r = embedding_tool(&mut p).await;
    let text = result_text(&r);
    let actual: serde_json::Value = serde_json::from_str(text).unwrap();
    let expected = serde_json::json!({
        "success": false,
        "error": "Failed to fetch embedding provider status: daemon offline"
    });
    assert_eq!(actual, expected);
}
