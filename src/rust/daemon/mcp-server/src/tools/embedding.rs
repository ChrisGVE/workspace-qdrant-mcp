//! `embedding` MCP tool handler.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/embedding.ts`.
//!
//! Calls `SystemService.GetEmbeddingProviderStatus` over gRPC and returns the
//! active provider's configuration plus its current probe state.
//!
//! # Result shape (TS `EmbeddingToolResult`, field order preserved for JSON parity)
//!
//! ```json
//! // success
//! { "success": true, "provider": "fastembed", "model": "all-MiniLM-L6-v2",
//!   "output_dim": 384, "base_url": "", "probe_status": "healthy",
//!   "probe_message": "probe succeeded" }
//!
//! // error (daemon down or gRPC failure)
//! { "success": false, "error": "Failed to fetch embedding provider status: <msg>" }
//! ```
//!
//! The tool catches all errors and returns an **in-band** error shape — it never
//! throws, matching the TS `try/catch` pattern in `handleEmbedding`.

use rmcp::model::CallToolResult;
use serde::Serialize;

use crate::tools::envelope::ok_text;

// ---------------------------------------------------------------------------
// Public trait — injectable for tests
// ---------------------------------------------------------------------------

/// Abstraction over the single gRPC call needed by the embedding tool.
///
/// Production code uses `DaemonClient` (via the blanket impl below).
/// Tests inject a stub.
pub trait EmbeddingStatusProvider {
    /// Fetch the embedding provider status from the daemon.
    ///
    /// Returns `(provider, model, output_dim, base_url, probe_status, probe_message)`.
    fn get_embedding_provider_status(
        &mut self,
    ) -> impl std::future::Future<Output = Result<EmbeddingProviderFields, String>> + Send;
}

/// Raw fields from `GetEmbeddingProviderStatusResponse`.
#[derive(Debug, Clone)]
pub struct EmbeddingProviderFields {
    pub provider: String,
    pub model: String,
    pub output_dim: u32,
    pub base_url: String,
    pub probe_status: String,
    pub probe_message: String,
}

// Blanket impl for `DaemonClient`.
impl EmbeddingStatusProvider for crate::grpc::DaemonClient {
    async fn get_embedding_provider_status(&mut self) -> Result<EmbeddingProviderFields, String> {
        self.get_embedding_provider_status()
            .await
            .map(|r| EmbeddingProviderFields {
                provider: r.provider,
                model: r.model,
                output_dim: r.output_dim,
                base_url: r.base_url,
                probe_status: r.probe_status,
                probe_message: r.probe_message,
            })
            .map_err(|s| s.message().to_string())
    }
}

// ---------------------------------------------------------------------------
// Result struct — field ORDER must match TS `EmbeddingToolResult` declaration
// ---------------------------------------------------------------------------

/// Result returned by the `embedding` tool.
///
/// Field declaration order matches `EmbeddingToolResult` in `embedding.ts` lines
/// 10-19 for byte-for-byte JSON parity (serde serializes fields in declaration
/// order).
///
/// On success: `success=true` + provider fields; `error` omitted.
/// On failure: `success=false` + `error`; all provider fields omitted.
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct EmbeddingToolResult {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dim: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool function
// ---------------------------------------------------------------------------

/// Execute the `embedding` tool.
///
/// Mirrors `handleEmbedding` in `embedding.ts`. Always returns a
/// `CallToolResult`; errors are returned **in-band** (no thrown errors):
/// - Success → pretty JSON of `EmbeddingToolResult` with `success=true`.
/// - gRPC error → pretty JSON of `EmbeddingToolResult` with `success=false`.
pub async fn embedding_tool<P>(provider: &mut P) -> CallToolResult
where
    P: EmbeddingStatusProvider,
{
    match provider.get_embedding_provider_status().await {
        Ok(fields) => {
            let result = EmbeddingToolResult {
                success: true,
                provider: Some(fields.provider),
                model: Some(fields.model),
                output_dim: Some(fields.output_dim),
                base_url: Some(fields.base_url),
                probe_status: Some(fields.probe_status),
                probe_message: Some(fields.probe_message),
                error: None,
            };
            ok_text(&result)
        }
        Err(msg) => {
            let result = EmbeddingToolResult {
                success: false,
                provider: None,
                model: None,
                output_dim: None,
                base_url: None,
                probe_status: None,
                probe_message: None,
                error: Some(format!("Failed to fetch embedding provider status: {msg}")),
            };
            // In-band error: TS catch returns the object, NOT a thrown error.
            // The CallToolResult has no isError flag set (ok_text).
            ok_text(&result)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "embedding_tests.rs"]
mod tests;
