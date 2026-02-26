//! LSP initialization protocol
//!
//! Handles LSP capability negotiation and workspace configuration
//! during the `initialize` / `initialized` handshake.

use tokio::time::timeout;
use tracing::debug;

use crate::lsp::{LspError, LspResult};

use super::process::ServerInstance;

impl ServerInstance {
    /// Initialize the LSP server with the initialization request
    pub(super) async fn initialize_lsp(&self) -> LspResult<()> {
        debug!("Sending LSP initialize request to {}", self.metadata.name);

        let init_params = build_initialize_params(&self.metadata.working_directory);

        // Send initialize request
        let _response = timeout(
            self.config.request_timeout,
            self.rpc_client.send_request("initialize", init_params),
        )
        .await
        .map_err(|_| LspError::Timeout {
            operation: "LSP initialize".to_string(),
        })??;

        // Send initialized notification
        self.rpc_client
            .send_notification("initialized", serde_json::json!({}))
            .await?;

        debug!("LSP server {} initialized successfully", self.metadata.name);
        Ok(())
    }

    /// Re-initialize server after a health-check shutdown/restart cycle
    pub(super) async fn reinitialize_after_health_check(&self) -> LspResult<()> {
        let _ = self.initialize_lsp().await;
        Ok(())
    }
}

/// Build the LSP `initialize` request parameters
///
/// Declares the minimal set of client capabilities needed for
/// code intelligence (definitions, references, hover, symbols).
fn build_initialize_params(working_directory: &std::path::Path) -> serde_json::Value {
    serde_json::json!({
        "processId": std::process::id(),
        "rootUri": format!("file://{}", working_directory.display()),
        "capabilities": {
            "textDocument": text_document_capabilities(),
            "workspace": workspace_capabilities()
        }
    })
}

/// Client capabilities for textDocument requests
fn text_document_capabilities() -> serde_json::Value {
    serde_json::json!({
        "synchronization": {
            "dynamicRegistration": false,
            "willSave": false,
            "willSaveWaitUntil": false,
            "didSave": false
        },
        "completion": {
            "dynamicRegistration": false,
            "completionItem": {
                "snippetSupport": false,
                "commitCharactersSupport": false,
                "documentationFormat": ["plaintext"],
                "deprecatedSupport": false,
                "preselectSupport": false
            },
            "contextSupport": false
        },
        "hover": {
            "dynamicRegistration": false,
            "contentFormat": ["plaintext"]
        },
        "definition": {
            "dynamicRegistration": false
        },
        "references": {
            "dynamicRegistration": false
        },
        "documentSymbol": {
            "dynamicRegistration": false
        }
    })
}

/// Client capabilities for workspace requests
fn workspace_capabilities() -> serde_json::Value {
    serde_json::json!({
        "applyEdit": false,
        "workspaceEdit": {
            "documentChanges": false
        },
        "didChangeConfiguration": {
            "dynamicRegistration": false
        },
        "didChangeWatchedFiles": {
            "dynamicRegistration": false
        },
        "symbol": {
            "dynamicRegistration": false
        },
        "executeCommand": {
            "dynamicRegistration": false
        }
    })
}
