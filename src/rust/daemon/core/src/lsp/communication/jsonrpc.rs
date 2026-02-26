//! JSON-RPC message types and parsing
//!
//! Defines the core JSON-RPC 2.0 message structures: requests, responses,
//! notifications, and errors. Provides parsing from JSON and serialization.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::lsp::{LspError, LspResult};

/// JSON-RPC request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version
    pub jsonrpc: String,
    /// Request ID for correlation
    pub id: JsonValue,
    /// Method name
    pub method: String,
    /// Request parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<JsonValue>,
}

/// JSON-RPC response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version
    pub jsonrpc: String,
    /// Request ID for correlation
    pub id: JsonValue,
    /// Success result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JsonValue>,
    /// Error result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC notification message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    /// JSON-RPC version
    pub jsonrpc: String,
    /// Method name
    pub method: String,
    /// Notification parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<JsonValue>,
}

/// JSON-RPC error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonValue>,
}

/// Union type for all JSON-RPC messages
#[derive(Debug, Clone)]
pub enum JsonRpcMessage {
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}

impl JsonRpcMessage {
    /// Parse a JSON-RPC message from JSON
    pub fn parse(json: &str) -> LspResult<Self> {
        let value: JsonValue = serde_json::from_str(json)?;

        // Check if it's a request (has method and id)
        if value.get("method").is_some() && value.get("id").is_some() {
            let request: JsonRpcRequest = serde_json::from_value(value)?;
            return Ok(JsonRpcMessage::Request(request));
        }

        // Check if it's a response (has id and result/error)
        if value.get("id").is_some()
            && (value.get("result").is_some() || value.get("error").is_some())
        {
            let response: JsonRpcResponse = serde_json::from_value(value)?;
            return Ok(JsonRpcMessage::Response(response));
        }

        // Check if it's a notification (has method but no id)
        if value.get("method").is_some() && value.get("id").is_none() {
            let notification: JsonRpcNotification = serde_json::from_value(value)?;
            return Ok(JsonRpcMessage::Notification(notification));
        }

        Err(LspError::JsonRpc {
            message: "Invalid JSON-RPC message format".to_string(),
        })
    }

    /// Serialize message to JSON
    pub fn to_json(&self) -> LspResult<String> {
        let json = match self {
            JsonRpcMessage::Request(req) => serde_json::to_string(req)?,
            JsonRpcMessage::Response(resp) => serde_json::to_string(resp)?,
            JsonRpcMessage::Notification(notif) => serde_json::to_string(notif)?,
        };
        Ok(json)
    }

    /// Get the method name if this is a request or notification
    pub fn method(&self) -> Option<&str> {
        match self {
            JsonRpcMessage::Request(req) => Some(&req.method),
            JsonRpcMessage::Notification(notif) => Some(&notif.method),
            JsonRpcMessage::Response(_) => None,
        }
    }

    /// Get the ID if this is a request or response
    pub fn id(&self) -> Option<&JsonValue> {
        match self {
            JsonRpcMessage::Request(req) => Some(&req.id),
            JsonRpcMessage::Response(resp) => Some(&resp.id),
            JsonRpcMessage::Notification(_) => None,
        }
    }
}
