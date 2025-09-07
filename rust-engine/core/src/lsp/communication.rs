//! JSON-RPC Communication Module
//!
//! This module provides JSON-RPC protocol implementation for LSP server communication
//! over stdio and TCP connections with timeout handling and correlation.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tokio::time::{timeout, Duration};
use tracing::{debug, error, trace, warn};

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
        if value.get("id").is_some() && 
           (value.get("result").is_some() || value.get("error").is_some()) {
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

/// Pending request awaiting response
#[derive(Debug)]
struct PendingRequest {
    sender: oneshot::Sender<JsonRpcResponse>,
    created_at: std::time::Instant,
}

/// JSON-RPC client for LSP communication
pub struct JsonRpcClient {
    /// Next request ID
    next_id: AtomicU64,
    /// Pending requests awaiting responses
    pending_requests: Arc<RwLock<HashMap<u64, PendingRequest>>>,
    /// Message sender for outgoing messages
    message_sender: Arc<Mutex<Option<mpsc::UnboundedSender<String>>>>,
    /// Notification handler
    notification_handler: Arc<Mutex<Option<Box<dyn Fn(JsonRpcNotification) + Send + Sync>>>>,
    /// Request timeout duration
    request_timeout: Duration,
}

impl JsonRpcClient {
    /// Create a new JSON-RPC client
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            message_sender: Arc::new(Mutex::new(None)),
            notification_handler: Arc::new(Mutex::new(None)),
            request_timeout: Duration::from_secs(30),
        }
    }

    /// Connect to an LSP server via stdio
    pub async fn connect_stdio(
        &self,
        mut stdin: ChildStdin,
        stdout: ChildStdout,
    ) -> LspResult<()> {
        debug!("Connecting JSON-RPC client via stdio");

        // Create message channel
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        *self.message_sender.lock().await = Some(tx);

        // Spawn task to write to stdin
        let stdin_task = async move {
            while let Some(message) = rx.recv().await {
                let content = format!("Content-Length: {}\r\n\r\n{}", message.len(), message);
                trace!("Sending: {}", content);
                
                if let Err(e) = stdin.write_all(content.as_bytes()).await {
                    error!("Error writing to LSP server stdin: {}", e);
                    break;
                }
                
                if let Err(e) = stdin.flush().await {
                    error!("Error flushing LSP server stdin: {}", e);
                    break;
                }
            }
        };

        // Spawn task to read from stdout
        let pending_requests = self.pending_requests.clone();
        let notification_handler = self.notification_handler.clone();
        
        let stdout_task = async move {
            let mut reader = BufReader::new(stdout);
            let mut buffer = String::new();

            loop {
                buffer.clear();
                
                // Read Content-Length header
                match reader.read_line(&mut buffer).await {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        if buffer.trim().is_empty() {
                            continue;
                        }
                        
                        // Parse Content-Length
                        let content_length = if buffer.starts_with("Content-Length: ") {
                            buffer[16..].trim().parse::<usize>().unwrap_or(0)
                        } else {
                            warn!("Invalid header: {}", buffer.trim());
                            continue;
                        };

                        if content_length == 0 {
                            continue;
                        }

                        // Read empty line
                        buffer.clear();
                        if reader.read_line(&mut buffer).await.is_err() {
                            break;
                        }

                        // Read message content
                        let mut content = vec![0u8; content_length];
                        if reader.read_exact(&mut content).await.is_err() {
                            break;
                        }

                        let message_text = String::from_utf8_lossy(&content);
                        trace!("Received: {}", message_text);

                        // Parse and handle message
                        if let Err(e) = Self::handle_incoming_message(
                            &message_text,
                            &pending_requests,
                            &notification_handler,
                        ).await {
                            warn!("Error handling message: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Error reading from LSP server stdout: {}", e);
                        break;
                    }
                }
            }

            debug!("LSP server stdout reader finished");
        };

        // Start both tasks
        tokio::spawn(stdin_task);
        tokio::spawn(stdout_task);

        debug!("JSON-RPC client connected via stdio");
        Ok(())
    }

    /// Handle incoming message from LSP server
    async fn handle_incoming_message(
        message_text: &str,
        pending_requests: &Arc<RwLock<HashMap<u64, PendingRequest>>>,
        notification_handler: &Arc<Mutex<Option<Box<dyn Fn(JsonRpcNotification) + Send + Sync>>>>,
    ) -> LspResult<()> {
        let message = JsonRpcMessage::parse(message_text)?;

        match message {
            JsonRpcMessage::Response(response) => {
                // Handle response to our request
                if let Some(id) = response.id.as_u64() {
                    let mut pending = pending_requests.write().await;
                    if let Some(pending_req) = pending.remove(&id) {
                        if let Err(_) = pending_req.sender.send(response) {
                            debug!("Failed to send response - receiver dropped");
                        }
                    } else {
                        warn!("Received response for unknown request ID: {}", id);
                    }
                }
            }
            JsonRpcMessage::Notification(notification) => {
                // Handle notification from server
                let handler = notification_handler.lock().await;
                if let Some(ref handler_fn) = *handler {
                    handler_fn(notification);
                } else {
                    debug!("Received notification but no handler set: {}", notification.method);
                }
            }
            JsonRpcMessage::Request(request) => {
                // Handle request from server (not typically expected in LSP)
                warn!("Received request from LSP server: {}", request.method);
            }
        }

        Ok(())
    }

    /// Send a request and wait for response
    pub async fn send_request(
        &self,
        method: &str,
        params: JsonValue,
    ) -> LspResult<JsonRpcResponse> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonValue::Number(id.into()),
            method: method.to_string(),
            params: if params.is_null() { None } else { Some(params) },
        };

        // Create response channel
        let (tx, rx) = oneshot::channel();
        
        // Store pending request
        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(id, PendingRequest {
                sender: tx,
                created_at: std::time::Instant::now(),
            });
        }

        // Send request
        let message_json = serde_json::to_string(&request)?;
        self.send_message(message_json).await?;

        // Wait for response with timeout
        match timeout(self.request_timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                // Clean up pending request
                self.pending_requests.write().await.remove(&id);
                Err(LspError::Communication {
                    message: "Response channel closed".to_string(),
                })
            }
            Err(_) => {
                // Clean up pending request
                self.pending_requests.write().await.remove(&id);
                Err(LspError::Timeout {
                    operation: format!("request: {}", method),
                })
            }
        }
    }

    /// Send a notification (no response expected)
    pub async fn send_notification(
        &self,
        method: &str,
        params: JsonValue,
    ) -> LspResult<()> {
        let notification = JsonRpcNotification {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params: if params.is_null() { None } else { Some(params) },
        };

        let message_json = serde_json::to_string(&notification)?;
        self.send_message(message_json).await
    }

    /// Send a raw message
    async fn send_message(&self, message: String) -> LspResult<()> {
        let sender = self.message_sender.lock().await;
        if let Some(ref tx) = *sender {
            tx.send(message).map_err(|_| LspError::Communication {
                message: "Message channel closed".to_string(),
            })?;
            Ok(())
        } else {
            Err(LspError::Communication {
                message: "Not connected".to_string(),
            })
        }
    }

    /// Set notification handler
    pub async fn set_notification_handler<F>(&self, handler: F)
    where
        F: Fn(JsonRpcNotification) + Send + Sync + 'static,
    {
        *self.notification_handler.lock().await = Some(Box::new(handler));
    }

    /// Get statistics
    pub async fn get_stats(&self) -> HashMap<String, JsonValue> {
        let mut stats = HashMap::new();
        
        let pending_count = self.pending_requests.read().await.len();
        stats.insert("pending_requests".to_string(), JsonValue::Number(pending_count.into()));
        
        let connected = self.message_sender.lock().await.is_some();
        stats.insert("connected".to_string(), JsonValue::Bool(connected));
        
        stats
    }

    /// Clean up expired requests
    pub async fn cleanup_expired_requests(&self) -> u32 {
        let mut pending = self.pending_requests.write().await;
        let now = std::time::Instant::now();
        let mut expired_count = 0;

        pending.retain(|_, pending_req| {
            if now.duration_since(pending_req.created_at) > self.request_timeout {
                expired_count += 1;
                false
            } else {
                true
            }
        });

        if expired_count > 0 {
            warn!("Cleaned up {} expired requests", expired_count);
        }

        expired_count
    }

    /// Set request timeout
    pub fn set_request_timeout(&mut self, timeout: Duration) {
        self.request_timeout = timeout;
    }

    /// Check if client is connected
    pub async fn is_connected(&self) -> bool {
        self.message_sender.lock().await.is_some()
    }

    /// Disconnect the client
    pub async fn disconnect(&self) {
        // Close the message sender
        *self.message_sender.lock().await = None;

        // Cancel all pending requests
        let mut pending = self.pending_requests.write().await;
        for (_, pending_req) in pending.drain() {
            // The oneshot channel will be dropped, causing the receiver to get an error
            drop(pending_req.sender);
        }

        debug!("JSON-RPC client disconnected");
    }
}

impl Default for JsonRpcClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rpc_request_creation() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonValue::Number(1.into()),
            method: "initialize".to_string(),
            params: Some(serde_json::json!({"foo": "bar"})),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("initialize"));
        assert!(json.contains("foo"));
    }

    #[test]
    fn test_json_rpc_message_parsing() {
        // Test request parsing
        let request_json = r#"{"jsonrpc":"2.0","id":1,"method":"test","params":{}}"#;
        let message = JsonRpcMessage::parse(request_json).unwrap();
        assert!(matches!(message, JsonRpcMessage::Request(_)));
        assert_eq!(message.method(), Some("test"));

        // Test response parsing
        let response_json = r#"{"jsonrpc":"2.0","id":1,"result":"success"}"#;
        let message = JsonRpcMessage::parse(response_json).unwrap();
        assert!(matches!(message, JsonRpcMessage::Response(_)));

        // Test notification parsing
        let notification_json = r#"{"jsonrpc":"2.0","method":"notify","params":{}}"#;
        let message = JsonRpcMessage::parse(notification_json).unwrap();
        assert!(matches!(message, JsonRpcMessage::Notification(_)));
        assert_eq!(message.method(), Some("notify"));
    }

    #[test]
    fn test_json_rpc_error() {
        let error = JsonRpcError {
            code: -1,
            message: "Test error".to_string(),
            data: Some(serde_json::json!({"details": "more info"})),
        };

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("Test error"));
        assert!(json.contains("details"));
    }

    #[tokio::test]
    async fn test_json_rpc_client_creation() {
        let client = JsonRpcClient::new();
        assert!(!client.is_connected().await);
        
        let stats = client.get_stats().await;
        assert_eq!(stats.get("pending_requests").unwrap().as_u64(), Some(0));
        assert_eq!(stats.get("connected").unwrap().as_bool(), Some(false));
    }

    #[tokio::test]
    async fn test_cleanup_expired_requests() {
        let mut client = JsonRpcClient::new();
        client.set_request_timeout(Duration::from_millis(1));

        // Add a mock pending request
        {
            let (tx, _rx) = oneshot::channel();
            let mut pending = client.pending_requests.write().await;
            pending.insert(1, PendingRequest {
                sender: tx,
                created_at: std::time::Instant::now() - Duration::from_secs(1),
            });
        }

        // Wait a bit to ensure expiry
        tokio::time::sleep(Duration::from_millis(10)).await;

        let expired = client.cleanup_expired_requests().await;
        assert_eq!(expired, 1);

        let stats = client.get_stats().await;
        assert_eq!(stats.get("pending_requests").unwrap().as_u64(), Some(0));
    }
}