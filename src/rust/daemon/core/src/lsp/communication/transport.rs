//! Stdio transport for JSON-RPC communication
//!
//! Implements the LSP stdio transport layer: framing messages with
//! `Content-Length` headers, reading/writing to child process streams,
//! and correlating request/response pairs via pending request tracking.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value as JsonValue;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tokio::time::{timeout, Duration};
use tracing::{debug, error, trace, warn};

use crate::lsp::{LspError, LspResult};

use super::jsonrpc::{JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};

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
            read_stdout_loop(
                &mut BufReader::new(stdout),
                &pending_requests,
                &notification_handler,
            )
            .await;
        };

        // Start both tasks
        tokio::spawn(stdin_task);
        tokio::spawn(stdout_task);

        debug!("JSON-RPC client connected via stdio");
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
            pending.insert(
                id,
                PendingRequest {
                    sender: tx,
                    created_at: std::time::Instant::now(),
                },
            );
        }

        // Send request
        let message_json = serde_json::to_string(&request)?;
        self.send_message(message_json).await?;

        // Wait for response with timeout
        match timeout(self.request_timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                self.pending_requests.write().await.remove(&id);
                Err(LspError::Communication {
                    message: "Response channel closed".to_string(),
                })
            }
            Err(_) => {
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
        stats.insert(
            "pending_requests".to_string(),
            JsonValue::Number(pending_count.into()),
        );

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

    /// Insert a fake expired pending request for testing
    #[cfg(test)]
    pub(super) async fn insert_test_pending_request(
        &self,
        id: u64,
        sender: oneshot::Sender<JsonRpcResponse>,
        created_at: std::time::Instant,
    ) {
        let mut pending = self.pending_requests.write().await;
        pending.insert(id, PendingRequest { sender, created_at });
    }
}

impl Default for JsonRpcClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Read loop for LSP server stdout
///
/// Reads Content-Length framed messages, parses them, and dispatches
/// responses to pending request channels or notifications to the handler.
async fn read_stdout_loop(
    reader: &mut BufReader<ChildStdout>,
    pending_requests: &Arc<RwLock<HashMap<u64, PendingRequest>>>,
    notification_handler: &Arc<
        Mutex<Option<Box<dyn Fn(JsonRpcNotification) + Send + Sync>>>,
    >,
) {
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
                if let Err(e) = handle_incoming_message(
                    &message_text,
                    pending_requests,
                    notification_handler,
                )
                .await
                {
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
}

/// Handle an incoming JSON-RPC message from the LSP server
async fn handle_incoming_message(
    message_text: &str,
    pending_requests: &Arc<RwLock<HashMap<u64, PendingRequest>>>,
    notification_handler: &Arc<
        Mutex<Option<Box<dyn Fn(JsonRpcNotification) + Send + Sync>>>,
    >,
) -> LspResult<()> {
    let message = JsonRpcMessage::parse(message_text)?;

    match message {
        JsonRpcMessage::Response(response) => {
            if let Some(id) = response.id.as_u64() {
                let mut pending = pending_requests.write().await;
                if let Some(pending_req) = pending.remove(&id) {
                    if pending_req.sender.send(response).is_err() {
                        debug!("Failed to send response - receiver dropped");
                    }
                } else {
                    warn!("Received response for unknown request ID: {}", id);
                }
            }
        }
        JsonRpcMessage::Notification(notification) => {
            let handler = notification_handler.lock().await;
            if let Some(ref handler_fn) = *handler {
                handler_fn(notification);
            } else {
                debug!(
                    "Received notification but no handler set: {}",
                    notification.method
                );
            }
        }
        JsonRpcMessage::Request(request) => {
            warn!("Received request from LSP server: {}", request.method);
        }
    }

    Ok(())
}
