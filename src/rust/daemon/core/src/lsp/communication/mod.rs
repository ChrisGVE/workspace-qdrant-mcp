//! JSON-RPC Communication Module
//!
//! This module provides JSON-RPC protocol implementation for LSP server communication
//! over stdio and TCP connections with timeout handling and correlation.
//!
//! # Submodules
//!
//! - [`jsonrpc`] - JSON-RPC 2.0 message types and parsing
//! - [`transport`] - Stdio transport, pending request correlation, client lifecycle

pub mod jsonrpc;
pub mod transport;

#[cfg(test)]
mod tests;

// Re-export all public types for backward compatibility
pub use jsonrpc::{
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
};
pub use transport::JsonRpcClient;
