//! Tests for JSON-RPC message parsing and client lifecycle.

use crate::lsp::{JsonRpcClient, JsonRpcMessage};

#[test]
fn test_json_rpc_message_parsing() {
    // Test request parsing
    let request_json =
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"foo":"bar"}}"#;
    let message = JsonRpcMessage::parse(request_json).unwrap();

    match message {
        JsonRpcMessage::Request(req) => {
            assert_eq!(req.method, "initialize");
            assert_eq!(req.id, serde_json::json!(1));
            assert!(req.params.is_some());
        }
        _ => panic!("Expected request message"),
    }

    // Test response parsing
    let response_json = r#"{"jsonrpc":"2.0","id":1,"result":"success"}"#;
    let message = JsonRpcMessage::parse(response_json).unwrap();

    match message {
        JsonRpcMessage::Response(resp) => {
            assert_eq!(resp.id, serde_json::json!(1));
            assert_eq!(resp.result, Some(serde_json::json!("success")));
            assert!(resp.error.is_none());
        }
        _ => panic!("Expected response message"),
    }

    // Test notification parsing
    let notification_json =
        r#"{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{}}"#;
    let message = JsonRpcMessage::parse(notification_json).unwrap();

    match message {
        JsonRpcMessage::Notification(notif) => {
            assert_eq!(notif.method, "textDocument/didOpen");
            assert!(notif.params.is_some());
        }
        _ => panic!("Expected notification message"),
    }

    // Test error response parsing
    let error_json =
        r#"{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"Test error"}}"#;
    let message = JsonRpcMessage::parse(error_json).unwrap();

    match message {
        JsonRpcMessage::Response(resp) => {
            assert!(resp.result.is_none());
            assert!(resp.error.is_some());
            let error = resp.error.unwrap();
            assert_eq!(error.code, -1);
            assert_eq!(error.message, "Test error");
        }
        _ => panic!("Expected error response"),
    }
}

#[tokio::test]
async fn test_json_rpc_client_lifecycle() {
    let client = JsonRpcClient::new();

    // Test initial state
    assert!(!client.is_connected().await);

    let stats = client.get_stats().await;
    assert_eq!(stats.get("pending_requests").unwrap().as_u64(), Some(0));
    assert_eq!(stats.get("connected").unwrap().as_bool(), Some(false));

    // Test cleanup of expired requests
    let expired = client.cleanup_expired_requests().await;
    assert_eq!(expired, 0);

    // Test disconnection
    client.disconnect().await;
    assert!(!client.is_connected().await);
}
