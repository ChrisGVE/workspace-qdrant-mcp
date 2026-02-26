//! Unit tests for JSON-RPC communication

use tokio::sync::oneshot;
use tokio::time::Duration;

use super::jsonrpc::*;
use super::transport::*;

#[test]
fn test_json_rpc_request_creation() {
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::Value::Number(1.into()),
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

    // Add a mock pending request via the test helper
    {
        let (tx, _rx) = oneshot::channel();
        client
            .insert_test_pending_request(
                1,
                tx,
                std::time::Instant::now() - Duration::from_secs(1),
            )
            .await;
    }

    // Wait a bit to ensure expiry
    tokio::time::sleep(Duration::from_millis(10)).await;

    let expired = client.cleanup_expired_requests().await;
    assert_eq!(expired, 1);

    let stats = client.get_stats().await;
    assert_eq!(stats.get("pending_requests").unwrap().as_u64(), Some(0));
}
