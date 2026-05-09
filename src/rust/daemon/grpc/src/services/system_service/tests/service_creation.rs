//! Tests for SystemServiceImpl construction and Default trait.

use std::time::SystemTime;

use super::super::service_impl::SystemServiceImpl;

#[tokio::test]
async fn test_service_creation() {
    let service = SystemServiceImpl::new();
    assert!(service.start_time <= SystemTime::now());
}

#[tokio::test]
async fn test_default_trait() {
    let _service = SystemServiceImpl::default();
    // Should not panic
}
