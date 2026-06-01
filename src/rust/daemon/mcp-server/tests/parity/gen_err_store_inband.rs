//! GENERATED per-case error-parity wrappers for `store_inband` (do not edit by hand).
//! Regenerate via the tmp capture+codegen harness. One test per corpus row.

use super::error_asserts::assert_store_inband;

#[tokio::test]
async fn esi_content_missing() {
    assert_store_inband("content_missing").await;
}

#[tokio::test]
async fn esi_content_null() {
    assert_store_inband("content_null").await;
}

#[tokio::test]
async fn esi_content_empty() {
    assert_store_inband("content_empty").await;
}

#[tokio::test]
async fn esi_content_whitespace() {
    assert_store_inband("content_whitespace").await;
}

#[tokio::test]
async fn esi_content_tab() {
    assert_store_inband("content_tab").await;
}

#[tokio::test]
async fn esi_content_newlines() {
    assert_store_inband("content_newlines").await;
}

#[tokio::test]
async fn esi_content_mixed_ws() {
    assert_store_inband("content_mixed_ws").await;
}

#[tokio::test]
async fn esi_lib_missing() {
    assert_store_inband("lib_missing").await;
}

#[tokio::test]
async fn esi_lib_empty() {
    assert_store_inband("lib_empty").await;
}

#[tokio::test]
async fn esi_lib_whitespace() {
    assert_store_inband("lib_whitespace").await;
}

#[tokio::test]
async fn esi_lib_tab() {
    assert_store_inband("lib_tab").await;
}

#[tokio::test]
async fn esi_lib_null() {
    assert_store_inband("lib_null").await;
}

#[tokio::test]
async fn esi_lib_missing_unicode_content() {
    assert_store_inband("lib_missing_unicode_content").await;
}

#[tokio::test]
async fn esi_forproject_no_pid() {
    assert_store_inband("forproject_no_pid").await;
}

#[tokio::test]
async fn esi_forproject_empty_pid() {
    assert_store_inband("forproject_empty_pid").await;
}

#[tokio::test]
async fn esi_forproject_ws_pid() {
    assert_store_inband("forproject_ws_pid").await;
}

#[tokio::test]
async fn esi_forproject_tab_pid() {
    assert_store_inband("forproject_tab_pid").await;
}
