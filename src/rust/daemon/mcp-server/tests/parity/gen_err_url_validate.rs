//! GENERATED per-case error-parity wrappers for `url_validate` (do not edit by hand).
//! Regenerate via the tmp capture+codegen harness. One test per corpus row.

use super::error_asserts::assert_url_validate;

#[tokio::test]
async fn euv_missing() {
    assert_url_validate("missing").await;
}

#[tokio::test]
async fn euv_null() {
    assert_url_validate("null").await;
}

#[tokio::test]
async fn euv_empty() {
    assert_url_validate("empty").await;
}

#[tokio::test]
async fn euv_whitespace() {
    assert_url_validate("whitespace").await;
}

#[tokio::test]
async fn euv_tab() {
    assert_url_validate("tab").await;
}

#[tokio::test]
async fn euv_number() {
    assert_url_validate("number").await;
}

#[tokio::test]
async fn euv_array() {
    assert_url_validate("array").await;
}

#[tokio::test]
async fn euv_object() {
    assert_url_validate("object").await;
}

#[tokio::test]
async fn euv_valid_http() {
    assert_url_validate("valid_http").await;
}

#[tokio::test]
async fn euv_valid_https_path() {
    assert_url_validate("valid_https_path").await;
}

#[tokio::test]
async fn euv_valid_caps_scheme() {
    assert_url_validate("valid_caps_scheme").await;
}

#[tokio::test]
async fn euv_valid_port() {
    assert_url_validate("valid_port").await;
}

#[tokio::test]
async fn euv_valid_subdomain() {
    assert_url_validate("valid_subdomain").await;
}

#[tokio::test]
async fn euv_valid_localhost() {
    assert_url_validate("valid_localhost").await;
}

#[tokio::test]
async fn euv_valid_with_surrounding_ws() {
    assert_url_validate("valid_with_surrounding_ws").await;
}

#[tokio::test]
async fn euv_scheme_ftp() {
    assert_url_validate("scheme_ftp").await;
}

#[tokio::test]
async fn euv_scheme_file() {
    assert_url_validate("scheme_file").await;
}

#[tokio::test]
async fn euv_scheme_ssh() {
    assert_url_validate("scheme_ssh").await;
}

#[tokio::test]
async fn euv_scheme_gopher() {
    assert_url_validate("scheme_gopher").await;
}

#[tokio::test]
async fn euv_scheme_custom_plus() {
    assert_url_validate("scheme_custom_plus").await;
}

#[tokio::test]
async fn euv_no_scheme_sep() {
    assert_url_validate("no_scheme_sep").await;
}

#[tokio::test]
async fn euv_plain_text() {
    assert_url_validate("plain_text").await;
}

#[tokio::test]
async fn euv_dots_only_host() {
    assert_url_validate("dots_only_host").await;
}

#[tokio::test]
async fn euv_valid_query_frag() {
    assert_url_validate("valid_query_frag").await;
}

#[tokio::test]
async fn euv_valid_ip_host() {
    assert_url_validate("valid_ip_host").await;
}

#[tokio::test]
async fn euv_valid_trailing_slash() {
    assert_url_validate("valid_trailing_slash").await;
}

#[tokio::test]
async fn euv_valid_uppercase_https() {
    assert_url_validate("valid_uppercase_https").await;
}

#[tokio::test]
async fn euv_valid_deep_path() {
    assert_url_validate("valid_deep_path").await;
}

#[tokio::test]
async fn euv_scheme_data() {
    assert_url_validate("scheme_data").await;
}

#[tokio::test]
async fn euv_scheme_ws() {
    assert_url_validate("scheme_ws").await;
}

#[tokio::test]
async fn euv_scheme_wss() {
    assert_url_validate("scheme_wss").await;
}

#[tokio::test]
async fn euv_valid_userinfo() {
    assert_url_validate("valid_userinfo").await;
}

#[tokio::test]
async fn euv_empty_after_trim_newline() {
    assert_url_validate("empty_after_trim_newline").await;
}

#[tokio::test]
async fn euv_valid_https_root() {
    assert_url_validate("valid_https_root").await;
}

#[tokio::test]
async fn euv_valid_http_port_only() {
    assert_url_validate("valid_http_port_only").await;
}

#[tokio::test]
async fn euv_scheme_telnet() {
    assert_url_validate("scheme_telnet").await;
}

#[tokio::test]
async fn euv_scheme_redis() {
    assert_url_validate("scheme_redis").await;
}

#[tokio::test]
async fn euv_valid_hyphen_host() {
    assert_url_validate("valid_hyphen_host").await;
}

#[tokio::test]
async fn euv_valid_numeric_subpath() {
    assert_url_validate("valid_numeric_subpath").await;
}

#[tokio::test]
async fn euv_scheme_jdbc() {
    assert_url_validate("scheme_jdbc").await;
}
