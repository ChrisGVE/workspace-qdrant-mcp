//! GENERATED per-case parity wrappers for `idempotency` (do not edit by hand).
//! Regenerate via tmp capture harness. One #[test] per corpus row.

use super::asserts::assert_idempotency;

#[test]
fn idem_text_add_simple() {
    assert_idempotency("text_add_simple");
}

#[test]
fn idem_text_update() {
    assert_idempotency("text_update");
}

#[test]
fn idem_text_delete() {
    assert_idempotency("text_delete");
}

#[test]
fn idem_text_uplift() {
    assert_idempotency("text_uplift");
}

#[test]
fn idem_file_add() {
    assert_idempotency("file_add");
}

#[test]
fn idem_file_rename() {
    assert_idempotency("file_rename");
}

#[test]
fn idem_url_add() {
    assert_idempotency("url_add");
}

#[test]
fn idem_website_scan() {
    assert_idempotency("website_scan");
}

#[test]
fn idem_doc_delete() {
    assert_idempotency("doc_delete");
}

#[test]
fn idem_folder_scan() {
    assert_idempotency("folder_scan");
}

#[test]
fn idem_tenant_add() {
    assert_idempotency("tenant_add");
}

#[test]
fn idem_collection_uplift() {
    assert_idempotency("collection_uplift");
}

#[test]
fn idem_unsorted_payload_keys() {
    assert_idempotency("unsorted_payload_keys");
}

#[test]
fn idem_nested_payload() {
    assert_idempotency("nested_payload");
}

#[test]
fn idem_unicode_payload() {
    assert_idempotency("unicode_payload");
}

#[test]
fn idem_empty_payload() {
    assert_idempotency("empty_payload");
}

#[test]
fn idem_payload_with_null() {
    assert_idempotency("payload_with_null");
}

#[test]
fn idem_payload_with_array() {
    assert_idempotency("payload_with_array");
}

#[test]
fn idem_payload_bool_num() {
    assert_idempotency("payload_bool_num");
}

#[test]
fn idem_tenant_special_chars() {
    assert_idempotency("tenant_special_chars");
}
