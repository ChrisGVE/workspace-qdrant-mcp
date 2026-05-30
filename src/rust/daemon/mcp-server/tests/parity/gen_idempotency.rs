//! GENERATED per-case parity wrappers for `idempotency` (do not edit by hand).
//! Regenerate via the tmp capture+codegen harness. One #[test] per corpus row.

use super::asserts::assert_idempotency;

#[test]
fn idem_pair_text_add() {
    assert_idempotency("pair_text_add");
}

#[test]
fn idem_pair_text_update() {
    assert_idempotency("pair_text_update");
}

#[test]
fn idem_pair_text_delete() {
    assert_idempotency("pair_text_delete");
}

#[test]
fn idem_pair_text_uplift() {
    assert_idempotency("pair_text_uplift");
}

#[test]
fn idem_pair_file_add() {
    assert_idempotency("pair_file_add");
}

#[test]
fn idem_pair_file_update() {
    assert_idempotency("pair_file_update");
}

#[test]
fn idem_pair_file_delete() {
    assert_idempotency("pair_file_delete");
}

#[test]
fn idem_pair_file_rename() {
    assert_idempotency("pair_file_rename");
}

#[test]
fn idem_pair_file_uplift() {
    assert_idempotency("pair_file_uplift");
}

#[test]
fn idem_pair_url_add() {
    assert_idempotency("pair_url_add");
}

#[test]
fn idem_pair_url_update() {
    assert_idempotency("pair_url_update");
}

#[test]
fn idem_pair_url_delete() {
    assert_idempotency("pair_url_delete");
}

#[test]
fn idem_pair_url_uplift() {
    assert_idempotency("pair_url_uplift");
}

#[test]
fn idem_pair_website_add() {
    assert_idempotency("pair_website_add");
}

#[test]
fn idem_pair_website_update() {
    assert_idempotency("pair_website_update");
}

#[test]
fn idem_pair_website_delete() {
    assert_idempotency("pair_website_delete");
}

#[test]
fn idem_pair_website_scan() {
    assert_idempotency("pair_website_scan");
}

#[test]
fn idem_pair_website_uplift() {
    assert_idempotency("pair_website_uplift");
}

#[test]
fn idem_pair_doc_delete() {
    assert_idempotency("pair_doc_delete");
}

#[test]
fn idem_pair_doc_uplift() {
    assert_idempotency("pair_doc_uplift");
}

#[test]
fn idem_pair_folder_delete() {
    assert_idempotency("pair_folder_delete");
}

#[test]
fn idem_pair_folder_scan() {
    assert_idempotency("pair_folder_scan");
}

#[test]
fn idem_pair_folder_rename() {
    assert_idempotency("pair_folder_rename");
}

#[test]
fn idem_pair_tenant_add() {
    assert_idempotency("pair_tenant_add");
}

#[test]
fn idem_pair_tenant_update() {
    assert_idempotency("pair_tenant_update");
}

#[test]
fn idem_pair_tenant_delete() {
    assert_idempotency("pair_tenant_delete");
}

#[test]
fn idem_pair_tenant_scan() {
    assert_idempotency("pair_tenant_scan");
}

#[test]
fn idem_pair_tenant_rename() {
    assert_idempotency("pair_tenant_rename");
}

#[test]
fn idem_pair_tenant_uplift() {
    assert_idempotency("pair_tenant_uplift");
}

#[test]
fn idem_pair_collection_uplift() {
    assert_idempotency("pair_collection_uplift");
}

#[test]
fn idem_pair_collection_reset() {
    assert_idempotency("pair_collection_reset");
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
fn idem_deep_nested_payload() {
    assert_idempotency("deep_nested_payload");
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
fn idem_payload_nested_array_of_objects() {
    assert_idempotency("payload_nested_array_of_objects");
}

#[test]
fn idem_payload_bool_num() {
    assert_idempotency("payload_bool_num");
}

#[test]
fn idem_payload_negative_zero() {
    assert_idempotency("payload_negative_zero");
}

#[test]
fn idem_tenant_special_chars() {
    assert_idempotency("tenant_special_chars");
}

#[test]
fn idem_collection_special_in_payload() {
    assert_idempotency("collection_special_in_payload");
}

#[test]
fn idem_payload_escaped_strings() {
    assert_idempotency("payload_escaped_strings");
}

#[test]
fn idem_payload_empty_string_key() {
    assert_idempotency("payload_empty_string_key");
}

#[test]
fn idem_tenant_unicode() {
    assert_idempotency("tenant_unicode");
}

#[test]
fn idem_collection_with_spaces() {
    assert_idempotency("collection_with_spaces");
}

#[test]
fn idem_payload_long_content() {
    assert_idempotency("payload_long_content");
}
