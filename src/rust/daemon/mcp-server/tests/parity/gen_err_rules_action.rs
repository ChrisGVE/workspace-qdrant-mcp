//! GENERATED per-case error-parity wrappers for `rules_action` (do not edit by hand).
//! Regenerate via the tmp capture+codegen harness. One test per corpus row.

use super::error_asserts::assert_rules_action;

#[test]
fn era_missing() {
    assert_rules_action("missing");
}

#[test]
fn era_null() {
    assert_rules_action("null");
}

#[test]
fn era_empty_string() {
    assert_rules_action("empty_string");
}

#[test]
fn era_number() {
    assert_rules_action("number");
}

#[test]
fn era_zero() {
    assert_rules_action("zero");
}

#[test]
fn era_negative() {
    assert_rules_action("negative");
}

#[test]
fn era_float() {
    assert_rules_action("float");
}

#[test]
fn era_bool_true() {
    assert_rules_action("bool_true");
}

#[test]
fn era_bool_false() {
    assert_rules_action("bool_false");
}

#[test]
fn era_array_one() {
    assert_rules_action("array_one");
}

#[test]
fn era_array_many() {
    assert_rules_action("array_many");
}

#[test]
fn era_array_empty() {
    assert_rules_action("array_empty");
}

#[test]
fn era_array_with_null() {
    assert_rules_action("array_with_null");
}

#[test]
fn era_array_numbers() {
    assert_rules_action("array_numbers");
}

#[test]
fn era_array_nested() {
    assert_rules_action("array_nested");
}

#[test]
fn era_object() {
    assert_rules_action("object");
}

#[test]
fn era_object_nonempty() {
    assert_rules_action("object_nonempty");
}

#[test]
fn era_wrong_string() {
    assert_rules_action("wrong_string");
}

#[test]
fn era_add_caps() {
    assert_rules_action("add_caps");
}

#[test]
fn era_update_caps() {
    assert_rules_action("update_caps");
}

#[test]
fn era_add_leading_space() {
    assert_rules_action("add_leading_space");
}

#[test]
fn era_add_trailing_space() {
    assert_rules_action("add_trailing_space");
}

#[test]
fn era_list_typo() {
    assert_rules_action("list_typo");
}

#[test]
fn era_remove_typo() {
    assert_rules_action("remove_typo");
}

#[test]
fn era_unicode() {
    assert_rules_action("unicode");
}

#[test]
fn era_delete_not_remove() {
    assert_rules_action("delete_not_remove");
}

#[test]
fn era_create_not_add() {
    assert_rules_action("create_not_add");
}

#[test]
fn era_get_not_list() {
    assert_rules_action("get_not_list");
}

#[test]
fn era_valid_add() {
    assert_rules_action("valid_add");
}

#[test]
fn era_valid_update() {
    assert_rules_action("valid_update");
}

#[test]
fn era_valid_remove() {
    assert_rules_action("valid_remove");
}

#[test]
fn era_valid_list() {
    assert_rules_action("valid_list");
}
