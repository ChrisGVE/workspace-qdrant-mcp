//! GENERATED per-case parity wrappers for `stable_stringify` (do not edit by hand).
//! Regenerate via tmp capture harness. One #[test] per corpus row.

use super::asserts::assert_stable_stringify;

#[test]
fn ss_empty_object() {
    assert_stable_stringify("empty_object");
}

#[test]
fn ss_empty_array() {
    assert_stable_stringify("empty_array");
}

#[test]
fn ss_null() {
    assert_stable_stringify("null");
}

#[test]
fn ss_bool_true() {
    assert_stable_stringify("bool_true");
}

#[test]
fn ss_bool_false() {
    assert_stable_stringify("bool_false");
}

#[test]
fn ss_int() {
    assert_stable_stringify("int");
}

#[test]
fn ss_negative_int() {
    assert_stable_stringify("negative_int");
}

#[test]
fn ss_zero() {
    assert_stable_stringify("zero");
}

#[test]
fn ss_float() {
    assert_stable_stringify("float");
}

#[test]
fn ss_string_simple() {
    assert_stable_stringify("string_simple");
}

#[test]
fn ss_string_empty() {
    assert_stable_stringify("string_empty");
}

#[test]
fn ss_string_quotes() {
    assert_stable_stringify("string_quotes");
}

#[test]
fn ss_string_backslash() {
    assert_stable_stringify("string_backslash");
}

#[test]
fn ss_string_newline() {
    assert_stable_stringify("string_newline");
}

#[test]
fn ss_string_tab() {
    assert_stable_stringify("string_tab");
}

#[test]
fn ss_string_unicode() {
    assert_stable_stringify("string_unicode");
}

#[test]
fn ss_string_emoji_surrogate() {
    assert_stable_stringify("string_emoji_surrogate");
}

#[test]
fn ss_keys_unsorted() {
    assert_stable_stringify("keys_unsorted");
}

#[test]
fn ss_keys_caps_vs_lower() {
    assert_stable_stringify("keys_caps_vs_lower");
}

#[test]
fn ss_keys_numeric_strings() {
    assert_stable_stringify("keys_numeric_strings");
}

#[test]
fn ss_keys_unicode() {
    assert_stable_stringify("keys_unicode");
}

#[test]
fn ss_nested_object() {
    assert_stable_stringify("nested_object");
}

#[test]
fn ss_array_of_objects() {
    assert_stable_stringify("array_of_objects");
}

#[test]
fn ss_array_mixed() {
    assert_stable_stringify("array_mixed");
}

#[test]
fn ss_deep_nesting() {
    assert_stable_stringify("deep_nesting");
}

#[test]
fn ss_value_with_null_field() {
    assert_stable_stringify("value_with_null_field");
}

#[test]
fn ss_value_with_empty_nested() {
    assert_stable_stringify("value_with_empty_nested");
}

#[test]
fn ss_large_int() {
    assert_stable_stringify("large_int");
}

#[test]
fn ss_string_slash() {
    assert_stable_stringify("string_slash");
}

#[test]
fn ss_string_control() {
    assert_stable_stringify("string_control");
}

#[test]
fn ss_keys_with_spaces() {
    assert_stable_stringify("keys_with_spaces");
}

#[test]
fn ss_unicode_key_sort() {
    assert_stable_stringify("unicode_key_sort");
}
