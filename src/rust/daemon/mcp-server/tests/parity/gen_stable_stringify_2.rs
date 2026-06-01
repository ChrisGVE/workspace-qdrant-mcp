//! GENERATED per-case parity wrappers for `stable_stringify` (part 2) (do not edit by hand).
//! Regenerate via the tmp capture+codegen harness. One #[test] per corpus row.

use super::asserts::assert_stable_stringify;

#[test]
fn ss_boolean_array() {
    assert_stable_stringify("boolean_array");
}

#[test]
fn ss_numeric_array_sorted_already() {
    assert_stable_stringify("numeric_array_sorted_already");
}

#[test]
fn ss_negative_and_positive() {
    assert_stable_stringify("negative_and_positive");
}

#[test]
fn ss_object_numeric_string_keys_many() {
    assert_stable_stringify("object_numeric_string_keys_many");
}

#[test]
fn ss_array_empty_objects() {
    assert_stable_stringify("array_empty_objects");
}

#[test]
fn ss_array_empty_arrays() {
    assert_stable_stringify("array_empty_arrays");
}

#[test]
fn ss_object_value_is_array_of_objects() {
    assert_stable_stringify("object_value_is_array_of_objects");
}

#[test]
fn ss_deeply_unsorted_keys() {
    assert_stable_stringify("deeply_unsorted_keys");
}

#[test]
fn ss_long_key() {
    assert_stable_stringify("long_key");
}

#[test]
fn ss_key_with_quote() {
    assert_stable_stringify("key_with_quote");
}

#[test]
fn ss_key_with_backslash() {
    assert_stable_stringify("key_with_backslash");
}

#[test]
fn ss_key_with_newline() {
    assert_stable_stringify("key_with_newline");
}

#[test]
fn ss_value_string_with_slash() {
    assert_stable_stringify("value_string_with_slash");
}

#[test]
fn ss_negative_floats_array() {
    assert_stable_stringify("negative_floats_array");
}

#[test]
fn ss_scientific_notation_value() {
    assert_stable_stringify("scientific_notation_value");
}

#[test]
fn ss_tiny_exponent_value() {
    assert_stable_stringify("tiny_exponent_value");
}

#[test]
fn ss_object_with_emoji_key_and_value() {
    assert_stable_stringify("object_with_emoji_key_and_value");
}

#[test]
fn ss_nested_unicode_keys() {
    assert_stable_stringify("nested_unicode_keys");
}

#[test]
fn ss_array_of_mixed_unicode() {
    assert_stable_stringify("array_of_mixed_unicode");
}

#[test]
fn ss_object_bool_and_null_mix() {
    assert_stable_stringify("object_bool_and_null_mix");
}

#[test]
fn ss_triple_nested_arrays() {
    assert_stable_stringify("triple_nested_arrays");
}

#[test]
fn ss_object_keys_case_collision() {
    assert_stable_stringify("object_keys_case_collision");
}

#[test]
fn ss_single_unicode_char() {
    assert_stable_stringify("single_unicode_char");
}

#[test]
fn ss_array_with_empty_strings() {
    assert_stable_stringify("array_with_empty_strings");
}
