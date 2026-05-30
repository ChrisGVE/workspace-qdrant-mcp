//! GENERATED per-case parity wrappers for `stable_stringify` (do not edit by hand).
//! Regenerate via the tmp capture+codegen harness. One #[test] per corpus row.

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

#[test]
fn ss_negative_zero() {
    assert_stable_stringify("negative_zero");
}

#[test]
fn ss_small_float() {
    assert_stable_stringify("small_float");
}

#[test]
fn ss_very_small_float() {
    assert_stable_stringify("very_small_float");
}

#[test]
fn ss_large_float() {
    assert_stable_stringify("large_float");
}

#[test]
fn ss_negative_float() {
    assert_stable_stringify("negative_float");
}

#[test]
fn ss_float_trailing_zero() {
    assert_stable_stringify("float_trailing_zero");
}

#[test]
fn ss_min_safe_int() {
    assert_stable_stringify("min_safe_int");
}

#[test]
fn ss_exponent_pos() {
    assert_stable_stringify("exponent_pos");
}

#[test]
fn ss_one_third_approx() {
    assert_stable_stringify("one_third_approx");
}

#[test]
fn ss_max_u32() {
    assert_stable_stringify("max_u32");
}

#[test]
fn ss_string_backspace() {
    assert_stable_stringify("string_backspace");
}

#[test]
fn ss_string_formfeed() {
    assert_stable_stringify("string_formfeed");
}

#[test]
fn ss_string_carriage_return() {
    assert_stable_stringify("string_carriage_return");
}

#[test]
fn ss_string_all_escapes() {
    assert_stable_stringify("string_all_escapes");
}

#[test]
fn ss_string_null_char() {
    assert_stable_stringify("string_null_char");
}

#[test]
fn ss_string_unit_separator() {
    assert_stable_stringify("string_unit_separator");
}

#[test]
fn ss_string_vertical_tab() {
    assert_stable_stringify("string_vertical_tab");
}

#[test]
fn ss_string_escape_char() {
    assert_stable_stringify("string_escape_char");
}

#[test]
fn ss_string_start_of_heading() {
    assert_stable_stringify("string_start_of_heading");
}

#[test]
fn ss_string_mixed_control() {
    assert_stable_stringify("string_mixed_control");
}

#[test]
fn ss_string_combining_mark() {
    assert_stable_stringify("string_combining_mark");
}

#[test]
fn ss_string_rtl() {
    assert_stable_stringify("string_rtl");
}

#[test]
fn ss_string_cjk_mixed() {
    assert_stable_stringify("string_cjk_mixed");
}

#[test]
fn ss_string_multi_emoji() {
    assert_stable_stringify("string_multi_emoji");
}

#[test]
fn ss_string_zwj_family() {
    assert_stable_stringify("string_zwj_family");
}

#[test]
fn ss_string_skin_tone() {
    assert_stable_stringify("string_skin_tone");
}

#[test]
fn ss_string_bmp_and_astral() {
    assert_stable_stringify("string_bmp_and_astral");
}

#[test]
fn ss_string_surrogate_pair_only() {
    assert_stable_stringify("string_surrogate_pair_only");
}

#[test]
fn ss_string_nbsp() {
    assert_stable_stringify("string_nbsp");
}

#[test]
fn ss_string_bom() {
    assert_stable_stringify("string_bom");
}

#[test]
fn ss_empty_string_key() {
    assert_stable_stringify("empty_string_key");
}

#[test]
fn ss_keys_digits_mixed_case() {
    assert_stable_stringify("keys_digits_mixed_case");
}

#[test]
fn ss_keys_underscore_dash() {
    assert_stable_stringify("keys_underscore_dash");
}

#[test]
fn ss_keys_dot_paths() {
    assert_stable_stringify("keys_dot_paths");
}

#[test]
fn ss_keys_utf16_codeunit_sort() {
    assert_stable_stringify("keys_utf16_codeunit_sort");
}

#[test]
fn ss_keys_emoji() {
    assert_stable_stringify("keys_emoji");
}

#[test]
fn ss_keys_leading_digit() {
    assert_stable_stringify("keys_leading_digit");
}

#[test]
fn ss_keys_whitespace_variants() {
    assert_stable_stringify("keys_whitespace_variants");
}

#[test]
fn ss_deep_nesting_5() {
    assert_stable_stringify("deep_nesting_5");
}

#[test]
fn ss_nested_arrays() {
    assert_stable_stringify("nested_arrays");
}

#[test]
fn ss_array_of_nested_objects_unsorted() {
    assert_stable_stringify("array_of_nested_objects_unsorted");
}

#[test]
fn ss_mixed_deep() {
    assert_stable_stringify("mixed_deep");
}

#[test]
fn ss_object_with_all_value_types() {
    assert_stable_stringify("object_with_all_value_types");
}

#[test]
fn ss_array_with_nulls() {
    assert_stable_stringify("array_with_nulls");
}

#[test]
fn ss_array_of_strings_escaped() {
    assert_stable_stringify("array_of_strings_escaped");
}

#[test]
fn ss_object_unicode_values() {
    assert_stable_stringify("object_unicode_values");
}

#[test]
fn ss_object_key_and_value_unicode() {
    assert_stable_stringify("object_key_and_value_unicode");
}

#[test]
fn ss_single_key_object() {
    assert_stable_stringify("single_key_object");
}

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
