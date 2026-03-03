//! Unit tests for CollectionService validation helpers

use super::validation::{
    map_distance_metric, validate_alias_name, validate_collection_name, validate_vector_size,
};

#[test]
fn test_validate_collection_name() {
    // Valid names
    assert!(validate_collection_name("test_collection").is_ok());
    assert!(validate_collection_name("my-collection").is_ok());
    assert!(validate_collection_name("collection123").is_ok());
    assert!(validate_collection_name("a_b_c").is_ok());

    // Invalid: too short
    assert!(validate_collection_name("ab").is_err());

    // Invalid: starts with number
    assert!(validate_collection_name("1collection").is_err());

    // Invalid: special characters
    assert!(validate_collection_name("collection@123").is_err());
    assert!(validate_collection_name("collection.name").is_err());

    // Invalid: too long
    let long_name = "a".repeat(256);
    assert!(validate_collection_name(&long_name).is_err());
}

#[test]
fn test_validate_canonical_collection_names() {
    // Canonical collection names per ADR-001 must be valid
    assert!(validate_collection_name("projects").is_ok());
    assert!(validate_collection_name("libraries").is_ok());
    assert!(validate_collection_name("rules").is_ok());

    // Legacy underscore-prefixed names are syntactically valid for migration compatibility
    assert!(validate_collection_name("_projects").is_ok());
    assert!(validate_collection_name("_libraries").is_ok());
    assert!(validate_collection_name("_rules").is_ok());
}

#[test]
fn test_validate_vector_size() {
    // Valid sizes
    assert!(validate_vector_size(384).is_ok());
    assert!(validate_vector_size(768).is_ok());
    assert!(validate_vector_size(1536).is_ok());
    assert!(validate_vector_size(512).is_ok());

    // Invalid: zero or negative
    assert!(validate_vector_size(0).is_err());
    assert!(validate_vector_size(-1).is_err());

    // Invalid: too large
    assert!(validate_vector_size(10001).is_err());
}

#[test]
fn test_map_distance_metric() {
    // Valid metrics
    assert_eq!(map_distance_metric("Cosine").unwrap(), "Cosine");
    assert_eq!(map_distance_metric("Euclidean").unwrap(), "Euclid");
    assert_eq!(map_distance_metric("Dot").unwrap(), "Dot");

    // Invalid metric
    assert!(map_distance_metric("Invalid").is_err());
    assert!(map_distance_metric("manhattan").is_err());
}

#[test]
fn test_validate_alias_name_rejects_canonical() {
    // Canonical collection names must be rejected as aliases
    assert!(validate_alias_name("projects").is_err());
    assert!(validate_alias_name("libraries").is_err());
    assert!(validate_alias_name("rules").is_err());
}

#[test]
fn test_validate_alias_name_allows_non_canonical() {
    // Non-canonical names should be allowed
    assert!(validate_alias_name("my-alias").is_ok());
    assert!(validate_alias_name("project_v2").is_ok());
    assert!(validate_alias_name("lib-backup").is_ok());
    assert!(validate_alias_name("_projects").is_ok());
}
