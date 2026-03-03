use super::super::*;

#[test]
fn test_sql_constant_is_valid() {
    assert!(CREATE_SCHEMA_VERSION_SQL.contains("CREATE TABLE"));
    assert!(CREATE_SCHEMA_VERSION_SQL.contains("schema_version"));
    assert!(CREATE_SCHEMA_VERSION_SQL.contains("version INTEGER PRIMARY KEY"));
}

#[test]
fn test_current_version_is_positive() {
    assert!(CURRENT_SCHEMA_VERSION > 0);
}

#[test]
fn test_build_registry_has_all_migrations() {
    let registry = SchemaManager::build_registry();
    for v in 1..=CURRENT_SCHEMA_VERSION {
        assert!(
            registry.get(v).is_some(),
            "Migration v{} should be registered",
            v
        );
        assert_eq!(registry.get(v).unwrap().version(), v);
    }
}
