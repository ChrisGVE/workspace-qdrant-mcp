//! Migration trait and registry for schema version management.
//!
//! Defines the `Migration` trait that each per-version migration implements,
//! and the `MigrationRegistry` for version-based dispatch.

use async_trait::async_trait;
use sqlx::SqlitePool;

use super::SchemaError;

/// Trait for individual schema migrations.
///
/// Each migration version implements this trait in its own file (v01.rs, v02.rs, etc.).
#[async_trait]
pub trait Migration: Send + Sync {
    /// Execute the migration.
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError>;

    /// The version number this migration upgrades to.
    fn version(&self) -> i32;

    /// Human-readable description of the migration.
    fn description(&self) -> &'static str;
}

/// Registry of all available migrations, keyed by version number.
pub struct MigrationRegistry {
    migrations: Vec<Box<dyn Migration>>,
}

impl MigrationRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            migrations: Vec::new(),
        }
    }

    /// Register a migration.
    pub fn register(&mut self, migration: Box<dyn Migration>) {
        self.migrations.push(migration);
    }

    /// Get a migration by version number.
    pub fn get(&self, version: i32) -> Option<&dyn Migration> {
        self.migrations
            .iter()
            .find(|m| m.version() == version)
            .map(|m| m.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestMigration {
        ver: i32,
    }

    #[async_trait]
    impl Migration for TestMigration {
        async fn up(&self, _pool: &SqlitePool) -> Result<(), SchemaError> {
            Ok(())
        }
        fn version(&self) -> i32 {
            self.ver
        }
        fn description(&self) -> &'static str {
            "test migration"
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = MigrationRegistry::new();
        assert!(registry.get(1).is_none());
    }

    #[test]
    fn test_registry_register_and_lookup() {
        let mut registry = MigrationRegistry::new();
        registry.register(Box::new(TestMigration { ver: 1 }));
        registry.register(Box::new(TestMigration { ver: 2 }));

        assert!(registry.get(1).is_some());
        assert_eq!(registry.get(1).unwrap().version(), 1);
        assert!(registry.get(2).is_some());
        assert_eq!(registry.get(2).unwrap().version(), 2);
    }

    #[test]
    fn test_registry_missing_version() {
        let mut registry = MigrationRegistry::new();
        registry.register(Box::new(TestMigration { ver: 1 }));
        assert!(registry.get(99).is_none());
    }
}
