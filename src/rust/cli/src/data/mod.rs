//! Shared data access layer for the CLI.
//!
//! Provides unified database connections and common query helpers
//! to eliminate duplicate connection code across command modules.

pub mod db;
pub mod health;
pub mod queries;
pub mod tenants;
