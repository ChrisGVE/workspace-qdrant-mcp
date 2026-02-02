//! Periodic path validation for project cleanup.
//!
//! This module provides functionality to periodically validate that watched project
//! paths still exist on the filesystem. When a project folder is deleted or moved
//! externally (cross-filesystem move), this validator detects the orphaned project
//! and triggers cleanup.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during path validation
#[derive(Error, Debug)]
pub enum PathValidatorError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Validation error: {0}")]
    Validation(String),
}

/// Configuration for the path validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathValidatorConfig {
    /// Validation interval in hours
    pub validation_interval_hours: u64,

    /// Whether path validation is enabled
    pub enabled: bool,

    /// Grace period before marking a project as orphaned (minutes)
    /// This allows for temporary unmounts or brief unavailability
    pub grace_period_minutes: u64,

    /// Maximum number of paths to validate per cycle
    pub max_paths_per_cycle: usize,
}

impl Default for PathValidatorConfig {
    fn default() -> Self {
        Self {
            validation_interval_hours: 1,
            enabled: true,
            grace_period_minutes: 5,
            max_paths_per_cycle: 1000,
        }
    }
}

/// Represents a project that may have been orphaned (path no longer exists)
#[derive(Debug, Clone)]
pub struct OrphanedProject {
    /// Project identifier
    pub project_id: String,

    /// Original path on disk
    pub path: PathBuf,

    /// When the path was first detected as missing
    pub first_missing: Instant,

    /// Number of consecutive validation failures
    pub failure_count: u32,
}

/// Information about a registered project for validation
#[derive(Debug, Clone)]
pub struct RegisteredProject {
    /// Project identifier (tenant_id)
    pub project_id: String,

    /// Project root path
    pub path: PathBuf,

    /// Whether the project is currently active (has watchers)
    pub is_active: bool,
}

/// Validates that project paths still exist on the filesystem
pub struct PathValidator {
    /// Configuration
    config: PathValidatorConfig,

    /// Last validation time
    last_validation: Arc<RwLock<Instant>>,

    /// Timer reset flag (set by folder operations)
    timer_reset_pending: Arc<RwLock<bool>>,

    /// Projects that have been detected as missing but are within grace period
    pending_orphans: Arc<RwLock<Vec<OrphanedProject>>>,
}

impl PathValidator {
    /// Create a new path validator with default configuration
    pub fn new() -> Self {
        Self::with_config(PathValidatorConfig::default())
    }

    /// Create a new path validator with custom configuration
    pub fn with_config(config: PathValidatorConfig) -> Self {
        Self {
            config,
            last_validation: Arc::new(RwLock::new(Instant::now())),
            timer_reset_pending: Arc::new(RwLock::new(false)),
            pending_orphans: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Check if validation is due based on the interval
    pub async fn is_validation_due(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Check if timer was reset
        let mut reset_pending = self.timer_reset_pending.write().await;
        if *reset_pending {
            *reset_pending = false;
            let mut last = self.last_validation.write().await;
            *last = Instant::now();
            return false;
        }

        let last = self.last_validation.read().await;
        let interval = Duration::from_secs(self.config.validation_interval_hours * 3600);
        last.elapsed() >= interval
    }

    /// Reset the validation timer (call on folder operations)
    pub async fn reset_timer(&self) {
        let mut reset_pending = self.timer_reset_pending.write().await;
        *reset_pending = true;
    }

    /// Validate a batch of registered projects
    ///
    /// Returns projects that are confirmed orphaned (path doesn't exist
    /// and grace period has expired)
    pub async fn validate_projects(
        &self,
        projects: Vec<RegisteredProject>,
    ) -> Result<Vec<OrphanedProject>, PathValidatorError> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut last = self.last_validation.write().await;
        *last = Instant::now();
        drop(last);

        let grace_period = Duration::from_secs(self.config.grace_period_minutes * 60);
        let mut confirmed_orphans = Vec::new();
        let mut new_pending = Vec::new();

        // Get current pending orphans
        let mut pending = self.pending_orphans.write().await;

        // Create a set of pending project IDs for quick lookup
        let pending_ids: HashSet<String> = pending
            .iter()
            .map(|o| o.project_id.clone())
            .collect();

        // Limit the number of paths to validate
        let projects_to_check: Vec<_> = projects
            .into_iter()
            .take(self.config.max_paths_per_cycle)
            .collect();

        for project in projects_to_check {
            let path_exists = project.path.exists();

            if path_exists {
                // Path exists - remove from pending if it was there
                pending.retain(|o| o.project_id != project.project_id);
            } else {
                // Path doesn't exist
                if let Some(existing) = pending.iter_mut().find(|o| o.project_id == project.project_id) {
                    // Already pending - check grace period
                    existing.failure_count += 1;

                    if existing.first_missing.elapsed() >= grace_period {
                        // Grace period expired - confirmed orphan
                        confirmed_orphans.push(OrphanedProject {
                            project_id: project.project_id.clone(),
                            path: project.path.clone(),
                            first_missing: existing.first_missing,
                            failure_count: existing.failure_count,
                        });

                        tracing::info!(
                            project_id = %project.project_id,
                            path = %project.path.display(),
                            failures = existing.failure_count,
                            "Project confirmed orphaned after grace period"
                        );
                    }
                } else if !pending_ids.contains(&project.project_id) {
                    // New missing path - add to pending
                    new_pending.push(OrphanedProject {
                        project_id: project.project_id.clone(),
                        path: project.path.clone(),
                        first_missing: Instant::now(),
                        failure_count: 1,
                    });

                    tracing::warn!(
                        project_id = %project.project_id,
                        path = %project.path.display(),
                        "Project path missing, starting grace period"
                    );
                }
            }
        }

        // Remove confirmed orphans from pending
        pending.retain(|o| !confirmed_orphans.iter().any(|c| c.project_id == o.project_id));

        // Add new pending orphans
        pending.extend(new_pending);

        Ok(confirmed_orphans)
    }

    /// Get the list of currently pending (possibly orphaned) projects
    pub async fn get_pending_orphans(&self) -> Vec<OrphanedProject> {
        let pending = self.pending_orphans.read().await;
        pending.clone()
    }

    /// Clear all pending orphans
    pub async fn clear_pending(&self) {
        let mut pending = self.pending_orphans.write().await;
        pending.clear();
    }

    /// Get validation statistics
    pub async fn stats(&self) -> PathValidatorStats {
        let last = self.last_validation.read().await;
        let pending = self.pending_orphans.read().await;

        PathValidatorStats {
            enabled: self.config.enabled,
            validation_interval_hours: self.config.validation_interval_hours,
            time_since_last_validation_secs: last.elapsed().as_secs(),
            pending_orphan_count: pending.len(),
        }
    }
}

impl Default for PathValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the path validator
#[derive(Debug, Clone, Default)]
pub struct PathValidatorStats {
    pub enabled: bool,
    pub validation_interval_hours: u64,
    pub time_since_last_validation_secs: u64,
    pub pending_orphan_count: usize,
}

/// Actions to perform when a project is confirmed orphaned
#[derive(Debug, Clone)]
pub struct OrphanCleanupActions {
    /// Project ID to clean up
    pub project_id: String,

    /// Original path
    pub path: PathBuf,
}

impl OrphanCleanupActions {
    /// Get SQL statements to clean up an orphaned project from SQLite
    ///
    /// Returns a list of (table_name, condition) tuples for deletion
    pub fn sqlite_cleanup_statements(&self) -> Vec<(&'static str, String)> {
        let project_id = &self.project_id;
        let path_str = self.path.to_string_lossy();

        // NOTE: registered_projects table has been consolidated into watch_folders
        // per WORKSPACE_QDRANT_MCP.md v1.6.2+
        vec![
            ("unified_queue", format!("tenant_id = '{}'", project_id)),
            ("watch_folders", format!("path = '{}' OR path LIKE '{}/%'", path_str, path_str)),
        ]
    }

    /// Get Qdrant filter for deleting tenant data
    pub fn qdrant_tenant_filter(&self) -> String {
        format!("tenant_id == \"{}\"", self.project_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::thread::sleep;

    #[tokio::test]
    async fn test_new_validator() {
        let validator = PathValidator::new();
        let stats = validator.stats().await;

        assert!(stats.enabled);
        assert_eq!(stats.validation_interval_hours, 1);
        assert_eq!(stats.pending_orphan_count, 0);
    }

    #[tokio::test]
    async fn test_validation_not_due_initially() {
        let validator = PathValidator::new();

        // Validation should not be due immediately after creation
        assert!(!validator.is_validation_due().await);
    }

    #[tokio::test]
    async fn test_timer_reset() {
        let config = PathValidatorConfig {
            validation_interval_hours: 0, // Would be due immediately
            enabled: true,
            ..Default::default()
        };
        let validator = PathValidator::with_config(config);

        // Reset the timer
        validator.reset_timer().await;

        // Next check should return false due to reset
        assert!(!validator.is_validation_due().await);
    }

    #[tokio::test]
    async fn test_validate_existing_paths() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        let validator = PathValidator::new();

        let projects = vec![RegisteredProject {
            project_id: "test-project".to_string(),
            path: path.clone(),
            is_active: true,
        }];

        let orphans = validator.validate_projects(projects).await.unwrap();

        // Path exists, so no orphans
        assert!(orphans.is_empty());
        assert!(validator.get_pending_orphans().await.is_empty());
    }

    #[tokio::test]
    async fn test_validate_missing_path_enters_grace_period() {
        let validator = PathValidator::new();

        let projects = vec![RegisteredProject {
            project_id: "missing-project".to_string(),
            path: PathBuf::from("/nonexistent/path/that/does/not/exist"),
            is_active: true,
        }];

        let orphans = validator.validate_projects(projects).await.unwrap();

        // Path doesn't exist, but grace period hasn't expired
        assert!(orphans.is_empty());

        // Should be in pending
        let pending = validator.get_pending_orphans().await;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].project_id, "missing-project");
    }

    #[tokio::test]
    async fn test_orphan_confirmed_after_grace_period() {
        let config = PathValidatorConfig {
            validation_interval_hours: 1,
            enabled: true,
            grace_period_minutes: 0, // Immediate confirmation for testing
            max_paths_per_cycle: 1000,
        };
        let validator = PathValidator::with_config(config);

        let projects = vec![RegisteredProject {
            project_id: "missing-project".to_string(),
            path: PathBuf::from("/nonexistent/path"),
            is_active: true,
        }];

        // First validation - enters grace period
        let orphans = validator.validate_projects(projects.clone()).await.unwrap();
        assert!(orphans.is_empty());

        // Small delay to ensure time passes
        sleep(Duration::from_millis(10));

        // Second validation - grace period expired
        let orphans = validator.validate_projects(projects).await.unwrap();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].project_id, "missing-project");
    }

    #[tokio::test]
    async fn test_path_recovers_during_grace_period() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        let config = PathValidatorConfig {
            validation_interval_hours: 1,
            enabled: true,
            grace_period_minutes: 60, // Long grace period
            max_paths_per_cycle: 1000,
        };
        let validator = PathValidator::with_config(config);

        // First - path doesn't exist
        let projects = vec![RegisteredProject {
            project_id: "recovering-project".to_string(),
            path: PathBuf::from("/nonexistent/path"),
            is_active: true,
        }];

        validator.validate_projects(projects).await.unwrap();
        assert_eq!(validator.get_pending_orphans().await.len(), 1);

        // Second - path now exists (use temp_dir which exists)
        let projects = vec![RegisteredProject {
            project_id: "recovering-project".to_string(),
            path,
            is_active: true,
        }];

        validator.validate_projects(projects).await.unwrap();

        // Should be removed from pending
        assert!(validator.get_pending_orphans().await.is_empty());
    }

    #[tokio::test]
    async fn test_cleanup_statements() {
        let actions = OrphanCleanupActions {
            project_id: "test-project".to_string(),
            path: PathBuf::from("/path/to/project"),
        };

        let statements = actions.sqlite_cleanup_statements();
        // NOTE: registered_projects removed - consolidated into watch_folders
        assert_eq!(statements.len(), 2);

        // Check that statements contain expected tables
        assert!(statements.iter().any(|(table, _)| *table == "unified_queue"));
        assert!(statements.iter().any(|(table, _)| *table == "watch_folders"));
    }

    #[tokio::test]
    async fn test_disabled_validator() {
        let config = PathValidatorConfig {
            enabled: false,
            ..Default::default()
        };
        let validator = PathValidator::with_config(config);

        // Should never be due when disabled
        assert!(!validator.is_validation_due().await);

        // Validation should return empty when disabled
        let projects = vec![RegisteredProject {
            project_id: "test".to_string(),
            path: PathBuf::from("/nonexistent"),
            is_active: true,
        }];

        let orphans = validator.validate_projects(projects).await.unwrap();
        assert!(orphans.is_empty());
    }
}
