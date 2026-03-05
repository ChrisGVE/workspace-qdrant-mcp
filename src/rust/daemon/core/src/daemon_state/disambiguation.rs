//! Project disambiguation methods (Task 3).

use wqm_common::constants::COLLECTION_PROJECTS;

use super::{DaemonStateManager, DaemonStateResult, WatchFolderRecord};

impl DaemonStateManager {
    /// Find existing watch folders (clones) with the same remote_hash
    ///
    /// Used for duplicate detection when registering a new project.
    /// Returns all top-level watch folders (parent_watch_id IS NULL) with matching remote_hash.
    pub async fn find_clones_by_remote_hash(
        &self,
        remote_hash: &str,
    ) -> DaemonStateResult<Vec<WatchFolderRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived, last_commit_hash, is_git_tracked,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders
            WHERE remote_hash = ?1 AND parent_watch_id IS NULL AND collection = ?2
            ORDER BY created_at ASC
            "#,
        )
        .bind(remote_hash)
        .bind(COLLECTION_PROJECTS)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::new();
        for row in rows {
            results.push(self.row_to_watch_folder(&row)?);
        }

        Ok(results)
    }

    /// Update disambiguation_path and tenant_id for a watch folder
    ///
    /// Used when disambiguating clones. Updates the project_id (tenant_id)
    /// to include the disambiguation component.
    pub async fn update_project_disambiguation(
        &self,
        watch_id: &str,
        new_tenant_id: &str,
        disambiguation_path: &str,
    ) -> DaemonStateResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET tenant_id = ?1,
                disambiguation_path = ?2,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id = ?3
            "#,
        )
        .bind(new_tenant_id)
        .bind(disambiguation_path)
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Register a project with automatic duplicate detection and disambiguation
    ///
    /// This method handles the full disambiguation workflow:
    /// 1. Check for existing clones with the same remote_hash
    /// 2. If duplicates found, compute disambiguation paths for ALL clones
    /// 3. Update existing clones with new disambiguation paths
    /// 4. Store the new project with its disambiguation path
    ///
    /// Returns (WatchFolderRecord, Vec<(old_tenant_id, new_tenant_id)>) where the second
    /// element contains alias mappings for any updated existing projects.
    pub async fn register_project_with_disambiguation(
        &self,
        mut record: WatchFolderRecord,
    ) -> DaemonStateResult<(WatchFolderRecord, Vec<(String, String)>)> {
        use crate::project_disambiguation::{DisambiguationPathComputer, ProjectIdCalculator};
        use std::path::PathBuf;

        let mut aliases: Vec<(String, String)> = Vec::new();

        // Only handle disambiguation for projects with git remotes
        if let Some(ref remote_hash) = record.remote_hash {
            // Find existing clones with the same remote
            let existing_clones = self.find_clones_by_remote_hash(remote_hash).await?;

            if !existing_clones.is_empty() {
                // Collect all paths including the new one
                let mut all_paths: Vec<PathBuf> = existing_clones
                    .iter()
                    .map(|c| PathBuf::from(&c.path))
                    .collect();
                all_paths.push(PathBuf::from(&record.path));

                // Recompute disambiguation for all clones
                let disambig_map = DisambiguationPathComputer::recompute_all(&all_paths);

                let calculator = ProjectIdCalculator::new();

                // Update existing clones with new disambiguation paths
                for clone in &existing_clones {
                    let clone_path = PathBuf::from(&clone.path);
                    if let Some(new_disambig) = disambig_map.get(&clone_path) {
                        // Skip if disambiguation hasn't changed
                        let current_disambig = clone.disambiguation_path.as_deref().unwrap_or("");
                        if new_disambig == current_disambig {
                            continue;
                        }

                        // Calculate new tenant_id with disambiguation
                        let new_tenant_id = calculator.calculate(
                            &clone_path,
                            clone.git_remote_url.as_deref(),
                            if new_disambig.is_empty() {
                                None
                            } else {
                                Some(new_disambig)
                            },
                        );

                        // Record the alias (old_id -> new_id)
                        if clone.tenant_id != new_tenant_id {
                            aliases.push((clone.tenant_id.clone(), new_tenant_id.clone()));
                        }

                        // Update the existing clone
                        self.update_project_disambiguation(
                            &clone.watch_id,
                            &new_tenant_id,
                            new_disambig,
                        )
                        .await?;
                    }
                }

                // Set disambiguation path for the new project
                let new_path = PathBuf::from(&record.path);
                if let Some(new_disambig) = disambig_map.get(&new_path) {
                    record.disambiguation_path = if new_disambig.is_empty() {
                        None
                    } else {
                        Some(new_disambig.clone())
                    };

                    // Recalculate tenant_id with disambiguation
                    record.tenant_id = calculator.calculate(
                        &new_path,
                        record.git_remote_url.as_deref(),
                        record.disambiguation_path.as_deref(),
                    );
                }
            }
        }

        // Store the new project
        self.store_watch_folder(&record).await?;

        Ok((record, aliases))
    }
}
