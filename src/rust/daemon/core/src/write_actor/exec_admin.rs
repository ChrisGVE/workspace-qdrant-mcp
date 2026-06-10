//! SQL execution for AdminWriteService commands.

use super::actor::WriteActor;
use super::commands::*;

impl WriteActor {
    pub(super) async fn exec_rename_tenant_admin(
        &self,
        data: RenameTenantAdminData,
    ) -> WriteResult<RenameTenantAdminResult> {
        if data.old_tenant_id.is_empty() || data.new_tenant_id.is_empty() {
            return Err("old_tenant_id and new_tenant_id must not be empty".into());
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("transaction error: {}", e))?;

        let mut total = 0u32;

        let count = sqlx::query("UPDATE watch_folders SET tenant_id = ?1 WHERE tenant_id = ?2")
            .bind(&data.new_tenant_id)
            .bind(&data.old_tenant_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("database error: {}", e))?
            .rows_affected() as u32;
        total += count;

        let count = sqlx::query("UPDATE unified_queue SET tenant_id = ?1 WHERE tenant_id = ?2")
            .bind(&data.new_tenant_id)
            .bind(&data.old_tenant_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("database error: {}", e))?
            .rows_affected() as u32;
        total += count;

        // tracked_files may not have a tenant_id column in all schema versions
        match sqlx::query("UPDATE tracked_files SET tenant_id = ?1 WHERE tenant_id = ?2")
            .bind(&data.new_tenant_id)
            .bind(&data.old_tenant_id)
            .execute(&mut *tx)
            .await
        {
            Ok(r) => total += r.rows_affected() as u32,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("no such column") || msg.contains("has no column named") {
                    // Table may lack tenant_id column in older schema versions
                } else {
                    return Err(format!("database error updating tracked_files: {}", e));
                }
            }
        }

        tx.commit()
            .await
            .map_err(|e| format!("commit error: {}", e))?;

        Ok(RenameTenantAdminResult {
            success: true,
            total_rows_updated: total,
            message: format!(
                "Renamed tenant '{}' -> '{}' ({} rows)",
                data.old_tenant_id, data.new_tenant_id, total
            ),
        })
    }

    pub(super) async fn exec_rebalance_idf(
        &self,
        data: RebalanceIdfData,
    ) -> WriteResult<RebalanceIdfResult> {
        sqlx::query("UPDATE corpus_statistics SET last_corrected_n = ?1 WHERE collection = ?2")
            .bind(data.last_corrected_n)
            .bind(&data.collection)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

        Ok(RebalanceIdfResult {
            success: true,
            message: format!(
                "Updated last_corrected_n to {} for collection '{}'",
                data.last_corrected_n, data.collection
            ),
        })
    }

    /// Reapply ignore rules across all active projects.
    ///
    /// Calls into `startup::reconciliation::reconcile_all_ignore_rules`, which
    /// iterates `watch_folders WHERE collection='projects' AND enabled=1`,
    /// loads the current global + per-project ignore rules, and enqueues
    /// `file/delete` for newly-excluded paths and `file/add` for newly-included
    /// paths. Constructs a fresh `QueueManager` over the actor's pool — the
    /// manager is a stateless wrapper, so this is safe alongside the daemon's
    /// long-lived queue processor.
    pub(super) async fn exec_reapply_ignore_rules(&self) -> WriteResult<ReapplyIgnoreRulesResult> {
        let queue_manager = std::sync::Arc::new(crate::queue_operations::QueueManager::new(
            self.pool.clone(),
        ));

        // Count active projects up-front so we can report it (the reconciler
        // itself only returns stale/missing totals, not project count).
        let projects_processed: u32 = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM watch_folders WHERE collection = 'projects' AND enabled = 1",
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?
            as u32;

        let stats =
            crate::startup::reconciliation::reconcile_all_ignore_rules(&self.pool, &queue_manager)
                .await
                .map_err(|e| format!("ignore reconciliation failed: {}", e))?;

        // Prune documents for branches deleted from git. The file watcher
        // excludes `.git/`, so ref deletions are never observed live and the
        // branch-lifecycle event pipeline has no runtime consumer — without this
        // a deleted branch leaves its indexed documents orphaned forever. Runs
        // alongside (startup + on-demand) the ignore reconcile. Non-fatal: a
        // failure here must not discard the ignore-reconcile result above.
        if let Err(e) = crate::startup::reconciliation::branch_prune::prune_orphaned_branches(
            &self.pool,
            &queue_manager,
        )
        .await
        {
            tracing::warn!("branch prune reconciliation failed: {}", e);
        }

        Ok(ReapplyIgnoreRulesResult {
            projects_processed,
            stale_deleted: stats.stale_deleted as u32,
            missing_added: stats.missing_added as u32,
        })
    }

    /// Re-embed a single project in place.
    ///
    /// Enqueues a `folder|scan` for each of the tenant's enabled `watch_folders`
    /// rows so the pipeline re-reads, re-chunks and re-embeds its files. Unlike
    /// the global reembed pipeline this is **non-destructive** — it does not
    /// drop/recreate Qdrant collections (the re-process upserts points by id).
    /// Reuses the canonical `QueueManager::enqueue_unified` so idempotency keys,
    /// dedup, and metrics match the normal ingest path.
    pub(super) async fn exec_reembed_tenant(
        &self,
        data: ReembedTenantData,
    ) -> WriteResult<ReembedTenantResult> {
        let tenant_id = data.tenant_id.trim();
        if tenant_id.is_empty() {
            return Err("tenant_id must not be empty".into());
        }

        let folders = sqlx::query_as::<_, (String, String)>(
            "SELECT path, collection FROM watch_folders WHERE enabled = 1 AND tenant_id = ?1",
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        if folders.is_empty() {
            return Ok(ReembedTenantResult {
                files_enqueued: 0,
                message: format!("no enabled watch folders for tenant '{}'", tenant_id),
            });
        }

        let queue_manager = crate::queue_operations::QueueManager::new(self.pool.clone());
        let mut enqueued = 0u32;
        for (path, collection) in &folders {
            // `folder_path` is deliberately OMITTED: `FolderPayload.folder_path`
            // is `Option<RelativePath>` and `None` means "scan the watch_folder
            // root itself" (spec 16 §3.3). `path` here is the watch root's
            // ABSOLUTE path — putting it in the payload made the folder
            // strategy re-anchor it onto the root (<root>/<root>/…) and the
            // scan no-op'd with "Folder scan target is not a directory", so
            // per-tenant reembed silently did nothing.
            let payload = serde_json::json!({
                "recursive": true,
                "recursive_depth": 10,
                "patterns": [],
                "ignore_patterns": []
            })
            .to_string();
            // Label re-embedded project files under the repo's ACTUAL current
            // branch. Passing None here makes enqueue_unified fall back to "main"
            // (its default for an absent branch); for a repo whose real branch is
            // e.g. "dev-clean"/"master" that silently mislabels the entire corpus
            // under a non-existent "main" — which splits content off the branch the
            // daemon searches AND made deleted-branch reconciliation treat the
            // corpus as an orphan. Resolve the branch like the file-watcher path
            // (enqueue_tenant_scan) does. Libraries are branch-agnostic (None).
            let branch = if collection == "projects" {
                Some(crate::watching_queue::get_current_branch(std::path::Path::new(
                    path,
                )))
            } else {
                None
            };
            let (_, is_new) = queue_manager
                .enqueue_unified(
                    crate::unified_queue_schema::ItemType::Folder,
                    crate::unified_queue_schema::QueueOperation::Scan,
                    tenant_id,
                    collection,
                    &payload,
                    branch.as_deref(),
                    None,
                )
                .await
                .map_err(|e| format!("enqueue folder scan failed: {}", e))?;
            if is_new {
                enqueued += 1;
            }
        }

        Ok(ReembedTenantResult {
            files_enqueued: enqueued,
            message: format!(
                "re-enqueued {} folder scan(s) for tenant '{}' (re-embed in place)",
                enqueued, tenant_id
            ),
        })
    }
}
