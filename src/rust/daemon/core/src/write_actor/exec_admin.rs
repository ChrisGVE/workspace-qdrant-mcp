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

        // tracked_files has no tenant_id column: rows reach their tenant
        // through watch_folders.tenant_id (updated above), so no per-row
        // update is needed here.

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
}
