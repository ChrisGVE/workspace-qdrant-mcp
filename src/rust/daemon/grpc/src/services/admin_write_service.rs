//! AdminWriteService gRPC implementation
//!
//! Daemon-exclusive administrative mutations spanning multiple tables.
//! Replaces direct SQLite writes from CLI admin commands.

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};

use crate::proto::{
    admin_write_service_server::AdminWriteService, RebalanceIdfRequest, RebalanceIdfResponse,
    RenameTenantAdminRequest, RenameTenantAdminResponse,
};

pub struct AdminWriteServiceImpl {
    pool: SqlitePool,
}

impl AdminWriteServiceImpl {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[tonic::async_trait]
impl AdminWriteService for AdminWriteServiceImpl {
    async fn rename_tenant_admin(
        &self,
        request: Request<RenameTenantAdminRequest>,
    ) -> Result<Response<RenameTenantAdminResponse>, Status> {
        let req = request.into_inner();

        if req.old_tenant_id.is_empty() || req.new_tenant_id.is_empty() {
            return Err(Status::invalid_argument(
                "old_tenant_id and new_tenant_id must not be empty",
            ));
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Status::internal(format!("transaction error: {}", e)))?;

        let mut total = 0u32;

        let count = sqlx::query("UPDATE watch_folders SET tenant_id = ?1 WHERE tenant_id = ?2")
            .bind(&req.new_tenant_id)
            .bind(&req.old_tenant_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?
            .rows_affected() as u32;
        total += count;

        let count = sqlx::query("UPDATE unified_queue SET tenant_id = ?1 WHERE tenant_id = ?2")
            .bind(&req.new_tenant_id)
            .bind(&req.old_tenant_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?
            .rows_affected() as u32;
        total += count;

        // tracked_files may not have a tenant_id column in all schema versions
        match sqlx::query("UPDATE tracked_files SET tenant_id = ?1 WHERE tenant_id = ?2")
            .bind(&req.new_tenant_id)
            .bind(&req.old_tenant_id)
            .execute(&mut *tx)
            .await
        {
            Ok(r) => total += r.rows_affected() as u32,
            Err(_) => {} // Table may lack column — ignore
        }

        tx.commit()
            .await
            .map_err(|e| Status::internal(format!("commit error: {}", e)))?;

        Ok(Response::new(RenameTenantAdminResponse {
            success: true,
            total_rows_updated: total,
            message: format!(
                "Renamed tenant '{}' → '{}' ({} rows)",
                req.old_tenant_id, req.new_tenant_id, total
            ),
        }))
    }

    async fn rebalance_idf(
        &self,
        request: Request<RebalanceIdfRequest>,
    ) -> Result<Response<RebalanceIdfResponse>, Status> {
        let req = request.into_inner();

        sqlx::query("UPDATE corpus_statistics SET last_corrected_n = ?1 WHERE collection = ?2")
            .bind(req.last_corrected_n)
            .bind(&req.collection)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(RebalanceIdfResponse {
            success: true,
            message: format!(
                "Updated last_corrected_n to {} for collection '{}'",
                req.last_corrected_n, req.collection
            ),
        }))
    }
}
