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
    #[allow(dead_code)]
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
        _request: Request<RenameTenantAdminRequest>,
    ) -> Result<Response<RenameTenantAdminResponse>, Status> {
        Err(Status::unimplemented(
            "RenameTenantAdmin not yet implemented",
        ))
    }

    async fn rebalance_idf(
        &self,
        _request: Request<RebalanceIdfRequest>,
    ) -> Result<Response<RebalanceIdfResponse>, Status> {
        Err(Status::unimplemented("RebalanceIdf not yet implemented"))
    }
}
