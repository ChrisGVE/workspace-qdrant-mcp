//! AdminWriteService gRPC implementation
//!
//! Delegates all state.db mutations to the WriteActor via WriteActorHandle.

use tonic::{Request, Response, Status};
use workspace_qdrant_core::write_actor::{
    RebalanceIdfData, RenameTenantAdminData, WriteActorHandle,
};

use crate::proto::{
    admin_write_service_server::AdminWriteService, RebalanceIdfRequest, RebalanceIdfResponse,
    RenameTenantAdminRequest, RenameTenantAdminResponse, TriggerReembedRequest,
    TriggerReembedResponse,
};

pub struct AdminWriteServiceImpl {
    write_actor: WriteActorHandle,
}

impl AdminWriteServiceImpl {
    pub fn new(write_actor: WriteActorHandle) -> Self {
        Self { write_actor }
    }
}

fn to_status(err: String) -> Status {
    if err.contains("must not be empty") {
        Status::invalid_argument(err)
    } else {
        Status::internal(err)
    }
}

#[tonic::async_trait]
impl AdminWriteService for AdminWriteServiceImpl {
    async fn rename_tenant_admin(
        &self,
        request: Request<RenameTenantAdminRequest>,
    ) -> Result<Response<RenameTenantAdminResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .rename_tenant_admin(RenameTenantAdminData {
                old_tenant_id: req.old_tenant_id,
                new_tenant_id: req.new_tenant_id,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(RenameTenantAdminResponse {
            success: result.success,
            total_rows_updated: result.total_rows_updated,
            message: result.message,
        }))
    }

    async fn rebalance_idf(
        &self,
        request: Request<RebalanceIdfRequest>,
    ) -> Result<Response<RebalanceIdfResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .rebalance_idf(RebalanceIdfData {
                collection: req.collection,
                last_corrected_n: req.last_corrected_n,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(RebalanceIdfResponse {
            success: result.success,
            message: result.message,
        }))
    }

    async fn trigger_reembed(
        &self,
        _request: Request<TriggerReembedRequest>,
    ) -> Result<Response<TriggerReembedResponse>, Status> {
        Err(Status::unimplemented("TriggerReembed flow not yet wired"))
    }
}
