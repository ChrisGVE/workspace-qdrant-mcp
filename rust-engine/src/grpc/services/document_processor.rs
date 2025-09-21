//! Document processing gRPC service implementation

use crate::daemon::WorkspaceDaemon;
use crate::proto::{
    document_processor_server::DocumentProcessor,
    ProcessDocumentRequest, ProcessDocumentResponse,
    ProcessingStatusRequest, ProcessingStatusResponse,
    CancelProcessingRequest,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info};

/// Document processor service implementation
#[derive(Debug)]
pub struct DocumentProcessorImpl {
    daemon: Arc<WorkspaceDaemon>,
}

impl DocumentProcessorImpl {
    pub fn new(daemon: Arc<WorkspaceDaemon>) -> Self {
        Self { daemon }
    }
}

#[tonic::async_trait]
impl DocumentProcessor for DocumentProcessorImpl {
    async fn process_document(
        &self,
        request: Request<ProcessDocumentRequest>,
    ) -> Result<Response<ProcessDocumentResponse>, Status> {
        let req = request.into_inner();
        debug!("Processing document: {:?}", req.file_path);

        // TODO: Implement actual document processing
        // This is a placeholder implementation
        let response = ProcessDocumentResponse {
            document_id: uuid::Uuid::new_v4().to_string(),
            status: crate::proto::ProcessingStatus::Completed as i32,
            error_message: String::new(),
            chunks_created: 1,
            extracted_metadata: std::collections::HashMap::new(),
            processed_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    type ProcessDocumentsStream = tokio_stream::wrappers::ReceiverStream<Result<ProcessDocumentResponse, Status>>;

    async fn process_documents(
        &self,
        request: Request<tonic::Streaming<ProcessDocumentRequest>>,
    ) -> Result<Response<Self::ProcessDocumentsStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(128);

        // Spawn task to process the stream
        tokio::spawn(async move {
            while let Some(req) = stream.message().await.transpose() {
                match req {
                    Ok(req) => {
                        debug!("Processing document in stream: {:?}", req.file_path);

                        // TODO: Implement actual processing
                        let response = ProcessDocumentResponse {
                            document_id: uuid::Uuid::new_v4().to_string(),
                            status: crate::proto::ProcessingStatus::Completed as i32,
                            error_message: String::new(),
                            chunks_created: 1,
                            extracted_metadata: std::collections::HashMap::new(),
                            processed_at: Some(prost_types::Timestamp {
                                seconds: chrono::Utc::now().timestamp(),
                                nanos: 0,
                            }),
                        };

                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    },
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn get_processing_status(
        &self,
        request: Request<ProcessingStatusRequest>,
    ) -> Result<Response<ProcessingStatusResponse>, Status> {
        let req = request.into_inner();
        debug!("Getting processing status for: {}", req.operation_id);

        // TODO: Implement actual status tracking
        let response = ProcessingStatusResponse {
            operation_id: req.operation_id,
            status: crate::proto::ProcessingStatus::Completed as i32,
            total_documents: 1,
            processed_documents: 1,
            failed_documents: 0,
            error_messages: vec![],
            started_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp() - 60,
                nanos: 0,
            }),
            updated_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn cancel_processing(
        &self,
        request: Request<CancelProcessingRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        info!("Cancelling processing operation: {}", req.operation_id);

        // TODO: Implement actual cancellation logic

        Ok(Response::new(()))
    }
}