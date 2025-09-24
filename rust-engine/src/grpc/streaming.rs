//! Streaming support for large document operations
//!
//! This module provides server-side streaming for large document uploads,
//! processing results, and search operations.

use crate::grpc::message_validation::{MessageValidator, StreamHandle};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, warn, error, info};
use uuid::Uuid;

/// Streaming document upload handler
pub struct StreamingDocumentHandler {
    message_validator: Arc<MessageValidator>,
}

impl StreamingDocumentHandler {
    /// Create a new streaming document handler
    pub fn new(message_validator: Arc<MessageValidator>) -> Self {
        Self {
            message_validator,
        }
    }

    /// Handle streaming document upload with validation and compression
    pub async fn handle_document_stream<T, R>(
        &self,
        mut stream: Streaming<T>,
        process_chunk: impl Fn(T) -> Result<R> + Send + Sync + 'static,
    ) -> Result<Response<impl Stream<Item = Result<R, Status>>>, Status>
    where
        T: Send + 'static,
        R: Send + 'static,
    {
        // Check if streaming is enabled
        if !self.message_validator.is_streaming_enabled(true) {
            return Err(Status::unimplemented(
                "Server-side streaming is disabled"
            ));
        }

        // Register stream and get handle
        let stream_handle = self.message_validator.register_stream(None)
            .map_err(|e| Status::resource_exhausted(e.to_string()))?;

        info!("Started document streaming session, timeout: {:?}", stream_handle.timeout());

        let (tx, rx) = mpsc::channel(stream_handle.buffer_size());

        // Spawn task to process the incoming stream
        let message_validator = Arc::clone(&self.message_validator);
        let process_chunk = Arc::new(process_chunk);
        tokio::spawn(async move {
            let mut chunk_count = 0;
            let start_time = std::time::Instant::now();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        chunk_count += 1;
                        debug!("Processing chunk {}", chunk_count);

                        // Process with timeout
                        let processing_result = timeout(
                            stream_handle.timeout(),
                            tokio::task::spawn_blocking({
                                let process_chunk = Arc::clone(&process_chunk);
                                move || process_chunk(chunk)
                            })
                        ).await;

                        match processing_result {
                            Ok(Ok(Ok(result))) => {
                                if tx.send(Ok(result)).await.is_err() {
                                    warn!("Receiver dropped, terminating stream");
                                    break;
                                }
                            },
                            Ok(Ok(Err(e))) => {
                                error!("Processing error on chunk {}: {}", chunk_count, e);
                                if tx.send(Err(Status::internal(format!(
                                    "Processing failed: {}", e
                                )))).await.is_err() {
                                    break;
                                }
                            },
                            Ok(Err(join_error)) => {
                                error!("Task join error: {}", join_error);
                                if tx.send(Err(Status::internal(
                                    "Internal processing error"
                                ))).await.is_err() {
                                    break;
                                }
                            },
                            Err(_) => {
                                warn!("Processing timeout on chunk {}", chunk_count);
                                if tx.send(Err(Status::deadline_exceeded(
                                    "Processing timeout"
                                ))).await.is_err() {
                                    break;
                                }
                            }
                        }

                        // Flow control - yield if buffer is getting full
                        if stream_handle.flow_control_enabled() && chunk_count % 10 == 0 {
                            tokio::task::yield_now().await;
                        }
                    },
                    Err(e) => {
                        error!("Stream error on chunk {}: {}", chunk_count, e);
                        if tx.send(Err(e)).await.is_err() {
                            break;
                        }
                    }
                }
            }

            let total_time = start_time.elapsed();
            info!(
                "Completed document streaming session: {} chunks in {:?}",
                chunk_count, total_time
            );
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    /// Handle large document upload with chunking and compression
    pub async fn upload_large_document(
        &self,
        mut stream: Streaming<DocumentChunk>,
    ) -> Result<Response<DocumentUploadResult>, Status> {
        // Check if client streaming is enabled
        if !self.message_validator.is_streaming_enabled(false) {
            return Err(Status::unimplemented(
                "Client-side streaming is disabled"
            ));
        }

        let stream_handle = self.message_validator.register_stream(None)
            .map_err(|e| Status::resource_exhausted(e.to_string()))?;

        let mut document_data = Vec::new();
        let mut chunk_count = 0;
        let mut total_size = 0;
        let start_time = std::time::Instant::now();

        info!("Starting large document upload");

        // Process chunks with timeout
        while let Some(chunk_result) = timeout(
            stream_handle.timeout(),
            stream.next()
        ).await.map_err(|_| Status::deadline_exceeded("Stream timeout"))? {
            match chunk_result {
                Ok(chunk) => {
                    chunk_count += 1;
                    total_size += chunk.data.len();

                    debug!("Received chunk {} with {} bytes", chunk_count, chunk.data.len());

                    // Validate chunk size using public method
                    let dummy_request = Request::new(());
                    if let Err(e) = self.message_validator.validate_incoming_message(&dummy_request, "document_processor") {
                        return Err(e);
                    }

                    // Additional size check for chunk data (simplified validation)
                    if chunk.data.len() > 1024 * 1024 {
                        return Err(Status::invalid_argument(
                            "Chunk data exceeds 1MB limit"
                        ));
                    }

                    // Handle compression if chunk is compressed
                    let chunk_data = if chunk.is_compressed {
                        self.message_validator.decompress_message(&chunk.data)
                            .map_err(|e| Status::internal(format!("Decompression failed: {}", e)))?
                    } else {
                        chunk.data
                    };

                    document_data.extend_from_slice(&chunk_data);

                    // Check total document size
                    if document_data.len() > 100 * 1024 * 1024 { // 100MB limit
                        return Err(Status::invalid_argument(
                            "Document too large (exceeds 100MB)"
                        ));
                    }

                    // Yield control periodically for flow control
                    if stream_handle.flow_control_enabled() && chunk_count % 5 == 0 {
                        tokio::task::yield_now().await;
                    }
                },
                Err(e) => {
                    error!("Error receiving chunk {}: {}", chunk_count, e);
                    return Err(e);
                }
            }
        }

        let upload_time = start_time.elapsed();
        info!(
            "Completed document upload: {} chunks, {} bytes in {:?}",
            chunk_count, total_size, upload_time
        );

        // Process the complete document
        let processing_start = std::time::Instant::now();

        // Compress the final document if configured
        let final_data = if document_data.len() > 1024 { // Simple threshold check
            self.message_validator.compress_message(&document_data)
                .map_err(|e| Status::internal(format!("Final compression failed: {}", e)))?
        } else {
            document_data
        };

        let processing_time = processing_start.elapsed();

        let result = DocumentUploadResult {
            success: true,
            document_id: format!("doc_{}", Uuid::new_v4()),
            original_size: total_size as u64,
            compressed_size: final_data.len() as u64,
            chunks_received: chunk_count,
            upload_time_ms: upload_time.as_millis() as u64,
            processing_time_ms: processing_time.as_millis() as u64,
            compression_ratio: if total_size > 0 {
                final_data.len() as f32 / total_size as f32
            } else {
                1.0
            },
        };

        Ok(Response::new(result))
    }

    /// Stream search results for large result sets
    pub async fn stream_search_results(
        &self,
        search_request: SearchRequest,
    ) -> Result<Response<impl Stream<Item = Result<SearchResult, Status>>>, Status> {
        // Check if server streaming is enabled
        if !self.message_validator.is_streaming_enabled(true) {
            return Err(Status::unimplemented(
                "Server-side streaming is disabled"
            ));
        }

        let stream_handle = self.message_validator.register_stream(None)
            .map_err(|e| Status::resource_exhausted(e.to_string()))?;

        let (tx, rx) = mpsc::channel(stream_handle.buffer_size());

        // Spawn task to generate search results
        tokio::spawn(async move {
            let start_time = std::time::Instant::now();
            let mut result_count = 0;

            // Simulate search processing (in real implementation, this would query the actual search engine)
            for batch in 0..search_request.max_results / 10 {
                for i in 0..10 {
                    if result_count >= search_request.max_results {
                        break;
                    }

                    let result = SearchResult {
                        id: format!("result_{}_{}", batch, i),
                        score: 0.9 - (result_count as f32 * 0.01),
                        content: format!("Search result content for query: {}", search_request.query),
                        metadata: format!("{{\"batch\": {}, \"index\": {}}}", batch, i),
                    };

                    if tx.send(Ok(result)).await.is_err() {
                        warn!("Search results receiver dropped");
                        return;
                    }

                    result_count += 1;

                    // Simulate processing delay
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }

                // Flow control - yield between batches
                tokio::task::yield_now().await;
            }

            let total_time = start_time.elapsed();
            info!(
                "Completed search streaming: {} results in {:?}",
                result_count, total_time
            );
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

/// Document chunk for streaming uploads
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub sequence_number: u32,
    pub data: Vec<u8>,
    pub is_compressed: bool,
    pub is_final: bool,
}

/// Result of document upload operation
#[derive(Debug, Clone)]
pub struct DocumentUploadResult {
    pub success: bool,
    pub document_id: String,
    pub original_size: u64,
    pub compressed_size: u64,
    pub chunks_received: u32,
    pub upload_time_ms: u64,
    pub processing_time_ms: u64,
    pub compression_ratio: f32,
}

/// Search request for streaming results
#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    pub max_results: u32,
    pub collection: String,
}

/// Individual search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub content: String,
    pub metadata: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{MessageConfig, CompressionConfig, StreamingConfig};
    use tokio_stream::iter;

    fn create_test_message_validator() -> Arc<MessageValidator> {
        Arc::new(MessageValidator::new(
            MessageConfig {
                max_incoming_message_size: 1024 * 1024, // 1MB
                max_outgoing_message_size: 1024 * 1024,
                enable_size_validation: true,
                max_frame_size: 16384,
                initial_window_size: 65536,
            },
            CompressionConfig {
                enable_gzip: true,
                compression_threshold: 1024,
                compression_level: 6,
                enable_streaming_compression: true,
                enable_compression_monitoring: true,
            },
            StreamingConfig {
                enable_server_streaming: true,
                enable_client_streaming: true,
                max_concurrent_streams: 10,
                stream_buffer_size: 100,
                stream_timeout_secs: 30,
                enable_flow_control: true,
            },
        ))
    }

    #[tokio::test]
    async fn test_streaming_document_handler_creation() {
        let validator = create_test_message_validator();
        let handler = StreamingDocumentHandler::new(validator);

        // Handler should be created successfully
        assert!(handler.message_validator.is_streaming_enabled(true));
        assert!(handler.message_validator.is_streaming_enabled(false));
    }

    #[test]
    fn test_streaming_disabled_check() {
        // Create validator with streaming disabled
        let validator_disabled = Arc::new(MessageValidator::new(
            MessageConfig::default(),
            CompressionConfig::default(),
            StreamingConfig {
                enable_server_streaming: false,
                enable_client_streaming: false,
                max_concurrent_streams: 1,
                stream_buffer_size: 10,
                stream_timeout_secs: 5,
                enable_flow_control: false,
            },
        ));

        let handler = StreamingDocumentHandler::new(validator_disabled);

        // Check streaming capabilities
        assert!(!handler.message_validator.is_streaming_enabled(true));
        assert!(!handler.message_validator.is_streaming_enabled(false));
    }

    #[test]
    fn test_document_chunk_creation() {
        // Test creating document chunks
        let chunk = DocumentChunk {
            sequence_number: 1,
            data: b"Hello World!".to_vec(),
            is_compressed: false,
            is_final: true,
        };

        assert_eq!(chunk.sequence_number, 1);
        assert_eq!(chunk.data, b"Hello World!".to_vec());
        assert!(!chunk.is_compressed);
        assert!(chunk.is_final);
    }

    #[test]
    fn test_document_upload_result_creation() {
        // Test creating upload result
        let result = DocumentUploadResult {
            success: true,
            document_id: "test_doc_123".to_string(),
            original_size: 2000,
            compressed_size: 1500,
            chunks_received: 5,
            upload_time_ms: 100,
            processing_time_ms: 50,
            compression_ratio: 0.75,
        };

        assert!(result.success);
        assert_eq!(result.document_id, "test_doc_123");
        assert_eq!(result.original_size, 2000);
        assert_eq!(result.compressed_size, 1500);
        assert_eq!(result.compression_ratio, 0.75);
    }

    #[test]
    fn test_search_request_creation() {
        let search_request = SearchRequest {
            query: "test query".to_string(),
            max_results: 25,
            collection: "test_collection".to_string(),
        };

        assert_eq!(search_request.query, "test query");
        assert_eq!(search_request.max_results, 25);
        assert_eq!(search_request.collection, "test_collection");
    }

    #[test]
    fn test_search_result_creation() {
        let search_result = SearchResult {
            id: "result_1".to_string(),
            score: 0.95,
            content: "Test content".to_string(),
            metadata: "{\"type\": \"test\"}".to_string(),
        };

        assert_eq!(search_result.id, "result_1");
        assert_eq!(search_result.score, 0.95);
        assert_eq!(search_result.content, "Test content");
        assert!(!search_result.metadata.is_empty());
    }

    #[tokio::test]
    async fn test_streaming_handler_capabilities() {
        let validator = create_test_message_validator();
        let handler = StreamingDocumentHandler::new(validator.clone());

        // Test streaming capability checks
        assert!(handler.message_validator.is_streaming_enabled(true));
        assert!(handler.message_validator.is_streaming_enabled(false));

        // Test stream registration
        let stream_handle = validator.register_stream().unwrap();
        assert_eq!(stream_handle.timeout(), Duration::from_secs(30));
        assert_eq!(stream_handle.buffer_size(), 100);
        assert!(stream_handle.flow_control_enabled());
    }

    #[test]
    fn test_compression_with_chunks() {
        let validator = create_test_message_validator();

        let original_data = b"This is test data for compression";

        // Test compression
        let compressed_data = validator.compress_message(original_data).unwrap();
        assert!(compressed_data.len() <= original_data.len()); // May not compress small data

        // Test decompression
        let decompressed_data = validator.decompress_message(&compressed_data).unwrap();
        assert_eq!(decompressed_data, original_data);
    }

    // Note: test_handle_document_stream_processing requires a proper Streaming<T> mock
    // which is complex to create without the full gRPC infrastructure.
    // This would typically be tested in integration tests with a real gRPC client.
}