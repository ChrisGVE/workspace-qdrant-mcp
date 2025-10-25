//! Qdrant client operations for vector database management
//!
//! This module provides a comprehensive interface for interacting with Qdrant
//! vector database, including collection management, vector operations, and search.

pub mod client;
pub mod config;
pub mod error;
pub mod operations;

pub use client::QdrantClient;
pub use config::QdrantClientConfig;
pub use error::{QdrantError, QdrantResult};
pub use operations::{VectorOperation, SearchOperation, CollectionOperation};

/// Re-export commonly used types from qdrant-client
pub use qdrant_client::{
    qdrant::{
        vectors::VectorsOptions,
        vectors_config::Config as VectorsConfig,
        CollectionOperationResponse,
        CreateCollection,
        DeleteCollection,
        Distance,
        OptimizersConfigDiff,
        PointId,
        PointStruct,
        SearchParams,
        SearchPoints,
        UpsertPoints,
        VectorParams,
        VectorsConfigDiff,
    },
};