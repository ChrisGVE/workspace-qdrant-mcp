//! Types for network-based service discovery

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};

use crate::service_discovery::registry::ServiceInfo;

/// Network discovery errors
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Socket bind error: {0}")]
    BindError(#[from] std::io::Error),

    #[error("Message serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Invalid multicast address: {0}")]
    InvalidMulticastAddress(String),

    #[error("Network timeout")]
    Timeout,

    #[error("Discovery request failed: {0}")]
    DiscoveryFailed(String),
}

/// Discovery message types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DiscoveryMessageType {
    /// Request to discover services
    DiscoveryRequest,
    /// Response with service information
    DiscoveryResponse,
    /// Health ping between services
    HealthPing,
    /// Service announcement
    ServiceAnnouncement,
    /// Service shutdown notification
    ServiceShutdown,
}

/// Network discovery message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryMessage {
    /// Message type
    pub message_type: DiscoveryMessageType,

    /// Source service name
    pub service_name: String,

    /// Message timestamp (ISO 8601)
    pub timestamp: String,

    /// Message payload
    pub payload: DiscoveryPayload,

    /// Authentication token (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_token: Option<String>,
}

/// Discovery message payload
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DiscoveryPayload {
    /// Service information payload
    ServiceInfo(ServiceInfo),

    /// Request payload with filters
    Request {
        /// Requested service names (empty = all services)
        service_names: Vec<String>,
        /// Request ID for correlation
        request_id: String,
    },

    /// Health check payload
    Health {
        /// Service status
        status: String,
        /// Additional health metrics
        metrics: HashMap<String, String>,
    },

    /// Empty payload for simple messages
    Empty,
}

/// Network discovery manager
pub struct NetworkDiscovery {
    /// Multicast address for discovery
    pub(super) multicast_addr: SocketAddr,

    /// UDP socket for sending/receiving messages
    pub(super) socket: Arc<RwLock<Option<UdpSocket>>>,

    /// Discovered services cache
    pub(super) discovered_services: Arc<RwLock<HashMap<String, (ServiceInfo, SystemTime)>>>,

    /// Message sender for broadcasting discovery events
    pub(super) event_sender: broadcast::Sender<DiscoveryEvent>,

    /// Authentication token for secure communication
    pub(super) auth_token: Option<String>,

    /// Cache timeout for discovered services
    pub(super) cache_timeout: Duration,

    /// Locally registered services (services this node can respond for)
    pub(super) local_services: Arc<RwLock<HashMap<String, ServiceInfo>>>,
}

/// Discovery events broadcast to subscribers
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    /// Service discovered
    ServiceDiscovered {
        service_name: String,
        service_info: ServiceInfo,
    },

    /// Service updated
    ServiceUpdated {
        service_name: String,
        service_info: ServiceInfo,
    },

    /// Service lost (timeout or shutdown)
    ServiceLost { service_name: String },

    /// Health ping received
    HealthPing {
        service_name: String,
        status: String,
        metrics: HashMap<String, String>,
    },
}
