//! Network-based Service Discovery
//!
//! This module implements UDP multicast-based network discovery for services
//! on the local network. It allows services to announce themselves and discover
//! other services without relying on a central registry file.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};

use super::registry::ServiceInfo;

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
    multicast_addr: SocketAddr,
    
    /// UDP socket for sending/receiving messages
    socket: Arc<RwLock<Option<UdpSocket>>>,
    
    /// Discovered services cache
    discovered_services: Arc<RwLock<HashMap<String, (ServiceInfo, SystemTime)>>>,
    
    /// Message sender for broadcasting discovery events
    event_sender: broadcast::Sender<DiscoveryEvent>,
    
    /// Authentication token for secure communication
    auth_token: Option<String>,
    
    /// Cache timeout for discovered services
    cache_timeout: Duration,

    /// Locally registered services (services this node can respond for)
    local_services: Arc<RwLock<HashMap<String, ServiceInfo>>>,
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
    ServiceLost {
        service_name: String,
    },
    
    /// Health ping received
    HealthPing {
        service_name: String,
        status: String,
        metrics: HashMap<String, String>,
    },
}

impl NetworkDiscovery {
    /// Create a new network discovery instance
    pub fn new(
        multicast_address: &str,
        port: u16,
        auth_token: Option<String>,
    ) -> Result<Self, NetworkError> {
        // Parse multicast address
        let multicast_ip: Ipv4Addr = multicast_address.parse()
            .map_err(|_| NetworkError::InvalidMulticastAddress(multicast_address.to_string()))?;
        
        // Validate multicast address range
        if !multicast_ip.is_multicast() {
            return Err(NetworkError::InvalidMulticastAddress(
                format!("{} is not a valid multicast address", multicast_address)
            ));
        }
        
        let multicast_addr = SocketAddr::new(IpAddr::V4(multicast_ip), port);
        let (event_sender, _) = broadcast::channel(1000);
        
        info!("Network discovery initialized for {}", multicast_addr);
        
        Ok(Self {
            multicast_addr,
            socket: Arc::new(RwLock::new(None)),
            discovered_services: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            auth_token,
            cache_timeout: Duration::from_secs(300), // 5 minutes
            local_services: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start the network discovery service
    pub async fn start(&self) -> Result<(), NetworkError> {
        info!("Starting network discovery on {}", self.multicast_addr);
        
        // Create UDP socket
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.set_nonblocking(true)?;
        
        // Join multicast group
        let ip = self.multicast_addr.ip();
        if let IpAddr::V4(ipv4) = ip {
            socket.join_multicast_v4(&ipv4, &Ipv4Addr::UNSPECIFIED)?;
        } else {
            return Err(NetworkError::InvalidMulticastAddress("Multicast address must be IPv4".to_string()));
        }
        
        {
            let mut socket_guard = self.socket.write().await;
            *socket_guard = Some(socket);
        }
        
        // Start cache cleanup task
        self.start_cache_cleanup().await;
        
        info!("Network discovery service started successfully");
        Ok(())
    }

    /// Stop the network discovery service
    pub async fn stop(&self) -> Result<(), NetworkError> {
        info!("Stopping network discovery service");
        
        let mut socket_guard = self.socket.write().await;
        *socket_guard = None;
        
        info!("Network discovery service stopped");
        Ok(())
    }

    /// Announce service availability
    pub async fn announce_service(
        &self,
        service_name: &str,
        service_info: &ServiceInfo,
    ) -> Result<(), NetworkError> {
        debug!("Announcing service: {}", service_name);

        // Track as local service so we can respond to discovery requests
        {
            let mut local = self.local_services.write().await;
            local.insert(service_name.to_string(), service_info.clone());
        }

        let message = DiscoveryMessage {
            message_type: DiscoveryMessageType::ServiceAnnouncement,
            service_name: service_name.to_string(),
            timestamp: current_iso_timestamp(),
            payload: DiscoveryPayload::ServiceInfo(service_info.clone()),
            auth_token: self.auth_token.clone(),
        };

        self.send_message(&message).await?;

        debug!("Service {} announced successfully", service_name);
        Ok(())
    }

    /// Discover services on the network
    pub async fn discover_services(
        &self,
        service_names: Vec<String>,
        discover_timeout: Duration,
    ) -> Result<HashMap<String, ServiceInfo>, NetworkError> {
        info!("Discovering services: {:?}", service_names);
        
        let request_id = uuid::Uuid::new_v4().to_string();
        let message = DiscoveryMessage {
            message_type: DiscoveryMessageType::DiscoveryRequest,
            service_name: "discovery-client".to_string(),
            timestamp: current_iso_timestamp(),
            payload: DiscoveryPayload::Request {
                service_names: service_names.clone(),
                request_id: request_id.clone(),
            },
            auth_token: self.auth_token.clone(),
        };
        
        // Send discovery request
        self.send_message(&message).await?;
        
        // Wait for responses
        let start_time = SystemTime::now();
        let mut discovered = HashMap::new();
        
        // Subscribe to discovery events
        let mut event_receiver = self.event_sender.subscribe();
        
        while start_time.elapsed().unwrap_or(Duration::MAX) < discover_timeout {
            match timeout(Duration::from_millis(100), event_receiver.recv()).await {
                Ok(Ok(DiscoveryEvent::ServiceDiscovered { service_name, service_info })) => {
                    if service_names.is_empty() || service_names.contains(&service_name) {
                        discovered.insert(service_name, service_info);
                    }
                }
                Ok(Ok(DiscoveryEvent::ServiceUpdated { service_name, service_info })) => {
                    if service_names.is_empty() || service_names.contains(&service_name) {
                        discovered.insert(service_name, service_info);
                    }
                }
                _ => continue,
            }
            
            // Break if we found all requested services
            if !service_names.is_empty() && 
               service_names.iter().all(|name| discovered.contains_key(name)) {
                break;
            }
        }
        
        info!("Discovery completed, found {} services", discovered.len());
        Ok(discovered)
    }

    /// Send health ping
    pub async fn send_health_ping(
        &self,
        service_name: &str,
        status: &str,
        metrics: HashMap<String, String>,
    ) -> Result<(), NetworkError> {
        debug!("Sending health ping for service: {}", service_name);
        
        let message = DiscoveryMessage {
            message_type: DiscoveryMessageType::HealthPing,
            service_name: service_name.to_string(),
            timestamp: current_iso_timestamp(),
            payload: DiscoveryPayload::Health {
                status: status.to_string(),
                metrics,
            },
            auth_token: self.auth_token.clone(),
        };
        
        self.send_message(&message).await?;
        Ok(())
    }

    /// Announce service shutdown
    pub async fn announce_shutdown(&self, service_name: &str) -> Result<(), NetworkError> {
        info!("Announcing shutdown for service: {}", service_name);

        // Remove from local services
        {
            let mut local = self.local_services.write().await;
            local.remove(service_name);
        }

        let message = DiscoveryMessage {
            message_type: DiscoveryMessageType::ServiceShutdown,
            service_name: service_name.to_string(),
            timestamp: current_iso_timestamp(),
            payload: DiscoveryPayload::Empty,
            auth_token: self.auth_token.clone(),
        };

        self.send_message(&message).await?;
        Ok(())
    }

    /// Get currently discovered services from cache
    pub async fn get_cached_services(&self) -> HashMap<String, ServiceInfo> {
        let services_guard = self.discovered_services.read().await;
        services_guard.iter()
            .map(|(name, (info, _))| (name.clone(), info.clone()))
            .collect()
    }

    /// Subscribe to discovery events
    pub fn subscribe_events(&self) -> broadcast::Receiver<DiscoveryEvent> {
        self.event_sender.subscribe()
    }

    /// Send a message via multicast
    async fn send_message(&self, message: &DiscoveryMessage) -> Result<(), NetworkError> {
        self.send_message_to(message, self.multicast_addr).await
    }

    /// Send a message to a specific address (for unicast responses)
    async fn send_message_to(&self, message: &DiscoveryMessage, addr: SocketAddr) -> Result<(), NetworkError> {
        let socket_guard = self.socket.read().await;
        let socket = socket_guard.as_ref()
            .ok_or_else(|| NetworkError::DiscoveryFailed("Socket not initialized".to_string()))?;

        let message_data = serde_json::to_vec(message)?;
        socket.send_to(&message_data, addr)?;

        debug!("Sent {} message for service {} to {}",
               serde_json::to_string(&message.message_type).unwrap_or_default(),
               message.service_name,
               addr);

        Ok(())
    }

    /// Process received discovery message
    async fn _process_message(&self, message: DiscoveryMessage, sender_addr: SocketAddr) -> Result<(), NetworkError> {
        debug!("Processing {} message from {} for service {}", 
               serde_json::to_string(&message.message_type).unwrap_or_default(),
               sender_addr,
               message.service_name);

        // Verify authentication token if required
        if self.auth_token.is_some() && message.auth_token != self.auth_token {
            warn!("Authentication failed for message from {}", sender_addr);
            return Ok(()); // Ignore unauthorized messages
        }

        match message.message_type {
            DiscoveryMessageType::ServiceAnnouncement => {
                if let DiscoveryPayload::ServiceInfo(service_info) = message.payload {
                    self._cache_discovered_service(&message.service_name, service_info.clone()).await;
                    
                    let _ = self.event_sender.send(DiscoveryEvent::ServiceDiscovered {
                        service_name: message.service_name.clone(),
                        service_info,
                    });
                }
            }
            
            DiscoveryMessageType::DiscoveryRequest => {
                debug!("Received discovery request from {}", sender_addr);

                if let DiscoveryPayload::Request { service_names, .. } = &message.payload {
                    let local = self.local_services.read().await;

                    // Respond with matching local services
                    for (name, info) in local.iter() {
                        // If service_names is empty, respond with all; otherwise filter
                        if service_names.is_empty() || service_names.contains(name) {
                            let response = DiscoveryMessage {
                                message_type: DiscoveryMessageType::DiscoveryResponse,
                                service_name: name.clone(),
                                timestamp: current_iso_timestamp(),
                                payload: DiscoveryPayload::ServiceInfo(info.clone()),
                                auth_token: self.auth_token.clone(),
                            };

                            if let Err(e) = self.send_message_to(&response, sender_addr).await {
                                warn!("Failed to send discovery response to {}: {}", sender_addr, e);
                            }
                        }
                    }
                }
            }

            DiscoveryMessageType::DiscoveryResponse => {
                // Handle discovery response (same as announcement)
                if let DiscoveryPayload::ServiceInfo(service_info) = message.payload {
                    self._cache_discovered_service(&message.service_name, service_info.clone()).await;

                    let _ = self.event_sender.send(DiscoveryEvent::ServiceDiscovered {
                        service_name: message.service_name.clone(),
                        service_info,
                    });
                }
            }
            
            DiscoveryMessageType::HealthPing => {
                if let DiscoveryPayload::Health { status, metrics } = message.payload {
                    let _ = self.event_sender.send(DiscoveryEvent::HealthPing {
                        service_name: message.service_name,
                        status,
                        metrics,
                    });
                }
            }
            
            DiscoveryMessageType::ServiceShutdown => {
                self._remove_cached_service(&message.service_name).await;
                let _ = self.event_sender.send(DiscoveryEvent::ServiceLost {
                    service_name: message.service_name,
                });
            }
            
            _ => {
                debug!("Unhandled message type: {:?}", message.message_type);
            }
        }

        Ok(())
    }

    /// Cache a discovered service
    async fn _cache_discovered_service(&self, service_name: &str, service_info: ServiceInfo) {
        let mut services_guard = self.discovered_services.write().await;
        services_guard.insert(service_name.to_string(), (service_info, SystemTime::now()));
    }

    /// Remove cached service
    async fn _remove_cached_service(&self, service_name: &str) {
        let mut services_guard = self.discovered_services.write().await;
        services_guard.remove(service_name);
    }

    /// Start cache cleanup task to remove expired entries
    async fn start_cache_cleanup(&self) {
        let discovered_services = Arc::clone(&self.discovered_services);
        let event_sender = self.event_sender.clone();
        let cache_timeout = self.cache_timeout;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Clean every minute
            
            loop {
                interval.tick().await;
                
                let mut services_guard = discovered_services.write().await;
                let mut expired_services = Vec::new();
                
                services_guard.retain(|service_name, (_, last_seen)| {
                    if last_seen.elapsed().unwrap_or(Duration::MAX) > cache_timeout {
                        expired_services.push(service_name.clone());
                        false
                    } else {
                        true
                    }
                });
                
                drop(services_guard);
                
                // Send service lost events for expired services
                for service_name in expired_services {
                    let _ = event_sender.send(DiscoveryEvent::ServiceLost { service_name });
                }
            }
        });
    }
}

/// Get current ISO 8601 timestamp
fn current_iso_timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string()
}