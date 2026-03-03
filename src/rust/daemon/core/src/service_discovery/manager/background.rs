//! Background task management: network event processing and periodic cleanup.

use std::sync::Arc;
use tokio::time::interval;
use tracing::warn;

use super::super::registry::ServiceRegistry;
use super::super::network::DiscoveryEvent;
use super::core::DiscoveryManager;
use super::types::{DiscoveryStrategy, ServiceDiscoveryEvent};

impl DiscoveryManager {
    /// Start processing network discovery events
    pub(super) async fn start_network_event_processing(&self) {
        let mut event_receiver = self.network_discovery.subscribe_events();
        let event_sender = self.event_sender.clone();
        let known_services = Arc::clone(&self.known_services);

        let handle = tokio::spawn(async move {
            while let Ok(event) = event_receiver.recv().await {
                match event {
                    DiscoveryEvent::ServiceDiscovered { service_name, service_info } => {
                        {
                            let mut known = known_services.write().await;
                            known.insert(
                                service_name.clone(),
                                (service_info.clone(), DiscoveryStrategy::Network),
                            );
                        }

                        let _ = event_sender.send(ServiceDiscoveryEvent::ServiceDiscovered {
                            service_name,
                            service_info,
                            strategy: DiscoveryStrategy::Network,
                        });
                    }

                    DiscoveryEvent::ServiceLost { service_name } => {
                        {
                            let mut known = known_services.write().await;
                            known.remove(&service_name);
                        }

                        let _ = event_sender.send(ServiceDiscoveryEvent::ServiceLost {
                            service_name,
                            reason: "Network discovery lost".to_string(),
                        });
                    }

                    _ => {
                        // Handle other events as needed
                    }
                }
            }
        });

        self.monitoring_handles.write().await.push(handle);
    }

    /// Start periodic cleanup tasks
    pub(super) async fn start_periodic_cleanup(&self) {
        let registry = ServiceRegistry::new(self.config.registry_path.as_ref()).unwrap();
        let cleanup_interval = self.config.cleanup_interval;
        let event_sender = self.event_sender.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(cleanup_interval);

            loop {
                interval.tick().await;

                match registry.cleanup_stale_entries() {
                    Ok(removed_services) => {
                        for service_name in removed_services {
                            let _ = event_sender.send(ServiceDiscoveryEvent::ServiceLost {
                                service_name,
                                reason: "Stale registry entry".to_string(),
                            });
                        }
                    }
                    Err(e) => {
                        warn!("Registry cleanup failed: {}", e);
                    }
                }
            }
        });

        self.monitoring_handles.write().await.push(handle);
    }
}
