//! Network-based Service Discovery
//!
//! This module implements UDP multicast-based network discovery for services
//! on the local network. It allows services to announce themselves and discover
//! other services without relying on a central registry file.

mod types;
mod operations;

pub use types::{
    DiscoveryEvent,
    DiscoveryMessage,
    DiscoveryMessageType,
    DiscoveryPayload,
    NetworkDiscovery,
    NetworkError,
};
