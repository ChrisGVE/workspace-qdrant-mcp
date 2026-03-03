//! Service Discovery Manager
//!
//! This module provides the main orchestrator for service discovery, combining
//! file-based registry, network discovery, health checking, and configuration
//! fallback mechanisms into a unified discovery system.

mod core;
mod types;
mod lifecycle;
mod api;
mod strategies;
mod background;

pub use core::DiscoveryManager;
pub use types::{DiscoveryStrategy, ServiceDiscoveryEvent};
