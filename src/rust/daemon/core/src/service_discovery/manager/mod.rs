//! Service Discovery Manager
//!
//! This module provides the main orchestrator for service discovery, combining
//! file-based registry, network discovery, health checking, and configuration
//! fallback mechanisms into a unified discovery system.

mod api;
mod background;
mod core;
mod lifecycle;
mod strategies;
mod types;

pub use core::DiscoveryManager;
pub use types::{DiscoveryStrategy, ServiceDiscoveryEvent};
