//! Processing loop submodules for the unified queue processor.
//!
//! Split from the original `processing_loop.rs` to stay within the 500-line file limit.

mod batch_processing;
mod circuit_breakers;
mod health_probes;
mod idle_work;
mod item_dispatch;
mod loop_core;
mod loop_state;
mod memory_pressure;
