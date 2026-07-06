//! wqm-proto -- the N24 gRPC contract source of truth (ARCH rev08 §9.1).
//!
//! Owns the `.proto` definitions, the code generated from them, and the `Address`
//! enum. It declares no wqm-crate dependencies (generated code only) so it can be
//! linked by every surface without dragging in a closure.
//!
//! Per the grow-per-phase model (PRD F-00), Phase 0 only declares the crate; the
//! 13 services and the tonic/prost build step arrive at F-13.
