//! wqm-proto -- the N24 gRPC contract source of truth (ARCH rev15 §9.1).
//!
//! Owns the `.proto` definitions, the code generated from them, and the `Address`
//! enum. It declares no wqm-crate dependencies (generated code only) so it can be
//! linked by every surface without dragging in a closure.
//!
//! `P04-GT001` only declares the crate; the services and the tonic/prost build
//! step arrive with N24's slice, `P04-GT011`.
