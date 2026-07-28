//! Codegen for N24's proto source of truth.
//!
//! The generated code lands in `OUT_DIR` and is `include!`d by `src/lib.rs`, so it
//! is never committed and can never drift from the `.proto` beside it -- the proto
//! is the single producer (FP-2), and a stale checked-in copy is the classic way
//! that stops being true.

use std::io::Result;

fn main() -> Result<()> {
    let proto = "proto/wqm/v1/system.proto";

    // Re-run only when the contract changes.
    println!("cargo:rerun-if-changed={proto}");
    println!("cargo:rerun-if-changed=proto");

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&[proto], &["proto"])
}
