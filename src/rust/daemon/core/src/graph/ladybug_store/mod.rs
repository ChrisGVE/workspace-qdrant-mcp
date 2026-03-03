/// LadybugDB (Kuzu fork) graph store implementation.
///
/// Provides a `GraphStore` implementation backed by LadybugDB's in-process
/// property graph engine with native Cypher query support.
///
/// Gated behind the `ladybug` feature flag. Requires C++ compiler (Clang/LLVM)
/// at build time.

pub mod config;
pub mod store;

#[cfg(test)]
mod tests;

pub use config::LadybugConfig;
pub use store::LadybugGraphStore;
