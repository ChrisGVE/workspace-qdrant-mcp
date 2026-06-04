//! LanguageService gRPC implementation (WI-e1, #82).
//!
//! Tree-sitter grammar lifecycle (install / remove / list / query status) plus
//! language-registry summary, exposed so the CLI can drop its direct dependency
//! on `workspace-qdrant-core`. The grammar engine stays daemon-side; this
//! service is a thin RPC surface over a shared `GrammarManager` and the core
//! security gate.

mod handlers;
mod service_impl;

#[cfg(test)]
mod tests;

pub use service_impl::LanguageServiceImpl;
