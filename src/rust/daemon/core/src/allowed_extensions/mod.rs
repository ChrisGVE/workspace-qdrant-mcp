//! File Type Allowlist
//!
//! Provides a two-tier allowlist of file extensions that the daemon will process.
//! The library allowlist is a superset of the project allowlist, so reference
//! material containing code examples can be fully processed. Files whose
//! extensions are not in the appropriate allowlist are silently skipped.

mod extensions;
mod types;

#[cfg(test)]
mod tests;

pub use extensions::AllowedExtensions;
pub use types::FileRoute;
