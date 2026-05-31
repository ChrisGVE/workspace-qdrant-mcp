//! Canonical JSON serialization and payload builders for MCP queue operations.
//!
//! Provides byte-for-byte parity with the TypeScript MCP server's live
//! `stableStringify` function (`queue-operations.ts` lines 36-47) so that
//! idempotency keys computed here match those the daemon receives and stores.

pub mod payload_builders;
pub mod stable_stringify;

#[cfg(test)]
mod tests;
