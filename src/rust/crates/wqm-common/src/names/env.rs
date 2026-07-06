//! Environment-variable keys (N8).
//!
//! The five keys the daemon, CLI, and MCP server read runtime configuration from.
//! N7 (F-07) owns the *resolution* of these into a typed config; N8 owns only the
//! key each is spelled with, so a rename is a one-site edit.

/// A configuration environment variable. The set is the one documented in the
/// project README; N7 (F-07) resolves these into the typed config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvVar {
    /// Qdrant server URL (`QDRANT_URL`).
    QdrantUrl,
    /// Qdrant API key (`QDRANT_API_KEY`), required for Qdrant Cloud.
    QdrantApiKey,
    /// FastEmbed model id (`FASTEMBED_MODEL`).
    FastembedModel,
    /// Override for the storedb path (`WQM_DATABASE_PATH`).
    DatabasePath,
    /// Log-level override (`WQM_LOG_LEVEL`).
    LogLevel,
}

impl EnvVar {
    /// Every configuration environment variable, in declaration order.
    pub const ALL: [EnvVar; 5] = [
        EnvVar::QdrantUrl,
        EnvVar::QdrantApiKey,
        EnvVar::FastembedModel,
        EnvVar::DatabasePath,
        EnvVar::LogLevel,
    ];

    /// The environment-variable key this variable is read from.
    pub const fn key(self) -> &'static str {
        match self {
            EnvVar::QdrantUrl => "QDRANT_URL",
            EnvVar::QdrantApiKey => "QDRANT_API_KEY",
            EnvVar::FastembedModel => "FASTEMBED_MODEL",
            EnvVar::DatabasePath => "WQM_DATABASE_PATH",
            EnvVar::LogLevel => "WQM_LOG_LEVEL",
        }
    }
}
