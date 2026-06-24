//! Manifest type for full backup bundles (F20 / AC-F20.1).
//!
//! The manifest is written as `manifest.json` INSIDE the tar archive so any
//! consumer can inspect the bundle contents without a full decompress.
//!
//! ## Fields (AC-F20.1 contract)
//!
//! - `wqm_version`           -- semver string from `CARGO_PKG_VERSION`
//! - `stores`                -- list of SQLite stores bundled
//! - `qdrant_snapshot_name`  -- name of the Qdrant snapshot file in the archive
//!                              (`null` when Qdrant is unreachable and the
//!                              snapshot was skipped)
//! - `archive_timestamp`     -- RFC 3339 UTC timestamp of when the backup ran
//! - `compressor`            -- name of the external compressor used
//! - `daemon_running`        -- whether the daemon was running at backup time
//!                              (DATA-N01 temporal-skew note: a `true` value
//!                              means the SQLite stores and the Qdrant snapshot
//!                              may have been captured at slightly different
//!                              instants; run `wqm admin rebuild` after restore
//!                              to re-derive a consistent Qdrant index)

use chrono::Utc;
use serde::{Deserialize, Serialize};

/// One SQLite store entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct StoreEntry {
    /// Path of the store relative to the data directory (e.g. `state.db`,
    /// `projects/abc123/store.db`).
    pub rel_path: String,
    /// Tenant / namespace identifier for per-project stores; `null` for the
    /// central `state.db`.
    pub tenant_id: Option<String>,
    /// `content_key_version` column value as recorded in the store schema
    /// metadata; `null` when the store pre-dates F20 or the column is absent.
    pub content_key_version: Option<u32>,
}

/// Full-backup manifest written as `manifest.json` inside the tar archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BackupManifest {
    /// `wqm` version that produced the archive (from `CARGO_PKG_VERSION`).
    pub wqm_version: String,
    /// SQLite stores included in the bundle.
    pub stores: Vec<StoreEntry>,
    /// Name of the Qdrant snapshot file inside the tar archive; `None` if
    /// Qdrant was unreachable and the snapshot leg was skipped.
    pub qdrant_snapshot_name: Option<String>,
    /// RFC 3339 UTC timestamp of when the backup was created.
    pub archive_timestamp: String,
    /// Name of the external compressor used (`"zstd"`, `"xz"`, `"gzip"`).
    pub compressor: String,
    /// Whether the daemon was running at backup time (DATA-N01).
    pub daemon_running: bool,
}

impl BackupManifest {
    /// Construct a new manifest stamped with the current UTC time.
    pub(crate) fn new(
        stores: Vec<StoreEntry>,
        qdrant_snapshot_name: Option<String>,
        compressor: &str,
        daemon_running: bool,
    ) -> Self {
        Self {
            wqm_version: env!("CARGO_PKG_VERSION").to_string(),
            stores,
            qdrant_snapshot_name,
            archive_timestamp: Utc::now().to_rfc3339(),
            compressor: compressor.to_string(),
            daemon_running,
        }
    }

    /// Serialize to pretty-printed JSON bytes.
    pub(crate) fn to_json_bytes(&self) -> anyhow::Result<Vec<u8>> {
        serde_json::to_vec_pretty(self).map_err(Into::into)
    }

    /// Deserialize from JSON bytes.
    pub(crate) fn from_json_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        serde_json::from_slice(bytes).map_err(Into::into)
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> BackupManifest {
        BackupManifest::new(
            vec![
                StoreEntry {
                    rel_path: "state.db".into(),
                    tenant_id: None,
                    content_key_version: None,
                },
                StoreEntry {
                    rel_path: "projects/abc123/store.db".into(),
                    tenant_id: Some("abc123".into()),
                    content_key_version: Some(1),
                },
            ],
            Some("snapshot-2026-01-01.snapshot".into()),
            "zstd",
            false,
        )
    }

    /// AC-F20.1: manifest round-trips through JSON without loss.
    #[test]
    fn t_f20_manifest_round_trip() {
        let m = sample_manifest();
        let bytes = m.to_json_bytes().expect("serialize");
        let m2: BackupManifest = BackupManifest::from_json_bytes(&bytes).expect("deserialize");
        assert_eq!(m2.wqm_version, m.wqm_version);
        assert_eq!(m2.stores.len(), 2);
        assert_eq!(m2.compressor, "zstd");
        assert!(!m2.daemon_running);
        assert_eq!(
            m2.qdrant_snapshot_name.as_deref(),
            Some("snapshot-2026-01-01.snapshot")
        );
    }

    /// AC-F20.1: manifest records daemon_running correctly.
    #[test]
    fn t_f20_manifest_daemon_running_field() {
        let m_live = BackupManifest::new(vec![], None, "gzip", true);
        let m_stopped = BackupManifest::new(vec![], None, "gzip", false);

        let bytes_live = m_live.to_json_bytes().unwrap();
        let bytes_stopped = m_stopped.to_json_bytes().unwrap();

        let v_live: serde_json::Value = serde_json::from_slice(&bytes_live).unwrap();
        let v_stopped: serde_json::Value = serde_json::from_slice(&bytes_stopped).unwrap();

        assert_eq!(v_live["daemon_running"], true);
        assert_eq!(v_stopped["daemon_running"], false);
    }

    /// AC-F20.1: manifest JSON is parseable from raw bytes (asserts parseable).
    #[test]
    fn t_f20_manifest_json_is_parseable() {
        let m = sample_manifest();
        let bytes = m.to_json_bytes().unwrap();
        // Must parse without error.
        let parsed = BackupManifest::from_json_bytes(&bytes);
        assert!(
            parsed.is_ok(),
            "manifest JSON must be parseable: {:?}",
            parsed
        );
    }

    /// Verify wqm_version is non-empty (comes from CARGO_PKG_VERSION).
    #[test]
    fn t_f20_manifest_wqm_version_nonempty() {
        let m = BackupManifest::new(vec![], None, "xz", false);
        assert!(!m.wqm_version.is_empty(), "wqm_version must not be empty");
    }
}
