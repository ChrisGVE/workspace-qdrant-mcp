//! WI-c2 write-policy guard (#82 task 21).
//!
//! ## Policy (clarified with Chris, 2026-06-03)
//!
//! The forbidden thing is a **direct write to the Qdrant vector database or the
//! graph database** from a client. *Requesting* the daemon to perform a write —
//! via a gRPC write-service RPC (`QueueWriteService`, `WatchWriteService`,
//! `LibraryWriteService`, `TrackingWriteService`, `AdminWriteService`) or the
//! REST API — is the **sanctioned** pattern and does not violate the
//! single-writer principle: the daemon still executes the write. The same holds
//! for the MCP server, and for embedding (clients request embeddings from the
//! daemon rather than running a local model).
//!
//! Therefore wqm-client exposes every gRPC write-service accessor as legitimate.
//! What this guard enforces is the real invariant: **wqm-client must contain no
//! call to a direct Qdrant *client* write method** — the shared crate's only
//! Qdrant surface is the read-only [`crate::qdrant::QdrantReadClient`].
//!
//! Note: the Qdrant-client method names below are deliberately specific
//! (`upsert_points`, not bare `upsert`) so the gRPC RPC wrappers in
//! `grpc/write_methods.rs` — e.g. `upsert_rule_mirror`, `delete_rule_mirror`,
//! which merely *ask the daemon* to write — do not false-match.
//!
//! Direct-write enforcement across the `wqm-cli` / `wqm-mcp` *binaries* is the
//! job of the no-CLI-write CI grep (WI-e1, #82 task 38), after the remaining
//! direct-write violations (WI-f1 RebalanceIdf, etc.) are removed.

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    /// `qdrant_client::Qdrant` mutation methods. Specific enough that the gRPC
    /// write-service RPC wrappers (`upsert_rule_mirror`, `delete_scratchpad_mirror`,
    /// …) — the sanctioned "ask the daemon to write" calls — do not match.
    const FORBIDDEN_QDRANT_WRITES: &[&str] = &[
        "upsert_points",
        "delete_points",
        "set_payload",
        "delete_payload",
        "clear_payload",
        "overwrite_payload",
        "update_vectors",
        "delete_vectors",
        "create_collection",
        "delete_collection",
        "update_collection",
        "create_field_index",
        "delete_field_index",
    ];

    /// This file lists the tokens literally; skip it during the scan.
    const SELF_FILE: &str = "write_service_guard.rs";

    fn scan_dir(dir: &Path, hits: &mut Vec<String>) {
        for entry in fs::read_dir(dir).expect("read_dir src") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                scan_dir(&path, hits);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            if path.file_name().and_then(|n| n.to_str()) == Some(SELF_FILE) {
                continue;
            }
            let src = fs::read_to_string(&path).expect("read source file");
            for token in FORBIDDEN_QDRANT_WRITES {
                if src.contains(token) {
                    hits.push(format!("{}: `{token}`", path.display()));
                }
            }
        }
    }

    #[test]
    fn wqm_client_has_no_direct_qdrant_write_calls() {
        let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut hits = Vec::new();
        scan_dir(&src_root, &mut hits);
        assert!(
            hits.is_empty(),
            "wqm-client must not call a direct Qdrant write method — request the \
             daemon to write via a gRPC write-service instead (WI-c2). Found:\n{}",
            hits.join("\n")
        );
    }

    #[test]
    fn guard_detects_injected_direct_write() {
        // Negative test: a clean pass above is only meaningful if the scan would
        // actually catch a direct write token.
        let sample = "self.inner.upsert_points(builder).await?;";
        assert!(
            FORBIDDEN_QDRANT_WRITES.iter().any(|t| sample.contains(t)),
            "guard must detect an injected direct Qdrant write"
        );
    }
}
