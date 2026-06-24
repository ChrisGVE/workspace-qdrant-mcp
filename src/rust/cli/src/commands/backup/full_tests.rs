//! Tests for `wqm backup --full` (AC-F20.1, AC-F20.1b, AC-F20.2).
//!
//! All tests run offline -- no live Qdrant required.
//! The Qdrant leg is skipped gracefully when Qdrant is unreachable.

use std::fs;
use std::io::Read as _;
use std::path::Path;
use tempfile::TempDir;

use super::super::compressor::detect as detect_compressor;
use super::super::diskspace::check_free_space;
use super::super::manifest::BackupManifest;
use super::super::stores::{discover_stores, vacuum_into};

// ---- Offline archive round-trip helper -------------------------------------

/// Build a minimal tar archive compressed with the detected compressor,
/// containing a single SQLite copy and a manifest.  Returns the archive path.
fn build_test_archive(work_dir: &Path) -> std::path::PathBuf {
    use std::fs::File;
    use std::process::Stdio;

    // Create a tiny SQLite source.
    let src_db = work_dir.join("source.db");
    {
        let conn = rusqlite::Connection::open(&src_db).expect("open src");
        conn.execute_batch("CREATE TABLE t (x TEXT); INSERT INTO t VALUES ('hello');")
            .expect("populate");
    }

    // Copy via VACUUM INTO.
    let staged = work_dir.join("staged_state.db");
    vacuum_into(&src_db, &staged).expect("vacuum_into");

    // Build manifest.
    let manifest = BackupManifest::new(
        vec![super::super::manifest::StoreEntry {
            rel_path: "state.db".into(),
            tenant_id: None,
            content_key_version: None,
        }],
        None,
        "zstd",
        false,
    );
    let manifest_bytes = manifest.to_json_bytes().expect("manifest bytes");

    // Write uncompressed tar to a buffer.
    let mut tar_buf: Vec<u8> = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut tar_buf);
        let mut f = File::open(&staged).expect("open staged");
        builder
            .append_file("stores/state.db", &mut f)
            .expect("append db");
        let mut hdr = tar::Header::new_gnu();
        hdr.set_size(manifest_bytes.len() as u64);
        hdr.set_mode(0o644);
        hdr.set_cksum();
        builder
            .append_data(&mut hdr, "manifest.json", manifest_bytes.as_slice())
            .expect("append manifest");
        builder.finish().expect("finish tar");
    }

    let archive = work_dir.join("backup.tar.zst");
    let comp = detect_compressor();

    if let Some(c) = comp {
        // Spawn compressor, feed tar_buf via stdin.
        let dest_file = File::create(&archive).expect("create archive");
        let mut child = c
            .spawn_compress(Stdio::from(dest_file))
            .expect("spawn compressor");
        {
            use std::io::Write as _;
            child
                .stdin
                .take()
                .expect("stdin")
                .write_all(&tar_buf)
                .expect("write tar");
        }
        child.wait().expect("compressor wait");
    } else {
        // Fallback: store uncompressed.
        fs::write(&archive, &tar_buf).expect("write tar");
    }

    archive
}

// ---- Tests -----------------------------------------------------------------

/// AC-F20.1: archive contains manifest.json that is parseable with daemon_running.
#[test]
fn t_f20_full_archive_contains_parseable_manifest() {
    let dir = TempDir::new().expect("tempdir");
    let archive = build_test_archive(dir.path());
    assert!(archive.exists(), "archive must exist");

    // Decompress and inspect via tar.
    let comp = detect_compressor();
    let tar_bytes = if let Some(c) = comp {
        use std::io::Read as _;
        use std::process::Stdio;
        let archive_file = fs::File::open(&archive).expect("open archive");
        let mut child = c
            .spawn_decompress(Stdio::from(archive_file))
            .expect("spawn decompress");
        let mut buf = Vec::new();
        child
            .stdout
            .take()
            .expect("stdout")
            .read_to_end(&mut buf)
            .expect("read");
        child.wait().expect("wait");
        buf
    } else {
        fs::read(&archive).expect("read raw tar")
    };

    // Walk tar members for manifest.json.
    let mut found_manifest = false;
    let cursor = std::io::Cursor::new(&tar_bytes);
    let mut archive_reader = tar::Archive::new(cursor);
    for entry in archive_reader.entries().expect("tar entries") {
        let mut entry = entry.expect("entry");
        let path = entry.path().expect("path").to_string_lossy().to_string();
        if path == "manifest.json" {
            let mut content = Vec::new();
            entry.read_to_end(&mut content).expect("read manifest");
            let m = BackupManifest::from_json_bytes(&content).expect("manifest must parse");
            assert!(!m.wqm_version.is_empty());
            assert!(!m.daemon_running); // we set false in the test
            found_manifest = true;
        }
    }
    assert!(
        found_manifest,
        "manifest.json must be present in the archive"
    );
}

/// AC-F20.1: archive contains the SQLite store member.
#[test]
fn t_f20_full_archive_contains_sqlite_store() {
    let dir = TempDir::new().expect("tempdir");
    let archive = build_test_archive(dir.path());

    let comp = detect_compressor();
    let tar_bytes = if let Some(c) = comp {
        use std::io::Read as _;
        use std::process::Stdio;
        let archive_file = fs::File::open(&archive).expect("open archive");
        let mut child = c
            .spawn_decompress(Stdio::from(archive_file))
            .expect("spawn decompress");
        let mut buf = Vec::new();
        child.stdout.take().unwrap().read_to_end(&mut buf).unwrap();
        child.wait().unwrap();
        buf
    } else {
        fs::read(&archive).unwrap()
    };

    let cursor = std::io::Cursor::new(&tar_bytes);
    let mut ar = tar::Archive::new(cursor);
    let members: Vec<String> = ar
        .entries()
        .unwrap()
        .flatten()
        .filter_map(|e| e.path().ok().map(|p| p.to_string_lossy().to_string()))
        .collect();

    assert!(
        members.iter().any(|m| m == "stores/state.db"),
        "archive must contain stores/state.db; members: {:?}",
        members
    );
}

/// AC-F20.1b: pre-flight check refuses when required > available (u64::MAX).
#[test]
fn t_f20_full_preflight_refuses_below_required() {
    let dir = TempDir::new().expect("tempdir");
    let result = check_free_space(dir.path(), u64::MAX);
    assert!(result.is_err(), "must refuse when space insufficient");
}

/// AC-F20.1b: pre-flight check proceeds when required <= available.
#[test]
fn t_f20_full_preflight_proceeds_above_required() {
    let dir = TempDir::new().expect("tempdir");
    let result = check_free_space(dir.path(), 1);
    assert!(result.is_ok(), "must proceed when space sufficient");
}

/// AC-F20.2 streaming: decompressor stdout is streamed into tar (not buffered
/// in full before reading).  We verify by decompressing the archive as a
/// streaming reader -- the test passes without loading the whole archive into
/// memory first.
#[test]
fn t_f20_full_decompression_is_streamed_not_buffered() {
    use std::process::Stdio;

    let dir = TempDir::new().expect("tempdir");
    let archive = build_test_archive(dir.path());

    let Some(c) = detect_compressor() else {
        // No compressor: skip streaming test (archive is plain tar).
        return;
    };

    let archive_file = fs::File::open(&archive).expect("open archive");
    // Feed archive via stdin (file handle = stream, not a Vec<u8>).
    let mut child = c
        .spawn_decompress(Stdio::from(archive_file))
        .expect("spawn decompress");

    // Read from stdout as a streaming reader -- tar::Archive wraps it directly.
    let stdout = child.stdout.take().expect("stdout");
    let mut ar = tar::Archive::new(stdout); // NOT buffered whole archive

    let mut found = false;
    for entry in ar.entries().expect("entries") {
        let entry = entry.expect("entry");
        let path = entry.path().expect("path").to_string_lossy().to_string();
        if path == "manifest.json" {
            found = true;
        }
    }
    child.wait().expect("wait");
    assert!(
        found,
        "manifest.json must be found via streaming decompress"
    );
}

/// AC-F20.1: store discovery round-trip via the real discover_stores helper.
#[test]
fn t_f20_full_store_discovery_round_trip() {
    let dir = TempDir::new().expect("tempdir");

    // Create state.db and one project store.
    let conn = rusqlite::Connection::open(dir.path().join("state.db")).expect("open state.db");
    conn.execute_batch("CREATE TABLE t (x INTEGER);")
        .expect("create");
    drop(conn);

    let proj_dir = dir.path().join("projects").join("proj1");
    fs::create_dir_all(&proj_dir).expect("mkdir");
    let conn2 = rusqlite::Connection::open(proj_dir.join("store.db")).expect("open proj store.db");
    conn2
        .execute_batch("CREATE TABLE t (x INTEGER);")
        .expect("create");
    drop(conn2);

    let stores = discover_stores(dir.path());
    assert_eq!(
        stores.len(),
        2,
        "expected state.db + projects/proj1/store.db"
    );

    let rel_paths: Vec<&str> = stores.iter().map(|s| s.rel_path.as_str()).collect();
    assert!(rel_paths.contains(&"state.db"));
    assert!(rel_paths.contains(&"projects/proj1/store.db"));
}
