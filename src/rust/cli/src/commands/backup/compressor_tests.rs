//! Tests for compressor detection and argv shape (AC-F20.3).

use super::detect;

/// AC-F20.3: detection falls through zstd -> xz -> gzip in preference order.
/// At least one of those is present on any CI/dev machine (gzip is ubiquitous).
#[test]
fn t_f20_compressor_detect_finds_at_least_one() {
    let c = detect();
    assert!(
        c.is_some(),
        "expected at least gzip to be installed but detect() returned None"
    );
}

/// AC-F20.3: detection order is zstd -> xz -> gzip; if zstd is installed it
/// must be preferred over gzip (verifies order, not just presence).
#[test]
fn t_f20_compressor_detection_order_zstd_before_gzip() {
    // If zstd is available it must win over gzip regardless of PATH order.
    let zstd_available = which::which("zstd").is_ok();
    let detected = detect();
    if zstd_available {
        let c = detected.expect("detect must return Some when zstd present");
        assert_eq!(c.name, "zstd", "zstd must be preferred over xz/gzip");
    } else {
        // zstd absent -- gzip or xz may still be present; just note the result.
        // No assertion: we cannot guarantee any compressor in a minimal env.
        let _ = detected;
    }
}

/// AC-F20.3: compress argv uses fixed tokens only -- no shell string, no user
/// data.  Verify the returned argv list contains only static ASCII tokens.
#[test]
fn t_f20_compressor_compress_argv_is_fixed() {
    if let Some(c) = detect() {
        let args = c.compress_args();
        for arg in &args {
            let s = arg.to_string_lossy();
            // Each arg must be a short flag or "-" -- never a path or user data.
            assert!(
                s.starts_with('-') || s == "-",
                "compress arg '{}' looks like it could be user-influenced",
                s
            );
        }
    }
}

/// AC-F20.3: decompress argv is also fixed -- includes -d/-decompress flag.
#[test]
fn t_f20_compressor_decompress_argv_contains_decompress_flag() {
    if let Some(c) = detect() {
        let args = c.decompress_args();
        let has_decompress = args.iter().any(|a| *a == "-d" || *a == "--decompress");
        assert!(
            has_decompress,
            "decompress argv must contain -d or --decompress; got: {:?}",
            args
        );
    }
}

/// Verify `detect()` returns a binary that actually exists at the resolved path.
#[test]
fn t_f20_compressor_bin_path_exists() {
    if let Some(c) = detect() {
        assert!(
            c.bin.exists(),
            "resolved compressor binary '{}' does not exist",
            c.bin.display()
        );
        assert!(
            c.bin.is_absolute(),
            "compressor binary path '{}' must be absolute",
            c.bin.display()
        );
    }
}
