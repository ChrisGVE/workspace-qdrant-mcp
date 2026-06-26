//! `wqm docker generate-compose` — emit / verify / delete the compose override.
//!
//! Design source: `docs/specs/16-path-abstraction.md` §§9, 9.1, 9.2, 10.1.
//!
//! Three modes, mutually exclusive:
//!
//! * default — read `config.yaml`, render an override YAML, write it next
//!   to `docker-compose.yaml` (or `--output`).
//! * `--check` — re-hash the live config's mount section and compare it to
//!   the `# wqm-config-hash:` header in the existing override; exit 1 on
//!   mismatch.
//! * `--clean` — delete the override (no-op if absent).
//!
//! The control-port publish (`127.0.0.1:7799:7799`) is **mandatory** per
//! spec §10.1: without it, a docker memexd and a host memexd could both
//! bind their respective `7799` and corrupt SQLite.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use serde::Deserialize;

use wqm_common::paths::{find_config_file, mount_section_hash};
use wqm_common::yaml_defaults::YamlMountEntry;

use crate::output;

/// Default control-port value mirrored from `memexd::control_port::DEFAULT_CONTROL_PORT`.
///
/// Hard-coded here to avoid a `wqm-cli → memexd` crate dependency. Kept in
/// sync via the constant's spec reference (16 §10.1).
const DEFAULT_CONTROL_PORT: u16 = 7799;

/// Default override-file name.
const DEFAULT_OVERRIDE_FILENAME: &str = "docker-compose.override.yaml";

/// Container-side bind-mount targets (fixed, spec §9.2 / §9.1).
const CONTAINER_CONFIG_PATH: &str = "/etc/wqm/config.yaml";
const CONTAINER_STATE_DIR: &str = "/var/lib/wqm";
const CONTAINER_QDRANT_DIR: &str = "/qdrant/storage";
/// Container-side bind target for the generated override file. The image
/// entrypoint (docker/memexd-entrypoint.sh) reads `# wqm-config-hash:`
/// from this path to detect override drift (spec §9.1 layer 1).
const CONTAINER_OVERRIDE_PATH: &str = "/etc/docker-compose-wqm.override.yaml";

/// Host-side state-bind-mount defaults (spec §9.2).
const HOST_STATE_DIR_DEFAULT: &str = "~/.local/share/workspace-qdrant";
const HOST_QDRANT_DIR_DEFAULT: &str = "~/.local/share/qdrant";

/// Hash header line prefix written into the generated override.
const HASH_HEADER_PREFIX: &str = "# wqm-config-hash:";

/// CLI arguments for `wqm docker generate-compose`.
#[derive(Args, Debug)]
pub struct GenerateComposeArgs {
    /// Verify the existing override's hash matches the live config (exit 1
    /// on drift); no write.
    #[arg(long, conflicts_with = "clean")]
    pub check: bool,

    /// Delete the override file (no-op if absent).
    #[arg(long, conflicts_with = "check")]
    pub clean: bool,

    /// Override file path (default: `./docker-compose.override.yaml`).
    #[arg(long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Explicit `config.yaml` path. Overrides `WQM_CONFIG_PATH` and XDG
    /// search, only for this invocation.
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
}

/// Subset of `config.yaml` parsed for compose generation.
///
/// Only fields that influence the override are deserialised:
/// `mounts`, `control_port`, `network_mode`. The rest of the config is
/// tolerated via `#[serde(default)]` on the top-level struct.
#[derive(Debug, Default, Deserialize)]
struct ComposeRelevantConfig {
    #[serde(default)]
    mounts: Vec<YamlMountEntry>,
    /// Mirrors `DaemonConfig.control_port`; `None` ⇒ daemon default 7799.
    #[serde(default)]
    control_port: Option<u16>,
    /// Docker `network_mode` to emit. `None` (or absent) ⇒ default bridge.
    /// `"none"` is rejected per spec §10.1.
    #[serde(default)]
    network_mode: Option<String>,
}

/// Execute `wqm docker generate-compose`.
pub async fn execute(args: GenerateComposeArgs) -> Result<()> {
    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OVERRIDE_FILENAME));

    if args.clean {
        return run_clean(&output_path);
    }

    let cfg_path = resolve_config_path(args.config.as_deref())?;
    let cfg = load_relevant_config(&cfg_path)
        .with_context(|| format!("failed to load config from {}", cfg_path.display()))?;

    if args.check {
        run_check(&cfg, &output_path)
    } else {
        run_generate(&cfg, &cfg_path, &output_path)
    }
}

// ────────────────────────────────────────────────────────────────────────
// Mode handlers
// ────────────────────────────────────────────────────────────────────────

fn run_generate(cfg: &ComposeRelevantConfig, cfg_path: &Path, output_path: &Path) -> Result<()> {
    validate_network_mode(cfg)?;

    let yaml = render_override_yaml(cfg, cfg_path, output_path)?;
    fs::write(output_path, &yaml)
        .with_context(|| format!("failed to write override file at {}", output_path.display()))?;
    output::success(format!(
        "Wrote {} ({} mount(s), control port {})",
        output_path.display(),
        cfg.mounts.len(),
        cfg.control_port.unwrap_or(DEFAULT_CONTROL_PORT)
    ));
    Ok(())
}

fn run_check(cfg: &ComposeRelevantConfig, output_path: &Path) -> Result<()> {
    let existing = match fs::read_to_string(output_path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Drift: override missing entirely.
            output::error(format!(
                "Override file {} is missing — run: wqm docker generate-compose",
                output_path.display()
            ));
            std::process::exit(1);
        }
        Err(e) => {
            return Err(anyhow!(e).context(format!(
                "failed to read override file at {}",
                output_path.display()
            )))
        }
    };
    let recorded = extract_hash_header(&existing).ok_or_else(|| {
        anyhow!(
            "override at {} is missing the `{} <hash>` header; \
             regenerate via: wqm docker generate-compose",
            output_path.display(),
            HASH_HEADER_PREFIX
        )
    })?;
    let live = mount_section_hash(&cfg.mounts);
    if recorded == live {
        output::success(format!(
            "Override at {} is up to date",
            output_path.display()
        ));
        Ok(())
    } else {
        output::error(format!(
            "Override at {} is stale (recorded {}…, live {}…) — run: wqm docker generate-compose",
            output_path.display(),
            &recorded[..recorded.len().min(12)],
            &live[..live.len().min(12)],
        ));
        std::process::exit(1);
    }
}

fn run_clean(output_path: &Path) -> Result<()> {
    match fs::remove_file(output_path) {
        Ok(()) => {
            output::success(format!("Removed {}", output_path.display()));
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            output::info(format!("No override file at {}", output_path.display()));
            Ok(())
        }
        Err(e) => Err(anyhow!(e).context(format!(
            "failed to remove override file at {}",
            output_path.display()
        ))),
    }
}

// ────────────────────────────────────────────────────────────────────────
// Config loading
// ────────────────────────────────────────────────────────────────────────

fn resolve_config_path(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            bail!("config file not found: {}", p.display());
        }
        return Ok(p.to_path_buf());
    }
    find_config_file().ok_or_else(|| {
        anyhow!(
            "no config.yaml found; set WQM_CONFIG_PATH or create one under \
             ~/.config/workspace-qdrant/"
        )
    })
}

fn load_relevant_config(path: &Path) -> Result<ComposeRelevantConfig> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    parse_relevant_config(&raw)
}

fn parse_relevant_config(raw: &str) -> Result<ComposeRelevantConfig> {
    serde_yaml_ng::from_str::<ComposeRelevantConfig>(raw)
        .context("config.yaml is not valid YAML or has incompatible types")
}

// ────────────────────────────────────────────────────────────────────────
// Validation
// ────────────────────────────────────────────────────────────────────────

fn validate_network_mode(cfg: &ComposeRelevantConfig) -> Result<()> {
    // Per spec §10.1: refuse anything that prevents the control-port bind
    // from arbitrating between host and docker memexd. Only `host` and the
    // default bridge (which we'll publish to 127.0.0.1:7799) are allowed.
    if let Some(mode) = cfg.network_mode.as_deref() {
        let normalized = mode.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "host" | "bridge" | "" => Ok(()),
            "none" => Err(anyhow!(
                "network_mode: none is incompatible with the cross-process \
                 control-port lock — see docs/specs/16-path-abstraction.md §10.1"
            )),
            other => Err(anyhow!(
                "network_mode `{}` is not supported by `wqm docker generate-compose`. \
                 Use `host` or the default bridge with the mandatory \
                 `127.0.0.1:7799:7799` publish. See docs/specs/16-path-abstraction.md §10.1",
                other
            )),
        }
    } else {
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────
// YAML rendering
// ────────────────────────────────────────────────────────────────────────

fn render_override_yaml(
    cfg: &ComposeRelevantConfig,
    cfg_path: &Path,
    output_path: &Path,
) -> Result<String> {
    let hash = mount_section_hash(&cfg.mounts);
    let control_port = cfg.control_port.unwrap_or(DEFAULT_CONTROL_PORT);
    let use_host_network = matches!(
        cfg.network_mode
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("host")
    );

    let mut out = String::new();
    out.push_str(HASH_HEADER_PREFIX);
    out.push(' ');
    out.push_str(&hash);
    out.push('\n');
    out.push_str("# Generated by `wqm docker generate-compose`.\n");
    out.push_str("# DO NOT EDIT MANUALLY — regenerate via: wqm docker generate-compose\n");
    out.push_str("# See docs/specs/16-path-abstraction.md §9 for the design.\n");
    out.push_str("services:\n");
    out.push_str("  memexd:\n");

    // Volumes section: mounts + config + state + qdrant.
    out.push_str("    volumes:\n");
    for entry in &cfg.mounts {
        out.push_str("      - \"");
        out.push_str(&escape_yaml_double(&entry.host));
        out.push(':');
        out.push_str(&escape_yaml_double(&entry.container));
        out.push_str("\"\n");
    }
    // Config-file bind mount (read-only) — host path of the *active* config.
    let cfg_display = cfg_path
        .to_str()
        .ok_or_else(|| anyhow!("config path is not valid UTF-8: {}", cfg_path.display()))?;
    out.push_str("      - \"");
    out.push_str(&escape_yaml_double(cfg_display));
    out.push(':');
    out.push_str(CONTAINER_CONFIG_PATH);
    out.push_str(":ro\"\n");
    // Self-bind the override file read-only into the container so the
    // entrypoint can hash-check it on startup (spec §9.1 layer 1). The
    // host path is `output_path` (absolute when supplied via --output;
    // relative paths are resolved against the current directory at
    // generation time).
    let override_host = if output_path.is_absolute() {
        output_path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("cannot resolve override host path (current dir unavailable)")?
            .join(output_path)
    };
    let override_display = override_host.to_str().ok_or_else(|| {
        anyhow!(
            "override path is not valid UTF-8: {}",
            override_host.display()
        )
    })?;
    out.push_str("      - \"");
    out.push_str(&escape_yaml_double(override_display));
    out.push(':');
    out.push_str(CONTAINER_OVERRIDE_PATH);
    out.push_str(":ro\"\n");
    // State bind mounts (unconditional, spec §9.2).
    out.push_str("      - \"");
    out.push_str(HOST_STATE_DIR_DEFAULT);
    out.push(':');
    out.push_str(CONTAINER_STATE_DIR);
    out.push_str("\"\n");
    out.push_str("      - \"");
    out.push_str(HOST_QDRANT_DIR_DEFAULT);
    out.push(':');
    out.push_str(CONTAINER_QDRANT_DIR);
    out.push_str("\"\n");

    // Network: either `network_mode: host` or default bridge + port publish.
    if use_host_network {
        out.push_str("    network_mode: \"host\"\n");
        out.push_str(&format!(
            "    # control port {} bound directly on host network namespace\n",
            control_port
        ));
    } else {
        // Default bridge with mandatory 127.0.0.1:7799:7799 publish (§10.1).
        out.push_str("    ports:\n");
        out.push_str(&format!("      - \"127.0.0.1:{0}:{0}\"\n", control_port));
    }

    Ok(out)
}

/// Extract the `# wqm-config-hash: <hex>` value from an override file body.
///
/// Returns `None` if no header line is present. The header is expected on
/// the first non-empty line (we wrote it there), but we accept it anywhere
/// to be lenient with hand-edits that prepend a blank line.
pub(crate) fn extract_hash_header(body: &str) -> Option<String> {
    for line in body.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(HASH_HEADER_PREFIX) {
            return Some(rest.trim().to_string());
        }
    }
    None
}

/// Escape a value for use inside a YAML double-quoted scalar.
///
/// Only `"` and `\` need escaping for our compose use-case (paths). We never
/// produce control characters. Keeping the escape table tiny avoids dragging
/// a full YAML emitter in for a single line per mount.
fn escape_yaml_double(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with_mounts(mounts: Vec<(&str, &str)>) -> ComposeRelevantConfig {
        ComposeRelevantConfig {
            mounts: mounts
                .into_iter()
                .map(|(h, c)| YamlMountEntry {
                    host: h.into(),
                    container: c.into(),
                })
                .collect(),
            control_port: None,
            network_mode: None,
        }
    }

    #[test]
    fn parse_relevant_config_ignores_unknown_top_level_keys() {
        let raw = r#"
qdrant:
  url: http://localhost:6333
performance:
  max_concurrent_tasks: 4
mounts:
  - host: /Users/username/dev
    container: /Users/username/dev
control_port: 8800
"#;
        let cfg = parse_relevant_config(raw).unwrap();
        assert_eq!(cfg.mounts.len(), 1);
        assert_eq!(cfg.mounts[0].host, "/Users/username/dev");
        assert_eq!(cfg.control_port, Some(8800));
    }

    #[test]
    fn parse_relevant_config_handles_missing_mounts() {
        let cfg = parse_relevant_config("performance: {}\n").unwrap();
        assert!(cfg.mounts.is_empty());
        assert!(cfg.control_port.is_none());
    }

    #[test]
    fn render_override_contains_hash_header_and_mandatory_port() {
        let cfg = cfg_with_mounts(vec![("/Users/username/dev", "/Users/username/dev")]);
        let rendered = render_override_yaml(
            &cfg,
            Path::new("/etc/wqm/config.yaml"),
            Path::new("/tmp/docker-compose.override.yaml"),
        )
        .unwrap();
        assert!(rendered.starts_with("# wqm-config-hash: "));
        assert!(rendered.contains("127.0.0.1:7799:7799"));
        assert!(rendered.contains("/Users/username/dev:/Users/username/dev"));
    }

    #[test]
    fn render_override_emits_state_bind_mounts() {
        let cfg = cfg_with_mounts(vec![]);
        let rendered = render_override_yaml(
            &cfg,
            Path::new("/etc/wqm/config.yaml"),
            Path::new("/tmp/docker-compose.override.yaml"),
        )
        .unwrap();
        assert!(rendered.contains("~/.local/share/workspace-qdrant:/var/lib/wqm"));
        assert!(rendered.contains("~/.local/share/qdrant:/qdrant/storage"));
        assert!(rendered.contains("/etc/wqm/config.yaml:/etc/wqm/config.yaml:ro"));
    }

    #[test]
    fn render_override_uses_custom_control_port() {
        let mut cfg = cfg_with_mounts(vec![]);
        cfg.control_port = Some(9000);
        let rendered = render_override_yaml(
            &cfg,
            Path::new("/etc/wqm/config.yaml"),
            Path::new("/tmp/docker-compose.override.yaml"),
        )
        .unwrap();
        assert!(rendered.contains("127.0.0.1:9000:9000"));
    }

    #[test]
    fn render_override_host_network_skips_ports_section() {
        let mut cfg = cfg_with_mounts(vec![]);
        cfg.network_mode = Some("host".into());
        let rendered = render_override_yaml(
            &cfg,
            Path::new("/etc/wqm/config.yaml"),
            Path::new("/tmp/docker-compose.override.yaml"),
        )
        .unwrap();
        assert!(rendered.contains("network_mode: \"host\""));
        assert!(!rendered.contains("ports:"));
    }

    #[test]
    fn validate_network_mode_rejects_none() {
        let mut cfg = cfg_with_mounts(vec![]);
        cfg.network_mode = Some("none".into());
        let err = validate_network_mode(&cfg).unwrap_err();
        assert!(err.to_string().contains("network_mode: none"));
    }

    #[test]
    fn validate_network_mode_rejects_custom() {
        let mut cfg = cfg_with_mounts(vec![]);
        cfg.network_mode = Some("my-custom-net".into());
        let err = validate_network_mode(&cfg).unwrap_err();
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn validate_network_mode_allows_host_and_bridge() {
        let mut cfg = cfg_with_mounts(vec![]);
        cfg.network_mode = Some("host".into());
        assert!(validate_network_mode(&cfg).is_ok());
        cfg.network_mode = Some("bridge".into());
        assert!(validate_network_mode(&cfg).is_ok());
        cfg.network_mode = None;
        assert!(validate_network_mode(&cfg).is_ok());
    }

    #[test]
    fn extract_hash_header_finds_first_line() {
        let body = "# wqm-config-hash: abc123\nservices:\n  memexd: {}\n";
        assert_eq!(extract_hash_header(body), Some("abc123".into()));
    }

    #[test]
    fn extract_hash_header_tolerates_leading_blank() {
        let body = "\n\n# wqm-config-hash: deadbeef\n";
        assert_eq!(extract_hash_header(body), Some("deadbeef".into()));
    }

    #[test]
    fn extract_hash_header_returns_none_when_absent() {
        let body = "services:\n  memexd: {}\n";
        assert_eq!(extract_hash_header(body), None);
    }

    #[test]
    fn escape_yaml_double_passes_paths_through() {
        assert_eq!(escape_yaml_double("/a/b/c"), "/a/b/c");
    }

    #[test]
    fn escape_yaml_double_escapes_quote_and_backslash() {
        assert_eq!(escape_yaml_double("a\"b"), "a\\\"b");
        assert_eq!(escape_yaml_double("a\\b"), "a\\\\b");
    }

    #[test]
    fn rendered_yaml_round_trips_through_yaml_parser() {
        let cfg = cfg_with_mounts(vec![
            ("/Users/username/dev", "/Users/username/dev"),
            ("/Volumes/External/books", "/mnt/books"),
        ]);
        let rendered = render_override_yaml(
            &cfg,
            Path::new("/etc/wqm/config.yaml"),
            Path::new("/tmp/docker-compose.override.yaml"),
        )
        .unwrap();
        // The hash header is a YAML comment, so the body is a single document.
        let parsed: serde_yaml_ng::Value =
            serde_yaml_ng::from_str(&rendered).expect("rendered override must parse as YAML");
        let services = parsed.get("services").expect("services key");
        let memexd = services.get("memexd").expect("memexd key");
        let volumes = memexd
            .get("volumes")
            .and_then(|v| v.as_sequence())
            .expect("volumes sequence");
        // 2 mounts + 1 config + 1 override-self-bind + 2 state = 6
        assert_eq!(volumes.len(), 6);
        let ports = memexd
            .get("ports")
            .and_then(|p| p.as_sequence())
            .expect("ports sequence");
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].as_str(), Some("127.0.0.1:7799:7799"));
    }

    #[test]
    fn rendered_yaml_preserves_hash_for_check_round_trip() {
        let cfg = cfg_with_mounts(vec![("/a", "/a")]);
        let rendered = render_override_yaml(
            &cfg,
            Path::new("/cfg.yaml"),
            Path::new("/tmp/override.yaml"),
        )
        .unwrap();
        let recorded = extract_hash_header(&rendered).expect("hash present");
        let live = mount_section_hash(&cfg.mounts);
        assert_eq!(recorded, live);
    }
}
