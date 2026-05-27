use workspace_qdrant_core::config::DaemonConfig;

fn main() {
    let cfg = DaemonConfig::default();
    println!(
        "{}",
        serde_yaml_ng::to_string(&cfg).expect("failed to serialize default config")
    );
}
