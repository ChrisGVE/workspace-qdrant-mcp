//! `wqm admin metrics` — fetch live Prometheus metrics from the daemon
//!
//! Connects to the daemon's HTTP metrics endpoint and displays the current
//! Prometheus metrics in human-readable or JSON format.

use anyhow::{Context, Result};

use crate::output;

/// Execute the metrics subcommand.
pub async fn execute(port: u16, json: bool) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/metrics", port);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .context("Failed to build HTTP client")?;

    let response = client.get(&url).send().await.map_err(|e| {
        if e.is_connect() {
            anyhow::anyhow!(
                "Cannot connect to metrics endpoint at {}. \
                 Is the daemon running with --metrics-port {}?",
                url,
                port
            )
        } else {
            anyhow::anyhow!("Failed to fetch metrics: {}", e)
        }
    })?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Metrics endpoint returned HTTP {}: {}",
            response.status(),
            response.text().await.unwrap_or_default()
        );
    }

    let body = response.text().await.context("Failed to read response")?;

    if json {
        print_json(&body);
    } else {
        print_human(&body, port);
    }

    Ok(())
}

/// Parse Prometheus text format and display as structured output.
fn print_human(body: &str, port: u16) {
    output::section(format!("Daemon Metrics (port {})", port));

    for line in body.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        // Metric line: name{labels} value
        let (name_labels, value) = match line.rsplit_once(' ') {
            Some((n, v)) => (n, v),
            None => continue,
        };

        let display = if let Some(brace_pos) = name_labels.find('{') {
            let name = &name_labels[..brace_pos];
            let labels = &name_labels[brace_pos..];
            format!("{} {}", name, labels)
        } else {
            name_labels.to_string()
        };

        output::kv(&display, value);
    }
}

/// Output raw Prometheus metrics as a JSON object with metric families.
fn print_json(body: &str) {
    let mut metrics: Vec<serde_json::Value> = Vec::new();

    for line in body.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        let (name_labels, value) = match line.rsplit_once(' ') {
            Some((n, v)) => (n, v),
            None => continue,
        };

        let (name, labels) = if let Some(brace_pos) = name_labels.find('{') {
            let name = &name_labels[..brace_pos];
            let label_str = &name_labels[brace_pos + 1..name_labels.len() - 1];
            let labels: serde_json::Map<String, serde_json::Value> = label_str
                .split(',')
                .filter_map(|pair| {
                    let (k, v) = pair.split_once('=')?;
                    let v = v.trim_matches('"');
                    Some((k.to_string(), serde_json::Value::String(v.to_string())))
                })
                .collect();
            (name, serde_json::Value::Object(labels))
        } else {
            (
                name_labels.as_ref(),
                serde_json::Value::Object(serde_json::Map::new()),
            )
        };

        let val: f64 = value.parse().unwrap_or(0.0);

        metrics.push(serde_json::json!({
            "name": name,
            "labels": labels,
            "value": val,
        }));
    }

    let obj = serde_json::json!({ "metrics": metrics });
    println!("{}", serde_json::to_string_pretty(&obj).unwrap());
}
