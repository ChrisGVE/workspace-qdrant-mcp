//! `wqm admin token-savings` — display token-economy aggregates.
//!
//! Reads the `token_savings` view created by migration v38 and aggregates
//! per-tool (default) or per-day/per-session (via flags). Renders a
//! borderless table or JSON.
//!
//! Spec: `docs/specs/20-token-economy-instrumentation.md` §4.
//!
//! Bytes are the source of truth in storage; the table also shows a
//! "≈ tokens" line derived as `bytes / 4` (a conservative proxy that
//! does not depend on a specific tokenizer — see spec §1 and §8).

use anyhow::{Context, Result};
use rusqlite::{params_from_iter, types::Value};
use serde::Serialize;
use tabled::Tabled;

use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::table::ColumnHints;
use crate::output::{self};

use super::perf_data::{fmt_thousands, fmt_thousands_f};

/// Borderless table row.
#[derive(Tabled, Serialize)]
struct TokenSavingsRow {
    #[tabled(rename = "Tool")]
    #[serde(rename = "tool")]
    tool: String,
    #[tabled(rename = "Calls")]
    #[serde(rename = "calls")]
    calls: String,
    #[tabled(rename = "Bytes in")]
    #[serde(rename = "bytes_in")]
    bytes_in: String,
    #[tabled(rename = "Bytes out")]
    #[serde(rename = "bytes_out")]
    bytes_out: String,
    #[tabled(rename = "Saved %")]
    #[serde(rename = "saved_pct")]
    saved_pct: String,
    #[tabled(rename = "Truncated")]
    #[serde(rename = "hits_truncated")]
    hits_truncated: String,
    #[tabled(rename = "Followup %")]
    #[serde(rename = "followup_pct")]
    followup_pct: String,
    #[tabled(rename = "Escalation %")]
    #[serde(rename = "escalation_pct")]
    escalation_pct: String,
}

impl ColumnHints for TokenSavingsRow {
    fn content_columns() -> &'static [usize] {
        &[0]
    }
    fn numeric_columns() -> &'static [usize] {
        &[1, 2, 3, 4, 5, 6, 7]
    }
}

/// One per-tool aggregate.
#[derive(Debug, Clone, Serialize)]
struct ToolAggregate {
    tool: String,
    calls: i64,
    bytes_in: i64,
    bytes_out: i64,
    hits_truncated: i64,
    followup_count: i64,
    escalation_count: i64,
}

impl ToolAggregate {
    fn savings_ratio(&self) -> Option<f64> {
        if self.bytes_in <= 0 {
            None
        } else {
            Some(1.0 - (self.bytes_out as f64 / self.bytes_in as f64))
        }
    }

    fn followup_pct(&self) -> Option<f64> {
        if self.calls <= 0 {
            None
        } else {
            Some(100.0 * (self.followup_count as f64 / self.calls as f64))
        }
    }

    fn escalation_pct(&self) -> Option<f64> {
        if self.calls <= 0 {
            None
        } else {
            Some(100.0 * (self.escalation_count as f64 / self.calls as f64))
        }
    }
}

/// Execute the token-savings subcommand.
pub async fn execute(
    window: String,
    json: bool,
    project: Option<String>,
    tool: Option<String>,
) -> Result<()> {
    let window_hours = parse_window(&window)?;
    let conn = open_state_db()?;
    ensure_view(&conn)?;

    let aggregates = query_aggregates(&conn, window_hours, project.as_deref(), tool.as_deref())
        .context("Failed to query token_savings view")?;

    if aggregates.is_empty() {
        if json {
            println!("{}", render_empty_json(window_hours, &project, &tool));
        } else {
            output::info(format!(
                "No token-economy events in the last {} hours.",
                fmt_thousands_f(window_hours)
            ));
        }
        return Ok(());
    }

    if json {
        print_json(&aggregates, window_hours, &project, &tool);
    } else {
        print_table(&aggregates, window_hours, &project, &tool);
    }

    Ok(())
}

// ── Parsing ─────────────────────────────────────────────────────────────────

/// Parse a window string like `7d`, `24h`, `30m`. Returns hours as f64.
fn parse_window(input: &str) -> Result<f64> {
    let s = input.trim();
    if s.is_empty() {
        anyhow::bail!("Empty --window value");
    }
    let (num_part, unit) = if let Some(stripped) = s.strip_suffix('d') {
        (stripped, 'd')
    } else if let Some(stripped) = s.strip_suffix('h') {
        (stripped, 'h')
    } else if let Some(stripped) = s.strip_suffix('m') {
        (stripped, 'm')
    } else {
        // Bare number → interpret as hours, matching wqm admin perf default.
        (s, 'h')
    };
    let n: f64 = num_part
        .parse()
        .with_context(|| format!("Invalid --window number: {}", num_part))?;
    if n <= 0.0 {
        anyhow::bail!("--window must be > 0 (got {})", n);
    }
    let hours = match unit {
        'd' => n * 24.0,
        'h' => n,
        'm' => n / 60.0,
        _ => unreachable!(),
    };
    Ok(hours)
}

// ── Storage ─────────────────────────────────────────────────────────────────

fn open_state_db() -> Result<rusqlite::Connection> {
    let db_path = crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;
    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }
    let conn =
        rusqlite::Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .context("Failed to open state database")?;
    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;
    Ok(conn)
}

fn ensure_view(conn: &rusqlite::Connection) -> Result<()> {
    let exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master \
             WHERE type='view' AND name='token_savings')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);
    if !exists {
        anyhow::bail!(
            "token_savings view not found. Daemon may be running an older schema (v37 or below). \
             Restart the daemon to apply migration v38."
        );
    }
    Ok(())
}

fn query_aggregates(
    conn: &rusqlite::Connection,
    window_hours: f64,
    project: Option<&str>,
    tool: Option<&str>,
) -> Result<Vec<ToolAggregate>> {
    // Build dynamic WHERE — keep `ts` filter first to leverage the index
    // on `(ts)` already present on `search_events`.
    let mut sql = String::from(
        "SELECT \
            tool, \
            COUNT(*)                                                       AS calls, \
            COALESCE(SUM(bytes_in),       0)                               AS sum_bytes_in, \
            COALESCE(SUM(bytes_out),      0)                               AS sum_bytes_out, \
            COALESCE(SUM(hits_truncated), 0)                               AS sum_hits_truncated, \
            SUM(CASE WHEN had_followup   THEN 1 ELSE 0 END)                AS followup_count, \
            SUM(CASE WHEN had_escalation THEN 1 ELSE 0 END)                AS escalation_count \
         FROM token_savings \
         WHERE ts >= datetime('now', ?1)",
    );
    let mut params: Vec<Value> = vec![Value::Text(format!("-{} hours", window_hours))];

    if let Some(p) = project {
        sql.push_str(" AND project_id = ?");
        sql.push_str(&(params.len() + 1).to_string());
        params.push(Value::Text(p.to_string()));
    }
    if let Some(t) = tool {
        sql.push_str(" AND tool = ?");
        sql.push_str(&(params.len() + 1).to_string());
        params.push(Value::Text(t.to_string()));
    }

    sql.push_str(" GROUP BY tool ORDER BY sum_bytes_in DESC");

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
        Ok(ToolAggregate {
            tool: row.get::<_, String>(0)?,
            calls: row.get::<_, i64>(1)?,
            bytes_in: row.get::<_, i64>(2)?,
            bytes_out: row.get::<_, i64>(3)?,
            hits_truncated: row.get::<_, i64>(4)?,
            followup_count: row.get::<_, i64>(5)?,
            escalation_count: row.get::<_, i64>(6)?,
        })
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

// ── Output: table ───────────────────────────────────────────────────────────

fn print_table(
    aggregates: &[ToolAggregate],
    window_hours: f64,
    project: &Option<String>,
    tool: &Option<String>,
) {
    canvas::print_title(&format!(
        "Token Savings (last {})",
        humanize_window(window_hours)
    ));
    canvas::print_blank();

    let rows: Vec<TokenSavingsRow> = aggregates.iter().map(aggregate_to_row).collect();
    output::print_table_auto(&rows);

    let totals = total_aggregate(aggregates);
    print_summary(&totals, project, tool);
}

fn aggregate_to_row(a: &ToolAggregate) -> TokenSavingsRow {
    TokenSavingsRow {
        tool: a.tool.clone(),
        calls: fmt_thousands(a.calls),
        bytes_in: fmt_bytes(a.bytes_in),
        bytes_out: fmt_bytes(a.bytes_out),
        saved_pct: fmt_pct(a.savings_ratio().map(|r| r * 100.0)),
        hits_truncated: fmt_thousands(a.hits_truncated),
        followup_pct: fmt_pct(a.followup_pct()),
        escalation_pct: fmt_pct(a.escalation_pct()),
    }
}

#[derive(Default)]
struct TotalsSnapshot {
    calls: i64,
    bytes_in: i64,
    bytes_out: i64,
    hits_truncated: i64,
    followup_count: i64,
    escalation_count: i64,
}

fn total_aggregate(aggregates: &[ToolAggregate]) -> TotalsSnapshot {
    let mut t = TotalsSnapshot::default();
    for a in aggregates {
        t.calls += a.calls;
        t.bytes_in += a.bytes_in;
        t.bytes_out += a.bytes_out;
        t.hits_truncated += a.hits_truncated;
        t.followup_count += a.followup_count;
        t.escalation_count += a.escalation_count;
    }
    t
}

fn print_summary(t: &TotalsSnapshot, project: &Option<String>, tool: &Option<String>) {
    let saved_pct = if t.bytes_in > 0 {
        Some(100.0 * (t.bytes_in - t.bytes_out) as f64 / t.bytes_in as f64)
    } else {
        None
    };

    // Conservative proxy: ~4 bytes per token. See spec §1 and §8.
    let tokens_in = t.bytes_in as f64 / 4.0;
    let tokens_out = t.bytes_out as f64 / 4.0;
    let tokens_saved = tokens_in - tokens_out;

    let mut builder = ColumnarBuilder::new()
        .section(Some("Totals"))
        .kv("Calls", fmt_thousands(t.calls))
        .kv("Bytes in", fmt_bytes(t.bytes_in))
        .kv("Bytes out", fmt_bytes(t.bytes_out))
        .kv("Saved", fmt_pct(saved_pct))
        .kv(
            "≈ Tokens (bytes ÷ 4)",
            format!(
                "{} in → {} out = ~{} saved",
                fmt_count(tokens_in),
                fmt_count(tokens_out),
                fmt_count(tokens_saved.max(0.0))
            ),
        );

    if let Some(p) = project {
        builder = builder.kv("Project filter", p.clone());
    }
    if let Some(t_) = tool {
        builder = builder.kv("Tool filter", t_.clone());
    }

    builder.render();
}

// ── Output: JSON ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct JsonAggregate<'a> {
    tool: &'a str,
    calls: i64,
    bytes_in: i64,
    bytes_out: i64,
    savings_bytes: i64,
    savings_ratio: Option<f64>,
    hits_truncated: i64,
    followup_count: i64,
    followup_ratio: Option<f64>,
    escalation_count: i64,
    escalation_ratio: Option<f64>,
}

#[derive(Serialize)]
struct JsonRoot<'a> {
    window_hours: f64,
    project: Option<&'a str>,
    tool: Option<&'a str>,
    tools: Vec<JsonAggregate<'a>>,
    totals: JsonTotals,
}

#[derive(Serialize)]
struct JsonTotals {
    calls: i64,
    bytes_in: i64,
    bytes_out: i64,
    savings_bytes: i64,
    savings_ratio: Option<f64>,
    tokens_in_approx: f64,
    tokens_out_approx: f64,
    tokens_saved_approx: f64,
}

fn print_json(
    aggregates: &[ToolAggregate],
    window_hours: f64,
    project: &Option<String>,
    tool: &Option<String>,
) {
    let tools: Vec<JsonAggregate> = aggregates
        .iter()
        .map(|a| JsonAggregate {
            tool: a.tool.as_str(),
            calls: a.calls,
            bytes_in: a.bytes_in,
            bytes_out: a.bytes_out,
            savings_bytes: a.bytes_in - a.bytes_out,
            savings_ratio: a.savings_ratio(),
            hits_truncated: a.hits_truncated,
            followup_count: a.followup_count,
            followup_ratio: a.followup_pct().map(|p| p / 100.0),
            escalation_count: a.escalation_count,
            escalation_ratio: a.escalation_pct().map(|p| p / 100.0),
        })
        .collect();

    let t = total_aggregate(aggregates);
    let totals = JsonTotals {
        calls: t.calls,
        bytes_in: t.bytes_in,
        bytes_out: t.bytes_out,
        savings_bytes: t.bytes_in - t.bytes_out,
        savings_ratio: if t.bytes_in > 0 {
            Some(1.0 - (t.bytes_out as f64 / t.bytes_in as f64))
        } else {
            None
        },
        tokens_in_approx: t.bytes_in as f64 / 4.0,
        tokens_out_approx: t.bytes_out as f64 / 4.0,
        tokens_saved_approx: ((t.bytes_in - t.bytes_out) as f64 / 4.0).max(0.0),
    };

    let root = JsonRoot {
        window_hours,
        project: project.as_deref(),
        tool: tool.as_deref(),
        tools,
        totals,
    };

    println!("{}", serde_json::to_string_pretty(&root).unwrap());
}

fn render_empty_json(window_hours: f64, project: &Option<String>, tool: &Option<String>) -> String {
    let root = JsonRoot {
        window_hours,
        project: project.as_deref(),
        tool: tool.as_deref(),
        tools: Vec::new(),
        totals: JsonTotals {
            calls: 0,
            bytes_in: 0,
            bytes_out: 0,
            savings_bytes: 0,
            savings_ratio: None,
            tokens_in_approx: 0.0,
            tokens_out_approx: 0.0,
            tokens_saved_approx: 0.0,
        },
    };
    serde_json::to_string_pretty(&root).unwrap()
}

// ── Formatting helpers ──────────────────────────────────────────────────────

/// Format bytes as KiB / MiB / GiB with one decimal.
pub(crate) fn fmt_bytes(n: i64) -> String {
    let n_abs = n.unsigned_abs() as f64;
    let sign = if n < 0 { "-" } else { "" };
    if n_abs < 1024.0 {
        format!("{}{} B", sign, n_abs as i64)
    } else if n_abs < 1024.0 * 1024.0 {
        format!("{}{:.1} KiB", sign, n_abs / 1024.0)
    } else if n_abs < 1024.0 * 1024.0 * 1024.0 {
        format!("{}{:.1} MiB", sign, n_abs / (1024.0 * 1024.0))
    } else {
        format!("{}{:.2} GiB", sign, n_abs / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Format a percentage with one decimal, or "—" for None.
pub(crate) fn fmt_pct(p: Option<f64>) -> String {
    match p {
        Some(v) => format!("{:.1}%", v),
        None => "—".to_string(),
    }
}

/// Format a token count as K/M (no decimal for < 1k, one decimal otherwise).
pub(crate) fn fmt_count(n: f64) -> String {
    if n < 1000.0 {
        format!("{}", n.round() as i64)
    } else if n < 1_000_000.0 {
        format!("{:.1}K", n / 1000.0)
    } else if n < 1_000_000_000.0 {
        format!("{:.2}M", n / 1_000_000.0)
    } else {
        format!("{:.2}B", n / 1_000_000_000.0)
    }
}

/// Human-readable window: "7d" / "24h" / "30m".
fn humanize_window(hours: f64) -> String {
    if hours >= 24.0 && (hours / 24.0).fract() == 0.0 {
        format!("{}d", (hours / 24.0) as i64)
    } else if hours >= 1.0 && hours.fract() == 0.0 {
        format!("{}h", hours as i64)
    } else if hours < 1.0 {
        format!("{}m", (hours * 60.0).round() as i64)
    } else {
        format!("{:.1}h", hours)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_window_accepts_d_h_m_and_bare_number() {
        assert_eq!(parse_window("7d").unwrap(), 7.0 * 24.0);
        assert_eq!(parse_window("24h").unwrap(), 24.0);
        assert_eq!(parse_window("30m").unwrap(), 0.5);
        assert_eq!(parse_window("12").unwrap(), 12.0); // bare → hours
    }

    #[test]
    fn parse_window_rejects_invalid() {
        assert!(parse_window("").is_err());
        assert!(parse_window("-1d").is_err());
        assert!(parse_window("0h").is_err());
        assert!(parse_window("abc").is_err());
    }

    #[test]
    fn fmt_bytes_picks_appropriate_unit() {
        assert_eq!(fmt_bytes(0), "0 B");
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(2_048), "2.0 KiB");
        assert_eq!(fmt_bytes(1_500_000), "1.4 MiB");
        assert_eq!(fmt_bytes(2_500_000_000), "2.33 GiB");
    }

    #[test]
    fn fmt_pct_handles_none_and_one_decimal() {
        assert_eq!(fmt_pct(None), "—");
        assert_eq!(fmt_pct(Some(88.6)), "88.6%");
        assert_eq!(fmt_pct(Some(0.0)), "0.0%");
    }

    #[test]
    fn fmt_count_picks_unit() {
        assert_eq!(fmt_count(0.0), "0");
        assert_eq!(fmt_count(500.0), "500");
        assert_eq!(fmt_count(7_600.0), "7.6K");
        assert_eq!(fmt_count(4_100_000.0), "4.10M");
    }

    #[test]
    fn humanize_window_rounds_to_d_h_m() {
        assert_eq!(humanize_window(168.0), "7d");
        assert_eq!(humanize_window(24.0), "1d");
        assert_eq!(humanize_window(12.0), "12h");
        assert_eq!(humanize_window(0.5), "30m");
        assert_eq!(humanize_window(1.5), "1.5h");
    }

    #[test]
    fn tool_aggregate_ratios_handle_zero_division() {
        let zero = ToolAggregate {
            tool: "x".into(),
            calls: 0,
            bytes_in: 0,
            bytes_out: 0,
            hits_truncated: 0,
            followup_count: 0,
            escalation_count: 0,
        };
        assert_eq!(zero.savings_ratio(), None);
        assert_eq!(zero.followup_pct(), None);
        assert_eq!(zero.escalation_pct(), None);

        let saved = ToolAggregate {
            tool: "search".into(),
            calls: 100,
            bytes_in: 10_000,
            bytes_out: 1_500,
            hits_truncated: 50,
            followup_count: 12,
            escalation_count: 4,
        };
        let r = saved.savings_ratio().unwrap();
        assert!((r - 0.85).abs() < 1e-9);
        let f = saved.followup_pct().unwrap();
        assert!((f - 12.0).abs() < 1e-9);
        let e = saved.escalation_pct().unwrap();
        assert!((e - 4.0).abs() < 1e-9);
    }
}
