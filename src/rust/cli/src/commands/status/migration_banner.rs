//! Relative-path migration banner for `wqm status health`.
//!
//! When a relative-path migration is in progress (the daemon's state.db
//! holds a row in the `relative_path_migration_in_progress` table), this
//! module renders a banner above the health output. The banner shows
//! progress as a percentage based on queue counts plus the
//! `initial_pending_count` captured when the initial walk finished.
//!
//! See `docs/specs/16-path-abstraction.md` §6.2.2.

use anyhow::Result;
use colored::Colorize;
use rusqlite::Connection;

use crate::config::get_database_path_checked;

/// Snapshot of the marker row as read by the CLI. Mirrors the daemon-side
/// `RelativePathMigrationStatus` but read-only and rusqlite-backed.
#[derive(Debug, Clone)]
pub struct MigrationBannerInfo {
    pub initial_walk_complete: bool,
    pub initial_pending_count: Option<i64>,
    pub pending_remaining: i64,
    pub in_progress_remaining: i64,
}

impl MigrationBannerInfo {
    /// Render the banner string. Returns `None` if the migration is not in
    /// progress (no marker row, or marker table absent).
    pub fn render(&self) -> String {
        let total = self.pending_remaining + self.in_progress_remaining;
        match self.initial_pending_count {
            Some(initial) if self.initial_walk_complete && initial > 0 => {
                let processed = (initial - total).max(0);
                let percent = ((processed as f64) / (initial as f64) * 100.0).clamp(0.0, 100.0);
                format!(
                    "⚠ Relative-path migration in progress: {:.0}% complete ({} of {} files processed)",
                    percent, processed, initial
                )
            }
            _ => {
                // Initial walk not done yet — we don't know the denominator.
                format!(
                    "⚠ Relative-path migration in progress: initial walk underway ({} queued)",
                    total
                )
            }
        }
    }

    /// Render the banner string with terminal styling (yellow, matching
    /// other `wqm status` warning-tier output).
    pub fn render_styled(&self) -> String {
        self.render().yellow().to_string()
    }
}

/// Look up the marker row and pending-queue counts from the daemon's
/// `state.db`. Returns `Ok(None)` for fresh databases, completed migrations,
/// or when the database file is not yet present.
pub fn fetch_migration_banner_info() -> Result<Option<MigrationBannerInfo>> {
    let db_path = match get_database_path_checked() {
        Ok(p) => p,
        // No database yet → nothing to report.
        Err(_) => return Ok(None),
    };

    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.execute_batch("PRAGMA busy_timeout=2000;")?;

    // Marker table may not exist on fresh databases that never ran v37.
    let marker_table_exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' \
         AND name='relative_path_migration_in_progress'",
        [],
        |r| r.get(0),
    )?;
    if marker_table_exists == 0 {
        return Ok(None);
    }

    let row: Option<(i64, Option<i64>)> = conn
        .query_row(
            "SELECT initial_walk_complete, initial_pending_count \
             FROM relative_path_migration_in_progress LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();
    let (initial_walk_complete_raw, initial_pending_count) = match row {
        Some(t) => t,
        None => return Ok(None),
    };

    // Count remaining work in the unified queue. Absent queue table → 0
    // (the migration just started and the queue hasn't been re-enqueued
    // yet).
    let queue_exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='unified_queue'",
        [],
        |r| r.get(0),
    )?;
    let (pending_remaining, in_progress_remaining) = if queue_exists > 0 {
        let pending: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let in_progress: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM unified_queue WHERE status = 'in_progress'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        (pending, in_progress)
    } else {
        (0, 0)
    };

    Ok(Some(MigrationBannerInfo {
        initial_walk_complete: initial_walk_complete_raw != 0,
        initial_pending_count,
        pending_remaining,
        in_progress_remaining,
    }))
}

/// Print the migration banner if active, with a trailing blank line. Silently
/// skips when the migration is not active. Fetch errors are logged at debug
/// level and swallowed — the banner must never block health output.
pub fn print_if_active() {
    match fetch_migration_banner_info() {
        Ok(Some(info)) => {
            println!("{}", info.render_styled());
            println!();
        }
        Ok(None) => {}
        Err(e) => {
            tracing::debug!(
                target: "wqm::status::migration_banner",
                "skipping relative-path migration banner: {e}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_with_known_total_shows_percent() {
        let info = MigrationBannerInfo {
            initial_walk_complete: true,
            initial_pending_count: Some(1000),
            pending_remaining: 250,
            in_progress_remaining: 50,
        };
        let s = info.render();
        // 1000 - 300 = 700 processed; 70% complete.
        assert!(s.contains("70%"), "expected 70% in: {s}");
        assert!(s.contains("700 of 1000"), "expected counts in: {s}");
        assert!(s.contains("Relative-path migration"));
    }

    #[test]
    fn render_before_initial_walk_shows_underway_state() {
        let info = MigrationBannerInfo {
            initial_walk_complete: false,
            initial_pending_count: None,
            pending_remaining: 42,
            in_progress_remaining: 0,
        };
        let s = info.render();
        assert!(s.contains("initial walk underway"), "got: {s}");
        assert!(s.contains("42 queued"), "got: {s}");
    }

    #[test]
    fn render_zero_initial_avoids_div_by_zero() {
        let info = MigrationBannerInfo {
            initial_walk_complete: true,
            initial_pending_count: Some(0),
            pending_remaining: 0,
            in_progress_remaining: 0,
        };
        let _ = info.render(); // must not panic
    }

    #[test]
    fn render_processed_clamped_at_zero() {
        // If the queue grew after the walk (shouldn't happen, but
        // defensive math): processed could go negative — clamp to 0.
        let info = MigrationBannerInfo {
            initial_walk_complete: true,
            initial_pending_count: Some(100),
            pending_remaining: 200,
            in_progress_remaining: 0,
        };
        let s = info.render();
        assert!(s.contains("0%"), "expected 0% clamp: {s}");
    }

    #[test]
    fn render_styled_preserves_payload_and_contains_ansi() {
        // Force colorized output even when the test runs in a non-TTY
        // environment (e.g. CI captured stdout) so the ANSI escapes are
        // present and the smoke test is deterministic.
        colored::control::set_override(true);
        let info = MigrationBannerInfo {
            initial_walk_complete: true,
            initial_pending_count: Some(1000),
            pending_remaining: 250,
            in_progress_remaining: 50,
        };
        let styled = info.render_styled();
        colored::control::unset_override();

        // Styled banner contains the same payload as the plain render.
        assert!(styled.contains("70%"), "payload missing in: {styled}");
        assert!(
            styled.contains("Relative-path migration"),
            "headline missing in: {styled}"
        );
        // ANSI yellow SGR sequence — ensures the styling wrapper actually
        // applied colour rather than passing through the plain string.
        assert!(
            styled.contains("\u{1b}["),
            "expected ANSI escape in styled banner: {styled:?}"
        );
    }
}
