//! Tests for debug subcommand modules

#[cfg(test)]
mod tests {
    use crate::commands::debug::log_parsing::*;
    use chrono::{Datelike, Duration, Timelike, Utc};
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    // -- parse_relative_duration --

    #[test]
    fn test_parse_relative_duration_seconds() {
        let d = parse_relative_duration("30s").unwrap();
        assert_eq!(d, Duration::seconds(30));
    }

    #[test]
    fn test_parse_relative_duration_minutes() {
        let d = parse_relative_duration("5m").unwrap();
        assert_eq!(d, Duration::minutes(5));
    }

    #[test]
    fn test_parse_relative_duration_hours() {
        let d = parse_relative_duration("2h").unwrap();
        assert_eq!(d, Duration::hours(2));
    }

    #[test]
    fn test_parse_relative_duration_days() {
        let d = parse_relative_duration("7d").unwrap();
        assert_eq!(d, Duration::days(7));
    }

    #[test]
    fn test_parse_relative_duration_weeks() {
        let d = parse_relative_duration("1w").unwrap();
        assert_eq!(d, Duration::weeks(1));
    }

    #[test]
    fn test_parse_relative_duration_milliseconds() {
        let d = parse_relative_duration("500ms").unwrap();
        assert_eq!(d, Duration::milliseconds(500));
    }

    #[test]
    fn test_parse_relative_duration_no_unit_defaults_to_seconds() {
        let d = parse_relative_duration("60").unwrap();
        assert_eq!(d, Duration::seconds(60));
    }

    #[test]
    fn test_parse_relative_duration_invalid_unit() {
        assert!(parse_relative_duration("5x").is_err());
    }

    #[test]
    fn test_parse_relative_duration_empty() {
        assert!(parse_relative_duration("").is_err());
    }

    // -- parse_since --

    #[test]
    fn test_parse_since_relative() {
        let before = Utc::now();
        let cutoff = parse_since("1h").unwrap();
        let expected_approx = before - Duration::hours(1);
        // Within 2 seconds tolerance
        assert!((cutoff - expected_approx).num_seconds().abs() < 2);
    }

    #[test]
    fn test_parse_since_rfc3339() {
        let cutoff = parse_since("2025-06-15T10:30:00Z").unwrap();
        assert_eq!(cutoff.year(), 2025);
        assert_eq!(cutoff.month(), 6);
        assert_eq!(cutoff.hour(), 10);
    }

    #[test]
    fn test_parse_since_naive_datetime() {
        let cutoff = parse_since("2025-06-15T10:30:00").unwrap();
        assert_eq!(cutoff.year(), 2025);
    }

    #[test]
    fn test_parse_since_space_separated() {
        let cutoff = parse_since("2025-06-15 10:30:00").unwrap();
        assert_eq!(cutoff.year(), 2025);
    }

    #[test]
    fn test_parse_since_invalid() {
        assert!(parse_since("not-a-time").is_err());
    }

    // -- LogLevel --

    #[test]
    fn test_log_level_from_string() {
        assert_eq!(LogLevel::from_str("ERROR"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("WARN"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("INFO"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("DEBUG"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_str("TRACE"), Some(LogLevel::Trace));
    }

    #[test]
    fn test_log_level_from_string_case_insensitive() {
        assert_eq!(LogLevel::from_str("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("warn"), Some(LogLevel::Warn));
    }

    #[test]
    fn test_log_level_from_pino_numeric() {
        // pino uses numeric levels: 50=error, 40=warn, 30=info, 20=debug, 10=trace
        let error = serde_json::json!(50);
        assert_eq!(LogLevel::from_json(&error), Some(LogLevel::Error));

        let warn = serde_json::json!(40);
        assert_eq!(LogLevel::from_json(&warn), Some(LogLevel::Warn));

        let info = serde_json::json!(30);
        assert_eq!(LogLevel::from_json(&info), Some(LogLevel::Info));
    }

    #[test]
    fn test_log_level_is_error_or_warn() {
        assert!(LogLevel::Error.is_error_or_warn());
        assert!(LogLevel::Warn.is_error_or_warn());
        assert!(!LogLevel::Info.is_error_or_warn());
        assert!(!LogLevel::Debug.is_error_or_warn());
    }

    // -- LogEntry parsing --

    #[test]
    fn test_log_entry_from_json_with_level() {
        let line = r#"{"timestamp":"2025-06-15T10:30:00Z","level":"ERROR","msg":"test"}"#;
        let entry = LogEntry::from_json_line(line, "daemon").unwrap();
        assert_eq!(entry.level, Some(LogLevel::Error));
        assert!(entry.parsed_time.is_some());
        assert_eq!(entry.component, "daemon");
    }

    #[test]
    fn test_log_entry_from_pino_json() {
        let line = r#"{"time":"2025-06-15T10:30:00Z","level":30,"msg":"hello"}"#;
        let entry = LogEntry::from_json_line(line, "mcp-server").unwrap();
        assert_eq!(entry.level, Some(LogLevel::Info));
    }

    #[test]
    fn test_log_entry_from_non_json() {
        let line = "plain text log line";
        let entry = LogEntry::from_json_line(line, "daemon").unwrap();
        assert!(entry.level.is_none());
        assert!(entry.parsed_time.is_none());
    }

    // -- LogFilter --

    #[test]
    fn test_log_filter_errors_only() {
        let filter = LogFilter {
            errors_only: true,
            since: None,
            session: None,
        };

        let error_entry = LogEntry {
            timestamp: String::new(),
            parsed_time: None,
            level: Some(LogLevel::Error),
            component: "daemon".to_string(),
            session_id: None,
            raw_line: "error".to_string(),
        };
        assert!(filter.matches(&error_entry));

        let info_entry = LogEntry {
            level: Some(LogLevel::Info),
            ..error_entry
        };
        assert!(!filter.matches(&info_entry));
    }

    #[test]
    fn test_log_filter_since() {
        let cutoff = Utc::now() - Duration::hours(1);
        let filter = LogFilter {
            errors_only: false,
            since: Some(cutoff),
            session: None,
        };

        let recent = LogEntry {
            timestamp: String::new(),
            parsed_time: Some(Utc::now()),
            level: Some(LogLevel::Info),
            component: "daemon".to_string(),
            session_id: None,
            raw_line: "recent".to_string(),
        };
        assert!(filter.matches(&recent));

        let old = LogEntry {
            parsed_time: Some(Utc::now() - Duration::hours(2)),
            ..recent
        };
        assert!(!filter.matches(&old));
    }

    #[test]
    fn test_log_filter_session() {
        let filter = LogFilter {
            errors_only: false,
            since: None,
            session: Some("abc123".to_string()),
        };

        let matching = LogEntry {
            timestamp: String::new(),
            parsed_time: None,
            level: None,
            component: "mcp".to_string(),
            session_id: Some("session-abc123-xyz".to_string()),
            raw_line: "test".to_string(),
        };
        assert!(filter.matches(&matching));

        let no_session = LogEntry {
            session_id: None,
            ..matching
        };
        assert!(!filter.matches(&no_session));
    }

    // -- discover_log_files --

    #[test]
    fn test_discover_log_files_current_only() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("daemon.jsonl");
        File::create(&log_path).unwrap();

        let files = discover_log_files(dir.path(), "daemon");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], log_path);
    }

    #[test]
    fn test_discover_log_files_with_rotated() {
        let dir = TempDir::new().unwrap();

        // Current + 2 rotated
        File::create(dir.path().join("daemon.jsonl")).unwrap();
        File::create(dir.path().join("daemon.jsonl.1")).unwrap();
        File::create(dir.path().join("daemon.jsonl.2")).unwrap();
        // .gz files should be ignored (not yet supported)
        File::create(dir.path().join("daemon.jsonl.3.gz")).unwrap();

        let files = discover_log_files(dir.path(), "daemon");
        assert_eq!(files.len(), 3);
        // Current first, then rotated in order
        assert!(files[0].ends_with("daemon.jsonl"));
        assert!(files[1].ends_with("daemon.jsonl.1"));
        assert!(files[2].ends_with("daemon.jsonl.2"));
    }

    #[test]
    fn test_discover_log_files_empty_dir() {
        let dir = TempDir::new().unwrap();
        let files = discover_log_files(dir.path(), "daemon");
        assert!(files.is_empty());
    }

    // -- read_log_file_filtered --

    #[test]
    fn test_read_log_file_filtered_errors_only() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&log_path).unwrap();
            writeln!(
                f,
                r#"{{"timestamp":"2025-06-15T10:00:00Z","level":"INFO","msg":"info line"}}"#
            )
            .unwrap();
            writeln!(
                f,
                r#"{{"timestamp":"2025-06-15T10:01:00Z","level":"ERROR","msg":"error line"}}"#
            )
            .unwrap();
            writeln!(
                f,
                r#"{{"timestamp":"2025-06-15T10:02:00Z","level":"WARN","msg":"warn line"}}"#
            )
            .unwrap();
            writeln!(
                f,
                r#"{{"timestamp":"2025-06-15T10:03:00Z","level":"DEBUG","msg":"debug line"}}"#
            )
            .unwrap();
        }

        let filter = LogFilter {
            errors_only: true,
            since: None,
            session: None,
        };
        let entries = read_log_file_filtered(&log_path, "test", 100, &filter);
        assert_eq!(entries.len(), 2);
        assert!(entries[0].raw_line.contains("error line"));
        assert!(entries[1].raw_line.contains("warn line"));
    }

    #[test]
    fn test_read_log_file_filtered_since() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&log_path).unwrap();
            // Old entry (2020)
            writeln!(
                f,
                r#"{{"timestamp":"2020-01-01T00:00:00Z","level":"INFO","msg":"old"}}"#
            )
            .unwrap();
            // Recent entry
            let recent = Utc::now().to_rfc3339();
            writeln!(
                f,
                r#"{{"timestamp":"{}","level":"INFO","msg":"recent"}}"#,
                recent
            )
            .unwrap();
        }

        let filter = LogFilter {
            errors_only: false,
            since: Some(Utc::now() - Duration::hours(1)),
            session: None,
        };
        let entries = read_log_file_filtered(&log_path, "test", 100, &filter);
        assert_eq!(entries.len(), 1);
        assert!(entries[0].raw_line.contains("recent"));
    }

    #[test]
    fn test_read_log_file_filtered_max_lines() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&log_path).unwrap();
            for i in 0..20 {
                writeln!(
                    f,
                    r#"{{"timestamp":"2025-06-15T10:{:02}:00Z","level":"INFO","msg":"line {}"}}"#,
                    i, i
                )
                .unwrap();
            }
        }

        let filter = LogFilter {
            errors_only: false,
            since: None,
            session: None,
        };
        let entries = read_log_file_filtered(&log_path, "test", 5, &filter);
        assert_eq!(entries.len(), 5);
        // Should be the last 5 lines (15-19)
        assert!(entries[0].raw_line.contains("line 15"));
        assert!(entries[4].raw_line.contains("line 19"));
    }

    #[test]
    fn test_read_log_files_filtered_with_rotated() {
        let dir = TempDir::new().unwrap();

        // Write old entries to rotated file
        {
            let mut f = File::create(dir.path().join("test.jsonl.1")).unwrap();
            writeln!(
                f,
                r#"{{"timestamp":"2025-06-15T08:00:00Z","level":"ERROR","msg":"old error"}}"#
            )
            .unwrap();
        }
        // Write recent entries to current file
        {
            let mut f = File::create(dir.path().join("test.jsonl")).unwrap();
            let recent = Utc::now().to_rfc3339();
            writeln!(
                f,
                r#"{{"timestamp":"{}","level":"ERROR","msg":"new error"}}"#,
                recent
            )
            .unwrap();
        }

        // With --since, should read from both files
        let filter = LogFilter {
            errors_only: true,
            since: Some(Utc::now() - Duration::days(365)),
            session: None,
        };
        let entries = read_log_files_filtered(dir.path(), "test", "test", 100, &filter);
        assert_eq!(entries.len(), 2);
    }

    // -- Combined filter --

    #[test]
    fn test_combined_errors_only_and_since() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&log_path).unwrap();
            // Old error
            writeln!(
                f,
                r#"{{"timestamp":"2020-01-01T00:00:00Z","level":"ERROR","msg":"old error"}}"#
            )
            .unwrap();
            // Recent info
            let recent = Utc::now().to_rfc3339();
            writeln!(
                f,
                r#"{{"timestamp":"{}","level":"INFO","msg":"recent info"}}"#,
                recent
            )
            .unwrap();
            // Recent error
            let recent2 = Utc::now().to_rfc3339();
            writeln!(
                f,
                r#"{{"timestamp":"{}","level":"ERROR","msg":"recent error"}}"#,
                recent2
            )
            .unwrap();
        }

        let filter = LogFilter {
            errors_only: true,
            since: Some(Utc::now() - Duration::hours(1)),
            session: None,
        };
        let entries = read_log_file_filtered(&log_path, "test", 100, &filter);
        assert_eq!(entries.len(), 1);
        assert!(entries[0].raw_line.contains("recent error"));
    }
}
