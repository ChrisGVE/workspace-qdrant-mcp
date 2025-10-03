-- Metric History Schema
-- Schema for storing historical queue metrics for trend analysis
-- Purpose: Track queue performance metrics over time for statistical analysis,
--          forecasting, and anomaly detection

-- =============================================================================
-- METRIC HISTORY TABLE
-- =============================================================================

-- Historical metric storage table for time-series analysis
-- Stores all trackable metrics with timestamps and optional metadata
CREATE TABLE IF NOT EXISTS metric_history (
    -- Primary identifier
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Timestamp when metric was recorded
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Metric identifier (e.g., 'queue_size', 'processing_rate', 'error_rate')
    metric_name TEXT NOT NULL CHECK (
        metric_name IN (
            'queue_size',
            'processing_rate',
            'error_rate',
            'latency',
            'success_rate',
            'resource_usage_cpu',
            'resource_usage_memory'
        )
    ),

    -- Numeric metric value
    value REAL NOT NULL,

    -- Additional context as JSON
    -- Examples:
    --   {"collection": "my-project", "tenant_id": "default"}
    --   {"priority": "high", "operation": "ingest"}
    --   {"source": "backpressure_detection"}
    metadata TEXT  -- JSON
);

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Composite index for metric queries by name and time range (most common query)
-- Ordered DESC for recent-first queries
CREATE INDEX IF NOT EXISTS idx_metric_history_name_timestamp
    ON metric_history(metric_name, timestamp DESC);

-- Index for timestamp-based cleanup operations
CREATE INDEX IF NOT EXISTS idx_metric_history_timestamp
    ON metric_history(timestamp);

-- Index for metric name filtering
CREATE INDEX IF NOT EXISTS idx_metric_history_metric_name
    ON metric_history(metric_name);

-- =============================================================================
-- RETENTION AND CLEANUP
-- =============================================================================

-- Retention policy notes:
-- - Default retention: 30 days (configurable via trend_analysis.retention_days)
-- - Cleanup runs daily via HistoricalTrendAnalyzer._cleanup_old_data()
-- - DELETE query: DELETE FROM metric_history WHERE timestamp < datetime('now', '-30 days')
-- - Vacuum recommended after cleanup: VACUUM metric_history

-- =============================================================================
-- CONVENIENCE VIEWS
-- =============================================================================

-- View for recent metric summary (last 24 hours)
CREATE VIEW IF NOT EXISTS recent_metrics AS
SELECT
    metric_name,
    COUNT(*) as data_points,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    MIN(timestamp) as earliest,
    MAX(timestamp) as latest
FROM metric_history
WHERE timestamp >= datetime('now', '-24 hours')
GROUP BY metric_name;

-- View for metric counts by type (helps monitor data volume)
CREATE VIEW IF NOT EXISTS metric_counts AS
SELECT
    metric_name,
    COUNT(*) as total_points,
    MIN(timestamp) as oldest_point,
    MAX(timestamp) as newest_point,
    (julianday(MAX(timestamp)) - julianday(MIN(timestamp))) as days_of_data
FROM metric_history
GROUP BY metric_name
ORDER BY metric_name;

-- =============================================================================
-- EXAMPLE QUERIES
-- =============================================================================

-- Get queue_size trend for last 24 hours:
-- SELECT timestamp, value FROM metric_history
-- WHERE metric_name = 'queue_size' AND timestamp >= datetime('now', '-24 hours')
-- ORDER BY timestamp ASC;

-- Calculate average processing rate for last week:
-- SELECT AVG(value) FROM metric_history
-- WHERE metric_name = 'processing_rate' AND timestamp >= datetime('now', '-7 days');

-- Find anomalies (values > 3 std deviations from mean):
-- WITH stats AS (
--     SELECT AVG(value) as mean, (AVG(value * value) - AVG(value) * AVG(value)) as variance
--     FROM metric_history WHERE metric_name = 'error_rate'
-- )
-- SELECT m.timestamp, m.value,
--        (m.value - stats.mean) / sqrt(stats.variance) as z_score
-- FROM metric_history m, stats
-- WHERE m.metric_name = 'error_rate'
--   AND ABS((m.value - stats.mean) / sqrt(stats.variance)) > 3;
