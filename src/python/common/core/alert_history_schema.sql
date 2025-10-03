-- =============================================================================
-- ALERT HISTORY SCHEMA
-- =============================================================================
-- Purpose: Stores alert history, alert rules, and notification delivery status
-- for the queue alerting system.
--
-- Tables:
--   - alert_rules: Alert rule definitions with thresholds and conditions
--   - alert_history: Historical record of triggered alerts
--   - alert_delivery_status: Notification delivery tracking
-- =============================================================================

-- Alert Rules Table
-- Stores configured alert rules with thresholds and notification settings
CREATE TABLE IF NOT EXISTS alert_rules (
    rule_id TEXT PRIMARY KEY,
    rule_name TEXT NOT NULL UNIQUE,
    description TEXT,
    enabled BOOLEAN NOT NULL DEFAULT 1,
    condition_logic TEXT NOT NULL DEFAULT 'AND',  -- 'AND' or 'OR'
    thresholds_json TEXT NOT NULL,  -- JSON array of AlertThreshold objects
    recipients_json TEXT NOT NULL,  -- JSON array of recipient configurations
    cooldown_minutes INTEGER NOT NULL DEFAULT 15,
    last_triggered_at REAL,  -- Unix timestamp of last alert
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Alert History Table
-- Records all triggered alerts with their metrics and status
CREATE TABLE IF NOT EXISTS alert_history (
    alert_id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    severity TEXT NOT NULL,  -- 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message TEXT NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    threshold_value REAL,
    threshold_operator TEXT,  -- '>', '<', '==', '>=', '<='
    details_json TEXT,  -- Additional context as JSON
    timestamp REAL NOT NULL,
    acknowledged BOOLEAN NOT NULL DEFAULT 0,
    acknowledged_at REAL,
    acknowledged_by TEXT,
    FOREIGN KEY (rule_id) REFERENCES alert_rules(rule_id) ON DELETE CASCADE
);

-- Alert Delivery Status Table
-- Tracks notification delivery attempts and status
CREATE TABLE IF NOT EXISTS alert_delivery_status (
    delivery_id TEXT PRIMARY KEY,
    alert_id TEXT NOT NULL,
    channel TEXT NOT NULL,  -- 'log', 'email', 'webhook', 'slack', 'pagerduty'
    status TEXT NOT NULL,  -- 'pending', 'success', 'failed', 'retrying'
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt_at REAL,
    delivered_at REAL,
    error_message TEXT,
    FOREIGN KEY (alert_id) REFERENCES alert_history(alert_id) ON DELETE CASCADE
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Alert history indexes for common queries
CREATE INDEX IF NOT EXISTS idx_alert_history_timestamp ON alert_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alert_history_severity ON alert_history(severity);
CREATE INDEX IF NOT EXISTS idx_alert_history_rule_id ON alert_history(rule_id);
CREATE INDEX IF NOT EXISTS idx_alert_history_acknowledged ON alert_history(acknowledged);
CREATE INDEX IF NOT EXISTS idx_alert_history_severity_timestamp
    ON alert_history(severity, timestamp DESC);

-- Alert rules indexes
CREATE INDEX IF NOT EXISTS idx_alert_rules_enabled ON alert_rules(enabled);
CREATE INDEX IF NOT EXISTS idx_alert_rules_last_triggered
    ON alert_rules(last_triggered_at);

-- Delivery status indexes
CREATE INDEX IF NOT EXISTS idx_delivery_status_alert ON alert_delivery_status(alert_id);
CREATE INDEX IF NOT EXISTS idx_delivery_status_channel_status
    ON alert_delivery_status(channel, status);
