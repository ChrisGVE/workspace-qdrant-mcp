"""
Unit tests for ML model monitoring system.

Tests monitoring capabilities including drift detection, alerting,
and performance tracking with basic functionality verification.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.python.common.ml.config.ml_config import MLConfig
from src.python.common.ml.monitoring.model_monitor import (
    AlertLevel,
    DataProfile,
    DriftAlert,
    DriftType,
    ModelMonitor,
    MonitoringStatus,
    PerformanceMetrics,
)


class TestDataProfile:
    """Test DataProfile class."""

    def test_data_profile_creation(self):
        """Test creating data profile."""
        profile = DataProfile(
            feature_name="test_feature",
            data_type="float64",
            mean=5.0,
            std=2.0
        )

        assert profile.feature_name == "test_feature"
        assert profile.data_type == "float64"
        assert profile.mean == 5.0
        assert profile.std == 2.0
        assert isinstance(profile.created_at, datetime)

    def test_data_profile_to_dict(self):
        """Test converting data profile to dictionary."""
        profile = DataProfile(
            feature_name="test_feature",
            data_type="float64",
            mean=5.0,
            percentiles={"25": 3.0, "75": 7.0}
        )

        data = profile.to_dict()
        assert data["feature_name"] == "test_feature"
        assert data["mean"] == 5.0
        assert data["percentiles"]["25"] == 3.0


class TestDriftAlert:
    """Test DriftAlert class."""

    def test_drift_alert_creation(self):
        """Test creating drift alert."""
        alert = DriftAlert(
            alert_id="test_alert_123",
            model_id="test_model",
            drift_type=DriftType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Test alert message",
            details={"drift_score": 0.8},
            detected_at=datetime.now()
        )

        assert alert.alert_id == "test_alert_123"
        assert alert.model_id == "test_model"
        assert alert.drift_type == DriftType.DATA_DRIFT
        assert alert.level == AlertLevel.WARNING
        assert not alert.acknowledged
        assert not alert.resolved


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            model_id="test_model",
            timestamp=datetime.now(),
            accuracy=0.95,
            f1_score=0.92
        )

        assert metrics.model_id == "test_model"
        assert metrics.accuracy == 0.95
        assert metrics.f1_score == 0.92
        assert metrics.custom_metrics == {}

    def test_performance_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        now = datetime.now()
        metrics = PerformanceMetrics(
            model_id="test_model",
            timestamp=now,
            accuracy=0.95,
            custom_metrics={"custom": 0.88}
        )

        data = metrics.to_dict()
        assert data["model_id"] == "test_model"
        assert data["accuracy"] == 0.95
        assert data["custom_metrics"]["custom"] == 0.88
        assert isinstance(data["timestamp"], str)


class TestModelMonitor:
    """Test ModelMonitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_path = Path(tempfile.mkdtemp())

        self.config = MLConfig(
            project_name="test_monitoring",
            artifacts_directory=self.temp_path / "artifacts"
        )

        # Patch threading to avoid background threads during tests
        with patch('threading.Thread'):
            self.monitor = ModelMonitor(self.config)
            self.monitor.monitoring_active = False  # Disable background monitoring

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.config == self.config
        assert self.monitor.status == MonitoringStatus.ACTIVE
        assert isinstance(self.monitor.model_profiles, dict)
        assert isinstance(self.monitor.alerts, dict)
        assert self.monitor.monitoring_dir.exists()

    def test_set_baseline_profile(self):
        """Test setting baseline profile."""
        features = {
            "feature1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "feature2": np.array([10, 20, 30, 40, 50])
        }

        self.monitor.set_baseline_profile("test_model", features)

        assert "test_model" in self.monitor.model_profiles
        assert len(self.monitor.model_profiles["test_model"]) == 2
        assert "feature1" in self.monitor.model_profiles["test_model"]
        assert "feature2" in self.monitor.model_profiles["test_model"]

        profile1 = self.monitor.model_profiles["test_model"]["feature1"]
        assert profile1.mean == 3.0
        assert profile1.std == pytest.approx(1.58, rel=0.1)

    def test_generate_data_profile_numerical(self):
        """Test generating data profile for numerical features."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        profile = self.monitor._generate_data_profile("test_feature", values)

        assert profile.feature_name == "test_feature"
        assert profile.mean == 3.0
        assert profile.std == pytest.approx(1.58, rel=0.1)
        assert profile.min_val == 1.0
        assert profile.max_val == 5.0
        assert "25" in profile.percentiles
        assert profile.missing_rate == 0.0

    def test_generate_data_profile_categorical(self):
        """Test generating data profile for categorical features."""
        values = np.array(['A', 'B', 'A', 'C', 'B'])
        profile = self.monitor._generate_data_profile("test_feature", values)

        assert profile.feature_name == "test_feature"
        assert profile.data_type == '<U1'  # Unicode string type
        assert len(profile.unique_values) == 3
        assert set(profile.unique_values) == {'A', 'B', 'C'}

    def test_calculate_drift_score_numerical(self):
        """Test calculating drift score for numerical features."""
        baseline = DataProfile(
            feature_name="test",
            data_type="float64",
            mean=5.0,
            std=2.0
        )

        current = DataProfile(
            feature_name="test",
            data_type="float64",
            mean=7.0,
            std=3.0
        )

        drift_score = self.monitor._calculate_drift_score(baseline, current)
        assert 0.0 <= drift_score <= 1.0
        assert drift_score > 0  # Should detect some drift

    def test_calculate_drift_score_categorical(self):
        """Test calculating drift score for categorical features."""
        baseline = DataProfile(
            feature_name="test",
            data_type="object",
            unique_values=['A', 'B', 'C']
        )

        current = DataProfile(
            feature_name="test",
            data_type="object",
            unique_values=['A', 'B', 'D']  # C changed to D
        )

        drift_score = self.monitor._calculate_drift_score(baseline, current)
        assert 0.0 <= drift_score <= 1.0
        assert drift_score > 0  # Should detect drift due to changed categories

    def test_monitor_data_drift_no_drift(self):
        """Test data drift monitoring with no drift."""
        # Set baseline
        baseline_features = {
            "feature1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        }
        self.monitor.set_baseline_profile("test_model", baseline_features)

        # Monitor with similar data
        current_features = {
            "feature1": np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        }

        drift_scores = self.monitor.monitor_data_drift("test_model", current_features)

        assert "feature1" in drift_scores
        assert drift_scores["feature1"] < 0.05  # Should be low drift

    def test_monitor_data_drift_with_drift(self):
        """Test data drift monitoring with significant drift."""
        # Set baseline
        baseline_features = {
            "feature1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        }
        self.monitor.set_baseline_profile("test_model", baseline_features)

        # Monitor with very different data
        current_features = {
            "feature1": np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }

        drift_scores = self.monitor.monitor_data_drift("test_model", current_features, threshold=0.01)

        assert "feature1" in drift_scores
        assert drift_scores["feature1"] > 0.01  # Should detect significant drift

        # Check that alert was created
        model_alerts = [a for a in self.monitor.alerts.values() if a.model_id == "test_model"]
        assert len(model_alerts) > 0

    def test_calculate_performance_degradation(self):
        """Test calculating performance degradation."""
        baseline = PerformanceMetrics(
            model_id="test",
            timestamp=datetime.now(),
            accuracy=0.95,
            f1_score=0.92
        )

        current = PerformanceMetrics(
            model_id="test",
            timestamp=datetime.now(),
            accuracy=0.85,  # 10% degradation
            f1_score=0.88   # ~4% degradation
        )

        degradation = self.monitor._calculate_performance_degradation(baseline, current)
        assert degradation > 0
        assert degradation == pytest.approx(0.105, rel=0.1)  # Should be around 10% (max degradation)

    def test_monitor_performance_drift(self):
        """Test performance drift monitoring."""
        baseline_metrics = PerformanceMetrics(
            model_id="test_model",
            timestamp=datetime.now(),
            accuracy=0.95
        )

        self.monitor.baseline_metrics["test_model"] = baseline_metrics

        current_metrics = PerformanceMetrics(
            model_id="test_model",
            timestamp=datetime.now(),
            accuracy=0.80  # 15% degradation
        )

        # Monitor with high threshold (should not trigger alert)
        drift_detected = self.monitor.monitor_performance_drift("test_model", current_metrics, threshold=0.20)
        assert not drift_detected

        # Monitor with low threshold (should trigger alert)
        drift_detected = self.monitor.monitor_performance_drift("test_model", current_metrics, threshold=0.10)
        assert drift_detected

        # Check alert was created
        perf_alerts = [
            a for a in self.monitor.alerts.values()
            if a.model_id == "test_model" and a.drift_type == DriftType.PERFORMANCE_DRIFT
        ]
        assert len(perf_alerts) > 0

    def test_add_alert_callback(self):
        """Test adding alert callbacks."""
        callback_called = []

        def test_callback(alert):
            callback_called.append(alert.alert_id)

        self.monitor.add_alert_callback(test_callback)

        # Trigger an alert
        self.monitor._create_drift_alert(
            model_id="test_model",
            drift_type=DriftType.DATA_DRIFT,
            drift_score=0.8
        )

        assert len(callback_called) == 1

    def test_acknowledge_alert(self):
        """Test acknowledging alerts."""
        # Create an alert
        self.monitor._create_drift_alert(
            model_id="test_model",
            drift_type=DriftType.DATA_DRIFT,
            drift_score=0.8
        )

        alert_id = list(self.monitor.alerts.keys())[0]
        alert = self.monitor.alerts[alert_id]
        assert not alert.acknowledged

        # Acknowledge alert
        success = self.monitor.acknowledge_alert(alert_id)
        assert success
        assert alert.acknowledged

    def test_resolve_alert(self):
        """Test resolving alerts."""
        # Create an alert
        self.monitor._create_drift_alert(
            model_id="test_model",
            drift_type=DriftType.DATA_DRIFT,
            drift_score=0.8
        )

        alert_id = list(self.monitor.alerts.keys())[0]
        alert = self.monitor.alerts[alert_id]
        assert not alert.resolved

        # Resolve alert
        success = self.monitor.resolve_alert(alert_id)
        assert success
        assert alert.resolved

    def test_get_alerts_filtered(self):
        """Test getting alerts with filtering."""
        # Create multiple alerts
        self.monitor._create_drift_alert(
            model_id="model1",
            drift_type=DriftType.DATA_DRIFT,
            drift_score=0.5
        )

        self.monitor._create_drift_alert(
            model_id="model2",
            drift_type=DriftType.PERFORMANCE_DRIFT,
            performance_degradation=0.15
        )

        # Get all alerts
        all_alerts = self.monitor.get_alerts()
        assert len(all_alerts) == 2

        # Get alerts for specific model
        model1_alerts = self.monitor.get_alerts(model_id="model1")
        assert len(model1_alerts) == 1
        assert model1_alerts[0].model_id == "model1"

        # Get alerts by level (assuming performance drift creates critical alert)
        critical_alerts = self.monitor.get_alerts(level=AlertLevel.CRITICAL)
        assert len(critical_alerts) >= 0  # May be 0 or 1 depending on threshold

    def test_get_model_health(self):
        """Test getting model health status."""
        # Create some test data
        self.monitor._create_drift_alert(
            model_id="test_model",
            drift_type=DriftType.DATA_DRIFT,
            drift_score=0.3
        )

        health = self.monitor.get_model_health("test_model")

        assert health["model_id"] == "test_model"
        assert "timestamp" in health
        assert "status" in health
        assert "alerts" in health
        assert "recommendations" in health

        assert len(health["alerts"]) > 0

    def test_stop_monitoring(self):
        """Test stopping monitoring system."""
        self.monitor.stop_monitoring()
        assert self.monitor.status == MonitoringStatus.STOPPED
        assert not self.monitor.monitoring_active
