"""
Advanced ML Model Monitoring System with drift detection and performance tracking.

This module provides comprehensive model monitoring including data drift detection,
concept drift identification, performance degradation alerts, and automated
remediation suggestions for production ML models.
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from ..config.ml_config import MLConfig


class DriftType(Enum):
    """Types of model drift."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringStatus(Enum):
    """Monitoring system status."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class DataProfile:
    """Statistical profile of data features."""
    feature_name: str
    data_type: str
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    percentiles: dict[str, float] = None  # e.g., {"25": 1.5, "50": 2.0, "75": 3.5}
    unique_values: list[str] | None = None  # For categorical features
    missing_rate: float = 0.0
    distribution_hash: str | None = None
    created_at: datetime = None

    def __post_init__(self):
        if self.percentiles is None:
            self.percentiles = {}
        if self.unique_values is None:
            self.unique_values = []
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    alert_id: str
    model_id: str
    drift_type: DriftType
    level: AlertLevel
    message: str
    details: dict[str, Any]
    detected_at: datetime
    acknowledged: bool = False
    resolved: bool = False
    remediation_suggestions: list[str] = None

    def __post_init__(self):
        if self.remediation_suggestions is None:
            self.remediation_suggestions = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['detected_at'] = self.detected_at.isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """Model performance metrics over time."""
    model_id: str
    timestamp: datetime
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    auc_roc: float | None = None
    mse: float | None = None
    mae: float | None = None
    custom_metrics: dict[str, float] = None
    prediction_count: int = 0
    error_rate: float = 0.0
    response_time_p95: float = 0.0

    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MonitoringError(Exception):
    """Base exception for monitoring errors."""
    pass


class DriftDetectionError(MonitoringError):
    """Exception for drift detection failures."""
    pass


class ModelMonitor:
    """
    Advanced ML model monitoring system with comprehensive drift detection.

    Features:
    - Real-time data drift detection using statistical tests
    - Concept drift identification through performance monitoring
    - Prediction drift analysis and alerting
    - Performance degradation detection with customizable thresholds
    - Automated remediation suggestions
    - Historical trend analysis and reporting
    - Integration with deployment systems for automated responses
    """

    def __init__(self, config: MLConfig):
        """
        Initialize model monitor.

        Args:
            config: ML configuration
        """
        self.config = config
        self.status = MonitoringStatus.ACTIVE
        self.model_profiles: dict[str, dict[str, DataProfile]] = {}  # model_id -> feature -> profile
        self.baseline_metrics: dict[str, PerformanceMetrics] = {}  # model_id -> metrics
        self.alerts: dict[str, DriftAlert] = {}
        self.alert_callbacks: list[Callable[[DriftAlert], None]] = []
        self.drift_thresholds = {
            "data_drift": 0.05,  # p-value threshold for KS test
            "performance_drift": 0.05,  # relative performance drop threshold
            "prediction_drift": 0.10,  # prediction distribution change threshold
        }

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create monitoring directory
        self.monitoring_dir = config.artifacts_directory / "monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db_path = self.monitoring_dir / "monitoring.db"
        self._init_database()

        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def set_baseline_profile(
        self,
        model_id: str,
        features: dict[str, np.ndarray],
        metrics: PerformanceMetrics | None = None
    ):
        """
        Set baseline data profile for a model.

        Args:
            model_id: Model identifier
            features: Dictionary of feature arrays {feature_name: values}
            metrics: Baseline performance metrics
        """
        try:
            self.logger.info(f"Setting baseline profile for model {model_id}")

            # Generate feature profiles
            profiles = {}
            for feature_name, values in features.items():
                profile = self._generate_data_profile(feature_name, values)
                profiles[feature_name] = profile

            self.model_profiles[model_id] = profiles

            # Store baseline metrics
            if metrics:
                self.baseline_metrics[model_id] = metrics

            # Persist to database
            self._save_baseline_profile(model_id, profiles, metrics)

            self.logger.info(f"Baseline profile set for model {model_id} with {len(profiles)} features")

        except Exception as e:
            error_msg = f"Failed to set baseline profile for {model_id}: {str(e)}"
            self.logger.error(error_msg)
            raise MonitoringError(error_msg)

    def monitor_data_drift(
        self,
        model_id: str,
        features: dict[str, np.ndarray],
        threshold: float | None = None
    ) -> dict[str, float]:
        """
        Monitor for data drift in model features.

        Args:
            model_id: Model identifier
            features: Current feature data
            threshold: Custom drift detection threshold

        Returns:
            Dictionary of drift scores per feature
        """
        try:
            if model_id not in self.model_profiles:
                raise MonitoringError(f"No baseline profile found for model {model_id}")

            threshold = threshold or self.drift_thresholds["data_drift"]
            drift_scores = {}
            baseline_profiles = self.model_profiles[model_id]

            for feature_name, values in features.items():
                if feature_name not in baseline_profiles:
                    self.logger.warning(f"No baseline for feature {feature_name}, skipping drift detection")
                    continue

                baseline_profile = baseline_profiles[feature_name]
                current_profile = self._generate_data_profile(feature_name, values)

                # Perform statistical test for drift detection
                drift_score = self._calculate_drift_score(baseline_profile, current_profile)
                drift_scores[feature_name] = drift_score

                # Check if drift exceeds threshold
                if drift_score > threshold:
                    self._create_drift_alert(
                        model_id=model_id,
                        drift_type=DriftType.DATA_DRIFT,
                        feature_name=feature_name,
                        drift_score=drift_score,
                        threshold=threshold
                    )

            self.logger.debug(f"Data drift monitoring completed for {model_id}: {drift_scores}")
            return drift_scores

        except Exception as e:
            error_msg = f"Data drift monitoring failed for {model_id}: {str(e)}"
            self.logger.error(error_msg)
            raise DriftDetectionError(error_msg)

    def monitor_performance_drift(
        self,
        model_id: str,
        current_metrics: PerformanceMetrics,
        threshold: float | None = None
    ) -> bool:
        """
        Monitor for performance drift in model predictions.

        Args:
            model_id: Model identifier
            current_metrics: Current performance metrics
            threshold: Custom performance drift threshold

        Returns:
            True if performance drift detected, False otherwise
        """
        try:
            if model_id not in self.baseline_metrics:
                self.logger.warning(f"No baseline metrics for model {model_id}, storing current as baseline")
                self.baseline_metrics[model_id] = current_metrics
                return False

            threshold = threshold or self.drift_thresholds["performance_drift"]
            baseline = self.baseline_metrics[model_id]

            # Calculate performance degradation
            degradation = self._calculate_performance_degradation(baseline, current_metrics)

            # Store current metrics
            self._store_performance_metrics(current_metrics)

            if degradation > threshold:
                self._create_drift_alert(
                    model_id=model_id,
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    performance_degradation=degradation,
                    threshold=threshold,
                    baseline_metrics=baseline.to_dict(),
                    current_metrics=current_metrics.to_dict()
                )
                return True

            return False

        except Exception as e:
            error_msg = f"Performance drift monitoring failed for {model_id}: {str(e)}"
            self.logger.error(error_msg)
            raise DriftDetectionError(error_msg)

    def monitor_prediction_drift(
        self,
        model_id: str,
        predictions: np.ndarray,
        threshold: float | None = None
    ) -> float:
        """
        Monitor for drift in model predictions.

        Args:
            model_id: Model identifier
            predictions: Model predictions array
            threshold: Custom prediction drift threshold

        Returns:
            Prediction drift score
        """
        try:
            threshold = threshold or self.drift_thresholds["prediction_drift"]

            # Get historical predictions
            historical_predictions = self._get_historical_predictions(model_id)
            if not historical_predictions:
                self.logger.info(f"No historical predictions for {model_id}, storing current batch")
                self._store_predictions(model_id, predictions)
                return 0.0

            # Calculate prediction drift using distribution comparison
            drift_score = self._compare_prediction_distributions(historical_predictions, predictions)

            # Store current predictions
            self._store_predictions(model_id, predictions)

            if drift_score > threshold:
                self._create_drift_alert(
                    model_id=model_id,
                    drift_type=DriftType.PREDICTION_DRIFT,
                    drift_score=drift_score,
                    threshold=threshold
                )

            return drift_score

        except Exception as e:
            error_msg = f"Prediction drift monitoring failed for {model_id}: {str(e)}"
            self.logger.error(error_msg)
            raise DriftDetectionError(error_msg)

    def get_model_health(self, model_id: str) -> dict[str, Any]:
        """
        Get comprehensive health status for a model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary containing health metrics and status
        """
        try:
            health_status = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "alerts": [],
                "metrics": {},
                "drift_scores": {},
                "recommendations": []
            }

            # Get active alerts for this model
            model_alerts = [
                alert for alert in self.alerts.values()
                if alert.model_id == model_id and not alert.resolved
            ]
            health_status["alerts"] = [alert.to_dict() for alert in model_alerts]

            # Determine overall health status
            if any(alert.level == AlertLevel.EMERGENCY for alert in model_alerts):
                health_status["status"] = "critical"
            elif any(alert.level == AlertLevel.CRITICAL for alert in model_alerts):
                health_status["status"] = "degraded"
            elif any(alert.level == AlertLevel.WARNING for alert in model_alerts):
                health_status["status"] = "warning"

            # Get recent performance metrics
            recent_metrics = self._get_recent_metrics(model_id)
            if recent_metrics:
                health_status["metrics"] = recent_metrics.to_dict()

            # Get recent drift scores
            health_status["drift_scores"] = self._get_recent_drift_scores(model_id)

            # Generate recommendations
            health_status["recommendations"] = self._generate_health_recommendations(
                model_id, model_alerts, recent_metrics
            )

            return health_status

        except Exception as e:
            error_msg = f"Failed to get health status for {model_id}: {str(e)}"
            self.logger.error(error_msg)
            raise MonitoringError(error_msg)

    def add_alert_callback(self, callback: Callable[[DriftAlert], None]):
        """
        Add callback function for drift alerts.

        Args:
            callback: Function to call when alerts are generated
        """
        self.alert_callbacks.append(callback)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            True if alert acknowledged successfully
        """
        try:
            if alert_id in self.alerts:
                self.alerts[alert_id].acknowledged = True
                self._update_alert_status(alert_id, acknowledged=True)
                self.logger.info(f"Alert {alert_id} acknowledged")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert {alert_id}: {str(e)}")
            return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.

        Args:
            alert_id: Alert identifier

        Returns:
            True if alert resolved successfully
        """
        try:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self._update_alert_status(alert_id, resolved=True)
                self.logger.info(f"Alert {alert_id} resolved")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to resolve alert {alert_id}: {str(e)}")
            return False

    def get_alerts(
        self,
        model_id: str | None = None,
        level: AlertLevel | None = None,
        resolved: bool | None = None
    ) -> list[DriftAlert]:
        """
        Get alerts with optional filtering.

        Args:
            model_id: Filter by model ID
            level: Filter by alert level
            resolved: Filter by resolution status

        Returns:
            List of alerts matching criteria
        """
        alerts = list(self.alerts.values())

        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        if level:
            alerts = [a for a in alerts if a.level == level]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        return sorted(alerts, key=lambda x: x.detected_at, reverse=True)

    def stop_monitoring(self):
        """Stop monitoring system."""
        self.monitoring_active = False
        self.status = MonitoringStatus.STOPPED
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

    def _generate_data_profile(self, feature_name: str, values: np.ndarray) -> DataProfile:
        """Generate statistical profile for feature data."""
        profile = DataProfile(
            feature_name=feature_name,
            data_type=str(values.dtype)
        )

        if np.issubdtype(values.dtype, np.number):
            # Numerical feature
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                profile.mean = float(np.mean(valid_values))
                profile.std = float(np.std(valid_values))
                profile.min_val = float(np.min(valid_values))
                profile.max_val = float(np.max(valid_values))
                profile.percentiles = {
                    "25": float(np.percentile(valid_values, 25)),
                    "50": float(np.percentile(valid_values, 50)),
                    "75": float(np.percentile(valid_values, 75)),
                    "95": float(np.percentile(valid_values, 95))
                }
        else:
            # Categorical feature
            unique_values = np.unique(values)
            profile.unique_values = [str(v) for v in unique_values[:100]]  # Limit to 100 values

        # Calculate missing rate
        profile.missing_rate = float(np.sum(np.isnan(values.astype(float, errors='ignore'))) / len(values))

        # Generate distribution hash for drift detection
        profile.distribution_hash = self._calculate_distribution_hash(values)

        return profile

    def _calculate_drift_score(self, baseline: DataProfile, current: DataProfile) -> float:
        """Calculate drift score between baseline and current data profiles."""
        # Simplified drift score calculation
        # In practice, this would use more sophisticated statistical tests like KS test

        if baseline.data_type != current.data_type:
            return 1.0  # Maximum drift if data types differ

        if baseline.data_type in ['float64', 'float32', 'int64', 'int32']:
            # Numerical drift detection
            if baseline.mean is None or current.mean is None:
                return 0.0

            # Calculate normalized difference in means and standard deviations
            mean_diff = abs(baseline.mean - current.mean) / (abs(baseline.mean) + 1e-8)
            std_diff = abs(baseline.std - current.std) / (baseline.std + 1e-8)

            return min(1.0, (mean_diff + std_diff) / 2)
        else:
            # Categorical drift detection
            baseline_set = set(baseline.unique_values)
            current_set = set(current.unique_values)

            if not baseline_set or not current_set:
                return 0.0

            # Jaccard distance
            intersection = len(baseline_set.intersection(current_set))
            union = len(baseline_set.union(current_set))

            return 1.0 - (intersection / union) if union > 0 else 0.0

    def _calculate_performance_degradation(
        self,
        baseline: PerformanceMetrics,
        current: PerformanceMetrics
    ) -> float:
        """Calculate performance degradation between baseline and current metrics."""
        degradations = []

        # Compare key metrics
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

        for metric in metrics_to_compare:
            baseline_val = getattr(baseline, metric, None)
            current_val = getattr(current, metric, None)

            if baseline_val is not None and current_val is not None and baseline_val > 0:
                # Calculate relative degradation (negative means improvement)
                degradation = (baseline_val - current_val) / baseline_val
                degradations.append(max(0, degradation))  # Only consider degradation, not improvement

        # For error metrics (lower is better)
        error_metrics = ['mse', 'mae', 'error_rate']
        for metric in error_metrics:
            baseline_val = getattr(baseline, metric, None)
            current_val = getattr(current, metric, None)

            if baseline_val is not None and current_val is not None and baseline_val > 0:
                # For error metrics, increase is degradation
                degradation = (current_val - baseline_val) / baseline_val
                degradations.append(max(0, degradation))

        return max(degradations) if degradations else 0.0

    def _compare_prediction_distributions(
        self,
        historical: np.ndarray,
        current: np.ndarray
    ) -> float:
        """Compare prediction distributions using statistical tests."""
        try:
            from scipy import stats

            # Use Kolmogorov-Smirnov test for distribution comparison
            statistic, p_value = stats.ks_2samp(historical, current)

            # Return 1 - p_value as drift score (higher score = more drift)
            return 1.0 - p_value

        except ImportError:
            # Fallback to simple statistical comparison if scipy not available
            hist_mean, hist_std = np.mean(historical), np.std(historical)
            curr_mean, curr_std = np.mean(current), np.std(current)

            if hist_std == 0 and curr_std == 0:
                return 0.0 if hist_mean == curr_mean else 1.0

            mean_diff = abs(hist_mean - curr_mean) / (abs(hist_mean) + 1e-8)
            std_diff = abs(hist_std - curr_std) / (hist_std + 1e-8)

            return min(1.0, (mean_diff + std_diff) / 2)

    def _calculate_distribution_hash(self, values: np.ndarray) -> str:
        """Calculate hash of data distribution for change detection."""
        # Create histogram and hash its string representation
        if np.issubdtype(values.dtype, np.number):
            hist, _ = np.histogram(values, bins=50)
            hist_str = str(hist.tolist())
        else:
            unique, counts = np.unique(values, return_counts=True)
            hist_str = str(dict(zip(unique, counts, strict=False)))

        return hashlib.md5(hist_str.encode()).hexdigest()

    def _create_drift_alert(
        self,
        model_id: str,
        drift_type: DriftType,
        **details
    ):
        """Create and process a drift alert."""
        alert_id = f"{model_id}_{drift_type.value}_{int(time.time())}"

        # Determine alert level based on drift type and severity
        if drift_type == DriftType.PERFORMANCE_DRIFT:
            level = AlertLevel.CRITICAL if details.get('performance_degradation', 0) > 0.2 else AlertLevel.WARNING
        elif drift_type == DriftType.DATA_DRIFT:
            level = AlertLevel.WARNING if details.get('drift_score', 0) > 0.1 else AlertLevel.INFO
        else:
            level = AlertLevel.INFO

        # Generate message and remediation suggestions
        message, suggestions = self._generate_alert_message_and_suggestions(drift_type, details)

        alert = DriftAlert(
            alert_id=alert_id,
            model_id=model_id,
            drift_type=drift_type,
            level=level,
            message=message,
            details=details,
            detected_at=datetime.now(),
            remediation_suggestions=suggestions
        )

        self.alerts[alert_id] = alert

        # Store in database
        self._store_alert(alert)

        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {str(e)}")

        self.logger.warning(f"Drift alert created: {alert_id} - {message}")

    def _generate_alert_message_and_suggestions(
        self,
        drift_type: DriftType,
        details: dict[str, Any]
    ) -> tuple[str, list[str]]:
        """Generate alert message and remediation suggestions."""
        if drift_type == DriftType.DATA_DRIFT:
            feature = details.get('feature_name', 'unknown')
            score = details.get('drift_score', 0)
            message = f"Data drift detected in feature '{feature}' (score: {score:.3f})"
            suggestions = [
                f"Review recent data changes for feature '{feature}'",
                "Check data collection pipeline for anomalies",
                "Consider retraining model with recent data",
                "Validate data preprocessing steps"
            ]

        elif drift_type == DriftType.PERFORMANCE_DRIFT:
            degradation = details.get('performance_degradation', 0)
            message = f"Performance degradation detected ({degradation:.1%} drop)"
            suggestions = [
                "Investigate recent data quality issues",
                "Consider emergency model rollback",
                "Schedule immediate model retraining",
                "Review prediction pipeline for errors"
            ]

        elif drift_type == DriftType.PREDICTION_DRIFT:
            score = details.get('drift_score', 0)
            message = f"Prediction drift detected (score: {score:.3f})"
            suggestions = [
                "Analyze recent prediction patterns",
                "Check for changes in input data distribution",
                "Consider gradual model update",
                "Monitor user behavior changes"
            ]

        else:
            message = f"Drift detected: {drift_type.value}"
            suggestions = ["Investigate model behavior", "Review recent changes"]

        return message, suggestions

    def _generate_health_recommendations(
        self,
        model_id: str,
        alerts: list[DriftAlert],
        metrics: PerformanceMetrics | None
    ) -> list[str]:
        """Generate health recommendations based on alerts and metrics."""
        recommendations = []

        # Alert-based recommendations
        if alerts:
            critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
            if critical_alerts:
                recommendations.append("URGENT: Address critical alerts immediately")
                recommendations.append("Consider emergency model rollback")

            warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
            if warning_alerts:
                recommendations.append("Schedule model retraining within 24 hours")

        # Performance-based recommendations
        if metrics:
            if metrics.error_rate > 0.05:  # 5% error rate threshold
                recommendations.append("High error rate detected - investigate prediction pipeline")

            if metrics.response_time_p95 > 1000:  # 1 second threshold
                recommendations.append("High response time - optimize model serving")

        # General maintenance recommendations
        if not recommendations:
            recommendations.append("Model health is good - continue regular monitoring")

        return recommendations

    # Database methods

    def _init_database(self):
        """Initialize monitoring database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_profiles (
                        model_id TEXT,
                        feature_name TEXT,
                        profile_data TEXT,
                        created_at TEXT,
                        PRIMARY KEY (model_id, feature_name)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT,
                        metrics_data TEXT,
                        timestamp TEXT
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        model_id TEXT,
                        alert_data TEXT,
                        detected_at TEXT,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT,
                        predictions TEXT,
                        timestamp TEXT
                    )
                """)

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise MonitoringError(f"Database initialization failed: {str(e)}")

    def _save_baseline_profile(
        self,
        model_id: str,
        profiles: dict[str, DataProfile],
        metrics: PerformanceMetrics | None
    ):
        """Save baseline profile to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for feature_name, profile in profiles.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO data_profiles (model_id, feature_name, profile_data, created_at) VALUES (?, ?, ?, ?)",
                        (model_id, feature_name, json.dumps(profile.to_dict()), profile.created_at.isoformat())
                    )

                if metrics:
                    cursor.execute(
                        "INSERT INTO performance_metrics (model_id, metrics_data, timestamp) VALUES (?, ?, ?)",
                        (model_id, json.dumps(metrics.to_dict()), metrics.timestamp.isoformat())
                    )

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to save baseline profile: {str(e)}")

    def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO performance_metrics (model_id, metrics_data, timestamp) VALUES (?, ?, ?)",
                    (metrics.model_id, json.dumps(metrics.to_dict()), metrics.timestamp.isoformat())
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to store performance metrics: {str(e)}")

    def _store_predictions(self, model_id: str, predictions: np.ndarray):
        """Store predictions for drift analysis."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO predictions (model_id, predictions, timestamp) VALUES (?, ?, ?)",
                    (model_id, json.dumps(predictions.tolist()), datetime.now().isoformat())
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to store predictions: {str(e)}")

    def _store_alert(self, alert: DriftAlert):
        """Store alert in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO alerts (alert_id, model_id, alert_data, detected_at) VALUES (?, ?, ?, ?)",
                    (alert.alert_id, alert.model_id, json.dumps(alert.to_dict()), alert.detected_at.isoformat())
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to store alert: {str(e)}")

    def _update_alert_status(self, alert_id: str, acknowledged: bool = False, resolved: bool = False):
        """Update alert status in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE alerts SET acknowledged = ?, resolved = ? WHERE alert_id = ?",
                    (acknowledged, resolved, alert_id)
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to update alert status: {str(e)}")

    def _get_historical_predictions(self, model_id: str, limit: int = 1000) -> np.ndarray | None:
        """Get historical predictions for drift analysis."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT predictions FROM predictions WHERE model_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (model_id, limit)
                )
                rows = cursor.fetchall()

                if not rows:
                    return None

                # Combine all historical predictions
                all_predictions = []
                for row in rows:
                    predictions = json.loads(row[0])
                    all_predictions.extend(predictions)

                return np.array(all_predictions)

        except Exception as e:
            self.logger.error(f"Failed to get historical predictions: {str(e)}")
            return None

    def _get_recent_metrics(self, model_id: str) -> PerformanceMetrics | None:
        """Get most recent performance metrics for a model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT metrics_data FROM performance_metrics WHERE model_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (model_id,)
                )
                row = cursor.fetchone()

                if not row:
                    return None

                metrics_data = json.loads(row[0])
                metrics_data['timestamp'] = datetime.fromisoformat(metrics_data['timestamp'])

                return PerformanceMetrics(**metrics_data)

        except Exception as e:
            self.logger.error(f"Failed to get recent metrics: {str(e)}")
            return None

    def _get_recent_drift_scores(self, model_id: str) -> dict[str, float]:
        """Get recent drift scores for a model."""
        # This would typically calculate drift scores from recent data
        # For now, return empty dict as a placeholder
        return {}

    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while self.monitoring_active:
            try:
                # Perform periodic monitoring tasks
                self._cleanup_old_data()
                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)  # Wait before retrying

    def _cleanup_old_data(self):
        """Clean up old data from database to prevent unbounded growth."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Clean old performance metrics
                cursor.execute(
                    "DELETE FROM performance_metrics WHERE timestamp < ?",
                    (cutoff_date,)
                )

                # Clean old predictions
                cursor.execute(
                    "DELETE FROM predictions WHERE timestamp < ?",
                    (cutoff_date,)
                )

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {str(e)}")
