#!/usr/bin/env python3
"""
Data Analytics and Intelligence Framework for Workspace Qdrant MCP

Implements advanced data analytics for document processing insights, intelligence framework
for pattern recognition and trend analysis, predictive analytics for system optimization,
data visualization and reporting capabilities, and anomaly detection and alerting systems.

Created: 2025-09-25T18:01:09+02:00
"""

import asyncio
import json
import logging
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class AnalyticsLevel(Enum):
    """Analytics processing levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnalyticsMetric:
    """Individual analytics metric"""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]
    level: AnalyticsLevel


@dataclass
class PatternInsight:
    """Pattern recognition insight"""
    pattern_type: str
    confidence: float
    description: str
    data_points: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class PredictionResult:
    """Prediction result with confidence intervals"""
    predicted_value: float
    confidence_interval: Tuple[float, float]
    model_accuracy: float
    prediction_horizon: timedelta
    metadata: Dict[str, Any]


@dataclass
class AnomalyAlert:
    """Anomaly detection alert"""
    severity: AlertSeverity
    description: str
    anomaly_score: float
    affected_metrics: List[str]
    timestamp: datetime
    recommended_actions: List[str]


class AdvancedAnalyticsEngine:
    """Advanced analytics engine with statistical analysis capabilities"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.scaler = StandardScaler()

    def calculate_descriptive_statistics(self, data: List[float], metric_name: str) -> Dict[str, float]:
        """Calculate comprehensive descriptive statistics with edge case handling"""
        if not data:
            return {"error": "empty_dataset", "count": 0}

        # Handle edge cases
        data = [x for x in data if not (math.isnan(x) if isinstance(x, float) else False)]

        if not data:
            return {"error": "no_valid_data", "count": 0}

        try:
            stats = {
                "count": len(data),
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "mode": statistics.mode(data) if len(set(data)) < len(data) else None,
                "std_dev": statistics.stdev(data) if len(data) > 1 else 0.0,
                "variance": statistics.variance(data) if len(data) > 1 else 0.0,
                "min": min(data),
                "max": max(data),
                "range": max(data) - min(data),
                "q1": np.percentile(data, 25),
                "q3": np.percentile(data, 75),
                "iqr": np.percentile(data, 75) - np.percentile(data, 25),
                "skewness": self._calculate_skewness(data),
                "kurtosis": self._calculate_kurtosis(data)
            }

            # Handle division by zero in coefficient of variation
            if stats["mean"] != 0:
                stats["coefficient_of_variation"] = stats["std_dev"] / abs(stats["mean"])
            else:
                stats["coefficient_of_variation"] = float('inf') if stats["std_dev"] > 0 else 0.0

            return stats

        except Exception as e:
            self.logger.error(f"Statistics calculation error for {metric_name}: {e}")
            return {"error": str(e), "count": len(data)}

    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness with edge case handling"""
        if len(data) < 3:
            return 0.0

        try:
            mean_val = statistics.mean(data)
            std_val = statistics.stdev(data)

            if std_val == 0:
                return 0.0

            n = len(data)
            skew = sum(((x - mean_val) / std_val) ** 3 for x in data) / n
            return skew

        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis with edge case handling"""
        if len(data) < 4:
            return 0.0

        try:
            mean_val = statistics.mean(data)
            std_val = statistics.stdev(data)

            if std_val == 0:
                return 0.0

            n = len(data)
            kurt = sum(((x - mean_val) / std_val) ** 4 for x in data) / n - 3
            return kurt

        except (ZeroDivisionError, ValueError):
            return 0.0

    def perform_correlation_analysis(self, datasets: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Perform correlation analysis between multiple datasets"""
        if len(datasets) < 2:
            return {"error": "insufficient_datasets"}

        correlations = {}

        for name1, data1 in datasets.items():
            correlations[name1] = {}
            for name2, data2 in datasets.items():
                if name1 == name2:
                    correlations[name1][name2] = 1.0
                    continue

                try:
                    # Handle different data lengths
                    min_len = min(len(data1), len(data2))
                    if min_len < 2:
                        correlations[name1][name2] = 0.0
                        continue

                    data1_subset = data1[:min_len]
                    data2_subset = data2[:min_len]

                    # Filter out NaN values
                    valid_pairs = [(x, y) for x, y in zip(data1_subset, data2_subset)
                                  if not (math.isnan(x) or math.isnan(y))]

                    if len(valid_pairs) < 2:
                        correlations[name1][name2] = 0.0
                        continue

                    x_vals, y_vals = zip(*valid_pairs)
                    correlation = np.corrcoef(x_vals, y_vals)[0, 1]

                    # Handle NaN correlation (constant data)
                    correlations[name1][name2] = 0.0 if math.isnan(correlation) else correlation

                except Exception as e:
                    self.logger.error(f"Correlation calculation error between {name1} and {name2}: {e}")
                    correlations[name1][name2] = 0.0

        return correlations


class IntelligenceFramework:
    """Intelligence framework for pattern recognition and ML-based insights"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.pattern_models = {}
        self.trained_models = {}

    def detect_patterns(self, data: List[Dict[str, Any]], pattern_types: Optional[List[str]] = None) -> List[PatternInsight]:
        """Detect patterns in data with ML-based insights"""
        if not data:
            return []

        patterns = []
        pattern_types = pattern_types or ["trend", "seasonality", "clustering", "outliers"]

        try:
            # Convert data to DataFrame for analysis
            df = pd.DataFrame(data)

            for pattern_type in pattern_types:
                insight = self._analyze_pattern_type(df, pattern_type)
                if insight:
                    patterns.append(insight)

        except Exception as e:
            self.logger.error(f"Pattern detection error: {e}")

        return patterns

    def _analyze_pattern_type(self, df: pd.DataFrame, pattern_type: str) -> Optional[PatternInsight]:
        """Analyze specific pattern type with edge case handling"""
        try:
            if pattern_type == "trend":
                return self._detect_trend_patterns(df)
            elif pattern_type == "seasonality":
                return self._detect_seasonal_patterns(df)
            elif pattern_type == "clustering":
                return self._detect_clustering_patterns(df)
            elif pattern_type == "outliers":
                return self._detect_outlier_patterns(df)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Pattern analysis error for {pattern_type}: {e}")
            return None

    def _detect_trend_patterns(self, df: pd.DataFrame) -> Optional[PatternInsight]:
        """Detect trend patterns with edge case handling"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return None

        trends = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue

            # Calculate trend using linear regression
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series, 1)
            slope = coeffs[0]

            if abs(slope) > 0.01:  # Significant trend threshold
                direction = "increasing" if slope > 0 else "decreasing"
                trends.append({
                    "column": col,
                    "direction": direction,
                    "slope": slope,
                    "strength": abs(slope)
                })

        if not trends:
            return None

        return PatternInsight(
            pattern_type="trend",
            confidence=min(1.0, sum(t["strength"] for t in trends) / len(trends)),
            description=f"Detected {len(trends)} significant trend patterns",
            data_points=trends,
            recommendations=[f"Monitor {t['column']} {t['direction']} trend" for t in trends[:3]]
        )

    def _detect_seasonal_patterns(self, df: pd.DataFrame) -> Optional[PatternInsight]:
        """Detect seasonal patterns with edge case handling"""
        if 'timestamp' not in df.columns:
            return None

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            seasonal_patterns = []

            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) < 24:  # Need sufficient data
                    continue

                # Simple seasonal detection using autocorrelation
                autocorr = [series.autocorr(lag) for lag in range(1, min(len(series)//2, 24))]
                max_autocorr = max(autocorr) if autocorr else 0

                if max_autocorr > 0.3:  # Seasonal threshold
                    seasonal_patterns.append({
                        "column": col,
                        "autocorr_strength": max_autocorr,
                        "estimated_period": autocorr.index(max_autocorr) + 1
                    })

            if not seasonal_patterns:
                return None

            return PatternInsight(
                pattern_type="seasonality",
                confidence=sum(p["autocorr_strength"] for p in seasonal_patterns) / len(seasonal_patterns),
                description=f"Detected seasonal patterns in {len(seasonal_patterns)} metrics",
                data_points=seasonal_patterns,
                recommendations=["Consider seasonal adjustments in forecasting models"]
            )

        except Exception as e:
            self.logger.error(f"Seasonal pattern detection error: {e}")
            return None

    def _detect_clustering_patterns(self, df: pd.DataFrame) -> Optional[PatternInsight]:
        """Detect clustering patterns with edge case handling"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2 or len(df) < 3:
            return None

        try:
            # Prepare data for clustering
            data_for_clustering = df[numeric_cols].dropna()
            if len(data_for_clustering) < 3:
                return None

            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_for_clustering)

            # Determine optimal number of clusters
            max_clusters = min(5, len(data_for_clustering) - 1)
            if max_clusters < 2:
                return None

            inertias = []
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)

            # Use elbow method to find optimal clusters
            optimal_k = 2  # Default
            if len(inertias) > 1:
                diffs = np.diff(inertias)
                if len(diffs) > 1:
                    elbow_idx = np.argmax(np.diff(diffs)) + 1
                    optimal_k = elbow_idx + 2

            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)

            cluster_info = []
            for i in range(optimal_k):
                cluster_mask = clusters == i
                cluster_size = np.sum(cluster_mask)
                cluster_info.append({
                    "cluster_id": i,
                    "size": int(cluster_size),
                    "percentage": float(cluster_size / len(clusters) * 100)
                })

            return PatternInsight(
                pattern_type="clustering",
                confidence=1.0 - (min(inertias) / max(inertias)) if inertias else 0.5,
                description=f"Identified {optimal_k} distinct data clusters",
                data_points=cluster_info,
                recommendations=[f"Analyze cluster {i} characteristics separately" for i in range(min(3, optimal_k))]
            )

        except Exception as e:
            self.logger.error(f"Clustering pattern detection error: {e}")
            return None

    def _detect_outlier_patterns(self, df: pd.DataFrame) -> Optional[PatternInsight]:
        """Detect outlier patterns with edge case handling"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0 or len(df) < 3:
            return None

        try:
            outlier_info = []

            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) < 3:
                    continue

                # Use IQR method for outlier detection
                q1, q3 = np.percentile(series, [25, 75])
                iqr = q3 - q1

                if iqr == 0:  # Handle constant data
                    continue

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_percentage = len(outliers) / len(series) * 100

                if outlier_percentage > 0:
                    outlier_info.append({
                        "column": col,
                        "outlier_count": len(outliers),
                        "outlier_percentage": outlier_percentage,
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    })

            if not outlier_info:
                return None

            total_outlier_rate = sum(info["outlier_percentage"] for info in outlier_info) / len(outlier_info)

            return PatternInsight(
                pattern_type="outliers",
                confidence=min(1.0, total_outlier_rate / 10.0),  # Scale confidence
                description=f"Detected outliers in {len(outlier_info)} metrics",
                data_points=outlier_info,
                recommendations=["Investigate outlier causes", "Consider outlier treatment strategies"]
            )

        except Exception as e:
            self.logger.error(f"Outlier pattern detection error: {e}")
            return None


class PredictiveAnalyticsSystem:
    """Predictive analytics system for performance and usage forecasting"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.models = {}

    def forecast_time_series(self, data: List[Tuple[datetime, float]],
                           forecast_horizon: timedelta,
                           confidence_level: float = 0.95) -> Optional[PredictionResult]:
        """Forecast time series with confidence intervals and edge case handling"""
        if len(data) < 3:
            return None

        try:
            # Sort data by timestamp
            data = sorted(data, key=lambda x: x[0])
            timestamps, values = zip(*data)

            # Filter out invalid values
            valid_data = [(t, v) for t, v in data if not math.isnan(v)]
            if len(valid_data) < 3:
                return None

            timestamps, values = zip(*valid_data)

            # Convert to numerical format for analysis
            time_deltas = [(t - timestamps[0]).total_seconds() for t in timestamps]

            # Simple linear trend forecasting with seasonal adjustment
            coeffs = np.polyfit(time_deltas, values, min(len(values) - 1, 2))

            # Calculate forecast point
            forecast_seconds = (timestamps[-1] - timestamps[0] + forecast_horizon).total_seconds()
            forecast_seconds_from_start = (timestamps[-1] - timestamps[0]).total_seconds() + forecast_horizon.total_seconds()

            predicted_value = np.polyval(coeffs, forecast_seconds_from_start)

            # Calculate prediction intervals using residuals
            fitted_values = [np.polyval(coeffs, td) for td in time_deltas]
            residuals = [actual - fitted for actual, fitted in zip(values, fitted_values)]
            residual_std = statistics.stdev(residuals) if len(residuals) > 1 else 0.0

            # Confidence interval calculation
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
            margin_error = z_score * residual_std

            confidence_interval = (
                predicted_value - margin_error,
                predicted_value + margin_error
            )

            # Calculate model accuracy using R-squared
            ss_res = sum(r**2 for r in residuals)
            ss_tot = sum((v - statistics.mean(values))**2 for v in values)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            return PredictionResult(
                predicted_value=predicted_value,
                confidence_interval=confidence_interval,
                model_accuracy=max(0.0, r_squared),
                prediction_horizon=forecast_horizon,
                metadata={
                    "data_points": len(valid_data),
                    "trend_coefficient": coeffs[0] if len(coeffs) > 1 else 0.0,
                    "residual_std": residual_std,
                    "confidence_level": confidence_level
                }
            )

        except Exception as e:
            self.logger.error(f"Time series forecasting error: {e}")
            return None

    def predict_resource_usage(self, historical_data: Dict[str, List[float]],
                             prediction_steps: int = 10) -> Dict[str, PredictionResult]:
        """Predict resource usage with edge case handling"""
        predictions = {}

        for resource_name, values in historical_data.items():
            if not values or len(values) < 3:
                continue

            try:
                # Filter out invalid values
                valid_values = [v for v in values if not math.isnan(v) and v >= 0]
                if len(valid_values) < 3:
                    continue

                # Simple exponential smoothing for resource prediction
                alpha = 0.3  # Smoothing parameter
                smoothed = [valid_values[0]]

                for i in range(1, len(valid_values)):
                    smoothed_val = alpha * valid_values[i] + (1 - alpha) * smoothed[i-1]
                    smoothed.append(smoothed_val)

                # Predict next value
                predicted_value = smoothed[-1]

                # Calculate prediction error using recent history
                recent_errors = []
                for i in range(max(1, len(valid_values) - 10), len(valid_values)):
                    if i < len(smoothed):
                        error = abs(valid_values[i] - smoothed[i])
                        recent_errors.append(error)

                prediction_error = statistics.mean(recent_errors) if recent_errors else 0.0

                # Handle edge cases for confidence interval
                if prediction_error == 0:
                    confidence_interval = (predicted_value * 0.95, predicted_value * 1.05)
                else:
                    confidence_interval = (
                        max(0, predicted_value - 2 * prediction_error),
                        predicted_value + 2 * prediction_error
                    )

                # Calculate accuracy based on recent predictions vs actual
                accuracy = max(0.0, 1.0 - (prediction_error / max(predicted_value, 1e-6)))

                predictions[resource_name] = PredictionResult(
                    predicted_value=predicted_value,
                    confidence_interval=confidence_interval,
                    model_accuracy=accuracy,
                    prediction_horizon=timedelta(minutes=prediction_steps),
                    metadata={
                        "method": "exponential_smoothing",
                        "alpha": alpha,
                        "recent_error": prediction_error,
                        "data_points": len(valid_values)
                    }
                )

            except Exception as e:
                self.logger.error(f"Resource prediction error for {resource_name}: {e}")
                continue

        return predictions


class DataVisualizationFramework:
    """Data visualization framework with interactive dashboards"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def create_analytics_dashboard(self, metrics: List[AnalyticsMetric],
                                 patterns: List[PatternInsight],
                                 predictions: Dict[str, PredictionResult]) -> Dict[str, Any]:
        """Create interactive analytics dashboard with edge case handling"""
        try:
            dashboard_data = {
                "metrics_overview": self._create_metrics_overview(metrics),
                "pattern_insights": self._create_pattern_visualization(patterns),
                "prediction_charts": self._create_prediction_visualization(predictions),
                "correlation_heatmap": self._create_correlation_heatmap(metrics),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "total_metrics": len(metrics),
                    "total_patterns": len(patterns),
                    "total_predictions": len(predictions)
                }
            }

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Dashboard creation error: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _create_metrics_overview(self, metrics: List[AnalyticsMetric]) -> Dict[str, Any]:
        """Create metrics overview with edge case handling"""
        if not metrics:
            return {"error": "no_metrics", "charts": []}

        try:
            # Group metrics by name
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric)

            charts = []
            for metric_name, metric_list in metric_groups.items():
                if not metric_list:
                    continue

                values = [m.value for m in metric_list if not math.isnan(m.value)]
                timestamps = [m.timestamp.isoformat() for m in metric_list]

                if not values:
                    continue

                chart_data = {
                    "name": metric_name,
                    "type": "line",
                    "data": {
                        "x": timestamps,
                        "y": values,
                        "mode": "lines+markers"
                    },
                    "stats": {
                        "current": values[-1] if values else 0,
                        "average": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "stable"
                    }
                }
                charts.append(chart_data)

            return {"charts": charts, "total_metrics": len(metrics)}

        except Exception as e:
            self.logger.error(f"Metrics overview creation error: {e}")
            return {"error": str(e), "charts": []}

    def _create_pattern_visualization(self, patterns: List[PatternInsight]) -> Dict[str, Any]:
        """Create pattern visualization with edge case handling"""
        if not patterns:
            return {"error": "no_patterns", "insights": []}

        try:
            insights = []
            for pattern in patterns:
                insight_data = {
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "description": pattern.description,
                    "data_points": pattern.data_points[:10],  # Limit data points
                    "recommendations": pattern.recommendations[:5]  # Limit recommendations
                }
                insights.append(insight_data)

            return {"insights": insights, "total_patterns": len(patterns)}

        except Exception as e:
            self.logger.error(f"Pattern visualization error: {e}")
            return {"error": str(e), "insights": []}

    def _create_prediction_visualization(self, predictions: Dict[str, PredictionResult]) -> Dict[str, Any]:
        """Create prediction visualization with edge case handling"""
        if not predictions:
            return {"error": "no_predictions", "forecasts": []}

        try:
            forecasts = []
            for name, prediction in predictions.items():
                forecast_data = {
                    "metric_name": name,
                    "predicted_value": prediction.predicted_value,
                    "confidence_interval": prediction.confidence_interval,
                    "accuracy": prediction.model_accuracy,
                    "horizon": str(prediction.prediction_horizon),
                    "metadata": prediction.metadata
                }
                forecasts.append(forecast_data)

            return {"forecasts": forecasts, "total_predictions": len(predictions)}

        except Exception as e:
            self.logger.error(f"Prediction visualization error: {e}")
            return {"error": str(e), "forecasts": []}

    def _create_correlation_heatmap(self, metrics: List[AnalyticsMetric]) -> Dict[str, Any]:
        """Create correlation heatmap with edge case handling"""
        if len(metrics) < 2:
            return {"error": "insufficient_metrics", "correlations": {}}

        try:
            # Group metrics by name
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric.value)

            if len(metric_groups) < 2:
                return {"error": "insufficient_metric_types", "correlations": {}}

            # Calculate correlations
            correlations = {}
            metric_names = list(metric_groups.keys())

            for i, name1 in enumerate(metric_names):
                correlations[name1] = {}
                for name2 in metric_names:
                    if name1 == name2:
                        correlations[name1][name2] = 1.0
                    else:
                        values1 = metric_groups[name1]
                        values2 = metric_groups[name2]

                        # Handle different lengths
                        min_len = min(len(values1), len(values2))
                        if min_len < 2:
                            correlations[name1][name2] = 0.0
                            continue

                        values1_subset = values1[:min_len]
                        values2_subset = values2[:min_len]

                        # Filter NaN values
                        valid_pairs = [(v1, v2) for v1, v2 in zip(values1_subset, values2_subset)
                                      if not (math.isnan(v1) or math.isnan(v2))]

                        if len(valid_pairs) < 2:
                            correlations[name1][name2] = 0.0
                        else:
                            x_vals, y_vals = zip(*valid_pairs)
                            corr = np.corrcoef(x_vals, y_vals)[0, 1]
                            correlations[name1][name2] = 0.0 if math.isnan(corr) else corr

            return {"correlations": correlations, "metric_count": len(metric_names)}

        except Exception as e:
            self.logger.error(f"Correlation heatmap creation error: {e}")
            return {"error": str(e), "correlations": {}}


class AnomalyDetectionSystem:
    """Anomaly detection and alerting system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.baseline_stats = {}

    def detect_anomalies(self, metrics: List[AnalyticsMetric],
                        sensitivity: float = 0.1) -> List[AnomalyAlert]:
        """Detect anomalies with configurable sensitivity and edge case handling"""
        if not metrics:
            return []

        alerts = []

        try:
            # Group metrics by name for analysis
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric)

            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) < 5:  # Need minimum data for anomaly detection
                    continue

                anomaly_alerts = self._detect_metric_anomalies(metric_name, metric_list, sensitivity)
                alerts.extend(anomaly_alerts)

            # Sort alerts by severity and anomaly score
            alerts.sort(key=lambda x: (x.severity.value, -x.anomaly_score))

            return alerts[:50]  # Limit to top 50 alerts

        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return []

    def _detect_metric_anomalies(self, metric_name: str, metrics: List[AnalyticsMetric],
                               sensitivity: float) -> List[AnomalyAlert]:
        """Detect anomalies for a specific metric with edge case handling"""
        alerts = []

        try:
            # Extract values and filter invalid data
            values = [m.value for m in metrics if not math.isnan(m.value)]
            if len(values) < 5:
                return []

            # Calculate baseline statistics
            baseline_mean = statistics.mean(values)
            baseline_std = statistics.stdev(values) if len(values) > 1 else 0.0

            # Store baseline for future reference
            self.baseline_stats[metric_name] = {
                "mean": baseline_mean,
                "std": baseline_std,
                "updated": datetime.now()
            }

            # Statistical anomaly detection using z-score
            z_threshold = 2.0 / sensitivity  # Adjust threshold based on sensitivity

            for i, metric in enumerate(metrics):
                if math.isnan(metric.value):
                    continue

                # Calculate z-score
                if baseline_std > 0:
                    z_score = abs(metric.value - baseline_mean) / baseline_std
                else:
                    z_score = 0  # No variation in data

                if z_score > z_threshold:
                    severity = self._calculate_alert_severity(z_score, z_threshold)

                    alert = AnomalyAlert(
                        severity=severity,
                        description=f"Statistical anomaly detected in {metric_name}",
                        anomaly_score=z_score,
                        affected_metrics=[metric_name],
                        timestamp=metric.timestamp,
                        recommended_actions=self._get_anomaly_recommendations(metric_name, z_score)
                    )
                    alerts.append(alert)

            # Use Isolation Forest for additional anomaly detection
            if len(values) >= 10:  # Need sufficient data for ML approach
                isolation_alerts = self._isolation_forest_detection(metric_name, metrics)
                alerts.extend(isolation_alerts)

            return alerts

        except Exception as e:
            self.logger.error(f"Metric anomaly detection error for {metric_name}: {e}")
            return []

    def _isolation_forest_detection(self, metric_name: str,
                                  metrics: List[AnalyticsMetric]) -> List[AnomalyAlert]:
        """Use Isolation Forest for anomaly detection with edge case handling"""
        alerts = []

        try:
            # Prepare data for Isolation Forest
            features = []
            for metric in metrics:
                if not math.isnan(metric.value):
                    # Create feature vector with value and metadata
                    feature_vector = [metric.value]

                    # Add metadata features if available
                    if metric.metadata:
                        feature_vector.extend([
                            len(str(metric.metadata.get('source', ''))),
                            metric.metadata.get('processing_time', 0),
                            metric.level.value == 'advanced'  # Convert to binary
                        ])
                    else:
                        feature_vector.extend([0, 0, 0])  # Default values

                    features.append(feature_vector)

            if len(features) < 10:  # Need minimum samples
                return []

            # Fit Isolation Forest
            features_array = np.array(features)
            outliers = self.isolation_forest.fit_predict(features_array)
            anomaly_scores = self.isolation_forest.score_samples(features_array)

            # Create alerts for detected outliers
            valid_metric_idx = 0
            for i, metric in enumerate(metrics):
                if math.isnan(metric.value):
                    continue

                if valid_metric_idx < len(outliers) and outliers[valid_metric_idx] == -1:
                    # Outlier detected
                    anomaly_score = abs(anomaly_scores[valid_metric_idx])
                    severity = self._calculate_alert_severity(anomaly_score * 10, 2.0)  # Scale score

                    alert = AnomalyAlert(
                        severity=severity,
                        description=f"ML-based anomaly detected in {metric_name}",
                        anomaly_score=anomaly_score,
                        affected_metrics=[metric_name],
                        timestamp=metric.timestamp,
                        recommended_actions=["Investigate metric source", "Check data quality"]
                    )
                    alerts.append(alert)

                valid_metric_idx += 1

            return alerts

        except Exception as e:
            self.logger.error(f"Isolation Forest detection error for {metric_name}: {e}")
            return []

    def _calculate_alert_severity(self, anomaly_score: float, threshold: float) -> AlertSeverity:
        """Calculate alert severity based on anomaly score"""
        if anomaly_score > threshold * 3:
            return AlertSeverity.CRITICAL
        elif anomaly_score > threshold * 2:
            return AlertSeverity.HIGH
        elif anomaly_score > threshold * 1.5:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _get_anomaly_recommendations(self, metric_name: str, anomaly_score: float) -> List[str]:
        """Get recommendations based on anomaly type and score"""
        recommendations = []

        if anomaly_score > 5:
            recommendations.append("Immediate investigation required")
            recommendations.append("Check system health and resources")
        elif anomaly_score > 3:
            recommendations.append("Monitor closely for pattern continuation")
            recommendations.append("Review recent configuration changes")
        else:
            recommendations.append("Log for trend analysis")
            recommendations.append("Consider baseline adjustment if persistent")

        # Metric-specific recommendations
        if "memory" in metric_name.lower():
            recommendations.append("Check for memory leaks or excessive usage")
        elif "cpu" in metric_name.lower():
            recommendations.append("Analyze CPU-intensive processes")
        elif "response" in metric_name.lower():
            recommendations.append("Check network latency and server load")

        return recommendations[:5]  # Limit recommendations


class DataAnalyticsFramework:
    """Main framework coordinating all analytics components"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize all components
        self.analytics_engine = AdvancedAnalyticsEngine(config.get('analytics', {}))
        self.intelligence_framework = IntelligenceFramework(config.get('intelligence', {}))
        self.predictive_system = PredictiveAnalyticsSystem(config.get('predictive', {}))
        self.visualization_framework = DataVisualizationFramework(config.get('visualization', {}))
        self.anomaly_system = AnomalyDetectionSystem(config.get('anomaly', {}))

    async def run_comprehensive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive analysis with all components"""
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "analytics": {},
                "patterns": [],
                "predictions": {},
                "visualizations": {},
                "anomalies": [],
                "summary": {}
            }

            # Extract metrics from data
            metrics = self._extract_metrics(data)

            if metrics:
                # Run analytics engine
                results["analytics"] = await self._run_analytics_analysis(metrics)

                # Run pattern detection
                pattern_data = self._prepare_pattern_data(data)
                results["patterns"] = self.intelligence_framework.detect_patterns(pattern_data)

                # Run predictive analytics
                time_series_data = self._extract_time_series_data(data)
                results["predictions"] = await self._run_predictive_analysis(time_series_data)

                # Create visualizations
                results["visualizations"] = self.visualization_framework.create_analytics_dashboard(
                    metrics, results["patterns"], results["predictions"]
                )

                # Detect anomalies
                results["anomalies"] = self.anomaly_system.detect_anomalies(metrics)

                # Generate summary
                results["summary"] = self._generate_analysis_summary(results)

            return results

        except Exception as e:
            self.logger.error(f"Comprehensive analysis error: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def _run_analytics_analysis(self, metrics: List[AnalyticsMetric]) -> Dict[str, Any]:
        """Run analytics analysis with edge case handling"""
        try:
            # Group metrics by name for statistical analysis
            metric_data = defaultdict(list)
            for metric in metrics:
                if not math.isnan(metric.value):
                    metric_data[metric.name].append(metric.value)

            analytics_results = {}
            for metric_name, values in metric_data.items():
                stats = self.analytics_engine.calculate_descriptive_statistics(values, metric_name)
                analytics_results[metric_name] = stats

            # Run correlation analysis if multiple metrics
            if len(metric_data) > 1:
                correlations = self.analytics_engine.perform_correlation_analysis(dict(metric_data))
                analytics_results["correlations"] = correlations

            return analytics_results

        except Exception as e:
            self.logger.error(f"Analytics analysis error: {e}")
            return {"error": str(e)}

    async def _run_predictive_analysis(self, time_series_data: Dict[str, List[Tuple[datetime, float]]]) -> Dict[str, PredictionResult]:
        """Run predictive analysis with edge case handling"""
        predictions = {}

        try:
            for metric_name, data in time_series_data.items():
                if len(data) >= 3:
                    # Forecast next hour
                    prediction = self.predictive_system.forecast_time_series(
                        data, timedelta(hours=1)
                    )
                    if prediction:
                        predictions[metric_name] = asdict(prediction)

            # Resource usage predictions
            resource_data = {name: [value for _, value in data]
                           for name, data in time_series_data.items()}

            resource_predictions = self.predictive_system.predict_resource_usage(resource_data)
            for name, prediction in resource_predictions.items():
                predictions[f"{name}_resource"] = asdict(prediction)

            return predictions

        except Exception as e:
            self.logger.error(f"Predictive analysis error: {e}")
            return {}

    def _extract_metrics(self, data: Dict[str, Any]) -> List[AnalyticsMetric]:
        """Extract metrics from raw data with edge case handling"""
        metrics = []

        try:
            # Handle different data structures
            if "metrics" in data and isinstance(data["metrics"], list):
                for item in data["metrics"]:
                    metric = self._convert_to_metric(item)
                    if metric:
                        metrics.append(metric)
            elif isinstance(data, dict):
                # Try to extract metrics from dict structure
                for key, value in data.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        metric = AnalyticsMetric(
                            name=key,
                            value=float(value),
                            timestamp=datetime.now(),
                            metadata={},
                            level=AnalyticsLevel.BASIC
                        )
                        metrics.append(metric)

            return metrics

        except Exception as e:
            self.logger.error(f"Metric extraction error: {e}")
            return []

    def _convert_to_metric(self, item: Dict[str, Any]) -> Optional[AnalyticsMetric]:
        """Convert data item to AnalyticsMetric with edge case handling"""
        try:
            name = item.get("name", "unknown")
            value = item.get("value", 0)

            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    return None

            if math.isnan(value):
                return None

            timestamp_str = item.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            level_str = item.get("level", "basic")
            try:
                level = AnalyticsLevel(level_str)
            except ValueError:
                level = AnalyticsLevel.BASIC

            return AnalyticsMetric(
                name=name,
                value=value,
                timestamp=timestamp,
                metadata=item.get("metadata", {}),
                level=level
            )

        except Exception as e:
            self.logger.error(f"Metric conversion error: {e}")
            return None

    def _prepare_pattern_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare data for pattern detection with edge case handling"""
        try:
            if "pattern_data" in data and isinstance(data["pattern_data"], list):
                return data["pattern_data"]

            # Convert metrics data for pattern detection
            pattern_data = []
            if "metrics" in data and isinstance(data["metrics"], list):
                for item in data["metrics"]:
                    if isinstance(item, dict):
                        pattern_data.append(item)

            return pattern_data

        except Exception as e:
            self.logger.error(f"Pattern data preparation error: {e}")
            return []

    def _extract_time_series_data(self, data: Dict[str, Any]) -> Dict[str, List[Tuple[datetime, float]]]:
        """Extract time series data with edge case handling"""
        time_series = {}

        try:
            if "time_series" in data and isinstance(data["time_series"], dict):
                for name, series_data in data["time_series"].items():
                    if isinstance(series_data, list):
                        converted_series = []
                        for point in series_data:
                            if isinstance(point, dict) and "timestamp" in point and "value" in point:
                                try:
                                    timestamp = datetime.fromisoformat(
                                        point["timestamp"].replace('Z', '+00:00')
                                    )
                                    value = float(point["value"])
                                    if not math.isnan(value):
                                        converted_series.append((timestamp, value))
                                except (ValueError, TypeError):
                                    continue

                        if converted_series:
                            time_series[name] = converted_series

            return time_series

        except Exception as e:
            self.logger.error(f"Time series extraction error: {e}")
            return {}

    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary with edge case handling"""
        try:
            summary = {
                "total_metrics": 0,
                "total_patterns": len(results.get("patterns", [])),
                "total_predictions": len(results.get("predictions", {})),
                "total_anomalies": len(results.get("anomalies", [])),
                "key_insights": [],
                "recommendations": [],
                "health_score": 100.0
            }

            # Count metrics
            analytics = results.get("analytics", {})
            if isinstance(analytics, dict):
                summary["total_metrics"] = len([k for k in analytics.keys() if k != "correlations"])

            # Extract key insights
            patterns = results.get("patterns", [])
            for pattern in patterns[:3]:  # Top 3 patterns
                if isinstance(pattern, dict):
                    summary["key_insights"].append(
                        f"{pattern.get('pattern_type', 'Unknown')} pattern detected with "
                        f"{pattern.get('confidence', 0)*100:.1f}% confidence"
                    )

            # Calculate health score based on anomalies
            anomalies = results.get("anomalies", [])
            if anomalies:
                critical_count = sum(1 for a in anomalies if isinstance(a, dict) and
                                   a.get("severity") == "critical")
                high_count = sum(1 for a in anomalies if isinstance(a, dict) and
                               a.get("severity") == "high")

                health_penalty = critical_count * 20 + high_count * 10
                summary["health_score"] = max(0.0, 100.0 - health_penalty)

            # Generate recommendations
            if summary["health_score"] < 80:
                summary["recommendations"].append("Address critical and high-severity anomalies")
            if summary["total_patterns"] > 0:
                summary["recommendations"].append("Leverage detected patterns for optimization")
            if summary["total_predictions"] > 0:
                summary["recommendations"].append("Use predictions for proactive planning")

            return summary

        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return {"error": str(e)}


# Main execution function for testing
async def main():
    """Main function for testing the framework"""
    # Initialize framework
    framework = DataAnalyticsFramework()

    # Sample test data with edge cases
    test_data = {
        "metrics": [
            {
                "name": "cpu_usage",
                "value": 45.2,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"source": "system_monitor"},
                "level": "basic"
            },
            {
                "name": "memory_usage",
                "value": 78.5,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"source": "system_monitor"},
                "level": "intermediate"
            },
            {
                "name": "cpu_usage",
                "value": float('nan'),  # Edge case: NaN value
                "timestamp": datetime.now().isoformat(),
                "metadata": {},
                "level": "basic"
            }
        ],
        "time_series": {
            "response_time": [
                {
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                    "value": 100 + i * 2 + (i % 3) * 10  # Trend with noise
                }
                for i in range(10)
            ]
        },
        "pattern_data": [
            {"cpu": 45.2, "memory": 78.5, "timestamp": datetime.now().isoformat()},
            {"cpu": 47.1, "memory": 82.3, "timestamp": datetime.now().isoformat()},
            {"cpu": 44.8, "memory": 76.9, "timestamp": datetime.now().isoformat()}
        ]
    }

    # Run comprehensive analysis
    results = await framework.run_comprehensive_analysis(test_data)

    print("Analytics Framework Test Results:")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())