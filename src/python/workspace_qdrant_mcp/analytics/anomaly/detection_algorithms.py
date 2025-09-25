"""
Anomaly Detection Algorithms for identifying outliers and unusual patterns.

Provides advanced anomaly detection capabilities including:
- Statistical anomaly detection (Z-score, IQR, Isolation Forest)
- Time series anomaly detection with seasonal awareness
- Real-time streaming anomaly detection
- False positive/negative threshold optimization
- Multi-dimensional anomaly detection
- Contextual and collective anomaly detection
"""

import math
import statistics
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

import numpy as np


logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Enumeration for anomaly types."""
    POINT = "point"  # Individual data points
    CONTEXTUAL = "contextual"  # Anomalous in specific context
    COLLECTIVE = "collective"  # Pattern of points
    SEASONAL = "seasonal"  # Seasonal deviation
    TREND = "trend"  # Trend deviation


class DetectionMethod(Enum):
    """Enumeration for detection methods."""
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    SEASONAL_HYBRID = "seasonal_hybrid"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"


class AnomalySeverity(Enum):
    """Enumeration for anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""

    anomaly_indices: List[int]
    anomaly_values: List[float]
    anomaly_scores: List[float]
    anomaly_types: List[AnomalyType]
    severity_levels: List[AnomalySeverity]
    method_used: DetectionMethod
    threshold_used: float
    confidence_scores: List[float]
    detection_timestamps: List[datetime]
    context_info: Dict[str, Any]
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    total_anomalies: int = 0
    anomaly_percentage: float = 0.0
    baseline_statistics: Optional[Dict[str, float]] = None
    is_reliable: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        """Calculate derived fields."""
        self.total_anomalies = len(self.anomaly_indices)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'anomaly_indices': self.anomaly_indices,
            'anomaly_values': self.anomaly_values,
            'anomaly_scores': self.anomaly_scores,
            'anomaly_types': [t.value for t in self.anomaly_types],
            'severity_levels': [s.value for s in self.severity_levels],
            'method_used': self.method_used.value,
            'threshold_used': self.threshold_used,
            'confidence_scores': self.confidence_scores,
            'detection_timestamps': [dt.isoformat() for dt in self.detection_timestamps],
            'context_info': self.context_info,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'total_anomalies': self.total_anomalies,
            'anomaly_percentage': self.anomaly_percentage,
            'baseline_statistics': self.baseline_statistics,
            'is_reliable': self.is_reliable,
            'error_message': self.error_message
        }


@dataclass
class StreamingAnomalyState:
    """State container for streaming anomaly detection."""

    window_data: List[float]
    baseline_mean: float
    baseline_std: float
    baseline_median: float
    baseline_mad: float
    seasonal_components: Optional[List[float]]
    trend_component: Optional[List[float]]
    last_update: datetime
    adaptation_rate: float
    detection_count: int
    false_alarm_count: int

    def update_baseline(self, new_values: List[float]):
        """Update baseline statistics with new values."""
        try:
            # Exponential decay for adaptation
            decay_factor = 1.0 - self.adaptation_rate

            if new_values:
                new_mean = statistics.mean(new_values)
                new_std = statistics.stdev(new_values) if len(new_values) > 1 else 0.0
                new_median = statistics.median(new_values)
                new_mad = statistics.median([abs(x - new_median) for x in new_values])

                self.baseline_mean = decay_factor * self.baseline_mean + self.adaptation_rate * new_mean
                self.baseline_std = decay_factor * self.baseline_std + self.adaptation_rate * new_std
                self.baseline_median = decay_factor * self.baseline_median + self.adaptation_rate * new_median
                self.baseline_mad = decay_factor * self.baseline_mad + self.adaptation_rate * new_mad

            self.last_update = datetime.now()

        except Exception as e:
            logger.warning(f"Error updating baseline statistics: {e}")


class AnomalyDetector:
    """
    Advanced anomaly detection engine with comprehensive algorithm support.

    Handles false positives/negatives, threshold boundary conditions, and provides
    robust anomaly detection for various data patterns and use cases.
    """

    def __init__(self, default_method: DetectionMethod = DetectionMethod.Z_SCORE,
                 default_threshold: float = 3.0, min_data_points: int = 10):
        """
        Initialize anomaly detector.

        Args:
            default_method: Default detection method to use
            default_threshold: Default threshold for anomaly detection
            min_data_points: Minimum data points required for detection
        """
        self.default_method = default_method
        self.default_threshold = default_threshold
        self.min_data_points = min_data_points
        self._streaming_states = {}
        self._detection_history = {}
        self._threshold_cache = {}

    def detect_anomalies(self,
                        data: List[Union[int, float]],
                        method: Optional[DetectionMethod] = None,
                        threshold: Optional[float] = None,
                        seasonal_period: Optional[int] = None,
                        context_data: Optional[Dict[str, Any]] = None) -> AnomalyResult:
        """
        Detect anomalies in the provided data.

        Args:
            data: Input data for anomaly detection
            method: Detection method to use
            threshold: Detection threshold
            seasonal_period: Seasonal period for seasonal methods
            context_data: Additional context information

        Returns:
            AnomalyResult with detected anomalies and metadata
        """
        # Use defaults if not provided
        method = method or self.default_method
        threshold = threshold or self.default_threshold
        context_data = context_data or {}

        result = AnomalyResult(
            anomaly_indices=[],
            anomaly_values=[],
            anomaly_scores=[],
            anomaly_types=[],
            severity_levels=[],
            method_used=method,
            threshold_used=threshold,
            confidence_scores=[],
            detection_timestamps=[],
            context_info=context_data
        )

        try:
            # Validate input data
            if not data:
                result.error_message = "Empty dataset provided"
                result.is_reliable = False
                return result

            if len(data) < self.min_data_points:
                result.error_message = f"Insufficient data points (minimum {self.min_data_points} required)"
                result.is_reliable = False
                return result

            # Filter and clean data
            valid_data = []
            valid_indices = []

            for i, value in enumerate(data):
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid_data.append(float(value))
                    valid_indices.append(i)

            if len(valid_data) < self.min_data_points:
                result.error_message = "Insufficient valid data points after filtering"
                result.is_reliable = False
                return result

            # Calculate baseline statistics
            result.baseline_statistics = self._calculate_baseline_statistics(valid_data)

            # Apply selected detection method
            if method == DetectionMethod.Z_SCORE:
                anomalies = self._detect_z_score_anomalies(valid_data, threshold)
            elif method == DetectionMethod.MODIFIED_Z_SCORE:
                anomalies = self._detect_modified_z_score_anomalies(valid_data, threshold)
            elif method == DetectionMethod.IQR:
                anomalies = self._detect_iqr_anomalies(valid_data, threshold)
            elif method == DetectionMethod.ISOLATION_FOREST:
                anomalies = self._detect_isolation_forest_anomalies(valid_data, threshold)
            elif method == DetectionMethod.LOCAL_OUTLIER_FACTOR:
                anomalies = self._detect_lof_anomalies(valid_data, threshold)
            elif method == DetectionMethod.SEASONAL_HYBRID:
                anomalies = self._detect_seasonal_anomalies(valid_data, seasonal_period, threshold)
            elif method == DetectionMethod.MOVING_AVERAGE:
                anomalies = self._detect_moving_average_anomalies(valid_data, threshold)
            elif method == DetectionMethod.EXPONENTIAL_SMOOTHING:
                anomalies = self._detect_exponential_smoothing_anomalies(valid_data, threshold)
            else:
                result.error_message = f"Unsupported detection method: {method}"
                result.is_reliable = False
                return result

            if not anomalies:
                # No anomalies detected
                result.anomaly_percentage = 0.0
                return result

            # Map back to original indices
            anomaly_indices = [valid_indices[idx] for idx, _, _, _, _ in anomalies if idx < len(valid_indices)]
            anomaly_values = [value for _, value, _, _, _ in anomalies]
            anomaly_scores = [score for _, _, score, _, _ in anomalies]
            anomaly_types = [atype for _, _, _, atype, _ in anomalies]
            confidence_scores = [conf for _, _, _, _, conf in anomalies]

            # Assign severity levels
            severity_levels = self._assign_severity_levels(anomaly_scores, method)

            # Generate timestamps
            detection_timestamps = [datetime.now()] * len(anomaly_indices)

            # Update result
            result.anomaly_indices = anomaly_indices
            result.anomaly_values = anomaly_values
            result.anomaly_scores = anomaly_scores
            result.anomaly_types = anomaly_types
            result.severity_levels = severity_levels
            result.confidence_scores = confidence_scores
            result.detection_timestamps = detection_timestamps
            result.total_anomalies = len(anomaly_indices)
            result.anomaly_percentage = (len(anomaly_indices) / len(valid_data)) * 100

            # Estimate false positive rate
            result.false_positive_rate = self._estimate_false_positive_rate(
                result, valid_data, method, threshold)

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            result.error_message = f"Anomaly detection error: {str(e)}"
            result.is_reliable = False

        return result

    def detect_streaming_anomalies(self,
                                  new_data: List[Union[int, float]],
                                  stream_id: str,
                                  window_size: int = 100,
                                  adaptation_rate: float = 0.1) -> AnomalyResult:
        """
        Detect anomalies in streaming data with adaptive baselines.

        Args:
            new_data: New data points to analyze
            stream_id: Unique identifier for the data stream
            window_size: Size of sliding window for baseline calculation
            adaptation_rate: Rate of baseline adaptation (0.0 to 1.0)

        Returns:
            AnomalyResult for the new data points
        """
        try:
            # Initialize streaming state if not exists
            if stream_id not in self._streaming_states:
                if len(new_data) < self.min_data_points:
                    # Need more data to initialize
                    result = AnomalyResult(
                        anomaly_indices=[], anomaly_values=[], anomaly_scores=[],
                        anomaly_types=[], severity_levels=[], method_used=self.default_method,
                        threshold_used=self.default_threshold, confidence_scores=[],
                        detection_timestamps=[], context_info={}
                    )
                    result.error_message = "Insufficient data to initialize streaming detection"
                    return result

                # Initialize with first batch of data
                valid_data = [float(x) for x in new_data if isinstance(x, (int, float)) and math.isfinite(x)]

                if len(valid_data) < self.min_data_points:
                    result = AnomalyResult(
                        anomaly_indices=[], anomaly_values=[], anomaly_scores=[],
                        anomaly_types=[], severity_levels=[], method_used=self.default_method,
                        threshold_used=self.default_threshold, confidence_scores=[],
                        detection_timestamps=[], context_info={}
                    )
                    result.error_message = "Insufficient valid data to initialize streaming detection"
                    return result

                self._streaming_states[stream_id] = StreamingAnomalyState(
                    window_data=valid_data[-window_size:],
                    baseline_mean=statistics.mean(valid_data),
                    baseline_std=statistics.stdev(valid_data) if len(valid_data) > 1 else 0.0,
                    baseline_median=statistics.median(valid_data),
                    baseline_mad=statistics.median([abs(x - statistics.median(valid_data)) for x in valid_data]),
                    seasonal_components=None,
                    trend_component=None,
                    last_update=datetime.now(),
                    adaptation_rate=adaptation_rate,
                    detection_count=0,
                    false_alarm_count=0
                )

            state = self._streaming_states[stream_id]

            # Filter new data
            valid_new_data = [float(x) for x in new_data if isinstance(x, (int, float)) and math.isfinite(x)]

            if not valid_new_data:
                result = AnomalyResult(
                    anomaly_indices=[], anomaly_values=[], anomaly_scores=[],
                    anomaly_types=[], severity_levels=[], method_used=self.default_method,
                    threshold_used=self.default_threshold, confidence_scores=[],
                    detection_timestamps=[], context_info={}
                )
                return result

            # Detect anomalies using current baseline
            anomalies = []
            for i, value in enumerate(valid_new_data):
                # Z-score based detection using streaming baseline
                if state.baseline_std > 0:
                    z_score = abs(value - state.baseline_mean) / state.baseline_std
                    if z_score > self.default_threshold:
                        anomalies.append((
                            i, value, z_score, AnomalyType.POINT,
                            min(1.0, z_score / (self.default_threshold * 2))
                        ))

                # Modified Z-score using MAD
                if state.baseline_mad > 0:
                    modified_z = 0.6745 * (value - state.baseline_median) / state.baseline_mad
                    if abs(modified_z) > self.default_threshold:
                        if not any(idx == i for idx, _, _, _, _ in anomalies):  # Avoid duplicates
                            anomalies.append((
                                i, value, abs(modified_z), AnomalyType.POINT,
                                min(1.0, abs(modified_z) / (self.default_threshold * 2))
                            ))

            # Update sliding window and baseline
            state.window_data.extend(valid_new_data)
            state.window_data = state.window_data[-window_size:]  # Keep only recent data

            # Update baseline statistics
            normal_data = [x for i, x in enumerate(valid_new_data)
                          if not any(idx == i for idx, _, _, _, _ in anomalies)]
            if normal_data:
                state.update_baseline(normal_data)

            # Update detection counts
            state.detection_count += len(anomalies)

            # Create result
            result = AnomalyResult(
                anomaly_indices=[idx for idx, _, _, _, _ in anomalies],
                anomaly_values=[val for _, val, _, _, _ in anomalies],
                anomaly_scores=[score for _, _, score, _, _ in anomalies],
                anomaly_types=[atype for _, _, _, atype, _ in anomalies],
                severity_levels=self._assign_severity_levels(
                    [score for _, _, score, _, _ in anomalies], DetectionMethod.Z_SCORE),
                method_used=DetectionMethod.Z_SCORE,  # Streaming uses Z-score variant
                threshold_used=self.default_threshold,
                confidence_scores=[conf for _, _, _, _, conf in anomalies],
                detection_timestamps=[datetime.now()] * len(anomalies),
                context_info={'stream_id': stream_id, 'window_size': window_size},
                baseline_statistics={
                    'mean': state.baseline_mean,
                    'std': state.baseline_std,
                    'median': state.baseline_median,
                    'mad': state.baseline_mad
                }
            )

            result.anomaly_percentage = (len(anomalies) / len(valid_new_data)) * 100 if valid_new_data else 0.0

            return result

        except Exception as e:
            logger.error(f"Error in streaming anomaly detection: {e}")
            result = AnomalyResult(
                anomaly_indices=[], anomaly_values=[], anomaly_scores=[],
                anomaly_types=[], severity_levels=[], method_used=self.default_method,
                threshold_used=self.default_threshold, confidence_scores=[],
                detection_timestamps=[], context_info={}
            )
            result.error_message = f"Streaming anomaly detection error: {str(e)}"
            result.is_reliable = False
            return result

    def optimize_threshold(self,
                         training_data: List[Union[int, float]],
                         known_anomalies: List[int],
                         method: Optional[DetectionMethod] = None,
                         target_fpr: float = 0.05) -> float:
        """
        Optimize detection threshold to minimize false positives while maintaining sensitivity.

        Args:
            training_data: Training data with known anomalies
            known_anomalies: Indices of known anomalies in training data
            method: Detection method to optimize for
            target_fpr: Target false positive rate

        Returns:
            Optimized threshold value
        """
        method = method or self.default_method

        try:
            if not training_data or not known_anomalies:
                return self.default_threshold

            valid_data = [float(x) for x in training_data if isinstance(x, (int, float)) and math.isfinite(x)]

            if len(valid_data) < self.min_data_points:
                return self.default_threshold

            # Test different threshold values
            threshold_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            best_threshold = self.default_threshold
            best_score = float('inf')

            known_anomaly_set = set(known_anomalies)

            for threshold in threshold_range:
                # Detect anomalies with this threshold
                result = self.detect_anomalies(valid_data, method=method, threshold=threshold)

                if not result.is_reliable:
                    continue

                detected_anomalies = set(result.anomaly_indices)

                # Calculate precision, recall, and F1 score
                true_positives = len(detected_anomalies & known_anomaly_set)
                false_positives = len(detected_anomalies - known_anomaly_set)
                false_negatives = len(known_anomaly_set - detected_anomalies)

                if true_positives + false_positives > 0:
                    precision = true_positives / (true_positives + false_positives)
                else:
                    precision = 0.0

                if true_positives + false_negatives > 0:
                    recall = true_positives / (true_positives + false_negatives)
                else:
                    recall = 0.0

                # False positive rate
                total_normal = len(valid_data) - len(known_anomaly_set)
                if total_normal > 0:
                    fpr = false_positives / total_normal
                else:
                    fpr = 0.0

                # Calculate combined score (prioritize low FPR and high recall)
                if precision > 0 and recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0.0

                # Penalize high false positive rates
                fpr_penalty = max(0, fpr - target_fpr) * 10
                combined_score = -f1_score + fpr_penalty  # Lower is better

                if combined_score < best_score:
                    best_score = combined_score
                    best_threshold = threshold

            # Cache the optimized threshold
            cache_key = f"{method.value}_{hash(tuple(valid_data))}"
            self._threshold_cache[cache_key] = best_threshold

            return best_threshold

        except Exception as e:
            logger.error(f"Error optimizing threshold: {e}")
            return self.default_threshold

    def _calculate_baseline_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate baseline statistics for the dataset."""
        try:
            stats = {
                'mean': statistics.mean(data),
                'median': statistics.median(data),
                'min': min(data),
                'max': max(data),
                'count': len(data)
            }

            if len(data) > 1:
                stats['std'] = statistics.stdev(data)
                stats['variance'] = statistics.variance(data)
            else:
                stats['std'] = 0.0
                stats['variance'] = 0.0

            # Calculate MAD (Median Absolute Deviation)
            median_val = stats['median']
            stats['mad'] = statistics.median([abs(x - median_val) for x in data])

            # Calculate quantiles
            try:
                quantiles = statistics.quantiles(data, n=4)
                stats['q1'] = quantiles[0]
                stats['q3'] = quantiles[2]
                stats['iqr'] = quantiles[2] - quantiles[0]
            except Exception:
                stats['q1'] = stats['median']
                stats['q3'] = stats['median']
                stats['iqr'] = 0.0

            return stats

        except Exception as e:
            logger.warning(f"Error calculating baseline statistics: {e}")
            return {
                'mean': 0.0, 'median': 0.0, 'std': 0.0, 'variance': 0.0,
                'min': 0.0, 'max': 0.0, 'mad': 0.0, 'q1': 0.0, 'q3': 0.0,
                'iqr': 0.0, 'count': len(data)
            }

    def _detect_z_score_anomalies(self, data: List[float], threshold: float) -> List[Tuple]:
        """Detect anomalies using Z-score method."""
        try:
            if len(data) < 2:
                return []

            mean_val = statistics.mean(data)
            std_val = statistics.stdev(data)

            if std_val == 0:
                return []  # No variance, no outliers

            anomalies = []
            for i, value in enumerate(data):
                z_score = abs(value - mean_val) / std_val
                if z_score > threshold:
                    confidence = min(1.0, z_score / (threshold * 2))
                    anomalies.append((i, value, z_score, AnomalyType.POINT, confidence))

            return anomalies

        except Exception as e:
            logger.warning(f"Error in Z-score anomaly detection: {e}")
            return []

    def _detect_modified_z_score_anomalies(self, data: List[float], threshold: float) -> List[Tuple]:
        """Detect anomalies using Modified Z-score method (using median and MAD)."""
        try:
            median_val = statistics.median(data)
            mad = statistics.median([abs(x - median_val) for x in data])

            if mad == 0:
                return []  # No variation

            anomalies = []
            for i, value in enumerate(data):
                modified_z_score = 0.6745 * (value - median_val) / mad
                if abs(modified_z_score) > threshold:
                    confidence = min(1.0, abs(modified_z_score) / (threshold * 2))
                    anomalies.append((i, value, abs(modified_z_score), AnomalyType.POINT, confidence))

            return anomalies

        except Exception as e:
            logger.warning(f"Error in Modified Z-score anomaly detection: {e}")
            return []

    def _detect_iqr_anomalies(self, data: List[float], threshold: float) -> List[Tuple]:
        """Detect anomalies using Interquartile Range method."""
        try:
            if len(data) < 4:
                return []

            q1 = statistics.quantiles(data, n=4)[0]
            q3 = statistics.quantiles(data, n=4)[2]
            iqr = q3 - q1

            if iqr == 0:
                return []  # No interquartile range

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            anomalies = []
            for i, value in enumerate(data):
                if value < lower_bound or value > upper_bound:
                    # Calculate how far outside the bounds
                    if value < lower_bound:
                        distance = (lower_bound - value) / iqr
                    else:
                        distance = (value - upper_bound) / iqr

                    confidence = min(1.0, distance / threshold)
                    anomalies.append((i, value, distance, AnomalyType.POINT, confidence))

            return anomalies

        except Exception as e:
            logger.warning(f"Error in IQR anomaly detection: {e}")
            return []

    def _detect_isolation_forest_anomalies(self, data: List[float], threshold: float) -> List[Tuple]:
        """Detect anomalies using simplified Isolation Forest approach."""
        try:
            # Simplified implementation without sklearn
            # Use random sampling to create isolation "trees"

            if len(data) < 10:
                return []

            n_trees = 10
            anomaly_scores = [0.0] * len(data)

            for _ in range(n_trees):
                # Create a random partition tree
                indices = list(range(len(data)))
                depths = self._calculate_isolation_depths(data, indices, max_depth=10)

                for i, depth in enumerate(depths):
                    # Lower depth = more isolated = higher anomaly score
                    expected_depth = math.log2(len(data))
                    isolation_score = expected_depth - depth if expected_depth > depth else 0
                    anomaly_scores[i] += isolation_score

            # Normalize scores
            max_score = max(anomaly_scores) if anomaly_scores else 1.0
            if max_score > 0:
                anomaly_scores = [score / max_score for score in anomaly_scores]

            # Convert threshold to percentile
            threshold_percentile = 1.0 - (threshold / 10.0)  # Scale threshold
            sorted_scores = sorted(anomaly_scores, reverse=True)
            if len(sorted_scores) > 1:
                cutoff_index = int(threshold_percentile * len(sorted_scores))
                score_threshold = sorted_scores[min(cutoff_index, len(sorted_scores) - 1)]
            else:
                score_threshold = 0.5

            anomalies = []
            for i, (value, score) in enumerate(zip(data, anomaly_scores)):
                if score > score_threshold:
                    confidence = min(1.0, score / (score_threshold * 2))
                    anomalies.append((i, value, score, AnomalyType.POINT, confidence))

            return anomalies

        except Exception as e:
            logger.warning(f"Error in Isolation Forest anomaly detection: {e}")
            return []

    def _calculate_isolation_depths(self, data: List[float], indices: List[int], max_depth: int) -> List[int]:
        """Calculate isolation depths for simplified isolation forest."""
        depths = [0] * len(indices)

        def isolate(current_indices, current_depth):
            if len(current_indices) <= 1 or current_depth >= max_depth:
                for idx in current_indices:
                    depths[idx] = current_depth
                return

            # Random split
            values = [data[i] for i in current_indices]
            min_val, max_val = min(values), max(values)
            if min_val == max_val:
                for idx in current_indices:
                    depths[idx] = current_depth
                return

            import random
            split_point = random.uniform(min_val, max_val)

            left_indices = [i for i in current_indices if data[i] < split_point]
            right_indices = [i for i in current_indices if data[i] >= split_point]

            if not left_indices:
                left_indices = current_indices[:1]
                right_indices = current_indices[1:]
            elif not right_indices:
                right_indices = current_indices[-1:]
                left_indices = current_indices[:-1]

            isolate(left_indices, current_depth + 1)
            isolate(right_indices, current_depth + 1)

        isolate(indices, 0)
        return depths

    def _detect_lof_anomalies(self, data: List[float], threshold: float) -> List[Tuple]:
        """Detect anomalies using simplified Local Outlier Factor approach."""
        try:
            # Simplified LOF implementation
            if len(data) < 5:
                return []

            k = min(5, len(data) - 1)  # Number of neighbors
            lof_scores = []

            for i, point in enumerate(data):
                # Calculate k-distance and k-nearest neighbors
                distances = sorted([(abs(point - other), j) for j, other in enumerate(data) if j != i])
                k_nearest = distances[:k]

                if not k_nearest:
                    lof_scores.append(1.0)
                    continue

                k_distance = k_nearest[-1][0] if k_nearest else 0.0

                # Calculate local reachability density
                reachability_distances = []
                for dist, neighbor_idx in k_nearest:
                    # Calculate reachability distance
                    neighbor = data[neighbor_idx]
                    neighbor_distances = sorted([abs(neighbor - other) for j, other in enumerate(data) if j != neighbor_idx])
                    neighbor_k_dist = neighbor_distances[min(k-1, len(neighbor_distances)-1)] if neighbor_distances else 0.0

                    reachability_dist = max(dist, neighbor_k_dist)
                    reachability_distances.append(reachability_dist)

                avg_reachability = statistics.mean(reachability_distances) if reachability_distances else 1.0
                lrd = 1.0 / (avg_reachability + 1e-10)  # Local reachability density

                # Calculate LOF
                neighbor_lrds = []
                for _, neighbor_idx in k_nearest:
                    # Simplified neighbor LRD calculation
                    neighbor_lrds.append(lrd)  # Approximation

                if neighbor_lrds:
                    avg_neighbor_lrd = statistics.mean(neighbor_lrds)
                    lof = avg_neighbor_lrd / (lrd + 1e-10)
                else:
                    lof = 1.0

                lof_scores.append(lof)

            # Detect anomalies based on LOF scores
            threshold_value = 1.0 + threshold / 3.0  # Scale threshold for LOF
            anomalies = []

            for i, (value, lof_score) in enumerate(zip(data, lof_scores)):
                if lof_score > threshold_value:
                    confidence = min(1.0, (lof_score - 1.0) / threshold)
                    anomalies.append((i, value, lof_score, AnomalyType.CONTEXTUAL, confidence))

            return anomalies

        except Exception as e:
            logger.warning(f"Error in LOF anomaly detection: {e}")
            return []

    def _detect_seasonal_anomalies(self, data: List[float], seasonal_period: Optional[int],
                                  threshold: float) -> List[Tuple]:
        """Detect anomalies with seasonal awareness."""
        try:
            if not seasonal_period or seasonal_period <= 0:
                seasonal_period = min(12, len(data) // 4) if len(data) >= 12 else len(data) // 2

            if len(data) < seasonal_period * 2:
                # Fall back to regular Z-score if insufficient data for seasonal analysis
                return self._detect_z_score_anomalies(data, threshold)

            anomalies = []

            # Extract seasonal components and detect anomalies
            for season_idx in range(seasonal_period):
                # Get values for this seasonal position
                seasonal_values = []
                seasonal_indices = []

                for i in range(season_idx, len(data), seasonal_period):
                    seasonal_values.append(data[i])
                    seasonal_indices.append(i)

                if len(seasonal_values) < 2:
                    continue

                # Detect anomalies within this seasonal component
                seasonal_mean = statistics.mean(seasonal_values)
                seasonal_std = statistics.stdev(seasonal_values) if len(seasonal_values) > 1 else 0.0

                if seasonal_std > 0:
                    for idx, (orig_idx, value) in enumerate(zip(seasonal_indices, seasonal_values)):
                        z_score = abs(value - seasonal_mean) / seasonal_std
                        if z_score > threshold:
                            confidence = min(1.0, z_score / (threshold * 2))
                            anomalies.append((orig_idx, value, z_score, AnomalyType.SEASONAL, confidence))

            return anomalies

        except Exception as e:
            logger.warning(f"Error in seasonal anomaly detection: {e}")
            return self._detect_z_score_anomalies(data, threshold)  # Fallback

    def _detect_moving_average_anomalies(self, data: List[float], threshold: float) -> List[Tuple]:
        """Detect anomalies using moving average method."""
        try:
            window_size = min(10, len(data) // 4) if len(data) >= 10 else 3

            if len(data) < window_size:
                return []

            anomalies = []

            for i in range(window_size, len(data)):
                # Calculate moving average for the window
                window = data[i-window_size:i]
                moving_avg = statistics.mean(window)
                window_std = statistics.stdev(window) if len(window) > 1 else 0.0

                if window_std > 0:
                    # Check if current point is anomalous
                    z_score = abs(data[i] - moving_avg) / window_std
                    if z_score > threshold:
                        confidence = min(1.0, z_score / (threshold * 2))
                        anomalies.append((i, data[i], z_score, AnomalyType.POINT, confidence))

            return anomalies

        except Exception as e:
            logger.warning(f"Error in moving average anomaly detection: {e}")
            return []

    def _detect_exponential_smoothing_anomalies(self, data: List[float], threshold: float) -> List[Tuple]:
        """Detect anomalies using exponential smoothing method."""
        try:
            if len(data) < 3:
                return []

            alpha = 0.3  # Smoothing parameter
            anomalies = []

            # Initialize exponential smoothing
            smoothed = data[0]
            errors = []

            for i in range(1, len(data)):
                # Calculate prediction error
                prediction_error = abs(data[i] - smoothed)
                errors.append(prediction_error)

                # Update smoothed value
                smoothed = alpha * data[i] + (1 - alpha) * smoothed

                # Check for anomaly after we have enough error history
                if len(errors) >= 3:
                    error_mean = statistics.mean(errors[-10:])  # Use recent errors
                    error_std = statistics.stdev(errors[-10:]) if len(errors[-10:]) > 1 else 0.0

                    if error_std > 0:
                        error_z_score = prediction_error / (error_std + 1e-10)
                        if error_z_score > threshold:
                            confidence = min(1.0, error_z_score / (threshold * 2))
                            anomalies.append((i, data[i], error_z_score, AnomalyType.POINT, confidence))

            return anomalies

        except Exception as e:
            logger.warning(f"Error in exponential smoothing anomaly detection: {e}")
            return []

    def _assign_severity_levels(self, anomaly_scores: List[float], method: DetectionMethod) -> List[AnomalySeverity]:
        """Assign severity levels based on anomaly scores."""
        if not anomaly_scores:
            return []

        severity_levels = []

        try:
            # Define thresholds based on method
            if method in [DetectionMethod.Z_SCORE, DetectionMethod.MODIFIED_Z_SCORE]:
                # For Z-score methods
                low_threshold = 2.0
                medium_threshold = 3.0
                high_threshold = 4.0
            elif method == DetectionMethod.IQR:
                # For IQR method
                low_threshold = 1.5
                medium_threshold = 2.0
                high_threshold = 3.0
            else:
                # For other methods, use normalized thresholds
                low_threshold = 0.3
                medium_threshold = 0.6
                high_threshold = 0.8

            for score in anomaly_scores:
                if score >= high_threshold:
                    severity_levels.append(AnomalySeverity.CRITICAL)
                elif score >= medium_threshold:
                    severity_levels.append(AnomalySeverity.HIGH)
                elif score >= low_threshold:
                    severity_levels.append(AnomalySeverity.MEDIUM)
                else:
                    severity_levels.append(AnomalySeverity.LOW)

        except Exception as e:
            logger.warning(f"Error assigning severity levels: {e}")
            # Default to medium severity for all
            severity_levels = [AnomalySeverity.MEDIUM] * len(anomaly_scores)

        return severity_levels

    def _estimate_false_positive_rate(self, result: AnomalyResult, data: List[float],
                                     method: DetectionMethod, threshold: float) -> Optional[float]:
        """Estimate false positive rate based on statistical properties."""
        try:
            if method == DetectionMethod.Z_SCORE:
                # For normal distribution, calculate expected false positive rate
                # P(|Z| > threshold) for standard normal distribution
                if threshold > 0:
                    # Approximation for false positive rate
                    fpr = 2 * (1 - self._normal_cdf(threshold))
                    return min(1.0, fpr)

            elif method == DetectionMethod.IQR:
                # For IQR method, estimate based on normal distribution assumption
                if threshold == 1.5:
                    return 0.007  # Approximately 0.7% for 1.5*IQR
                else:
                    # Scale proportionally
                    base_fpr = 0.007
                    return base_fpr * (1.5 / threshold) ** 2

            # For other methods, return None (unknown)
            return None

        except Exception as e:
            logger.warning(f"Error estimating false positive rate: {e}")
            return None

    def _normal_cdf(self, x: float) -> float:
        """Approximate cumulative distribution function for standard normal distribution."""
        # Abramowitz and Stegun approximation
        try:
            if x < 0:
                return 1 - self._normal_cdf(-x)

            # Constants
            b0 = 0.2316419
            b1 = 0.319381530
            b2 = -0.356563782
            b3 = 1.781477937
            b4 = -1.821255978
            b5 = 1.330274429

            t = 1 / (1 + b0 * x)
            phi = math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
            cdf = 1 - phi * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)

            return max(0.0, min(1.0, cdf))

        except Exception:
            # Fallback to simple approximation
            return 0.5 if x == 0 else (0.999 if x > 3 else 0.001)

    def get_streaming_state(self, stream_id: str) -> Optional[StreamingAnomalyState]:
        """Get the current streaming state for a stream."""
        return self._streaming_states.get(stream_id)

    def reset_streaming_state(self, stream_id: str):
        """Reset streaming state for a stream."""
        if stream_id in self._streaming_states:
            del self._streaming_states[stream_id]

    def clear_all_states(self):
        """Clear all streaming states and caches."""
        self._streaming_states.clear()
        self._detection_history.clear()
        self._threshold_cache.clear()