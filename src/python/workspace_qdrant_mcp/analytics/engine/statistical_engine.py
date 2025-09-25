"""
Statistical Engine for advanced data analytics.

Provides comprehensive statistical analysis capabilities with robust edge case handling
for NaN, infinity, zero division, and other numerical boundary conditions.
"""

import math
import statistics
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
import logging


logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""

    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[float] = None
    std_dev: Optional[float] = None
    variance: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    count: int = 0
    sum_value: Optional[float] = None
    percentiles: Optional[Dict[int, float]] = None
    is_valid: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mean': self.mean,
            'median': self.median,
            'mode': self.mode,
            'std_dev': self.std_dev,
            'variance': self.variance,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'count': self.count,
            'sum_value': self.sum_value,
            'percentiles': self.percentiles,
            'is_valid': self.is_valid,
            'error_message': self.error_message
        }


class StatisticalEngine:
    """
    Advanced statistical analysis engine with comprehensive edge case handling.

    Handles NaN, infinity, zero division, empty datasets, and other boundary conditions
    gracefully while providing accurate statistical insights.
    """

    def __init__(self, precision_digits: int = 6):
        """
        Initialize statistical engine.

        Args:
            precision_digits: Number of decimal places for result precision
        """
        self.precision_digits = precision_digits
        self._stats_cache = {}
        self._cache_ttl = timedelta(minutes=5)
        self._cache_timestamps = {}

    def calculate_basic_stats(self, data: List[Union[int, float]],
                            cache_key: Optional[str] = None) -> StatisticalResult:
        """
        Calculate comprehensive basic statistics for a dataset.

        Args:
            data: List of numerical values
            cache_key: Optional cache key for result caching

        Returns:
            StatisticalResult with comprehensive statistics
        """
        # Check cache if key provided
        if cache_key and self._is_cached_valid(cache_key):
            return self._stats_cache[cache_key]

        result = StatisticalResult()

        try:
            # Handle empty dataset
            if not data:
                result.is_valid = False
                result.error_message = "Empty dataset provided"
                return result

            # Filter out invalid values (NaN, infinity)
            valid_data = []
            for value in data:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid_data.append(float(value))

            if not valid_data:
                result.is_valid = False
                result.error_message = "No valid numerical values found"
                return result

            result.count = len(valid_data)

            # Basic calculations with error handling
            try:
                result.sum_value = sum(valid_data)
                result.mean = result.sum_value / len(valid_data)
                result.mean = round(result.mean, self.precision_digits)
            except (OverflowError, ZeroDivisionError) as e:
                logger.warning(f"Error calculating mean: {e}")
                result.mean = None

            try:
                result.median = statistics.median(valid_data)
                result.median = round(result.median, self.precision_digits)
            except Exception as e:
                logger.warning(f"Error calculating median: {e}")
                result.median = None

            try:
                result.mode = statistics.mode(valid_data)
                result.mode = round(result.mode, self.precision_digits)
            except (statistics.StatisticsError, Exception):
                # No unique mode or other error
                result.mode = None

            try:
                result.std_dev = statistics.stdev(valid_data) if len(valid_data) > 1 else 0.0
                result.std_dev = round(result.std_dev, self.precision_digits)
            except (statistics.StatisticsError, ValueError) as e:
                logger.warning(f"Error calculating standard deviation: {e}")
                result.std_dev = None

            try:
                result.variance = statistics.variance(valid_data) if len(valid_data) > 1 else 0.0
                result.variance = round(result.variance, self.precision_digits)
            except (statistics.StatisticsError, ValueError) as e:
                logger.warning(f"Error calculating variance: {e}")
                result.variance = None

            result.min_value = min(valid_data)
            result.max_value = max(valid_data)

            # Calculate percentiles
            result.percentiles = self._calculate_percentiles(valid_data)

            # Cache result if key provided
            if cache_key:
                self._cache_result(cache_key, result)

        except Exception as e:
            logger.error(f"Unexpected error in statistical calculation: {e}")
            result.is_valid = False
            result.error_message = f"Calculation error: {str(e)}"

        return result

    def calculate_correlation(self, x_data: List[Union[int, float]],
                            y_data: List[Union[int, float]]) -> Optional[float]:
        """
        Calculate Pearson correlation coefficient between two datasets.

        Args:
            x_data: First dataset
            y_data: Second dataset

        Returns:
            Correlation coefficient or None if calculation fails
        """
        try:
            if len(x_data) != len(y_data):
                logger.warning("Datasets have different lengths for correlation")
                return None

            if len(x_data) < 2:
                logger.warning("Insufficient data for correlation calculation")
                return None

            # Filter paired valid values
            paired_data = []
            for x, y in zip(x_data, y_data):
                if (isinstance(x, (int, float)) and math.isfinite(x) and
                    isinstance(y, (int, float)) and math.isfinite(y)):
                    paired_data.append((float(x), float(y)))

            if len(paired_data) < 2:
                logger.warning("Insufficient valid paired data for correlation")
                return None

            x_values = [pair[0] for pair in paired_data]
            y_values = [pair[1] for pair in paired_data]

            correlation = statistics.correlation(x_values, y_values)
            return round(correlation, self.precision_digits)

        except (statistics.StatisticsError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating correlation: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in correlation calculation: {e}")
            return None

    def calculate_moving_average(self, data: List[Union[int, float]],
                               window_size: int) -> List[Optional[float]]:
        """
        Calculate moving average with specified window size.

        Args:
            data: Input data series
            window_size: Size of moving window

        Returns:
            List of moving averages (None for invalid windows)
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")

        if not data:
            return []

        result = []

        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window_data = data[start_idx:i + 1]

            # Filter valid values in window
            valid_window = []
            for value in window_data:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid_window.append(float(value))

            if len(valid_window) > 0:
                try:
                    avg = sum(valid_window) / len(valid_window)
                    result.append(round(avg, self.precision_digits))
                except (OverflowError, ZeroDivisionError):
                    result.append(None)
            else:
                result.append(None)

        return result

    def detect_outliers(self, data: List[Union[int, float]],
                       method: str = "iqr",
                       threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers using specified method.

        Args:
            data: Input dataset
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection

        Returns:
            Dictionary containing outlier indices and statistics
        """
        if not data:
            return {"outliers": [], "outlier_values": [], "method": method, "threshold": threshold}

        # Filter valid data with original indices
        valid_data_with_indices = []
        for i, value in enumerate(data):
            if isinstance(value, (int, float)) and math.isfinite(value):
                valid_data_with_indices.append((i, float(value)))

        if len(valid_data_with_indices) < 3:
            return {"outliers": [], "outlier_values": [], "method": method, "threshold": threshold,
                   "warning": "Insufficient valid data for outlier detection"}

        outlier_indices = []
        outlier_values = []

        try:
            if method == "iqr":
                outlier_indices, outlier_values = self._detect_outliers_iqr(
                    valid_data_with_indices, threshold)
            elif method == "zscore":
                outlier_indices, outlier_values = self._detect_outliers_zscore(
                    valid_data_with_indices, threshold)
            elif method == "modified_zscore":
                outlier_indices, outlier_values = self._detect_outliers_modified_zscore(
                    valid_data_with_indices, threshold)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

        except Exception as e:
            logger.error(f"Error in outlier detection: {e}")
            return {"outliers": [], "outlier_values": [], "method": method, "threshold": threshold,
                   "error": str(e)}

        return {
            "outliers": outlier_indices,
            "outlier_values": outlier_values,
            "method": method,
            "threshold": threshold,
            "total_outliers": len(outlier_indices),
            "outlier_percentage": (len(outlier_indices) / len(valid_data_with_indices)) * 100
        }

    def _detect_outliers_iqr(self, data_with_indices: List[Tuple[int, float]],
                           threshold: float) -> Tuple[List[int], List[float]]:
        """Detect outliers using Interquartile Range method."""
        values = [item[1] for item in data_with_indices]

        q1 = statistics.quantiles(values, n=4)[0]  # 25th percentile
        q3 = statistics.quantiles(values, n=4)[2]  # 75th percentile
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outlier_indices = []
        outlier_values = []

        for original_idx, value in data_with_indices:
            if value < lower_bound or value > upper_bound:
                outlier_indices.append(original_idx)
                outlier_values.append(value)

        return outlier_indices, outlier_values

    def _detect_outliers_zscore(self, data_with_indices: List[Tuple[int, float]],
                              threshold: float) -> Tuple[List[int], List[float]]:
        """Detect outliers using Z-score method."""
        values = [item[1] for item in data_with_indices]

        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0

        if std_dev == 0:
            return [], []  # No outliers if no variation

        outlier_indices = []
        outlier_values = []

        for original_idx, value in data_with_indices:
            z_score = abs((value - mean_val) / std_dev)
            if z_score > threshold:
                outlier_indices.append(original_idx)
                outlier_values.append(value)

        return outlier_indices, outlier_values

    def _detect_outliers_modified_zscore(self, data_with_indices: List[Tuple[int, float]],
                                       threshold: float) -> Tuple[List[int], List[float]]:
        """Detect outliers using Modified Z-score method (using median)."""
        values = [item[1] for item in data_with_indices]

        median_val = statistics.median(values)
        mad = statistics.median([abs(x - median_val) for x in values])  # Median Absolute Deviation

        if mad == 0:
            return [], []  # No outliers if no variation

        outlier_indices = []
        outlier_values = []

        for original_idx, value in data_with_indices:
            modified_z_score = 0.6745 * (value - median_val) / mad
            if abs(modified_z_score) > threshold:
                outlier_indices.append(original_idx)
                outlier_values.append(value)

        return outlier_indices, outlier_values

    def _calculate_percentiles(self, data: List[float]) -> Dict[int, float]:
        """Calculate common percentiles for the dataset."""
        if len(data) < 1:
            return {}

        try:
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95, 99]:
                try:
                    # Use quantiles for percentile calculation
                    if p == 50:
                        percentiles[p] = round(statistics.median(data), self.precision_digits)
                    else:
                        # Convert percentile to quantile position
                        quantile_pos = p / 100.0
                        sorted_data = sorted(data)
                        n = len(sorted_data)

                        if quantile_pos == 0:
                            percentiles[p] = sorted_data[0]
                        elif quantile_pos == 1:
                            percentiles[p] = sorted_data[-1]
                        else:
                            # Linear interpolation
                            index = (n - 1) * quantile_pos
                            lower_index = int(math.floor(index))
                            upper_index = int(math.ceil(index))

                            if lower_index == upper_index:
                                percentiles[p] = sorted_data[lower_index]
                            else:
                                weight = index - lower_index
                                value = sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
                                percentiles[p] = round(value, self.precision_digits)

                except Exception as e:
                    logger.warning(f"Error calculating {p}th percentile: {e}")
                    continue

            return percentiles

        except Exception as e:
            logger.error(f"Error calculating percentiles: {e}")
            return {}

    def _is_cached_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self._stats_cache:
            return False

        if cache_key not in self._cache_timestamps:
            return False

        age = datetime.now() - self._cache_timestamps[cache_key]
        return age <= self._cache_ttl

    def _cache_result(self, cache_key: str, result: StatisticalResult):
        """Cache statistical result."""
        self._stats_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()

    def clear_cache(self):
        """Clear all cached results."""
        self._stats_cache.clear()
        self._cache_timestamps.clear()