//! Shared latency statistics for benchmark tools.

/// Computed statistics from a set of latency measurements.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub p95: f64,
    pub p99: f64,
}

impl LatencyStats {
    /// Compute statistics from a slice of latency values (in ms).
    ///
    /// Returns `None` if the input is empty.
    pub fn from_latencies(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let min = sorted[0];
        let max = sorted[n - 1];
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let median = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        // Standard deviation (population)
        let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        Some(Self {
            min,
            max,
            mean,
            median,
            std_dev,
            p95,
            p99,
        })
    }
}

/// Compute a percentile from a sorted slice using linear interpolation.
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = rank - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_basic() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = LatencyStats::from_latencies(&values).unwrap();
        assert!((stats.min - 10.0).abs() < 1e-6);
        assert!((stats.max - 50.0).abs() < 1e-6);
        assert!((stats.mean - 30.0).abs() < 1e-6);
        assert!((stats.median - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_single_value() {
        let stats = LatencyStats::from_latencies(&[42.0]).unwrap();
        assert!((stats.min - 42.0).abs() < 1e-6);
        assert!((stats.max - 42.0).abs() < 1e-6);
        assert!((stats.std_dev - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_empty() {
        assert!(LatencyStats::from_latencies(&[]).is_none());
    }

    #[test]
    fn test_percentile_p95() {
        // 20 values: 1..=20
        let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let p95 = percentile(&values, 95.0);
        // 95th percentile of 1..=20 with linear interpolation
        assert!(p95 > 18.0 && p95 < 20.0);
    }
}
