//! Parameter validation for GraphService gRPC handlers.
//!
//! Provides bounds-checking constants and validation functions for
//! PageRank, community detection, and betweenness centrality parameters.
//! Returns `Status::invalid_argument` on violation.

use tonic::Status;

// ─── PageRank bounds ────────────────────────────────────────────────

/// Maximum allowed PageRank iterations.
pub(super) const MAX_PAGERANK_ITERATIONS: u32 = 1000;

/// Minimum damping factor (below this, teleportation dominates and
/// PageRank scores become nearly uniform).
pub(super) const MIN_DAMPING: f64 = 0.1;

/// Maximum damping factor (at 1.0, dangling nodes cause division
/// issues; 0.99 keeps computation stable).
pub(super) const MAX_DAMPING: f64 = 0.99;

/// Minimum convergence tolerance (tighter than 1e-10 yields
/// negligible improvement at high iteration cost).
pub(super) const MIN_TOLERANCE: f64 = 1e-10;

/// Maximum convergence tolerance (above 0.1 the result is too
/// coarse to be meaningful).
pub(super) const MAX_TOLERANCE: f64 = 0.1;

// ─── Community detection bounds ─────────────────────────────────────

/// Maximum allowed label-propagation iterations.
pub(super) const MAX_COMMUNITY_ITERATIONS: u32 = 500;

/// Minimum community size filter (must include at least 1 node).
pub(super) const MIN_COMMUNITY_SIZE: u32 = 1;

/// Maximum community size filter.
pub(super) const MAX_COMMUNITY_SIZE: u32 = 10_000;

// ─── Materialization bounds ─────────────────────────────────────────

/// Default cap on nodes loaded into memory for graph algorithms.
pub(super) const DEFAULT_MAX_NODES: u32 = 100_000;

// ─── Validation functions ───────────────────────────────────────────

/// Validate PageRank parameters, returning the sanitised
/// `(damping, max_iterations, tolerance)` tuple.
///
/// - `damping` must be in `[MIN_DAMPING, MAX_DAMPING]`.
/// - `max_iterations` must be in `[1, MAX_PAGERANK_ITERATIONS]`.
/// - `tolerance` must be in `[MIN_TOLERANCE, MAX_TOLERANCE]`.
pub(super) fn validate_pagerank_params(
    damping: Option<f64>,
    max_iterations: Option<u32>,
    tolerance: Option<f64>,
) -> Result<(f64, usize, f64), Status> {
    let damping = damping.unwrap_or(0.85);
    if !(MIN_DAMPING..=MAX_DAMPING).contains(&damping) {
        return Err(Status::invalid_argument(format!(
            "damping must be between {} and {}, got {}",
            MIN_DAMPING, MAX_DAMPING, damping
        )));
    }

    let max_iter = max_iterations.unwrap_or(100);
    if max_iter == 0 || max_iter > MAX_PAGERANK_ITERATIONS {
        return Err(Status::invalid_argument(format!(
            "max_iterations must be between 1 and {}, got {}",
            MAX_PAGERANK_ITERATIONS, max_iter
        )));
    }

    let tol = tolerance.unwrap_or(1e-6);
    if !(MIN_TOLERANCE..=MAX_TOLERANCE).contains(&tol) {
        return Err(Status::invalid_argument(format!(
            "tolerance must be between {} and {}, got {}",
            MIN_TOLERANCE, MAX_TOLERANCE, tol
        )));
    }

    Ok((damping, max_iter as usize, tol))
}

/// Validate community-detection parameters, returning the sanitised
/// `(max_iterations, min_community_size)` tuple.
///
/// - `max_iterations` must be in `[1, MAX_COMMUNITY_ITERATIONS]`.
/// - `min_community_size` must be in `[MIN_COMMUNITY_SIZE, MAX_COMMUNITY_SIZE]`.
pub(super) fn validate_community_params(
    max_iterations: Option<u32>,
    min_community_size: Option<u32>,
) -> Result<(usize, usize), Status> {
    let max_iter = max_iterations.unwrap_or(50);
    if max_iter == 0 || max_iter > MAX_COMMUNITY_ITERATIONS {
        return Err(Status::invalid_argument(format!(
            "max_iterations must be between 1 and {}, got {}",
            MAX_COMMUNITY_ITERATIONS, max_iter
        )));
    }

    let min_size = min_community_size.unwrap_or(2);
    if min_size < MIN_COMMUNITY_SIZE || min_size > MAX_COMMUNITY_SIZE {
        return Err(Status::invalid_argument(format!(
            "min_community_size must be between {} and {}, got {}",
            MIN_COMMUNITY_SIZE, MAX_COMMUNITY_SIZE, min_size
        )));
    }

    Ok((max_iter as usize, min_size as usize))
}

/// Validate betweenness centrality `max_samples` parameter.
///
/// Returns `None` (meaning "use all nodes") when the input is `None`
/// or zero.  Otherwise validates upper bound.
pub(super) fn validate_betweenness_params(
    max_samples: Option<u32>,
) -> Result<Option<usize>, Status> {
    match max_samples.filter(|&v| v > 0) {
        Some(v) if v > DEFAULT_MAX_NODES => Err(Status::invalid_argument(format!(
            "max_samples must not exceed {}, got {}",
            DEFAULT_MAX_NODES, v
        ))),
        Some(v) => Ok(Some(v as usize)),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PageRank validation ────────────────────────────────────────

    #[test]
    fn pagerank_defaults_accepted() {
        let (d, i, t) = validate_pagerank_params(None, None, None).unwrap();
        assert!((d - 0.85).abs() < f64::EPSILON);
        assert_eq!(i, 100);
        assert!((t - 1e-6).abs() < f64::EPSILON);
    }

    #[test]
    fn pagerank_valid_custom_values() {
        let (d, i, t) = validate_pagerank_params(Some(0.5), Some(500), Some(1e-8)).unwrap();
        assert!((d - 0.5).abs() < f64::EPSILON);
        assert_eq!(i, 500);
        assert!((t - 1e-8).abs() < f64::EPSILON);
    }

    #[test]
    fn pagerank_boundary_damping_min() {
        let (d, _, _) = validate_pagerank_params(Some(MIN_DAMPING), None, None).unwrap();
        assert!((d - MIN_DAMPING).abs() < f64::EPSILON);
    }

    #[test]
    fn pagerank_boundary_damping_max() {
        let (d, _, _) = validate_pagerank_params(Some(MAX_DAMPING), None, None).unwrap();
        assert!((d - MAX_DAMPING).abs() < f64::EPSILON);
    }

    #[test]
    fn pagerank_damping_below_min_rejected() {
        let err = validate_pagerank_params(Some(0.05), None, None).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("damping"));
    }

    #[test]
    fn pagerank_damping_above_max_rejected() {
        let err = validate_pagerank_params(Some(1.0), None, None).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("damping"));
    }

    #[test]
    fn pagerank_iterations_zero_rejected() {
        let err = validate_pagerank_params(None, Some(0), None).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("max_iterations"));
    }

    #[test]
    fn pagerank_iterations_over_max_rejected() {
        let err =
            validate_pagerank_params(None, Some(MAX_PAGERANK_ITERATIONS + 1), None).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("max_iterations"));
    }

    #[test]
    fn pagerank_iterations_at_max_accepted() {
        let (_, i, _) =
            validate_pagerank_params(None, Some(MAX_PAGERANK_ITERATIONS), None).unwrap();
        assert_eq!(i, MAX_PAGERANK_ITERATIONS as usize);
    }

    #[test]
    fn pagerank_tolerance_below_min_rejected() {
        let err = validate_pagerank_params(None, None, Some(1e-11)).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("tolerance"));
    }

    #[test]
    fn pagerank_tolerance_above_max_rejected() {
        let err = validate_pagerank_params(None, None, Some(0.5)).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("tolerance"));
    }

    #[test]
    fn pagerank_tolerance_boundary_min() {
        let (_, _, t) = validate_pagerank_params(None, None, Some(MIN_TOLERANCE)).unwrap();
        assert!((t - MIN_TOLERANCE).abs() < f64::EPSILON);
    }

    #[test]
    fn pagerank_tolerance_boundary_max() {
        let (_, _, t) = validate_pagerank_params(None, None, Some(MAX_TOLERANCE)).unwrap();
        assert!((t - MAX_TOLERANCE).abs() < f64::EPSILON);
    }

    // ─── Community detection validation ─────────────────────────────

    #[test]
    fn community_defaults_accepted() {
        let (i, s) = validate_community_params(None, None).unwrap();
        assert_eq!(i, 50);
        assert_eq!(s, 2);
    }

    #[test]
    fn community_valid_custom_values() {
        let (i, s) = validate_community_params(Some(200), Some(5)).unwrap();
        assert_eq!(i, 200);
        assert_eq!(s, 5);
    }

    #[test]
    fn community_iterations_zero_rejected() {
        let err = validate_community_params(Some(0), None).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("max_iterations"));
    }

    #[test]
    fn community_iterations_over_max_rejected() {
        let err = validate_community_params(Some(MAX_COMMUNITY_ITERATIONS + 1), None).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("max_iterations"));
    }

    #[test]
    fn community_iterations_at_max_accepted() {
        let (i, _) = validate_community_params(Some(MAX_COMMUNITY_ITERATIONS), None).unwrap();
        assert_eq!(i, MAX_COMMUNITY_ITERATIONS as usize);
    }

    #[test]
    fn community_min_size_zero_rejected() {
        let err = validate_community_params(None, Some(0)).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("min_community_size"));
    }

    #[test]
    fn community_min_size_over_max_rejected() {
        let err = validate_community_params(None, Some(MAX_COMMUNITY_SIZE + 1)).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("min_community_size"));
    }

    #[test]
    fn community_min_size_at_boundaries() {
        let (_, s) = validate_community_params(None, Some(MIN_COMMUNITY_SIZE)).unwrap();
        assert_eq!(s, MIN_COMMUNITY_SIZE as usize);
        let (_, s) = validate_community_params(None, Some(MAX_COMMUNITY_SIZE)).unwrap();
        assert_eq!(s, MAX_COMMUNITY_SIZE as usize);
    }

    // ─── Betweenness validation ─────────────────────────────────────

    #[test]
    fn betweenness_none_returns_none() {
        assert!(validate_betweenness_params(None).unwrap().is_none());
    }

    #[test]
    fn betweenness_zero_returns_none() {
        assert!(validate_betweenness_params(Some(0)).unwrap().is_none());
    }

    #[test]
    fn betweenness_valid_value() {
        let result = validate_betweenness_params(Some(50)).unwrap();
        assert_eq!(result, Some(50));
    }

    #[test]
    fn betweenness_at_max_accepted() {
        let result = validate_betweenness_params(Some(DEFAULT_MAX_NODES)).unwrap();
        assert_eq!(result, Some(DEFAULT_MAX_NODES as usize));
    }

    #[test]
    fn betweenness_over_max_rejected() {
        let err = validate_betweenness_params(Some(DEFAULT_MAX_NODES + 1)).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("max_samples"));
    }
}
