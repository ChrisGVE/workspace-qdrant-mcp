//! Assertion helpers for floating-point and vector comparisons

use crate::TestResult;

/// Assert that two floating point values are approximately equal
pub fn assert_approx_eq(a: f32, b: f32, epsilon: f32) -> TestResult<()> {
    if (a - b).abs() <= epsilon {
        Ok(())
    } else {
        Err(format!(
            "Values not approximately equal: {} vs {} (epsilon: {})",
            a, b, epsilon
        )
        .into())
    }
}

/// Assert that a vector of floats are approximately equal
pub fn assert_vectors_approx_eq(a: &[f32], b: &[f32], epsilon: f32) -> TestResult<()> {
    if a.len() != b.len() {
        return Err(
            format!("Vectors have different lengths: {} vs {}", a.len(), b.len()).into(),
        );
    }

    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if (x - y).abs() > epsilon {
            return Err(format!(
                "Vectors differ at index {}: {} vs {} (epsilon: {})",
                i, x, y, epsilon
            )
            .into());
        }
    }

    Ok(())
}
