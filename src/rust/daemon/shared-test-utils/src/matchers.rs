//! Custom matchers and assertions for testing

use std::fmt;

/// Result of a custom assertion
pub type MatchResult = Result<(), String>;

/// Trait for custom matchers
pub trait Matcher<T> {
    fn matches(&self, actual: &T) -> MatchResult;
    fn description(&self) -> String;
}

/// Assert that a value matches a custom matcher
pub fn assert_that<T>(actual: T, matcher: impl Matcher<T>) -> MatchResult {
    matcher.matches(&actual)
}

/// Vector similarity matcher for embedding tests
pub struct VectorSimilarityMatcher {
    expected: Vec<f32>,
    threshold: f32,
}

impl VectorSimilarityMatcher {
    pub fn new(expected: Vec<f32>, threshold: f32) -> Self {
        Self { expected, threshold }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

impl Matcher<Vec<f32>> for VectorSimilarityMatcher {
    fn matches(&self, actual: &Vec<f32>) -> MatchResult {
        let similarity = Self::cosine_similarity(&self.expected, actual);
        if similarity >= self.threshold {
            Ok(())
        } else {
            Err(format!(
                "Expected cosine similarity >= {}, but got {}",
                self.threshold, similarity
            ))
        }
    }

    fn description(&self) -> String {
        format!("vector with cosine similarity >= {} to expected vector", self.threshold)
    }
}

/// Matcher for checking if a collection contains all expected items
pub struct ContainsAllMatcher<T> {
    expected: Vec<T>,
}

impl<T> ContainsAllMatcher<T> {
    pub fn new(expected: Vec<T>) -> Self {
        Self { expected }
    }
}

impl<T> Matcher<Vec<T>> for ContainsAllMatcher<T>
where
    T: PartialEq + fmt::Debug,
{
    fn matches(&self, actual: &Vec<T>) -> MatchResult {
        for item in &self.expected {
            if !actual.contains(item) {
                return Err(format!("Expected collection to contain {:?}, but it was missing", item));
            }
        }
        Ok(())
    }

    fn description(&self) -> String {
        format!("collection containing all of {:?}", self.expected)
    }
}

/// Matcher for checking response time constraints
pub struct ResponseTimeMatcher {
    max_duration_ms: u64,
}

impl ResponseTimeMatcher {
    pub fn new(max_duration_ms: u64) -> Self {
        Self { max_duration_ms }
    }
}

impl Matcher<u64> for ResponseTimeMatcher {
    fn matches(&self, actual: &u64) -> MatchResult {
        if *actual <= self.max_duration_ms {
            Ok(())
        } else {
            Err(format!(
                "Expected response time <= {}ms, but got {}ms",
                self.max_duration_ms, actual
            ))
        }
    }

    fn description(&self) -> String {
        format!("response time <= {}ms", self.max_duration_ms)
    }
}

/// Matcher for checking if a string matches a pattern
pub struct RegexMatcher {
    pattern: regex::Regex,
}

impl RegexMatcher {
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            pattern: regex::Regex::new(pattern)?,
        })
    }
}

impl Matcher<String> for RegexMatcher {
    fn matches(&self, actual: &String) -> MatchResult {
        if self.pattern.is_match(actual) {
            Ok(())
        } else {
            Err(format!(
                "Expected string to match pattern '{}', but got '{}'",
                self.pattern.as_str(),
                actual
            ))
        }
    }

    fn description(&self) -> String {
        format!("string matching pattern '{}'", self.pattern.as_str())
    }
}

impl Matcher<&str> for RegexMatcher {
    fn matches(&self, actual: &&str) -> MatchResult {
        if self.pattern.is_match(actual) {
            Ok(())
        } else {
            Err(format!(
                "Expected string to match pattern '{}', but got '{}'",
                self.pattern.as_str(),
                actual
            ))
        }
    }

    fn description(&self) -> String {
        format!("string matching pattern '{}'", self.pattern.as_str())
    }
}

/// Matcher for checking if a value is within a numeric range
pub struct RangeMatcher<T> {
    min: T,
    max: T,
}

impl<T> RangeMatcher<T> {
    pub fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

impl<T> Matcher<T> for RangeMatcher<T>
where
    T: PartialOrd + fmt::Display + Copy,
{
    fn matches(&self, actual: &T) -> MatchResult {
        if *actual >= self.min && *actual <= self.max {
            Ok(())
        } else {
            Err(format!(
                "Expected value to be in range [{}, {}], but got {}",
                self.min, self.max, actual
            ))
        }
    }

    fn description(&self) -> String {
        format!("value in range [{}, {}]", self.min, self.max)
    }
}

/// Matcher for checking if a collection has a specific size
pub struct SizeMatcher {
    expected_size: usize,
}

impl SizeMatcher {
    pub fn new(expected_size: usize) -> Self {
        Self { expected_size }
    }
}

impl<T> Matcher<Vec<T>> for SizeMatcher {
    fn matches(&self, actual: &Vec<T>) -> MatchResult {
        if actual.len() == self.expected_size {
            Ok(())
        } else {
            Err(format!(
                "Expected collection size {}, but got {}",
                self.expected_size,
                actual.len()
            ))
        }
    }

    fn description(&self) -> String {
        format!("collection with size {}", self.expected_size)
    }
}

/// Matcher for checking if a result is successful
pub struct SuccessMatcher;

impl<T, E> Matcher<Result<T, E>> for SuccessMatcher
where
    E: fmt::Debug,
{
    fn matches(&self, actual: &Result<T, E>) -> MatchResult {
        match actual {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Expected success, but got error: {:?}", e)),
        }
    }

    fn description(&self) -> String {
        "successful result".to_string()
    }
}

/// Matcher for checking if a result is an error
pub struct ErrorMatcher;

impl<T, E> Matcher<Result<T, E>> for ErrorMatcher
where
    T: fmt::Debug,
{
    fn matches(&self, actual: &Result<T, E>) -> MatchResult {
        match actual {
            Ok(v) => Err(format!("Expected error, but got success: {:?}", v)),
            Err(_) => Ok(()),
        }
    }

    fn description(&self) -> String {
        "error result".to_string()
    }
}

/// Convenience functions for creating matchers
pub mod matchers {
    use super::*;

    /// Create a vector similarity matcher
    pub fn similar_to(expected: Vec<f32>, threshold: f32) -> VectorSimilarityMatcher {
        VectorSimilarityMatcher::new(expected, threshold)
    }

    /// Create a contains all matcher
    pub fn contains_all<T>(expected: Vec<T>) -> ContainsAllMatcher<T> {
        ContainsAllMatcher::new(expected)
    }

    /// Create a response time matcher
    pub fn responds_within(max_duration_ms: u64) -> ResponseTimeMatcher {
        ResponseTimeMatcher::new(max_duration_ms)
    }

    /// Create a regex matcher
    pub fn matches_pattern(pattern: &str) -> Result<RegexMatcher, regex::Error> {
        RegexMatcher::new(pattern)
    }

    /// Create a range matcher
    pub fn in_range<T>(min: T, max: T) -> RangeMatcher<T> {
        RangeMatcher::new(min, max)
    }

    /// Create a size matcher
    pub fn has_size(expected_size: usize) -> SizeMatcher {
        SizeMatcher::new(expected_size)
    }

    /// Create a success matcher
    pub fn succeeds() -> SuccessMatcher {
        SuccessMatcher
    }

    /// Create an error matcher
    pub fn fails() -> ErrorMatcher {
        ErrorMatcher
    }
}

/// Macro for more natural assertion syntax
#[macro_export]
macro_rules! assert_that {
    ($actual:expr, $matcher:expr) => {
        match $crate::matchers::assert_that($actual, $matcher) {
            Ok(()) => (),
            Err(msg) => panic!("Assertion failed: {}", msg),
        }
    };
}

/// Macro for asserting vector similarity
#[macro_export]
macro_rules! assert_vectors_similar {
    ($actual:expr, $expected:expr, $threshold:expr) => {
        $crate::assert_that!(
            $actual,
            $crate::matchers::matchers::similar_to($expected, $threshold)
        );
    };
    ($actual:expr, $expected:expr) => {
        $crate::assert_vectors_similar!($actual, $expected, 0.9);
    };
}

/// Macro for asserting response times
#[macro_export]
macro_rules! assert_responds_within {
    ($duration:expr, $max_ms:expr) => {
        $crate::assert_that!(
            $duration,
            $crate::matchers::matchers::responds_within($max_ms)
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::matchers::*;

    #[test]
    fn test_vector_similarity_matcher() {
        let expected = vec![1.0, 0.0, 0.0];
        let similar = vec![0.9, 0.1, 0.0];
        let different = vec![0.0, 1.0, 0.0];

        let matcher = similar_to(expected, 0.8);
        assert!(matcher.matches(&similar).is_ok());
        assert!(matcher.matches(&different).is_err());
    }

    #[test]
    fn test_contains_all_matcher() {
        let actual = vec![1, 2, 3, 4, 5];
        let expected_subset = vec![2, 4];
        let expected_missing = vec![2, 6];

        let matcher = contains_all(expected_subset);
        assert!(matcher.matches(&actual).is_ok());

        let matcher = contains_all(expected_missing);
        assert!(matcher.matches(&actual).is_err());
    }

    #[test]
    fn test_response_time_matcher() {
        let matcher = responds_within(1000);
        assert!(matcher.matches(&500).is_ok());
        assert!(matcher.matches(&1500).is_err());
    }

    #[test]
    fn test_regex_matcher() {
        let matcher = matches_pattern(r"^\d{3}-\d{3}-\d{4}$").unwrap();
        assert!(matcher.matches(&"123-456-7890".to_string()).is_ok());
        assert!(matcher.matches(&"invalid-phone".to_string()).is_err());
    }

    #[test]
    fn test_range_matcher() {
        let matcher = in_range(10, 20);
        assert!(matcher.matches(&15).is_ok());
        assert!(matcher.matches(&5).is_err());
        assert!(matcher.matches(&25).is_err());
    }

    #[test]
    fn test_size_matcher() {
        let matcher = has_size(3);
        let vec_3 = vec![1, 2, 3];
        let vec_5 = vec![1, 2, 3, 4, 5];

        assert!(matcher.matches(&vec_3).is_ok());
        assert!(matcher.matches(&vec_5).is_err());
    }

    #[test]
    fn test_result_matchers() {
        let success: Result<i32, &str> = Ok(42);
        let error: Result<i32, &str> = Err("failed");

        assert!(succeeds().matches(&success).is_ok());
        assert!(succeeds().matches(&error).is_err());
        assert!(fails().matches(&error).is_ok());
        assert!(fails().matches(&success).is_err());
    }

    #[test]
    fn test_assert_that_macro() {
        let actual = vec![1.0, 0.0, 0.0];
        let expected = vec![0.9, 0.1, 0.0];

        // This should not panic
        assert_that!(actual, similar_to(expected, 0.8));
    }

    #[test]
    #[should_panic(expected = "Assertion failed")]
    fn test_assert_that_macro_failure() {
        let actual = vec![1.0, 0.0, 0.0];
        let expected = vec![0.0, 1.0, 0.0];

        // This should panic
        assert_that!(actual, similar_to(expected, 0.8));
    }
}