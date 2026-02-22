//! Specification Pattern: composable, testable business rule predicates.
//!
//! Each specification encapsulates a single business rule and can be combined
//! with `AndSpec`, `OrSpec`, `NotSpec` for complex conditions.
//!
//! # Submodules
//! - `collection` — canonical Collection enum for the 4 Qdrant collections
//! - `payload` — typed payload parsing helper for queue items
//!
//! # Future submodules
//! - `file_filter` — file exclusion/inclusion specs
//! - `priority` — priority computation spec

pub mod collection;
pub mod payload;

pub use collection::Collection;
pub use payload::parse_payload;

use std::marker::PhantomData;

/// A specification predicate over type `T`.
///
/// Specifications encapsulate business rules as composable, testable units.
/// Each spec answers "does this candidate satisfy the rule?"
pub trait Spec<T: ?Sized> {
    /// Check whether the candidate satisfies this specification.
    fn is_satisfied_by(&self, candidate: &T) -> bool;

    /// Optional human-readable reason when the candidate does NOT satisfy the spec.
    /// Returns `None` if satisfied.
    fn rejection_reason(&self, candidate: &T) -> Option<String> {
        if self.is_satisfied_by(candidate) {
            None
        } else {
            Some("specification not satisfied".to_string())
        }
    }
}

/// Logical AND of two specifications: both must be satisfied.
pub struct AndSpec<T: ?Sized, A: Spec<T>, B: Spec<T>> {
    left: A,
    right: B,
    _phantom: PhantomData<T>,
}

impl<T: ?Sized, A: Spec<T>, B: Spec<T>> AndSpec<T, A, B> {
    pub fn new(left: A, right: B) -> Self {
        Self {
            left,
            right,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Spec<T>, B: Spec<T>> Spec<T> for AndSpec<T, A, B> {
    fn is_satisfied_by(&self, candidate: &T) -> bool {
        self.left.is_satisfied_by(candidate) && self.right.is_satisfied_by(candidate)
    }

    fn rejection_reason(&self, candidate: &T) -> Option<String> {
        self.left
            .rejection_reason(candidate)
            .or_else(|| self.right.rejection_reason(candidate))
    }
}

/// Logical OR of two specifications: at least one must be satisfied.
pub struct OrSpec<T: ?Sized, A: Spec<T>, B: Spec<T>> {
    left: A,
    right: B,
    _phantom: PhantomData<T>,
}

impl<T: ?Sized, A: Spec<T>, B: Spec<T>> OrSpec<T, A, B> {
    pub fn new(left: A, right: B) -> Self {
        Self {
            left,
            right,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Spec<T>, B: Spec<T>> Spec<T> for OrSpec<T, A, B> {
    fn is_satisfied_by(&self, candidate: &T) -> bool {
        self.left.is_satisfied_by(candidate) || self.right.is_satisfied_by(candidate)
    }
}

/// Logical NOT of a specification: inverts the result.
pub struct NotSpec<T: ?Sized, S: Spec<T>> {
    inner: S,
    _phantom: PhantomData<T>,
}

impl<T: ?Sized, S: Spec<T>> NotSpec<T, S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized, S: Spec<T>> Spec<T> for NotSpec<T, S> {
    fn is_satisfied_by(&self, candidate: &T) -> bool {
        !self.inner.is_satisfied_by(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test spec that checks if a number is positive.
    struct IsPositive;
    impl Spec<i32> for IsPositive {
        fn is_satisfied_by(&self, candidate: &i32) -> bool {
            *candidate > 0
        }
        fn rejection_reason(&self, candidate: &i32) -> Option<String> {
            if *candidate > 0 {
                None
            } else {
                Some(format!("{} is not positive", candidate))
            }
        }
    }

    /// Test spec that checks if a number is even.
    struct IsEven;
    impl Spec<i32> for IsEven {
        fn is_satisfied_by(&self, candidate: &i32) -> bool {
            candidate % 2 == 0
        }
    }

    #[test]
    fn test_and_spec() {
        let spec = AndSpec::new(IsPositive, IsEven);
        assert!(spec.is_satisfied_by(&4));
        assert!(!spec.is_satisfied_by(&3)); // odd
        assert!(!spec.is_satisfied_by(&-2)); // negative
        assert!(!spec.is_satisfied_by(&-3)); // negative and odd
    }

    #[test]
    fn test_or_spec() {
        let spec = OrSpec::new(IsPositive, IsEven);
        assert!(spec.is_satisfied_by(&4)); // both
        assert!(spec.is_satisfied_by(&3)); // positive only
        assert!(spec.is_satisfied_by(&-2)); // even only
        assert!(!spec.is_satisfied_by(&-3)); // neither
    }

    #[test]
    fn test_not_spec() {
        let spec = NotSpec::new(IsPositive);
        assert!(!spec.is_satisfied_by(&5));
        assert!(spec.is_satisfied_by(&-1));
        assert!(spec.is_satisfied_by(&0));
    }

    #[test]
    fn test_rejection_reason() {
        let spec = AndSpec::new(IsPositive, IsEven);
        assert!(spec.rejection_reason(&4).is_none());
        assert!(spec.rejection_reason(&-1).is_some());
    }

    #[test]
    fn test_triple_composition() {
        // Positive AND even AND not zero — compose three specs
        struct NotZero;
        impl Spec<i32> for NotZero {
            fn is_satisfied_by(&self, candidate: &i32) -> bool {
                *candidate != 0
            }
        }

        let inner = AndSpec::new(IsPositive, IsEven);
        let spec = AndSpec::new(inner, NotZero);
        assert!(spec.is_satisfied_by(&4));
        assert!(!spec.is_satisfied_by(&0));
        assert!(!spec.is_satisfied_by(&3));
    }
}
