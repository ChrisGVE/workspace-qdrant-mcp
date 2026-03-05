//! Telemetry granularity levels (L0 through L4)

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Granularity levels for performance telemetry.
///
/// Each level subsumes the previous one: enabling L2 implicitly
/// enables L1 and L0.
///
/// ```
/// use workspace_qdrant_core::telemetry::TelemetryLevel;
///
/// assert!(TelemetryLevel::L0 < TelemetryLevel::L4);
/// assert!(TelemetryLevel::L2.includes(TelemetryLevel::L1));
/// assert!(!TelemetryLevel::L1.includes(TelemetryLevel::L3));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TelemetryLevel {
    /// Module entry/exit markers. Negligible overhead.
    L0 = 0,
    /// Function-level timing spans. Low overhead.
    L1 = 1,
    /// Data-flow metrics (queue depths, batch sizes). Moderate overhead.
    L2 = 2,
    /// Per-item tracing (individual file/document). Higher overhead.
    L3 = 3,
    /// Debug diagnostics (memory snapshots, state dumps). Full overhead.
    L4 = 4,
}

impl TelemetryLevel {
    /// Returns `true` when `self` is high enough to include `other`.
    ///
    /// For example, `L2.includes(L1)` is `true` because L2 implies L1.
    pub fn includes(self, other: Self) -> bool {
        self >= other
    }

    /// Numeric discriminant (0-4).
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// All levels in ascending order.
    pub const ALL: [TelemetryLevel; 5] = [Self::L0, Self::L1, Self::L2, Self::L3, Self::L4];
}

impl fmt::Display for TelemetryLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::L0 => write!(f, "L0"),
            Self::L1 => write!(f, "L1"),
            Self::L2 => write!(f, "L2"),
            Self::L3 => write!(f, "L3"),
            Self::L4 => write!(f, "L4"),
        }
    }
}

impl FromStr for TelemetryLevel {
    type Err = ParseTelemetryLevelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "L0" | "0" => Ok(Self::L0),
            "L1" | "1" => Ok(Self::L1),
            "L2" | "2" => Ok(Self::L2),
            "L3" | "3" => Ok(Self::L3),
            "L4" | "4" => Ok(Self::L4),
            _ => Err(ParseTelemetryLevelError(s.to_string())),
        }
    }
}

/// Error returned when parsing an invalid telemetry level string.
#[derive(Debug, Clone)]
pub struct ParseTelemetryLevelError(pub String);

impl fmt::Display for ParseTelemetryLevelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid telemetry level '{}': expected L0-L4 or 0-4",
            self.0
        )
    }
}

impl std::error::Error for ParseTelemetryLevelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_is_ascending() {
        assert!(TelemetryLevel::L0 < TelemetryLevel::L1);
        assert!(TelemetryLevel::L1 < TelemetryLevel::L2);
        assert!(TelemetryLevel::L2 < TelemetryLevel::L3);
        assert!(TelemetryLevel::L3 < TelemetryLevel::L4);
    }

    #[test]
    fn includes_lower_levels() {
        assert!(TelemetryLevel::L4.includes(TelemetryLevel::L0));
        assert!(TelemetryLevel::L2.includes(TelemetryLevel::L1));
        assert!(TelemetryLevel::L0.includes(TelemetryLevel::L0));
    }

    #[test]
    fn excludes_higher_levels() {
        assert!(!TelemetryLevel::L0.includes(TelemetryLevel::L1));
        assert!(!TelemetryLevel::L1.includes(TelemetryLevel::L3));
    }

    #[test]
    fn as_u8_matches_discriminant() {
        assert_eq!(TelemetryLevel::L0.as_u8(), 0);
        assert_eq!(TelemetryLevel::L4.as_u8(), 4);
    }

    #[test]
    fn display_format() {
        assert_eq!(TelemetryLevel::L0.to_string(), "L0");
        assert_eq!(TelemetryLevel::L3.to_string(), "L3");
    }

    #[test]
    fn from_str_variants() {
        assert_eq!("L0".parse::<TelemetryLevel>().unwrap(), TelemetryLevel::L0);
        assert_eq!("l2".parse::<TelemetryLevel>().unwrap(), TelemetryLevel::L2);
        assert_eq!("3".parse::<TelemetryLevel>().unwrap(), TelemetryLevel::L3);
        assert_eq!("L4".parse::<TelemetryLevel>().unwrap(), TelemetryLevel::L4);
    }

    #[test]
    fn from_str_invalid() {
        assert!("L5".parse::<TelemetryLevel>().is_err());
        assert!("foo".parse::<TelemetryLevel>().is_err());
        assert!("".parse::<TelemetryLevel>().is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let level = TelemetryLevel::L2;
        let json = serde_json::to_string(&level).unwrap();
        assert_eq!(json, r#""l2""#);
        let back: TelemetryLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(back, level);
    }

    #[test]
    fn all_levels_constant() {
        assert_eq!(TelemetryLevel::ALL.len(), 5);
        assert_eq!(TelemetryLevel::ALL[0], TelemetryLevel::L0);
        assert_eq!(TelemetryLevel::ALL[4], TelemetryLevel::L4);
    }
}
