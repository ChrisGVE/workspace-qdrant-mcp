"""
Depth validation utilities for folder watching operations.

This module provides comprehensive validation logic for recursive depth limits
including performance warnings and reasonable bounds checking.
"""

from common.logging.loguru_config import get_logger
from typing import Optional, Dict, Any

logger = get_logger(__name__)

# Constants for depth validation
MIN_DEPTH = -1  # -1 for unlimited
MAX_REASONABLE_DEPTH = 50  # Maximum reasonable depth
PERFORMANCE_WARNING_DEPTH = 20  # Depth at which to show performance warnings
SHALLOW_DEPTH_THRESHOLD = 3  # Threshold for shallow directory structures
DEEP_DEPTH_THRESHOLD = 10  # Threshold for deep directory structures


class DepthValidationError(Exception):
    """Raised when depth validation fails."""
    pass


class DepthValidationResult:
    """Result of depth validation with warnings and recommendations."""
    
    def __init__(
        self,
        is_valid: bool,
        depth: int,
        error_message: Optional[str] = None,
        warnings: Optional[list] = None,
        recommendations: Optional[list] = None,
        performance_impact: str = "low"
    ):
        self.is_valid = is_valid
        self.depth = depth
        self.error_message = error_message
        self.warnings = warnings or []
        self.recommendations = recommendations or []
        self.performance_impact = performance_impact
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "depth": self.depth,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "performance_impact": self.performance_impact
        }


def validate_recursive_depth(depth: int) -> DepthValidationResult:
    """
    Validate recursive depth parameter with comprehensive checks.
    
    Args:
        depth: The recursive depth to validate (-1 for unlimited, 0+ for specific depth)
    
    Returns:
        DepthValidationResult: Detailed validation result with warnings and recommendations
    
    Raises:
        DepthValidationError: If depth is critically invalid
    """
    warnings = []
    recommendations = []
    performance_impact = "low"
    
    # Basic validation
    if not isinstance(depth, int):
        return DepthValidationResult(
            is_valid=False,
            depth=depth,
            error_message="Depth must be an integer"
        )
    
    # Check minimum bounds
    if depth < MIN_DEPTH:
        return DepthValidationResult(
            is_valid=False,
            depth=depth,
            error_message=f"Depth must be -1 (unlimited) or a non-negative integer, got: {depth}"
        )
    
    # Check maximum reasonable bounds
    if depth > MAX_REASONABLE_DEPTH:
        return DepthValidationResult(
            is_valid=False,
            depth=depth,
            error_message=f"Depth {depth} exceeds maximum reasonable limit of {MAX_REASONABLE_DEPTH}"
        )
    
    # Unlimited depth (-1) analysis
    if depth == -1:
        warnings.append("Unlimited depth may impact performance on large directory structures")
        recommendations.extend([
            "Consider using a specific depth limit for better performance",
            "Monitor memory usage when using unlimited depth",
            "Ensure adequate ignore patterns to exclude large subdirectories"
        ])
        performance_impact = "high"
    
    # Performance warning for deep structures
    elif depth >= PERFORMANCE_WARNING_DEPTH:
        warnings.append(f"Depth {depth} may cause performance issues on large directory trees")
        recommendations.append("Consider reducing depth or adding specific ignore patterns")
        performance_impact = "medium"
    
    # Performance and depth-specific recommendations (only if not unlimited)
    if depth != -1:
        # Current directory only (depth 0) - special case
        if depth == 0:
            recommendations.append("Depth 0 will only watch the root directory, not subdirectories")
            performance_impact = "low"
        # Shallow depth recommendations
        elif depth <= SHALLOW_DEPTH_THRESHOLD:
            recommendations.append(f"Depth {depth} is quite shallow - consider if deeper nesting is needed")
            performance_impact = "low"
        # Deep but reasonable depth
        elif depth >= DEEP_DEPTH_THRESHOLD:
            recommendations.append(f"Depth {depth} will traverse deep directory structures")
            performance_impact = "medium"
    
    logger.debug(f"Depth validation for {depth}: valid={True}, warnings={len(warnings)}")
    
    return DepthValidationResult(
        is_valid=True,
        depth=depth,
        warnings=warnings,
        recommendations=recommendations,
        performance_impact=performance_impact
    )


def get_depth_recommendations(directory_structure_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get depth recommendations based on directory structure analysis.
    
    Args:
        directory_structure_info: Optional info about directory structure
            Expected keys: 'max_depth', 'file_count', 'directory_count'
    
    Returns:
        Dict with recommended depth and reasoning
    """
    recommendations = {
        "recommended_depth": 5,  # Default recommendation
        "reasoning": "Default recommendation for balanced performance",
        "scenarios": {
            "shallow": {"depth": 2, "description": "For mostly flat directory structures"},
            "medium": {"depth": 5, "description": "For typical nested project structures"},
            "deep": {"depth": 10, "description": "For heavily nested directory trees"},
            "unlimited": {"depth": -1, "description": "For unknown or highly variable structures"}
        }
    }
    
    if directory_structure_info:
        max_depth = directory_structure_info.get('max_depth', 0)
        file_count = directory_structure_info.get('file_count', 0)
        directory_count = directory_structure_info.get('directory_count', 0)
        
        # Adjust recommendation based on structure
        if max_depth <= 2:
            recommendations["recommended_depth"] = max_depth
            recommendations["reasoning"] = "Structure is shallow, limiting depth for efficiency"
        elif max_depth <= 5:
            recommendations["recommended_depth"] = max_depth
            recommendations["reasoning"] = "Structure matches medium depth recommendation"
        elif max_depth <= 15:
            recommendations["recommended_depth"] = min(max_depth, 10)
            recommendations["reasoning"] = "Structure is deep, limiting for performance"
        else:
            recommendations["recommended_depth"] = -1
            recommendations["reasoning"] = "Structure is very deep, unlimited depth may be needed"
        
        # Adjust for large file counts
        if file_count > 10000:
            if recommendations["recommended_depth"] > 5:
                recommendations["recommended_depth"] = 5
                recommendations["reasoning"] += " (reduced due to large file count)"
    
    return recommendations


def format_depth_display(depth: int) -> str:
    """
    Format depth value for user display.
    
    Args:
        depth: The depth value to format
    
    Returns:
        str: Human-readable depth description
    """
    if depth == -1:
        return "Unlimited"
    elif depth == 0:
        return "Current directory only"
    elif depth == 1:
        return "1 level deep"
    else:
        return f"{depth} levels deep"


def estimate_performance_impact(depth: int, estimated_directory_count: int = 100) -> Dict[str, Any]:
    """
    Estimate performance impact of a given depth setting.
    
    Args:
        depth: The recursive depth
        estimated_directory_count: Estimated number of directories at each level
    
    Returns:
        Dict with performance impact analysis
    """
    if depth == -1:
        # Unlimited depth - highest impact
        return {
            "impact_level": "high",
            "estimated_directories": "unlimited",
            "memory_usage": "high",
            "scan_time": "high",
            "recommendation": "Use specific depth limit for better performance"
        }
    
    estimated_total = min(estimated_directory_count ** depth, 1000000)  # Cap at 1M
    
    if depth <= 3:
        impact_level = "low"
        memory_usage = "low"
        scan_time = "fast"
    elif depth <= 10:
        impact_level = "medium"
        memory_usage = "medium"
        scan_time = "moderate"
    else:
        impact_level = "high"
        memory_usage = "high"
        scan_time = "slow"
    
    return {
        "impact_level": impact_level,
        "estimated_directories": estimated_total,
        "memory_usage": memory_usage,
        "scan_time": scan_time,
        "recommendation": f"Depth {depth} should be manageable for most systems"
    }