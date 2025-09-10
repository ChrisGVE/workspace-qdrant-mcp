"""
Comprehensive tests for depth validation functionality.

This module tests the depth validation logic for folder watching operations,
including edge cases, performance impact analysis, and recommendation systems.
"""

import pytest
from typing import Dict, Any

from workspace_qdrant_mcp.core.depth_validation import (
    validate_recursive_depth,
    get_depth_recommendations,
    format_depth_display,
    estimate_performance_impact,
    DepthValidationResult,
    DepthValidationError,
    MIN_DEPTH,
    MAX_REASONABLE_DEPTH,
    PERFORMANCE_WARNING_DEPTH,
    SHALLOW_DEPTH_THRESHOLD,
    DEEP_DEPTH_THRESHOLD
)


class TestDepthValidation:
    """Test depth validation functionality."""
    
    def test_valid_unlimited_depth(self):
        """Test validation of unlimited depth (-1)."""
        result = validate_recursive_depth(-1)
        
        assert result.is_valid is True
        assert result.depth == -1
        assert result.error_message is None
        assert "Unlimited depth may impact performance" in result.warnings[0]
        assert result.performance_impact == "high"
        assert len(result.recommendations) >= 3
    
    def test_valid_zero_depth(self):
        """Test validation of zero depth (current directory only)."""
        result = validate_recursive_depth(0)
        
        assert result.is_valid is True
        assert result.depth == 0
        assert result.error_message is None
        assert "only watch the root directory" in result.recommendations[0]
        assert result.performance_impact == "low"
    
    def test_valid_shallow_depth(self):
        """Test validation of shallow depths (1-3)."""
        for depth in range(1, SHALLOW_DEPTH_THRESHOLD + 1):
            result = validate_recursive_depth(depth)
            
            assert result.is_valid is True
            assert result.depth == depth
            assert result.error_message is None
            assert result.performance_impact == "low"
            if depth <= SHALLOW_DEPTH_THRESHOLD:
                assert any("shallow" in rec for rec in result.recommendations)
    
    def test_valid_medium_depth(self):
        """Test validation of medium depths (4-9)."""
        for depth in range(SHALLOW_DEPTH_THRESHOLD + 1, DEEP_DEPTH_THRESHOLD):
            result = validate_recursive_depth(depth)
            
            assert result.is_valid is True
            assert result.depth == depth
            assert result.error_message is None
            assert result.performance_impact in ["low", "medium"]
    
    def test_valid_deep_depth(self):
        """Test validation of deep depths (10-19)."""
        for depth in range(DEEP_DEPTH_THRESHOLD, PERFORMANCE_WARNING_DEPTH):
            result = validate_recursive_depth(depth)
            
            assert result.is_valid is True
            assert result.depth == depth
            assert result.error_message is None
            assert result.performance_impact == "medium"
            assert any("deep directory structures" in rec for rec in result.recommendations)
    
    def test_performance_warning_depth(self):
        """Test validation of depths that trigger performance warnings."""
        depth = PERFORMANCE_WARNING_DEPTH
        result = validate_recursive_depth(depth)
        
        assert result.is_valid is True
        assert result.depth == depth
        assert result.error_message is None
        assert result.performance_impact == "medium"
        assert any("performance issues" in warning for warning in result.warnings)
        assert any("reducing depth" in rec for rec in result.recommendations)
    
    def test_invalid_negative_depth(self):
        """Test validation of invalid negative depths."""
        invalid_depths = [-2, -5, -10]
        
        for depth in invalid_depths:
            result = validate_recursive_depth(depth)
            
            assert result.is_valid is False
            assert result.depth == depth
            assert "must be -1 (unlimited) or a non-negative integer" in result.error_message
    
    def test_invalid_excessive_depth(self):
        """Test validation of excessively large depths."""
        excessive_depths = [MAX_REASONABLE_DEPTH + 1, 100, 1000]
        
        for depth in excessive_depths:
            result = validate_recursive_depth(depth)
            
            assert result.is_valid is False
            assert result.depth == depth
            assert f"exceeds maximum reasonable limit of {MAX_REASONABLE_DEPTH}" in result.error_message
    
    def test_invalid_non_integer_depth(self):
        """Test validation of non-integer depth values."""
        invalid_types = [3.5, "5", None, [], {}]
        
        for depth in invalid_types:
            result = validate_recursive_depth(depth)
            
            assert result.is_valid is False
            assert result.depth == depth
            assert "Depth must be an integer" in result.error_message
    
    def test_depth_validation_result_to_dict(self):
        """Test DepthValidationResult to_dict conversion."""
        result = validate_recursive_depth(5)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["is_valid"] is True
        assert result_dict["depth"] == 5
        assert result_dict["error_message"] is None
        assert isinstance(result_dict["warnings"], list)
        assert isinstance(result_dict["recommendations"], list)
        assert result_dict["performance_impact"] in ["low", "medium", "high"]


class TestDepthRecommendations:
    """Test depth recommendation functionality."""
    
    def test_default_recommendations(self):
        """Test default recommendations without directory structure info."""
        recommendations = get_depth_recommendations()
        
        assert recommendations["recommended_depth"] == 5
        assert "Default recommendation" in recommendations["reasoning"]
        assert "scenarios" in recommendations
        assert len(recommendations["scenarios"]) == 4
        
        scenarios = recommendations["scenarios"]
        assert scenarios["shallow"]["depth"] == 2
        assert scenarios["medium"]["depth"] == 5
        assert scenarios["deep"]["depth"] == 10
        assert scenarios["unlimited"]["depth"] == -1
    
    def test_recommendations_for_shallow_structure(self):
        """Test recommendations for shallow directory structures."""
        structure_info = {
            "max_depth": 2,
            "file_count": 50,
            "directory_count": 10
        }
        
        recommendations = get_depth_recommendations(structure_info)
        
        assert recommendations["recommended_depth"] == 2  # max_depth
        assert "shallow" in recommendations["reasoning"]
    
    def test_recommendations_for_medium_structure(self):
        """Test recommendations for medium directory structures."""
        structure_info = {
            "max_depth": 4,
            "file_count": 500,
            "directory_count": 50
        }
        
        recommendations = get_depth_recommendations(structure_info)
        
        assert recommendations["recommended_depth"] == 4
        assert "medium depth" in recommendations["reasoning"]
    
    def test_recommendations_for_deep_structure(self):
        """Test recommendations for deep directory structures."""
        structure_info = {
            "max_depth": 12,
            "file_count": 1000,
            "directory_count": 100
        }
        
        recommendations = get_depth_recommendations(structure_info)
        
        assert recommendations["recommended_depth"] == 10  # Limited for performance
        assert "deep" in recommendations["reasoning"]
    
    def test_recommendations_for_very_deep_structure(self):
        """Test recommendations for very deep directory structures."""
        structure_info = {
            "max_depth": 25,
            "file_count": 2000,
            "directory_count": 200
        }
        
        recommendations = get_depth_recommendations(structure_info)
        
        assert recommendations["recommended_depth"] == -1  # Unlimited needed
        assert "very deep" in recommendations["reasoning"]
    
    def test_recommendations_for_large_file_count(self):
        """Test recommendations adjusted for large file counts."""
        structure_info = {
            "max_depth": 8,
            "file_count": 15000,  # Large file count
            "directory_count": 500
        }
        
        recommendations = get_depth_recommendations(structure_info)
        
        assert recommendations["recommended_depth"] == 5  # Reduced due to file count
        assert "large file count" in recommendations["reasoning"]


class TestDepthDisplayFormatting:
    """Test depth display formatting functionality."""
    
    def test_format_unlimited_depth(self):
        """Test formatting of unlimited depth."""
        assert format_depth_display(-1) == "Unlimited"
    
    def test_format_zero_depth(self):
        """Test formatting of zero depth."""
        assert format_depth_display(0) == "Current directory only"
    
    def test_format_single_depth(self):
        """Test formatting of single depth."""
        assert format_depth_display(1) == "1 level deep"
    
    def test_format_multiple_depths(self):
        """Test formatting of multiple depths."""
        assert format_depth_display(5) == "5 levels deep"
        assert format_depth_display(10) == "10 levels deep"
        assert format_depth_display(50) == "50 levels deep"


class TestPerformanceImpactEstimation:
    """Test performance impact estimation functionality."""
    
    def test_unlimited_depth_impact(self):
        """Test performance impact for unlimited depth."""
        impact = estimate_performance_impact(-1)
        
        assert impact["impact_level"] == "high"
        assert impact["estimated_directories"] == "unlimited"
        assert impact["memory_usage"] == "high"
        assert impact["scan_time"] == "high"
        assert "specific depth limit" in impact["recommendation"]
    
    def test_low_depth_impact(self):
        """Test performance impact for low depths."""
        for depth in [1, 2, 3]:
            impact = estimate_performance_impact(depth)
            
            assert impact["impact_level"] == "low"
            assert isinstance(impact["estimated_directories"], int)
            assert impact["memory_usage"] == "low"
            assert impact["scan_time"] == "fast"
    
    def test_medium_depth_impact(self):
        """Test performance impact for medium depths."""
        for depth in [5, 7, 10]:
            impact = estimate_performance_impact(depth)
            
            assert impact["impact_level"] == "medium"
            assert isinstance(impact["estimated_directories"], int)
            assert impact["memory_usage"] == "medium"
            assert impact["scan_time"] == "moderate"
    
    def test_high_depth_impact(self):
        """Test performance impact for high depths."""
        for depth in [15, 20, 25]:
            impact = estimate_performance_impact(depth)
            
            assert impact["impact_level"] == "high"
            assert isinstance(impact["estimated_directories"], int)
            assert impact["memory_usage"] == "high"
            assert impact["scan_time"] == "slow"
    
    def test_performance_impact_with_custom_directory_count(self):
        """Test performance impact with custom directory count estimation."""
        depth = 5
        custom_dir_count = 10
        
        impact = estimate_performance_impact(depth, custom_dir_count)
        
        # 10^5 = 100,000 estimated directories
        assert impact["estimated_directories"] == custom_dir_count ** depth
        assert isinstance(impact["estimated_directories"], int)
    
    def test_performance_impact_capped_estimation(self):
        """Test that performance impact estimation is capped at reasonable limits."""
        depth = 10
        large_dir_count = 1000  # Would result in 1000^10 without capping
        
        impact = estimate_performance_impact(depth, large_dir_count)
        
        # Should be capped at 1,000,000
        assert impact["estimated_directories"] <= 1000000


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_boundary_depths(self):
        """Test validation at boundary values."""
        boundary_depths = [
            MIN_DEPTH,
            0,
            SHALLOW_DEPTH_THRESHOLD,
            DEEP_DEPTH_THRESHOLD,
            PERFORMANCE_WARNING_DEPTH,
            MAX_REASONABLE_DEPTH
        ]
        
        for depth in boundary_depths:
            result = validate_recursive_depth(depth)
            assert result.is_valid is True
            assert result.depth == depth
    
    def test_just_over_boundary_depths(self):
        """Test validation just over boundary values."""
        invalid_depths = [
            MIN_DEPTH - 1,
            MAX_REASONABLE_DEPTH + 1
        ]
        
        for depth in invalid_depths:
            result = validate_recursive_depth(depth)
            assert result.is_valid is False
    
    def test_empty_directory_structure_info(self):
        """Test recommendations with empty directory structure info."""
        recommendations = get_depth_recommendations({})
        
        # Should fall back to defaults
        assert recommendations["recommended_depth"] == 5
        assert "Default recommendation" in recommendations["reasoning"]
    
    def test_partial_directory_structure_info(self):
        """Test recommendations with partial directory structure info."""
        structure_info = {"max_depth": 3}  # Missing file_count and directory_count
        
        recommendations = get_depth_recommendations(structure_info)
        
        # Should work with partial info
        assert recommendations["recommended_depth"] == 3  # max_depth
        assert isinstance(recommendations["reasoning"], str)