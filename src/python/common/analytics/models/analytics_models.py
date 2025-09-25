"""
Data models for analytics operations.

This module defines Pydantic models for analytics data structures,
metrics, patterns, and insights used throughout the system.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class MetricType(str, Enum):
    """Types of analytics metrics."""
    PERFORMANCE = "performance"
    USAGE = "usage"
    SEARCH = "search"
    DOCUMENT = "document"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM = "system"


class SeverityLevel(str, Enum):
    """Severity levels for alerts and anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternType(str, Enum):
    """Types of patterns detected in data."""
    TREND = "trend"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"


class AnalyticsMetrics(BaseModel):
    """Base model for analytics metrics."""

    metric_id: str = Field(..., description="Unique identifier for the metric")
    metric_type: MetricType = Field(..., description="Type of metric")
    timestamp: datetime = Field(default_factory=datetime.now, description="When metric was recorded")
    value: Union[float, int, str] = Field(..., description="Metric value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchPattern(BaseModel):
    """Model for search pattern analysis."""

    pattern_id: str = Field(..., description="Unique pattern identifier")
    query_text: str = Field(..., description="Search query text")
    query_type: str = Field(..., description="Type of search query (semantic, keyword, hybrid)")
    frequency: int = Field(..., description="Number of times pattern occurred")
    avg_response_time: float = Field(..., description="Average response time in milliseconds")
    success_rate: float = Field(..., ge=0, le=1, description="Success rate (0-1)")
    avg_results_count: float = Field(..., description="Average number of results returned")
    collections_accessed: List[str] = Field(default_factory=list, description="Collections searched")
    time_period: Dict[str, datetime] = Field(..., description="Start and end time of pattern")
    pattern_strength: float = Field(..., ge=0, le=1, description="Strength of the pattern (0-1)")

    @validator('time_period')
    def validate_time_period(cls, v):
        """Validate time period has start and end."""
        if 'start' not in v or 'end' not in v:
            raise ValueError("time_period must have 'start' and 'end' keys")
        return v


class DocumentInsight(BaseModel):
    """Model for document analysis insights."""

    insight_id: str = Field(..., description="Unique insight identifier")
    document_id: Optional[str] = Field(None, description="Document ID if insight is document-specific")
    collection_name: str = Field(..., description="Collection containing the document")
    insight_type: str = Field(..., description="Type of insight (content, structure, metadata)")
    summary: str = Field(..., description="Summary of the insight")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed insight data")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in the insight (0-1)")
    created_at: datetime = Field(default_factory=datetime.now, description="When insight was generated")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class PerformanceMetric(BaseModel):
    """Model for system performance metrics."""

    metric_name: str = Field(..., description="Name of the performance metric")
    component: str = Field(..., description="System component being measured")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(default_factory=datetime.now, description="When metric was recorded")
    threshold: Optional[float] = Field(None, description="Alert threshold if applicable")
    is_anomaly: bool = Field(default=False, description="Whether this reading is anomalous")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class AnomalyAlert(BaseModel):
    """Model for anomaly detection alerts."""

    alert_id: str = Field(..., description="Unique alert identifier")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed alert description")
    severity: SeverityLevel = Field(..., description="Severity level of the alert")
    component: str = Field(..., description="System component affected")
    metric_name: str = Field(..., description="Metric that triggered the alert")
    actual_value: float = Field(..., description="Actual value that triggered alert")
    expected_value: Optional[float] = Field(None, description="Expected value if known")
    threshold: Optional[float] = Field(None, description="Threshold that was exceeded")
    detected_at: datetime = Field(default_factory=datetime.now, description="When anomaly was detected")
    resolved_at: Optional[datetime] = Field(None, description="When anomaly was resolved")
    is_resolved: bool = Field(default=False, description="Whether anomaly is resolved")
    additional_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class SearchAnalytics(BaseModel):
    """Comprehensive search analytics data."""

    total_searches: int = Field(..., description="Total number of searches")
    unique_queries: int = Field(..., description="Number of unique queries")
    avg_response_time: float = Field(..., description="Average response time in ms")
    search_success_rate: float = Field(..., ge=0, le=1, description="Success rate (0-1)")
    top_queries: List[Dict[str, Any]] = Field(default_factory=list, description="Most frequent queries")
    query_types: Dict[str, int] = Field(default_factory=dict, description="Distribution of query types")
    collections_usage: Dict[str, int] = Field(default_factory=dict, description="Collection access frequency")
    time_distribution: Dict[str, int] = Field(default_factory=dict, description="Search frequency by hour")
    performance_trends: Dict[str, List[float]] = Field(default_factory=dict, description="Performance over time")


class DocumentAnalytics(BaseModel):
    """Document processing and content analytics."""

    total_documents: int = Field(..., description="Total number of documents")
    documents_by_type: Dict[str, int] = Field(default_factory=dict, description="Distribution by file type")
    total_size_bytes: int = Field(..., description="Total size of all documents")
    avg_document_size: float = Field(..., description="Average document size in bytes")
    processing_stats: Dict[str, float] = Field(default_factory=dict, description="Processing time statistics")
    content_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Detected content patterns")
    language_distribution: Dict[str, int] = Field(default_factory=dict, description="Language distribution")
    update_frequency: Dict[str, int] = Field(default_factory=dict, description="Document update patterns")


class UserBehaviorPattern(BaseModel):
    """User behavior pattern analysis."""

    pattern_type: PatternType = Field(..., description="Type of pattern detected")
    description: str = Field(..., description="Description of the behavior pattern")
    frequency: int = Field(..., description="How often pattern occurs")
    duration: float = Field(..., description="Average duration of the pattern")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in pattern detection")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence supporting the pattern")
    time_periods: List[Dict[str, datetime]] = Field(default_factory=list, description="Time periods when pattern occurs")
    impact_score: float = Field(..., ge=0, le=1, description="Impact score of the pattern")


class SystemPerformance(BaseModel):
    """Overall system performance analytics."""

    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    network_io: Dict[str, float] = Field(default_factory=dict, description="Network I/O statistics")
    qdrant_performance: Dict[str, float] = Field(default_factory=dict, description="Qdrant-specific metrics")
    concurrent_operations: int = Field(..., description="Number of concurrent operations")
    error_rate: float = Field(..., ge=0, description="System error rate")
    uptime_hours: float = Field(..., description="System uptime in hours")
    timestamp: datetime = Field(default_factory=datetime.now, description="When metrics were recorded")