"""Compatibility wrapper for common.utils.config_validator."""

from common.core.config import Config
from common.utils import config_validator as _impl

QdrantClient = _impl.QdrantClient
EmbeddingService = _impl.EmbeddingService
ProjectDetector = _impl.ProjectDetector


class ConfigValidator(_impl.ConfigValidator):
    """ConfigValidator wrapper that keeps patch paths stable."""

    def validate_qdrant_connection(self):
        _impl.QdrantClient = QdrantClient
        return super().validate_qdrant_connection()

    def validate_embedding_model(self):
        _impl.EmbeddingService = EmbeddingService
        return super().validate_embedding_model()

    def validate_project_detection(self):
        _impl.ProjectDetector = ProjectDetector
        return super().validate_project_detection()

    def validate_all(self):
        config_issues = []
        try:
            config_issues = Config.validate_config(self.config)
        except Exception:
            config_issues = []

        qdrant_valid, qdrant_message = self.validate_qdrant_connection()
        embed_valid, embed_message = self.validate_embedding_model()
        project_valid, project_message = self.validate_project_detection()

        issues = []
        warnings = []

        if not qdrant_valid:
            issues.append(qdrant_message)
        if not embed_valid:
            issues.append(embed_message)
        if not project_valid:
            warnings.append(project_message)
        issues.extend(config_issues or [])

        results = {
            "qdrant_connection": {"valid": qdrant_valid, "message": qdrant_message},
            "embedding_model": {"valid": embed_valid, "message": embed_message},
            "project_detection": {"valid": project_valid, "message": project_message},
            "issues": issues,
            "warnings": warnings,
            "suggestions": [],
        }

        return len(issues) == 0, results


__all__ = [
    "ConfigValidator",
    "QdrantClient",
    "EmbeddingService",
    "ProjectDetector",
]
