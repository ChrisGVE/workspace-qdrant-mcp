
from ...observability import get_logger
logger = get_logger(__name__)
"""Web interface for memory curation."""

from .server import create_web_app

__all__ = ["create_web_app"]
