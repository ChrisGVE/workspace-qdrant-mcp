
"""Minimal collections test."""
import pytest
from unittest.mock import Mock
from src.python.common.core.collections import CollectionConfig

def test_collection_config():
    config = CollectionConfig(
        name="test", description="test", collection_type="test"
    )
    assert config.name == "test"
