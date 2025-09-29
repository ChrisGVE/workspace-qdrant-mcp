"""Stub for yaml_config - backward compatibility shim for daemon_client.

This module provides backward compatibility for code that was written against
the old yaml_config-based configuration system. It wraps the new lua-style
ConfigManager to provide the old interface.

TODO: Update daemon_client.py to use ConfigManager directly and remove this stub.
"""

from typing import Any, Dict, Optional
from .config import get_config_manager, ConfigManager


class WorkspaceConfig:
    """Backward compatibility shim for old WorkspaceConfig class.

    This wraps the new ConfigManager to provide the old interface that
    daemon_client.py expects.
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize workspace config.

        Args:
            config_manager: Optional ConfigManager instance
        """
        self._config = config_manager or get_config_manager()

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            path: Configuration path (dot-separated)
            default: Default value if not found

        Returns:
            Configuration value
        """
        return self._config.get(path, default)

    def __getattr__(self, name: str) -> Any:
        """Get configuration attribute.

        Args:
            name: Attribute name

        Returns:
            Configuration value
        """
        # Map old attribute names to new config paths
        # Add mappings as needed
        attribute_map = {
            "qdrant_url": "qdrant.url",
            "qdrant_api_key": "qdrant.api_key",
            "grpc_host": "grpc.host",
            "grpc_port": "grpc.port",
            "workspace_collection": "workspace.collection_basename",
        }

        if name in attribute_map:
            return self.get(attribute_map[name])

        # Try direct path
        return self.get(name)


def load_config(config_file: Optional[str] = None) -> WorkspaceConfig:
    """Load workspace configuration.

    Args:
        config_file: Optional path to configuration file

    Returns:
        WorkspaceConfig instance
    """
    if config_file:
        config_manager = get_config_manager(config_file)
    else:
        config_manager = get_config_manager()

    return WorkspaceConfig(config_manager)