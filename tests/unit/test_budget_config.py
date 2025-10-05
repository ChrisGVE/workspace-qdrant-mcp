"""
Unit tests for BudgetConfigManager.

Tests hierarchical budget resolution, validation, persistence, and runtime adjustment.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.python.common.core.context_injection.budget_config import (
    BudgetConfig,
    BudgetConfigManager,
    BudgetScope,
)
from src.python.common.core.context_injection.token_usage_tracker import (
    OperationType,
)


class TestBudgetConfig:
    """Test BudgetConfig dataclass validation."""

    def test_valid_config(self):
        """Test creating valid budget config."""
        config = BudgetConfig(
            scope=BudgetScope.GLOBAL,
            name="global",
            default_budget=50000,
            max_budget=200000,
            min_budget=1000,
        )

        assert config.scope == BudgetScope.GLOBAL
        assert config.name == "global"
        assert config.default_budget == 50000
        assert config.max_budget == 200000
        assert config.min_budget == 1000
        assert config.enabled is True

    def test_default_budget_below_min(self):
        """Test validation: default_budget must be >= min_budget."""
        with pytest.raises(ValueError, match="must be >= min_budget"):
            BudgetConfig(
                scope=BudgetScope.GLOBAL,
                name="global",
                default_budget=500,
                min_budget=1000,
            )

    def test_default_budget_above_max(self):
        """Test validation: default_budget must be <= max_budget."""
        with pytest.raises(ValueError, match="must be <= max_budget"):
            BudgetConfig(
                scope=BudgetScope.GLOBAL,
                name="global",
                default_budget=100000,
                max_budget=50000,
            )

    def test_max_budget_below_min(self):
        """Test validation: max_budget must be >= min_budget."""
        with pytest.raises(ValueError, match="max_budget .* must be >= min_budget"):
            BudgetConfig(
                scope=BudgetScope.GLOBAL,
                name="global",
                default_budget=5000,
                max_budget=500,
                min_budget=1000,
            )

    def test_operation_budgets(self):
        """Test tool config with operation budgets."""
        config = BudgetConfig(
            scope=BudgetScope.TOOL,
            name="claude",
            default_budget=100000,
            operation_budgets={
                "context_injection": 50000,
                "user_query": 50000,
            },
        )

        assert config.operation_budgets["context_injection"] == 50000
        assert config.operation_budgets["user_query"] == 50000


class TestBudgetConfigManager:
    """Test BudgetConfigManager functionality."""

    def test_initialization_with_defaults(self):
        """Test manager initializes with sensible defaults."""
        manager = BudgetConfigManager()

        # Check global default
        global_budget = manager.get_budget()
        assert global_budget == 50000

        # Check tool defaults
        claude_budget = manager.get_budget("claude")
        assert claude_budget == 100000

        codex_budget = manager.get_budget("codex")
        assert codex_budget == 4000

        gemini_budget = manager.get_budget("gemini")
        assert gemini_budget == 32000

    def test_hierarchical_resolution_global(self):
        """Test budget resolution falls back to global."""
        manager = BudgetConfigManager()

        # Unknown tool should use global budget
        budget = manager.get_budget("unknown_tool")
        assert budget == 50000

    def test_hierarchical_resolution_tool(self):
        """Test budget resolution uses tool-specific budget."""
        manager = BudgetConfigManager()

        # Claude tool should use tool-specific budget
        budget = manager.get_budget("claude")
        assert budget == 100000

    def test_hierarchical_resolution_operation(self):
        """Test budget resolution uses operation-specific budget."""
        manager = BudgetConfigManager()

        # Claude context_injection should use operation-specific budget
        budget = manager.get_budget("claude", OperationType.CONTEXT_INJECTION)
        assert budget == 50000

        # Claude user_query should use operation-specific budget
        budget = manager.get_budget("claude", OperationType.USER_QUERY)
        assert budget == 50000

    def test_set_global_budget(self):
        """Test setting global budget at runtime."""
        manager = BudgetConfigManager()

        manager.set_budget(budget=75000)
        budget = manager.get_budget()
        assert budget == 75000

    def test_set_tool_budget(self):
        """Test setting tool-specific budget at runtime."""
        manager = BudgetConfigManager()

        manager.set_budget("claude", budget=120000)
        budget = manager.get_budget("claude")
        assert budget == 120000

    def test_set_operation_budget(self):
        """Test setting operation-specific budget at runtime."""
        manager = BudgetConfigManager()

        manager.set_budget("claude", OperationType.SEARCH, budget=30000)
        budget = manager.get_budget("claude", OperationType.SEARCH)
        assert budget == 30000

    def test_set_budget_validates_minimum(self):
        """Test budget validation enforces minimum."""
        manager = BudgetConfigManager()

        with pytest.raises(ValueError, match="must be >= 1000"):
            manager.set_budget("claude", budget=500)

    def test_set_budget_validates_maximum(self):
        """Test budget validation enforces maximum."""
        manager = BudgetConfigManager()

        with pytest.raises(ValueError, match="exceeds global max"):
            manager.set_budget(budget=300000)  # Global max is 200000

    def test_set_budget_creates_new_tool(self):
        """Test setting budget for new tool creates config."""
        manager = BudgetConfigManager()

        manager.set_budget("new_tool", budget=25000)
        budget = manager.get_budget("new_tool")
        assert budget == 25000

    def test_get_all_budgets(self):
        """Test getting complete budget hierarchy."""
        manager = BudgetConfigManager()

        all_budgets = manager.get_all_budgets()

        # Check global
        assert all_budgets["global"]["default"] == 50000
        assert all_budgets["global"]["max"] == 200000

        # Check tools
        assert "claude" in all_budgets["tools"]
        assert all_budgets["tools"]["claude"]["default"] == 100000
        assert "operations" in all_budgets["tools"]["claude"]

    def test_validate_budget_valid(self):
        """Test budget validation accepts valid budgets."""
        manager = BudgetConfigManager()

        assert manager.validate_budget(50000) is True
        assert manager.validate_budget(100000, "claude") is True

    def test_validate_budget_below_minimum(self):
        """Test budget validation rejects budgets below minimum."""
        manager = BudgetConfigManager()

        assert manager.validate_budget(500) is False

    def test_validate_budget_above_maximum(self):
        """Test budget validation rejects budgets above maximum."""
        manager = BudgetConfigManager()

        assert manager.validate_budget(300000) is False  # Global max is 200000
        assert manager.validate_budget(300000, "claude") is False  # Claude max is 200000

    def test_get_config_global(self):
        """Test getting global config object."""
        manager = BudgetConfigManager()

        config = manager.get_config()
        assert config is not None
        assert config.scope == BudgetScope.GLOBAL
        assert config.name == "global"

    def test_get_config_tool(self):
        """Test getting tool config object."""
        manager = BudgetConfigManager()

        config = manager.get_config("claude")
        assert config is not None
        assert config.scope == BudgetScope.TOOL
        assert config.name == "claude"
        assert config.default_budget == 100000

    def test_get_config_nonexistent_tool(self):
        """Test getting config for nonexistent tool returns None."""
        manager = BudgetConfigManager()

        config = manager.get_config("nonexistent")
        assert config is None

    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults."""
        manager = BudgetConfigManager()

        # Modify configuration
        manager.set_budget(budget=75000)
        manager.set_budget("claude", budget=120000)

        # Reset
        manager.reset_to_defaults()

        # Check defaults restored
        assert manager.get_budget() == 50000
        assert manager.get_budget("claude") == 100000


class TestBudgetConfigPersistence:
    """Test budget configuration persistence (load/save)."""

    def test_load_from_file(self):
        """Test loading budget config from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "budget_config.yaml"

            # Create test config file
            config_data = {
                "token_budgets": {
                    "global": {
                        "default": 60000,
                        "max": 250000,
                        "min": 2000,
                    },
                    "tools": {
                        "claude": {
                            "default": 110000,
                            "max": 220000,
                            "operations": {
                                "context_injection": 55000,
                                "user_query": 55000,
                            },
                        },
                        "custom_tool": {
                            "default": 40000,
                            "max": 80000,
                        },
                    },
                }
            }

            with open(config_file, "w") as f:
                yaml.safe_dump(config_data, f)

            # Load config
            manager = BudgetConfigManager(config_file=config_file)

            # Verify loaded values
            assert manager.get_budget() == 60000
            assert manager.get_budget("claude") == 110000
            assert manager.get_budget("claude", OperationType.CONTEXT_INJECTION) == 55000
            assert manager.get_budget("custom_tool") == 40000

    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file uses defaults."""
        manager = BudgetConfigManager(config_file=Path("/nonexistent/file.yaml"))

        # Should still have defaults
        assert manager.get_budget() == 50000
        assert manager.get_budget("claude") == 100000

    def test_save_to_file(self):
        """Test saving budget config to YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "budget_config.yaml"

            # Create manager with custom budgets
            manager = BudgetConfigManager()
            manager.set_budget(budget=70000)
            manager.set_budget("claude", budget=130000)
            manager.set_budget("claude", OperationType.SEARCH, budget=35000)

            # Save to file
            manager.save_to_file(config_file)

            # Load and verify
            with open(config_file) as f:
                saved_data = yaml.safe_load(f)

            assert saved_data["token_budgets"]["global"]["default"] == 70000
            assert saved_data["token_budgets"]["tools"]["claude"]["default"] == 130000
            assert (
                saved_data["token_budgets"]["tools"]["claude"]["operations"]["search"]
                == 35000
            )

    def test_save_and_reload(self):
        """Test save/load roundtrip preserves configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "budget_config.yaml"

            # Create and save
            manager1 = BudgetConfigManager()
            manager1.set_budget(budget=80000)
            manager1.set_budget("claude", budget=140000)
            manager1.set_budget("new_tool", OperationType.FORMATTING, budget=20000)
            manager1.save_to_file(config_file)

            # Load into new manager
            manager2 = BudgetConfigManager(config_file=config_file)

            # Verify all budgets match
            assert manager2.get_budget() == 80000
            assert manager2.get_budget("claude") == 140000
            assert manager2.get_budget("new_tool", OperationType.FORMATTING) == 20000


class TestBudgetConfigIntegration:
    """Test integration with other budget management components."""

    def test_config_usage_pattern(self):
        """Test typical usage pattern for budget configuration."""
        config_manager = BudgetConfigManager()

        # Get budget for specific tool and operation
        claude_budget = config_manager.get_budget(
            "claude", OperationType.CONTEXT_INJECTION
        )

        # Verify budget can be used in application logic
        assert claude_budget == 50000
        assert isinstance(claude_budget, int)
        assert claude_budget > 0

    def test_runtime_budget_adjustment(self):
        """Test runtime budget adjustment workflow."""
        config_manager = BudgetConfigManager()

        # Initial budget
        initial = config_manager.get_budget("claude", OperationType.USER_QUERY)
        assert initial == 50000

        # Adjust at runtime
        config_manager.set_budget("claude", OperationType.USER_QUERY, budget=60000)

        # Verify adjustment
        adjusted = config_manager.get_budget("claude", OperationType.USER_QUERY)
        assert adjusted == 60000

        # Other operations unaffected
        context_budget = config_manager.get_budget(
            "claude", OperationType.CONTEXT_INJECTION
        )
        assert context_budget == 50000

    def test_multi_tool_configuration(self):
        """Test managing budgets for multiple tools."""
        manager = BudgetConfigManager()

        # Configure multiple tools
        tools_config = {
            "tool1": 30000,
            "tool2": 40000,
            "tool3": 50000,
        }

        for tool, budget in tools_config.items():
            manager.set_budget(tool, budget=budget)

        # Verify all configurations
        for tool, expected_budget in tools_config.items():
            assert manager.get_budget(tool) == expected_budget

    def test_operation_specific_budgets_per_tool(self):
        """Test operation-specific budgets for multiple tools."""
        manager = BudgetConfigManager()

        # Configure different operations per tool
        configs = [
            ("claude", OperationType.CONTEXT_INJECTION, 50000),
            ("claude", OperationType.USER_QUERY, 50000),
            ("codex", OperationType.CONTEXT_INJECTION, 2000),
            ("codex", OperationType.USER_QUERY, 2000),
            ("gemini", OperationType.CONTEXT_INJECTION, 16000),
            ("gemini", OperationType.USER_QUERY, 16000),
        ]

        for tool, op_type, budget in configs:
            manager.set_budget(tool, op_type, budget=budget)

        # Verify all configurations
        for tool, op_type, expected_budget in configs:
            assert manager.get_budget(tool, op_type) == expected_budget


class TestBudgetConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_config_file(self):
        """Test loading empty config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "empty.yaml"
            config_file.write_text("")

            manager = BudgetConfigManager(config_file=config_file)

            # Should use defaults
            assert manager.get_budget() == 50000

    def test_malformed_config_file(self):
        """Test loading malformed YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "malformed.yaml"
            config_file.write_text("invalid: yaml: content: [")

            with pytest.raises(yaml.YAMLError):
                BudgetConfigManager(config_file=config_file)

    def test_config_file_without_token_budgets(self):
        """Test config file without token_budgets section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "no_budgets.yaml"

            with open(config_file, "w") as f:
                yaml.safe_dump({"other_section": {"key": "value"}}, f)

            manager = BudgetConfigManager(config_file=config_file)

            # Should use defaults
            assert manager.get_budget() == 50000

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        import threading

        manager = BudgetConfigManager()
        results = []

        def get_budget():
            budget = manager.get_budget("claude")
            results.append(budget)

        def set_budget():
            manager.set_budget("claude", budget=150000)

        # Create threads
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=get_budget))
            threads.append(threading.Thread(target=set_budget))

        # Run threads
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All operations should complete without errors
        assert len(results) == 5

    def test_budget_boundaries(self):
        """Test budget at exact boundaries."""
        manager = BudgetConfigManager()

        # Minimum boundary
        manager.set_budget("test_tool", budget=1000)
        assert manager.get_budget("test_tool") == 1000

        # Maximum boundary
        manager.set_budget("test_tool", budget=200000)
        assert manager.get_budget("test_tool") == 200000

    def test_all_operation_types(self):
        """Test budgets for all operation types."""
        manager = BudgetConfigManager()

        for op_type in OperationType:
            budget = 10000 + (op_type.value.__hash__() % 10000)
            manager.set_budget("test_tool", op_type, budget=budget)

            retrieved = manager.get_budget("test_tool", op_type)
            assert retrieved == budget
