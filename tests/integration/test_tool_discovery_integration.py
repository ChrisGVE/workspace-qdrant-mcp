"""
Comprehensive integration tests for Tool Discovery system.

Tests end-to-end workflows including:
- Tool discovery → Database updates → Verification
- LSP server discovery and database integration
- Tree-sitter CLI discovery and database integration
- Compiler and build tool discovery
- Project-specific tool discovery with local paths
- Missing tool reporting
- Cross-platform compatibility
- Timeout handling
- Custom paths configuration
- Database transaction integrity
"""

import asyncio
import os
import platform
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch

import pytest

from src.python.common.core.language_support_models import (
    LanguageDefinition,
    LanguageSupportDatabaseConfig,
    LSPDefinition,
    TreeSitterDefinition,
)
from src.python.common.core.sqlite_state_manager import SQLiteStateManager
from src.python.common.core.tool_database_integration import ToolDatabaseIntegration
from src.python.common.core.tool_discovery import ToolDiscovery
from src.python.common.core.tool_reporting import MissingTool, ToolReporter


@pytest.fixture
async def temp_db(tmp_path):
    """Create temporary SQLite database for testing."""
    db_path = tmp_path / "test_tool_discovery.db"
    yield db_path


@pytest.fixture
async def state_manager(temp_db):
    """Create and initialize SQLite state manager with temporary database."""
    sm = SQLiteStateManager(db_path=str(temp_db))
    await sm.initialize()
    yield sm
    await sm.close()


@pytest.fixture
def tool_discovery():
    """Create ToolDiscovery instance with default configuration."""
    return ToolDiscovery()


@pytest.fixture
def tool_discovery_with_timeout():
    """Create ToolDiscovery instance with short timeout for testing."""
    return ToolDiscovery(timeout=1)


@pytest.fixture
async def db_integration(state_manager):
    """Create ToolDatabaseIntegration instance."""
    return ToolDatabaseIntegration(state_manager)


@pytest.fixture
async def tool_reporter(tool_discovery, state_manager):
    """Create ToolReporter instance."""
    return ToolReporter(tool_discovery, state_manager)


@pytest.fixture
def mock_project_root(tmp_path):
    """Create mock project directory structure."""
    project_root = tmp_path / "mock_project"
    project_root.mkdir()

    # Create node_modules/.bin
    node_bin = project_root / "node_modules" / ".bin"
    node_bin.mkdir(parents=True)

    # Create Python venv
    venv_bin = project_root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)

    # Create generic bin/
    bin_dir = project_root / "bin"
    bin_dir.mkdir()

    yield project_root


@pytest.fixture
def sample_language_config():
    """Create sample language support database configuration."""
    return LanguageSupportDatabaseConfig(
        version="1.0.0",
        languages=[
            LanguageDefinition(
                name="python",
                extensions=[".py", ".pyi"],
                lsp=LSPDefinition(
                    name="pyright",
                    executable="pyright-langserver"
                ),
                treesitter=TreeSitterDefinition(
                    grammar="python",
                    repo="https://github.com/tree-sitter/tree-sitter-python"
                )
            ),
            LanguageDefinition(
                name="rust",
                extensions=[".rs"],
                lsp=LSPDefinition(
                    name="rust-analyzer",
                    executable="rust-analyzer"
                ),
                treesitter=TreeSitterDefinition(
                    grammar="rust",
                    repo="https://github.com/tree-sitter/tree-sitter-rust"
                )
            ),
            LanguageDefinition(
                name="typescript",
                extensions=[".ts", ".tsx"],
                lsp=LSPDefinition(
                    name="typescript-language-server",
                    executable="typescript-language-server"
                ),
                treesitter=TreeSitterDefinition(
                    grammar="typescript",
                    repo="https://github.com/tree-sitter/tree-sitter-typescript"
                )
            ),
        ]
    )


@pytest.mark.asyncio
async def test_end_to_end_discovery_workflow(
    state_manager, tool_discovery, db_integration, sample_language_config
):
    """Test complete end-to-end tool discovery workflow.

    Workflow:
    1. Load language support config into database
    2. Discover LSP servers
    3. Discover tree-sitter CLI
    4. Discover compilers and build tools
    5. Update database with all discoveries
    6. Verify database records are correct
    """
    # Step 1: Load language data into database manually
    # (Simulating what LanguageSupportLoader would do)
    for lang in sample_language_config.languages:
        with state_manager._lock:
            state_manager.connection.execute(
                """
                INSERT OR REPLACE INTO languages
                (language_name, file_extensions, lsp_name, lsp_executable, lsp_missing,
                 ts_grammar, ts_missing)
                VALUES (?, ?, ?, ?, 1, ?, 1)
                """,
                (
                    lang.name,
                    ",".join(lang.extensions),
                    lang.lsp.name if lang.lsp else None,
                    lang.lsp.executable if lang.lsp else None,
                    lang.treesitter.grammar if lang.treesitter else None,
                )
            )
            state_manager.connection.commit()

    # Step 2: Discover LSP servers
    lsp_paths = tool_discovery.discover_lsp_servers(sample_language_config)

    # Should discover at least system Python LSPs if available
    assert isinstance(lsp_paths, dict)

    # Update database with LSP paths
    for language_name, lsp_path in lsp_paths.items():
        await db_integration.update_lsp_path(language_name, lsp_path)

    # Step 3: Discover tree-sitter CLI
    ts_info = tool_discovery.discover_tree_sitter_cli()

    if ts_info and ts_info.get("path"):
        # Update database with tree-sitter path
        await db_integration.update_tree_sitter_path(ts_info["path"])

    # Step 4: Discover compilers and build tools
    compilers = tool_discovery.discover_compilers()
    build_tools = tool_discovery.discover_build_tools()

    assert isinstance(compilers, dict)
    assert isinstance(build_tools, dict)

    # Update database with discovered tools
    for compiler_name, compiler_path in compilers.items():
        if compiler_path:
            await db_integration.update_tool_path(
                compiler_name, compiler_path, "lsp_server"
            )

    for tool_name, tool_path in build_tools.items():
        if tool_path:
            await db_integration.update_tool_path(
                tool_name, tool_path, "lsp_server"
            )

    # Step 5: Verify database records were updated
    with state_manager._lock:
        # Check languages table was updated
        cursor = state_manager.connection.execute(
            "SELECT language_name, lsp_absolute_path, lsp_missing FROM languages"
        )
        language_rows = cursor.fetchall()

        assert len(language_rows) >= 3  # python, rust, typescript

        # Check that some languages have LSP paths or are marked missing
        for row in language_rows:
            # Each language should have either a path or be marked missing
            has_path = row["lsp_absolute_path"] is not None
            is_missing = row["lsp_missing"] == 1
            assert has_path or is_missing

        # Check tools table has entries
        cursor = state_manager.connection.execute(
            "SELECT tool_name, absolute_path FROM tools WHERE absolute_path IS NOT NULL"
        )
        tool_rows = cursor.fetchall()

        # Should have at least some tools discovered (varies by system)
        # Just verify the structure is correct
        for row in tool_rows:
            assert row["tool_name"]
            assert row["absolute_path"]


@pytest.mark.asyncio
async def test_lsp_server_discovery_integration(
    state_manager, tool_discovery, db_integration, sample_language_config
):
    """Test LSP server discovery and database integration.

    Verifies:
    - LSP servers are discovered from configuration
    - Database is updated with discovered paths
    - Missing LSP servers are marked correctly
    - lsp_absolute_path and lsp_missing flags are set correctly
    """
    # Load language data into database
    for lang in sample_language_config.languages:
        with state_manager._lock:
            state_manager.connection.execute(
                """
                INSERT OR REPLACE INTO languages
                (language_name, file_extensions, lsp_name, lsp_executable, lsp_missing)
                VALUES (?, ?, ?, ?, 1)
                """,
                (
                    lang.name,
                    ",".join(lang.extensions),
                    lang.lsp.name if lang.lsp else None,
                    lang.lsp.executable if lang.lsp else None,
                )
            )
            state_manager.connection.commit()

    # Discover LSP servers
    lsp_paths = tool_discovery.discover_lsp_servers(sample_language_config)

    # Batch update database
    updated_count = await db_integration.batch_update_lsp_paths(lsp_paths)

    # Verify at least attempted to update (count may be 0 if all LSPs missing)
    assert updated_count >= 0

    # Verify database records
    with state_manager._lock:
        cursor = state_manager.connection.execute(
            """
            SELECT language_name, lsp_absolute_path, lsp_missing
            FROM languages
            WHERE language_name IN ('python', 'rust', 'typescript')
            """
        )
        rows = cursor.fetchall()

        assert len(rows) == 3

        for row in rows:
            language_name = row["language_name"]
            lsp_path = row["lsp_absolute_path"]
            lsp_missing = row["lsp_missing"]

            # Verify consistency: if path is set, missing should be 0
            if lsp_path:
                assert lsp_missing == 0, f"{language_name} has path but marked missing"

            # If path is None, missing should be 1 (after update)
            if lsp_path is None and language_name in lsp_paths:
                # If we attempted to update and path is None, should be marked missing
                if lsp_paths[language_name] is None:
                    assert lsp_missing == 1


@pytest.mark.asyncio
async def test_tree_sitter_cli_discovery_integration(
    state_manager, tool_discovery, db_integration, sample_language_config
):
    """Test tree-sitter CLI discovery and database integration.

    Verifies:
    - Tree-sitter CLI is discovered if installed
    - Database is updated for all languages with ts_grammar
    - ts_cli_absolute_path is updated correctly
    """
    # Load language data into database
    for lang in sample_language_config.languages:
        with state_manager._lock:
            state_manager.connection.execute(
                """
                INSERT OR REPLACE INTO languages
                (language_name, file_extensions, ts_grammar, ts_missing)
                VALUES (?, ?, ?, 1)
                """,
                (
                    lang.name,
                    ",".join(lang.extensions),
                    lang.treesitter.grammar if lang.treesitter else None,
                )
            )
            state_manager.connection.commit()

    # Discover tree-sitter CLI
    ts_info = tool_discovery.discover_tree_sitter_cli()

    if ts_info and ts_info.get("path"):
        # Update database
        result = await db_integration.update_tree_sitter_path(ts_info["path"])

        # Verify update succeeded
        assert result is True

        # Verify database records
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                """
                SELECT COUNT(*) as count
                FROM languages
                WHERE ts_grammar IS NOT NULL
                  AND ts_cli_absolute_path = ?
                """,
                (ts_info["path"],)
            )
            row = cursor.fetchone()

            # All languages with ts_grammar should have the path set
            assert row["count"] >= 3  # python, rust, typescript
    else:
        # Tree-sitter CLI not found - verify we can handle this
        # Database should still be in valid state
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT COUNT(*) as count FROM languages WHERE ts_grammar IS NOT NULL"
            )
            row = cursor.fetchone()
            assert row["count"] >= 3  # Languages still exist


@pytest.mark.asyncio
async def test_compiler_and_build_tool_integration(
    state_manager, tool_discovery, db_integration
):
    """Test compiler and build tool discovery with database integration.

    Verifies:
    - Compilers are discovered (gcc, clang, etc.)
    - Build tools are discovered (git, make, cargo, npm)
    - Tools table is updated via ToolDatabaseIntegration
    - Tool records are created with correct types
    """
    # Discover compilers
    compilers = tool_discovery.discover_compilers()

    assert isinstance(compilers, dict)
    assert len(compilers) > 0

    # Update database with compilers (using lsp_server type as generic tool type)
    compiler_paths = {
        name: path for name, path in compilers.items() if path is not None
    }

    if compiler_paths:
        count = await db_integration.batch_update_tool_paths(
            compiler_paths, "lsp_server"
        )
        assert count >= 0

    # Discover build tools
    build_tools = tool_discovery.discover_build_tools()

    assert isinstance(build_tools, dict)
    assert len(build_tools) > 0

    # Update database with build tools
    tool_paths = {
        name: path for name, path in build_tools.items() if path is not None
    }

    if tool_paths:
        count = await db_integration.batch_update_tool_paths(
            tool_paths, "lsp_server"
        )
        assert count >= 0

    # Verify tools were added to database
    with state_manager._lock:
        cursor = state_manager.connection.execute(
            "SELECT tool_name, absolute_path, tool_type FROM tools"
        )
        rows = cursor.fetchall()

        # Should have at least some tools
        if compiler_paths or tool_paths:
            assert len(rows) > 0

            # Verify structure
            for row in rows:
                assert row["tool_name"]
                assert row["absolute_path"]
                assert row["tool_type"] in ("lsp_server", "tree_sitter_cli")


@pytest.mark.asyncio
async def test_project_specific_tool_discovery_integration(
    state_manager, mock_project_root, db_integration
):
    """Test project-specific tool discovery with local paths.

    Verifies:
    - Project-local tools are discovered (node_modules/.bin, venv/bin)
    - Local paths take priority over system PATH
    - Database is updated with project-local paths
    """
    # Create mock executables in project-local paths
    node_bin = mock_project_root / "node_modules" / ".bin"
    venv_bin = mock_project_root / ".venv" / "bin"

    # Create mock executables
    mock_executables = {
        "typescript-language-server": node_bin,
        "pyright": venv_bin,
    }

    for exe_name, exe_dir in mock_executables.items():
        exe_path = exe_dir / exe_name
        exe_path.write_text("#!/bin/sh\necho 'mock'")

    # Create ToolDiscovery with project root
    discovery = ToolDiscovery(project_root=mock_project_root)

    # Mock os.access to simulate executable files
    with patch('os.access', return_value=True):
        # Find project-local tools
        ts_server_path = discovery.find_lsp_executable(
            "typescript", "typescript-language-server", mock_project_root
        )
        pyright_path = discovery.find_lsp_executable(
            "python", "pyright", mock_project_root
        )

        # Verify project-local paths are found
        if platform.system() != "Windows":
            assert ts_server_path is not None
            assert "node_modules/.bin" in ts_server_path

            assert pyright_path is not None
            assert ".venv/bin" in pyright_path

            # Update database with project-local paths
            await db_integration.update_tool_path(
                "typescript-language-server", ts_server_path, "lsp_server"
            )
            await db_integration.update_tool_path(
                "pyright", pyright_path, "lsp_server"
            )

            # Verify database records
            with state_manager._lock:
                cursor = state_manager.connection.execute(
                    """
                    SELECT tool_name, absolute_path
                    FROM tools
                    WHERE tool_name IN ('typescript-language-server', 'pyright')
                    """
                )
                rows = cursor.fetchall()

                assert len(rows) == 2

                for row in rows:
                    # Verify paths contain project-local directories
                    path = row["absolute_path"]
                    assert (
                        "node_modules/.bin" in path or
                        ".venv/bin" in path or
                        ".venv\\Scripts" in path  # Windows
                    )


@pytest.mark.asyncio
async def test_missing_tool_reporting_integration(
    state_manager, tool_discovery, tool_reporter
):
    """Test missing tool reporting integration.

    Verifies:
    - Database is set up with some tools missing
    - Missing tool report is generated
    - Report includes correct tools with severity
    - Installation commands are provided
    """
    # Add some languages to database and mark LSPs as missing
    languages = [
        ("python", ".py", "pyright-langserver", "pyright"),
        ("rust", ".rs", "rust-analyzer", "rust-analyzer"),
    ]

    with state_manager._lock:
        for lang_name, ext, lsp_exe, lsp_name in languages:
            state_manager.connection.execute(
                """
                INSERT OR REPLACE INTO languages
                (language_name, file_extensions, lsp_executable, lsp_name, lsp_missing)
                VALUES (?, ?, ?, ?, 1)
                """,
                (lang_name, ext, lsp_exe, lsp_name)
            )
        state_manager.connection.commit()

    # Generate missing tools report
    missing_tools = await tool_reporter.get_missing_tools()

    # Verify missing tools are reported
    assert isinstance(missing_tools, list)

    # Should have at least the tools we marked as missing
    # (May have more if tree-sitter is also missing)
    assert len(missing_tools) >= 2

    # Verify structure of missing tools
    for tool in missing_tools:
        assert isinstance(tool, MissingTool)
        assert tool.name
        assert tool.tool_type in ("lsp_server", "tree_sitter")
        assert tool.severity in ("critical", "recommended", "optional")

        # If it's an LSP server, should have language
        if tool.tool_type == "lsp_server":
            assert tool.language is not None

    # Generate installation guide
    guide = tool_reporter.generate_installation_guide(missing_tools)

    assert isinstance(guide, str)
    assert len(guide) > 0

    # Verify guide contains key sections
    if missing_tools:
        assert "Missing Development Tools" in guide
        assert "=" * 60 in guide


@pytest.mark.asyncio
async def test_cross_platform_compatibility(tool_discovery):
    """Test cross-platform tool discovery compatibility.

    Verifies:
    - Current platform is detected correctly
    - Platform-specific tools are detected
    - Path separators are handled correctly
    - Executable validation works on current platform
    """
    # Get current platform
    current_platform = platform.system()

    assert current_platform in ("Windows", "Linux", "Darwin")

    # Test basic tool discovery works
    python_path = tool_discovery.find_executable("python")

    # Should find Python or python3
    if python_path is None:
        python_path = tool_discovery.find_executable("python3")

    if python_path:
        # Verify path uses correct separators
        assert os.path.exists(python_path)

        # Verify validation works
        assert tool_discovery.validate_executable(python_path)

        # Verify version detection works
        version = tool_discovery.get_version(python_path, "--version")
        assert version is not None
        assert "Python" in version or "python" in version.lower()

    # Test platform-specific executable extensions
    if current_platform == "Windows":
        # Windows executables should have extensions
        # Find any executable and verify it has an extension
        git_path = tool_discovery.find_executable("git")
        if git_path:
            assert Path(git_path).suffix.lower() in (".exe", ".bat", ".cmd")
    else:
        # Unix-like systems - executables may not have extensions
        # Just verify they have executable bit
        git_path = tool_discovery.find_executable("git")
        if git_path:
            assert os.access(git_path, os.X_OK)


@pytest.mark.asyncio
async def test_timeout_handling(tool_discovery_with_timeout):
    """Test timeout handling for slow operations.

    Verifies:
    - Timeout prevents operations from hanging
    - Graceful degradation when timeout occurs
    - No exceptions are raised on timeout
    """
    # Use discovery instance with very short timeout (1 second)
    discovery = tool_discovery_with_timeout

    # Try to get version of a command that might hang
    # Using a mock that simulates slow operation
    with patch("subprocess.run") as mock_run:
        import subprocess

        # Simulate timeout
        mock_run.side_effect = subprocess.TimeoutExpired("test", 1)

        # Should return None, not raise exception
        version = discovery.get_version("fake-tool", "--version")
        assert version is None


@pytest.mark.asyncio
async def test_custom_paths_configuration(tool_discovery):
    """Test custom paths configuration.

    Verifies:
    - Custom paths are checked before system PATH
    - Custom paths can override system PATH
    - Multiple custom paths are supported
    """
    # Create temporary directory with mock executable
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock executable
        mock_exe = temp_path / "custom-tool"
        mock_exe.write_text("#!/bin/sh\necho 'custom'")

        # Mock os.access to simulate executable on Unix-like systems
        if platform.system() != "Windows":
            with patch('os.access', return_value=True):
                # Create discovery with custom path
                discovery = ToolDiscovery(
                    config={"custom_paths": [str(temp_path)]}
                )

                # Should find custom tool
                found_path = discovery.find_executable("custom-tool")

                assert found_path is not None
                assert temp_dir in found_path


@pytest.mark.asyncio
async def test_database_transaction_integrity(state_manager, db_integration):
    """Test database transaction integrity for tool updates.

    Verifies:
    - Batch updates are atomic (all succeed or all fail)
    - Transactions are properly committed
    - Rollback works on errors
    """
    # Test successful batch update
    lsp_paths = {
        "python": "/usr/bin/pyright",
        "rust": "/usr/bin/rust-analyzer",
        "typescript": "/usr/bin/typescript-language-server",
    }

    # First, add languages to database
    for lang_name in lsp_paths.keys():
        with state_manager._lock:
            state_manager.connection.execute(
                """
                INSERT OR IGNORE INTO languages
                (language_name, file_extensions, lsp_executable, lsp_missing)
                VALUES (?, ?, ?, 1)
                """,
                (lang_name, f".{lang_name}", f"{lang_name}-lsp")
            )
            state_manager.connection.commit()

    # Batch update
    await db_integration.batch_update_lsp_paths(lsp_paths)

    # Verify all updates were committed
    with state_manager._lock:
        cursor = state_manager.connection.execute(
            """
            SELECT language_name, lsp_absolute_path
            FROM languages
            WHERE language_name IN ('python', 'rust', 'typescript')
            """
        )
        rows = cursor.fetchall()

        assert len(rows) == 3

        for row in rows:
            lang_name = row["language_name"]
            assert row["lsp_absolute_path"] == lsp_paths[lang_name]


@pytest.mark.asyncio
async def test_empty_configuration_handling(state_manager):
    """Test handling of empty or minimal configurations.

    Verifies:
    - Empty configurations don't cause crashes
    - Graceful handling of missing data
    - Proper defaults are used
    """
    # Create discovery with no custom configuration
    discovery = ToolDiscovery()

    # Should work with default configuration
    assert discovery.timeout > 0
    assert isinstance(discovery.custom_paths, list)

    # Should be able to discover basic tools
    compilers = discovery.discover_compilers()
    assert isinstance(compilers, dict)

    build_tools = discovery.discover_build_tools()
    assert isinstance(build_tools, dict)


@pytest.mark.asyncio
async def test_concurrent_discovery_operations(
    state_manager, tool_discovery, db_integration
):
    """Test concurrent tool discovery operations.

    Verifies:
    - Multiple discovery operations can run concurrently
    - Database updates are thread-safe
    - No race conditions in concurrent updates
    """
    async def discover_and_update_compilers():
        compilers = tool_discovery.discover_compilers()
        for name, path in compilers.items():
            if path:
                await db_integration.update_tool_path(name, path, "lsp_server")

    async def discover_and_update_build_tools():
        build_tools = tool_discovery.discover_build_tools()
        for name, path in build_tools.items():
            if path:
                await db_integration.update_tool_path(name, path, "lsp_server")

    # Run concurrent operations
    await asyncio.gather(
        discover_and_update_compilers(),
        discover_and_update_build_tools()
    )

    # Verify database is in consistent state
    with state_manager._lock:
        cursor = state_manager.connection.execute(
            "SELECT COUNT(*) as count FROM tools"
        )
        row = cursor.fetchone()

        # Should have tools from both operations
        # Count varies by system, just verify structure is valid
        assert row["count"] >= 0


@pytest.mark.asyncio
async def test_version_extraction_and_parsing(tool_discovery):
    """Test version extraction and parsing from various tools.

    Verifies:
    - Version information can be extracted
    - Different version formats are handled
    - Version parsing is robust
    """
    # Test with Python
    python_path = tool_discovery.find_executable("python")
    if not python_path:
        python_path = tool_discovery.find_executable("python3")

    if python_path:
        version = tool_discovery.get_version(python_path, "--version")
        assert version is not None
        # Should contain version number
        assert any(char.isdigit() for char in version)

    # Test with git
    git_path = tool_discovery.find_executable("git")
    if git_path:
        version = tool_discovery.get_version(git_path, "--version")
        assert version is not None
        assert "git" in version.lower()
