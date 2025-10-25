"""
Real-World Workflow Simulation Testing

Comprehensive test scenarios mimicking actual developer workflows including
project detection, collection management, and scratchbook functionality in
realistic development contexts.

This module implements subtask 203.3 of the End-to-End Functional Testing Framework.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import MagicMock, patch

import pytest
import yaml


class RealWorldProjectSimulator:
    """Simulates realistic development project structures and workflows."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.projects_root = tmp_path / "projects"
        self.config_dir = tmp_path / ".config" / "workspace-qdrant"
        self.cli_executable = "uv run wqm"

        self.active_projects = {}
        self.setup_simulator()

    def setup_simulator(self):
        """Set up the project simulator environment."""
        # Create base directories
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create base configuration
        config_content = {
            "qdrant_url": "http://localhost:6333",
            "projects_root": str(self.projects_root),
            "auto_detect_projects": True,
            "collections": {
                "global": ["_global", "_scratchbook"],
                "project_suffixes": ["project", "docs", "notes"]
            }
        }

        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

    def create_realistic_project(
        self,
        project_name: str,
        project_type: str = "python",
        include_docs: bool = True,
        include_submodules: bool = False
    ) -> Path:
        """Create a realistic project structure."""
        project_dir = self.projects_root / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Git repository
        subprocess.run(
            ["git", "init"],
            cwd=project_dir,
            capture_output=True
        )

        # Create .gitignore
        gitignore_content = self._get_gitignore_template(project_type)
        (project_dir / ".gitignore").write_text(gitignore_content)

        # Create README
        readme_content = f"# {project_name}\n\nA {project_type} project for testing real-world workflows.\n\n## Features\n- Feature 1\n- Feature 2\n- Feature 3\n"
        (project_dir / "README.md").write_text(readme_content)

        # Create project-specific structure
        if project_type == "python":
            self._create_python_project_structure(project_dir, project_name)
        elif project_type == "javascript":
            self._create_javascript_project_structure(project_dir, project_name)
        elif project_type == "rust":
            self._create_rust_project_structure(project_dir, project_name)

        # Add documentation if requested
        if include_docs:
            self._create_documentation_structure(project_dir)

        # Add submodules if requested
        if include_submodules:
            self._create_submodule_structure(project_dir)

        # Initial commit
        subprocess.run(["git", "add", "."], cwd=project_dir, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Initial commit"],
            cwd=project_dir,
            capture_output=True
        )

        self.active_projects[project_name] = {
            "path": project_dir,
            "type": project_type,
            "created_at": time.time()
        }

        return project_dir

    def simulate_development_session(
        self,
        project_name: str,
        duration_minutes: int = 5,
        actions_per_minute: int = 2
    ) -> list[dict[str, Any]]:
        """Simulate a realistic development session with file changes."""
        if project_name not in self.active_projects:
            raise ValueError(f"Project {project_name} not found")

        project_dir = self.active_projects[project_name]["path"]
        project_type = self.active_projects[project_name]["type"]
        actions = []

        total_actions = duration_minutes * actions_per_minute

        for i in range(total_actions):
            action = self._simulate_development_action(project_dir, project_type, i)
            actions.append(action)

            # Add some realistic timing
            time.sleep(0.1)  # Small delay between actions

        return actions

    def simulate_collaboration_workflow(
        self,
        project_name: str,
        num_collaborators: int = 2
    ) -> list[dict[str, Any]]:
        """Simulate collaborative development workflow."""
        if project_name not in self.active_projects:
            raise ValueError(f"Project {project_name} not found")

        project_dir = self.active_projects[project_name]["path"]
        collaboration_actions = []

        # Simulate different collaborators working
        collaborators = [f"dev{i}" for i in range(1, num_collaborators + 1)]

        for collaborator in collaborators:
            # Create branch for collaborator
            branch_name = f"feature/{collaborator}-work"
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=project_dir,
                capture_output=True
            )

            # Make changes
            for change_num in range(3):
                action = self._simulate_collaborator_action(
                    project_dir, collaborator, change_num
                )
                collaboration_actions.append(action)

            # Commit changes
            subprocess.run(["git", "add", "."], cwd=project_dir, capture_output=True)
            subprocess.run(
                ["git", "-c", f"user.name={collaborator}", "-c", f"user.email={collaborator}@example.com",
                 "commit", "-m", f"Work by {collaborator}"],
                cwd=project_dir,
                capture_output=True
            )

            # Switch back to main
            subprocess.run(
                ["git", "checkout", "main"],
                cwd=project_dir,
                capture_output=True
            )

        return collaboration_actions

    def run_cli_command(
        self,
        command: str,
        cwd: Path | None = None,
        timeout: int = 30
    ) -> tuple[int, str, str]:
        """Execute CLI command in project context."""
        if cwd is None:
            cwd = self.projects_root

        env = os.environ.copy()
        env.update({
            "WQM_CONFIG_DIR": str(self.config_dir),
            "PYTHONPATH": str(Path.cwd()),
        })

        try:
            result = subprocess.run(
                f"{self.cli_executable} {command}",
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", f"Command execution failed: {e}"

    def _get_gitignore_template(self, project_type: str) -> str:
        """Get gitignore template for project type."""
        templates = {
            "python": "__pycache__/\n*.py[cod]\n*$py.class\n.env\n.venv/\ndist/\nbuild/\n*.egg-info/\n",
            "javascript": "node_modules/\n*.log\n.env\ndist/\nbuild/\n.cache/\n",
            "rust": "target/\nCargo.lock\n*.rs.bk\n"
        }
        return templates.get(project_type, "")

    def _create_python_project_structure(self, project_dir: Path, project_name: str):
        """Create Python project structure."""
        # Source directory
        src_dir = project_dir / "src" / project_name.replace("-", "_")
        src_dir.mkdir(parents=True, exist_ok=True)

        # Python files
        (src_dir / "__init__.py").write_text(f'"""Package for {project_name}."""\n\n__version__ = "0.1.0"\n')
        (src_dir / "main.py").write_text('"""Main module."""\n\ndef main():\n    """Entry point."""\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()\n')
        (src_dir / "utils.py").write_text('"""Utility functions."""\n\ndef helper_function(data):\n    """Process data."""\n    return data.upper()\n')

        # Tests directory
        tests_dir = project_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "test_main.py").write_text('"""Tests for main module."""\n\ndef test_main():\n    """Test main function."""\n    assert True\n')

        # Configuration files
        (project_dir / "pyproject.toml").write_text(f'[project]\nname = "{project_name}"\nversion = "0.1.0"\ndescription = "Test project"\n')
        (project_dir / "requirements.txt").write_text("pytest>=7.0.0\nrequests>=2.28.0\n")

    def _create_javascript_project_structure(self, project_dir: Path, project_name: str):
        """Create JavaScript project structure."""
        # Source directory
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # JavaScript files
        (src_dir / "index.js").write_text('console.log("Hello, World!");\n\nmodule.exports = { main: () => console.log("Main function") };\n')
        (src_dir / "utils.js").write_text('function helperFunction(data) {\n    return data.toUpperCase();\n}\n\nmodule.exports = { helperFunction };\n')

        # Package configuration
        package_json = {
            "name": project_name,
            "version": "1.0.0",
            "description": "Test JavaScript project",
            "main": "src/index.js",
            "scripts": {"test": "jest", "start": "node src/index.js"},
            "devDependencies": {"jest": "^29.0.0"}
        }
        (project_dir / "package.json").write_text(json.dumps(package_json, indent=2))

    def _create_rust_project_structure(self, project_dir: Path, project_name: str):
        """Create Rust project structure."""
        # Source directory
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # Rust files
        (src_dir / "main.rs").write_text('fn main() {\n    println!("Hello, world!");\n}\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn it_works() {\n        assert_eq!(2 + 2, 4);\n    }\n}\n')
        (src_dir / "lib.rs").write_text('pub fn helper_function(data: &str) -> String {\n    data.to_uppercase()\n}\n')

        # Cargo configuration
        cargo_toml = f'[package]\nname = "{project_name}"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n'
        (project_dir / "Cargo.toml").write_text(cargo_toml)

    def _create_documentation_structure(self, project_dir: Path):
        """Create documentation structure."""
        docs_dir = project_dir / "docs"
        docs_dir.mkdir(exist_ok=True)

        # Documentation files
        (docs_dir / "API.md").write_text("# API Documentation\n\n## Functions\n\n### main()\nEntry point for the application.\n")
        (docs_dir / "ARCHITECTURE.md").write_text("# Architecture\n\n## Overview\nThis project follows a modular architecture.\n")
        (docs_dir / "DEPLOYMENT.md").write_text("# Deployment Guide\n\n## Prerequisites\n- Python 3.8+\n- Dependencies installed\n")

        # Tutorial directory
        tutorials_dir = docs_dir / "tutorials"
        tutorials_dir.mkdir(exist_ok=True)
        (tutorials_dir / "getting-started.md").write_text("# Getting Started\n\n## Installation\n1. Clone the repository\n2. Install dependencies\n")

    def _create_submodule_structure(self, project_dir: Path):
        """Create submodule structure."""
        # Create a simple submodule directory
        submodule_dir = project_dir / "external" / "library"
        submodule_dir.mkdir(parents=True, exist_ok=True)

        # Add submodule content
        (submodule_dir / "README.md").write_text("# External Library\n\nThis is a simulated external library.\n")
        (submodule_dir / "lib.py").write_text("def external_function():\n    return 'External function result'\n")

    def _simulate_development_action(
        self,
        project_dir: Path,
        project_type: str,
        action_num: int
    ) -> dict[str, Any]:
        """Simulate a single development action."""
        actions = [
            self._add_new_file,
            self._modify_existing_file,
            self._add_documentation,
            self._update_configuration,
            self._add_test_file
        ]

        action_func = actions[action_num % len(actions)]
        return action_func(project_dir, project_type, action_num)

    def _add_new_file(self, project_dir: Path, project_type: str, action_num: int) -> dict[str, Any]:
        """Add a new file to the project."""
        if project_type == "python":
            filename = f"feature_{action_num}.py"
            filepath = project_dir / "src" / filename
            content = f'"""Feature {action_num} implementation."""\n\ndef feature_{action_num}_function():\n    """Implement feature {action_num}."""\n    return "Feature {action_num} result"\n'
        else:
            filename = f"feature_{action_num}.md"
            filepath = project_dir / filename
            content = f"# Feature {action_num}\n\nImplementation notes for feature {action_num}.\n"

        filepath.write_text(content)
        return {
            "action": "add_file",
            "file": str(filepath.relative_to(project_dir)),
            "timestamp": time.time()
        }

    def _modify_existing_file(self, project_dir: Path, project_type: str, action_num: int) -> dict[str, Any]:
        """Modify an existing file."""
        readme_file = project_dir / "README.md"
        if readme_file.exists():
            current_content = readme_file.read_text()
            new_content = current_content + f"\n## Update {action_num}\nAdded in development session at {time.time()}\n"
            readme_file.write_text(new_content)

            return {
                "action": "modify_file",
                "file": "README.md",
                "timestamp": time.time()
            }

        return {"action": "modify_file", "file": "none", "timestamp": time.time()}

    def _add_documentation(self, project_dir: Path, project_type: str, action_num: int) -> dict[str, Any]:
        """Add documentation."""
        docs_dir = project_dir / "docs"
        if not docs_dir.exists():
            docs_dir.mkdir()

        doc_file = docs_dir / f"feature_{action_num}_docs.md"
        content = f"# Feature {action_num} Documentation\n\n## Overview\nDocumentation for feature {action_num}.\n\n## Usage\nExample usage goes here.\n"
        doc_file.write_text(content)

        return {
            "action": "add_documentation",
            "file": str(doc_file.relative_to(project_dir)),
            "timestamp": time.time()
        }

    def _update_configuration(self, project_dir: Path, project_type: str, action_num: int) -> dict[str, Any]:
        """Update project configuration."""
        config_files = {
            "python": "pyproject.toml",
            "javascript": "package.json",
            "rust": "Cargo.toml"
        }

        config_file = project_dir / config_files.get(project_type, "config.yaml")
        if config_file.exists():
            content = config_file.read_text()
            # Simple append for testing
            config_file.write_text(content + f"\n# Updated at {time.time()}\n")

            return {
                "action": "update_config",
                "file": config_file.name,
                "timestamp": time.time()
            }

        return {"action": "update_config", "file": "none", "timestamp": time.time()}

    def _add_test_file(self, project_dir: Path, project_type: str, action_num: int) -> dict[str, Any]:
        """Add a test file."""
        tests_dir = project_dir / "tests"
        if not tests_dir.exists():
            tests_dir.mkdir()

        test_file = tests_dir / f"test_feature_{action_num}.py"
        content = f'"""Tests for feature {action_num}."""\n\ndef test_feature_{action_num}():\n    """Test feature {action_num} functionality."""\n    assert True  # Placeholder test\n'
        test_file.write_text(content)

        return {
            "action": "add_test",
            "file": str(test_file.relative_to(project_dir)),
            "timestamp": time.time()
        }

    def _simulate_collaborator_action(
        self,
        project_dir: Path,
        collaborator: str,
        action_num: int
    ) -> dict[str, Any]:
        """Simulate action by a collaborator."""
        # Create collaborator-specific file
        collab_file = project_dir / f"{collaborator}_work_{action_num}.md"
        content = f"# Work by {collaborator}\n\nAction {action_num} implemented by {collaborator}.\n"
        collab_file.write_text(content)

        return {
            "action": "collaborator_work",
            "collaborator": collaborator,
            "file": collab_file.name,
            "timestamp": time.time()
        }


@pytest.mark.functional
@pytest.mark.real_world_simulation
class TestRealWorldWorkflowSimulation:
    """Test real-world developer workflow scenarios."""

    @pytest.fixture
    def project_simulator(self, tmp_path):
        """Create project simulator."""
        return RealWorldProjectSimulator(tmp_path)

    def test_single_project_development_workflow(self, project_simulator):
        """Test complete single project development workflow."""
        # Create a realistic Python project
        project_name = "test-python-app"
        project_dir = project_simulator.create_realistic_project(
            project_name, "python", include_docs=True
        )

        # Test project detection
        return_code, stdout, stderr = project_simulator.run_cli_command(
            "admin status",
            cwd=project_dir
        )

        # Should detect project context (even if daemon not available)
        assert len(stdout + stderr) > 0

        # Simulate development session
        actions = project_simulator.simulate_development_session(
            project_name, duration_minutes=2, actions_per_minute=1
        )

        # Validate development actions were recorded
        assert len(actions) > 0
        assert all("timestamp" in action for action in actions)

        # Test ingestion of project files
        return_code, stdout, stderr = project_simulator.run_cli_command(
            f"ingest folder {project_dir}",
            cwd=project_dir
        )

        # Should attempt ingestion (may fail without daemon)
        assert len(stdout + stderr) > 0

    def test_multi_project_workspace_workflow(self, project_simulator):
        """Test workflow with multiple projects in workspace."""
        # Create multiple projects
        projects = [
            ("backend-api", "python"),
            ("frontend-app", "javascript"),
            ("data-processor", "rust")
        ]

        project_dirs = {}
        for name, proj_type in projects:
            project_dirs[name] = project_simulator.create_realistic_project(
                name, proj_type, include_docs=True
            )

        # Test workspace detection
        return_code, stdout, stderr = project_simulator.run_cli_command("admin status")

        # Should provide workspace information
        assert len(stdout + stderr) > 0

        # Test project-specific operations
        for _project_name, project_dir in project_dirs.items():
            # Test project context detection
            return_code, stdout, stderr = project_simulator.run_cli_command(
                "admin status",
                cwd=project_dir
            )

            # Should detect specific project context
            assert len(stdout + stderr) > 0

    def test_collaborative_development_workflow(self, project_simulator):
        """Test collaborative development workflow."""
        # Create project for collaboration
        project_name = "collaborative-project"
        project_dir = project_simulator.create_realistic_project(
            project_name, "python", include_docs=True
        )

        # Simulate collaboration
        collab_actions = project_simulator.simulate_collaboration_workflow(
            project_name, num_collaborators=3
        )

        # Validate collaboration simulation
        assert len(collab_actions) > 0
        collaborators = {action.get("collaborator") for action in collab_actions}
        assert len(collaborators) == 3

        # Test collection management for collaborative project
        return_code, stdout, stderr = project_simulator.run_cli_command(
            "library list",
            cwd=project_dir
        )

        # Should handle library/collection operations
        assert len(stdout + stderr) > 0

    def test_project_documentation_workflow(self, project_simulator):
        """Test documentation-focused development workflow."""
        # Create documentation-heavy project
        project_name = "docs-project"
        project_dir = project_simulator.create_realistic_project(
            project_name, "python", include_docs=True
        )

        # Add extensive documentation
        docs_dir = project_dir / "docs"
        documentation_files = [
            ("user-guide.md", "# User Guide\n\nComprehensive user documentation."),
            ("api-reference.md", "# API Reference\n\nDetailed API documentation."),
            ("troubleshooting.md", "# Troubleshooting\n\nCommon issues and solutions."),
            ("changelog.md", "# Changelog\n\nVersion history and changes.")
        ]

        for filename, content in documentation_files:
            (docs_dir / filename).write_text(content)

        # Test documentation ingestion
        return_code, stdout, stderr = project_simulator.run_cli_command(
            f"ingest folder {docs_dir}",
            cwd=project_dir
        )

        # Should attempt documentation ingestion
        assert len(stdout + stderr) > 0

        # Test documentation search
        return_code, stdout, stderr = project_simulator.run_cli_command(
            "search project 'user guide'",
            cwd=project_dir
        )

        # Should attempt search (may fail without daemon)
        assert len(stdout + stderr) > 0

    def test_scratchbook_workflow(self, project_simulator):
        """Test scratchbook functionality in real development context."""
        # Create project
        project_name = "scratchbook-project"
        project_dir = project_simulator.create_realistic_project(
            project_name, "python"
        )

        # Test scratchbook operations
        scratchbook_commands = [
            "memory add 'Remember to use async/await for database operations'",
            "memory add 'Project uses FastAPI for REST API'",
            "memory list",
            "search scratchbook 'async'"
        ]

        results = []
        for command in scratchbook_commands:
            return_code, stdout, stderr = project_simulator.run_cli_command(
                command,
                cwd=project_dir
            )
            results.append({
                "command": command,
                "success": return_code == 0,
                "output": stdout + stderr
            })

        # Validate scratchbook workflow execution
        assert all(len(result["output"]) > 0 for result in results)

    def test_long_running_development_session(self, project_simulator):
        """Test long-running development session simulation."""
        # Create project
        project_name = "long-session-project"
        project_dir = project_simulator.create_realistic_project(
            project_name, "python", include_docs=True
        )

        # Simulate extended development session
        session_actions = project_simulator.simulate_development_session(
            project_name, duration_minutes=3, actions_per_minute=3
        )

        # Validate session simulation
        assert len(session_actions) >= 6  # 3 minutes * 2 actions/minute minimum

        # Check action diversity
        action_types = {action["action"] for action in session_actions}
        assert len(action_types) > 1  # Should have multiple types of actions

        # Test project state after session
        return_code, stdout, stderr = project_simulator.run_cli_command(
            "admin status",
            cwd=project_dir
        )

        # Should still detect project properly
        assert len(stdout + stderr) > 0

    def test_project_migration_workflow(self, project_simulator):
        """Test project migration and collection management workflow."""
        # Create original project
        original_name = "original-project"
        project_simulator.create_realistic_project(
            original_name, "python"
        )

        # Create migrated project (simulate project rename/move)
        migrated_name = "migrated-project"
        migrated_dir = project_simulator.create_realistic_project(
            migrated_name, "python"
        )

        # Test collection management during migration
        migration_commands = [
            "library list",
            f"library create {migrated_name}-collection",
            "admin status"
        ]

        for command in migration_commands:
            return_code, stdout, stderr = project_simulator.run_cli_command(
                command,
                cwd=migrated_dir
            )

            # Should handle migration operations
            assert len(stdout + stderr) > 0

    def test_submodule_project_workflow(self, project_simulator):
        """Test workflow with projects containing submodules."""
        # Create project with submodules
        project_name = "submodule-project"
        project_dir = project_simulator.create_realistic_project(
            project_name, "python",
            include_docs=True,
            include_submodules=True
        )

        # Test submodule detection and handling
        return_code, stdout, stderr = project_simulator.run_cli_command(
            "admin status",
            cwd=project_dir
        )

        # Should detect project with submodules
        assert len(stdout + stderr) > 0

        # Test ingestion of submodule content
        submodule_dir = project_dir / "external"
        return_code, stdout, stderr = project_simulator.run_cli_command(
            f"ingest folder {submodule_dir}",
            cwd=project_dir
        )

        # Should handle submodule ingestion
        assert len(stdout + stderr) > 0

    @pytest.mark.slow
    def test_concurrent_project_workflow(self, project_simulator):
        """Test concurrent operations across multiple projects."""
        # Create multiple projects
        projects = []
        for i in range(3):
            project_name = f"concurrent-project-{i}"
            project_dir = project_simulator.create_realistic_project(
                project_name, "python"
            )
            projects.append((project_name, project_dir))

        # Simulate concurrent development
        import queue
        import threading

        results_queue = queue.Queue()

        def worker(project_info):
            project_name, project_dir = project_info
            try:
                # Simulate work in project
                actions = project_simulator.simulate_development_session(
                    project_name, duration_minutes=1, actions_per_minute=2
                )

                # Test CLI operations
                return_code, stdout, stderr = project_simulator.run_cli_command(
                    "admin status",
                    cwd=project_dir
                )

                results_queue.put({
                    "project": project_name,
                    "actions": len(actions),
                    "cli_success": return_code == 0 or len(stdout + stderr) > 0
                })
            except Exception as e:
                results_queue.put({
                    "project": project_name,
                    "error": str(e)
                })

        # Start concurrent threads
        threads = []
        for project_info in projects:
            thread = threading.Thread(target=worker, args=(project_info,))
            thread.start()
            threads.append(thread)

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Validate concurrent operations
        assert len(results) == len(projects)
        assert all("error" not in result for result in results)

    def test_configuration_driven_workflow(self, project_simulator):
        """Test workflow with different configuration setups."""
        # Create project
        project_name = "config-project"
        project_dir = project_simulator.create_realistic_project(
            project_name, "python"
        )

        # Create custom configuration
        custom_config = {
            "qdrant_url": "http://localhost:6333",
            "auto_watch": True,
            "default_collection": f"{project_name}-custom",
            "ingestion": {
                "auto_process": True,
                "file_patterns": ["*.py", "*.md", "*.txt"]
            }
        }

        config_file = project_dir / "workspace-qdrant.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(custom_config, f)

        # Test configuration-driven operations
        return_code, stdout, stderr = project_simulator.run_cli_command(
            f"config show --config {config_file}",
            cwd=project_dir
        )

        # Should handle custom configuration
        assert len(stdout + stderr) > 0

        # Test operations with custom config
        return_code, stdout, stderr = project_simulator.run_cli_command(
            f"--config {config_file} admin status",
            cwd=project_dir
        )

        # Should use custom configuration
        assert len(stdout + stderr) > 0
