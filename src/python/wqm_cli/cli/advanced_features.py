"""Advanced CLI Features for wqm.

This module implements advanced CLI features including command suggestions,
smart defaults, configuration wizards, and enhanced user experience.

Task 251: Advanced CLI features for unified interface.
"""

import os
import sys
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich.tree import Tree
from rich.columns import Columns

console = Console()


class ConfigurationWizard:
    """Interactive configuration wizard for first-time setup."""

    def __init__(self):
        """Initialize configuration wizard."""
        self.config_data = {}

    def run_setup_wizard(self) -> Dict[str, Any]:
        """Run the complete setup wizard."""
        console.print(Panel.fit(
            "[bold blue]🚀 Welcome to wqm Setup Wizard[/bold blue]\n"
            "This wizard will help you configure wqm for your environment",
            border_style="blue"
        ))

        # Collect configuration step by step
        self._configure_qdrant()
        self._configure_embedding()
        self._configure_workspace()
        self._configure_performance()
        self._review_and_save()

        return self.config_data

    def _configure_qdrant(self) -> None:
        """Configure Qdrant connection settings."""
        console.print("\n📡 [bold blue]Qdrant Configuration[/bold blue]")

        # Auto-detect common Qdrant setups
        detected_setups = self._detect_qdrant_setups()

        if detected_setups:
            console.print("🔍 Detected possible Qdrant configurations:")
            for i, (name, config) in enumerate(detected_setups.items(), 1):
                console.print(f"  {i}. {name}: {config['url']}")

            choice = Prompt.ask(
                "Select a configuration or enter 'custom'",
                choices=[str(i) for i in range(1, len(detected_setups) + 1)] + ['custom'],
                default='1'
            )

            if choice != 'custom':
                selected_config = list(detected_setups.values())[int(choice) - 1]
                self.config_data['qdrant'] = selected_config
                console.print(f"✅ Using detected configuration: {selected_config['url']}")
                return

        # Manual configuration
        console.print("⚙️ Manual Qdrant Configuration")

        url = Prompt.ask(
            "Qdrant URL",
            default="http://localhost:6333"
        )

        api_key = Prompt.ask(
            "API Key (leave empty for none)",
            default="",
            password=True
        )

        prefer_grpc = Confirm.ask(
            "Use gRPC protocol for better performance?",
            default=True
        )

        timeout = IntPrompt.ask(
            "Connection timeout (seconds)",
            default=30
        )

        self.config_data['qdrant'] = {
            'url': url,
            'api_key': api_key if api_key else None,
            'prefer_grpc': prefer_grpc,
            'timeout': timeout
        }

    def _configure_embedding(self) -> None:
        """Configure embedding settings."""
        console.print("\n🧠 [bold blue]Embedding Configuration[/bold blue]")

        # Show popular models
        popular_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5"
        ]

        console.print("Popular embedding models:")
        for i, model in enumerate(popular_models, 1):
            console.print(f"  {i}. {model}")

        model_choice = Prompt.ask(
            "Select model or enter custom name",
            choices=[str(i) for i in range(1, len(popular_models) + 1)] + ['custom'],
            default='1'
        )

        if model_choice == 'custom':
            model_name = Prompt.ask("Custom model name")
        else:
            model_name = popular_models[int(model_choice) - 1]

        enable_sparse = Confirm.ask(
            "Enable sparse vectors for keyword search?",
            default=True
        )

        chunk_size = IntPrompt.ask(
            "Text chunk size (characters)",
            default=1000
        )

        chunk_overlap = IntPrompt.ask(
            "Chunk overlap size (characters)",
            default=200
        )

        self.config_data['embedding'] = {
            'model': model_name,
            'enable_sparse_vectors': enable_sparse,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }

    def _configure_workspace(self) -> None:
        """Configure workspace settings."""
        console.print("\n📁 [bold blue]Workspace Configuration[/bold blue]")

        # Auto-detect Git user
        github_user = self._detect_github_user()
        if github_user:
            use_detected = Confirm.ask(
                f"Use detected GitHub user '{github_user}'?",
                default=True
            )
            if not use_detected:
                github_user = Prompt.ask("GitHub username")
        else:
            github_user = Prompt.ask("GitHub username (optional)", default="")

        collection_suffixes = Prompt.ask(
            "Collection suffixes (comma-separated)",
            default="project,docs"
        ).split(',')

        auto_create = Confirm.ask(
            "Auto-create collections when needed?",
            default=True
        )

        self.config_data['workspace'] = {
            'github_user': github_user if github_user else None,
            'collection_suffixes': [s.strip() for s in collection_suffixes],
            'auto_create_collections': auto_create
        }

    def _configure_performance(self) -> None:
        """Configure performance settings."""
        console.print("\n⚡ [bold blue]Performance Configuration[/bold blue]")

        # Suggest settings based on system resources
        memory_gb = self._estimate_system_memory()
        cpu_cores = self._get_cpu_count()

        console.print(f"System info: ~{memory_gb}GB RAM, {cpu_cores} CPU cores")

        batch_size = IntPrompt.ask(
            "Embedding batch size",
            default=min(32, max(8, memory_gb * 2))
        )

        max_concurrent = IntPrompt.ask(
            "Maximum concurrent operations",
            default=min(8, cpu_cores)
        )

        enable_caching = Confirm.ask(
            "Enable embedding caching?",
            default=True
        )

        self.config_data['performance'] = {
            'batch_size': batch_size,
            'max_concurrent_operations': max_concurrent,
            'enable_caching': enable_caching
        }

    def _review_and_save(self) -> None:
        """Review configuration and save."""
        console.print("\n📋 [bold blue]Configuration Review[/bold blue]")

        # Display configuration summary
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        for section, settings in self.config_data.items():
            table.add_row(f"[bold]{section.upper()}[/bold]", "")
            for key, value in settings.items():
                if key == 'api_key' and value:
                    value = "***hidden***"
                table.add_row(f"  {key}", str(value))

        console.print(table)

        if Confirm.ask("\nSave this configuration?", default=True):
            config_path = Path.cwd() / "workspace_qdrant_config.yaml"

            # Convert to YAML format
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False)

            console.print(f"✅ Configuration saved to {config_path}")
        else:
            console.print("❌ Configuration not saved")

    def _detect_qdrant_setups(self) -> Dict[str, Dict[str, Any]]:
        """Auto-detect common Qdrant setups."""
        setups = {}

        # Check for local Docker container
        try:
            import subprocess
            result = subprocess.run(
                ['docker', 'ps', '--format', 'table {{.Names}}\t{{.Ports}}'],
                capture_output=True, text=True, timeout=5
            )
            if 'qdrant' in result.stdout.lower() and '6333' in result.stdout:
                setups['Docker (detected)'] = {
                    'url': 'http://localhost:6333',
                    'prefer_grpc': True
                }
        except:
            pass

        # Check for common cloud setups
        if os.getenv('QDRANT_URL'):
            setups['Environment Variable'] = {
                'url': os.getenv('QDRANT_URL'),
                'api_key': os.getenv('QDRANT_API_KEY')
            }

        # Add common defaults
        setups['Local Default'] = {
            'url': 'http://localhost:6333',
            'prefer_grpc': False
        }

        return setups

    def _detect_github_user(self) -> Optional[str]:
        """Try to detect GitHub username from git config."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'config', 'user.name'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None

    def _estimate_system_memory(self) -> int:
        """Estimate system memory in GB."""
        try:
            import psutil
            return int(psutil.virtual_memory().total / (1024**3))
        except:
            return 8  # Default assumption

    def _get_cpu_count(self) -> int:
        """Get CPU core count."""
        try:
            return os.cpu_count() or 4
        except:
            return 4


class SmartDefaults:
    """Smart default values based on context and user history."""

    def __init__(self):
        """Initialize smart defaults system."""
        self.usage_history_file = Path.home() / ".config" / "workspace-qdrant" / "usage_history.json"
        self.usage_history = self._load_usage_history()

    def _load_usage_history(self) -> Dict[str, Any]:
        """Load usage history from file."""
        try:
            if self.usage_history_file.exists():
                with open(self.usage_history_file) as f:
                    return json.load(f)
        except:
            pass
        return {
            'command_frequency': {},
            'flag_preferences': {},
            'collection_names': [],
            'file_paths': []
        }

    def _save_usage_history(self) -> None:
        """Save usage history to file."""
        try:
            self.usage_history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.usage_history_file, 'w') as f:
                json.dump(self.usage_history, f, indent=2)
        except:
            pass

    def record_command_usage(self, command: str, subcommand: str = None, flags: Dict[str, Any] = None) -> None:
        """Record command usage for learning preferences."""
        full_command = f"{command}"
        if subcommand:
            full_command += f" {subcommand}"

        # Update frequency
        freq = self.usage_history['command_frequency']
        freq[full_command] = freq.get(full_command, 0) + 1

        # Record flag preferences
        if flags:
            for flag, value in flags.items():
                flag_prefs = self.usage_history['flag_preferences']
                if flag not in flag_prefs:
                    flag_prefs[flag] = {}
                flag_prefs[flag][str(value)] = flag_prefs[flag].get(str(value), 0) + 1

        self._save_usage_history()

    def get_suggested_collection_name(self, context: str = None) -> str:
        """Get suggested collection name based on context."""
        recent_collections = self.usage_history.get('collection_names', [])

        if recent_collections:
            return recent_collections[-1]  # Most recent

        if context == "project":
            # Try to detect project name from git
            try:
                import subprocess
                result = subprocess.run(
                    ['git', 'remote', 'get-url', 'origin'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    url = result.stdout.strip()
                    # Extract repo name from URL
                    repo_name = url.split('/')[-1].replace('.git', '')
                    return repo_name
            except:
                pass

        return "default"

    def get_suggested_search_limit(self, search_type: str) -> int:
        """Get suggested search result limit based on search type."""
        defaults = {
            "project": 10,
            "global": 20,
            "collection": 15
        }

        # Could be enhanced with user history analysis
        return defaults.get(search_type, 10)

    def get_preferred_format(self, command: str) -> str:
        """Get user's preferred output format for a command."""
        flag_prefs = self.usage_history.get('flag_preferences', {})
        format_prefs = flag_prefs.get('format', {})

        if format_prefs:
            # Return most used format
            return max(format_prefs.items(), key=lambda x: x[1])[0]

        # Defaults by command type
        defaults = {
            "config": "yaml",
            "admin": "table",
            "search": "json"
        }

        return defaults.get(command, "yaml")


class CommandSuggestionSystem:
    """System for suggesting commands based on context and user intent."""

    def __init__(self):
        """Initialize command suggestion system."""
        self.command_relationships = self._build_command_relationships()
        self.context_patterns = self._build_context_patterns()

    def _build_command_relationships(self) -> Dict[str, List[str]]:
        """Build relationships between commands."""
        return {
            "after_config": ["admin status", "service restart"],
            "after_ingest": ["search project", "library list"],
            "after_service_install": ["service start", "admin status"],
            "after_error": ["admin diagnostics", "service logs"],
            "setup_workflow": ["config init-unified", "service install", "service start"],
        }

    def _build_context_patterns(self) -> Dict[str, List[str]]:
        """Build context-based command patterns."""
        return {
            "first_time_user": [
                "Configuration wizard to get started",
                "wqm help discover",
                "wqm config init-unified",
                "wqm service install"
            ],
            "debugging_connection": [
                "wqm admin status",
                "wqm config validate",
                "wqm service status",
                "wqm observability health"
            ],
            "adding_content": [
                "wqm ingest file <path>",
                "wqm library create <name>",
                "wqm watch add <directory>"
            ]
        }

    def suggest_next_commands(self, last_command: str) -> List[str]:
        """Suggest logical next commands based on what user just ran."""
        suggestions = []

        # Direct command relationships
        for pattern, commands in self.command_relationships.items():
            if pattern in last_command or any(part in last_command for part in pattern.split('_')):
                suggestions.extend(commands)

        # Context-based suggestions
        if "config" in last_command:
            suggestions.extend(["admin status", "service restart"])
        elif "ingest" in last_command:
            suggestions.extend(["search project", "admin collections"])
        elif "error" in last_command or "failed" in last_command:
            suggestions.extend(["admin diagnostics", "service logs"])

        # Remove duplicates and return top suggestions
        seen = set()
        unique_suggestions = []
        for cmd in suggestions:
            if cmd not in seen:
                seen.add(cmd)
                unique_suggestions.append(cmd)

        return unique_suggestions[:5]

    def suggest_for_context(self, context: str) -> List[str]:
        """Suggest commands for a given context."""
        return self.context_patterns.get(context, [])


def create_advanced_features_app() -> typer.Typer:
    """Create the advanced features command application."""
    features_app = typer.Typer(
        name="wizard",
        help="Advanced CLI features and setup wizards",
        no_args_is_help=True
    )

    @features_app.command("setup")
    def setup_wizard() -> None:
        """Run the interactive setup wizard."""
        wizard = ConfigurationWizard()
        config = wizard.run_setup_wizard()

        if config:
            console.print("\n🎉 [bold green]Setup completed successfully![/bold green]")
            console.print("Next steps:")
            console.print("  1. [cyan]wqm service install[/cyan] - Install the background service")
            console.print("  2. [cyan]wqm service start[/cyan] - Start the service")
            console.print("  3. [cyan]wqm admin status[/cyan] - Verify everything is working")
            console.print("  4. [cyan]wqm help discover[/cyan] - Explore available commands")

    @features_app.command("suggest")
    def command_suggestions(
        last_command: str = typer.Argument(..., help="Last command that was run"),
        context: Optional[str] = typer.Option(None, "--context", help="Additional context")
    ) -> None:
        """Get command suggestions based on your last action."""
        suggestion_system = CommandSuggestionSystem()

        suggestions = suggestion_system.suggest_next_commands(last_command)

        if suggestions:
            console.print(f"\n💡 Based on '[cyan]{last_command}[/cyan]', you might want to:")
            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"  {i}. [yellow]{suggestion}[/yellow]")

        if context:
            context_suggestions = suggestion_system.suggest_for_context(context)
            if context_suggestions:
                console.print(f"\n🎯 For '[cyan]{context}[/cyan]' context:")
                for suggestion in context_suggestions:
                    console.print(f"  • [green]{suggestion}[/green]")

        if not suggestions and not context:
            console.print("[yellow]No specific suggestions for that command[/yellow]")
            console.print("💡 Try: [cyan]wqm help discover[/cyan] to explore all commands")

    @features_app.command("defaults")
    def manage_smart_defaults(
        action: str = typer.Argument(..., help="Action: show, reset, or configure"),
    ) -> None:
        """Manage smart defaults and preferences."""
        smart_defaults = SmartDefaults()

        if action == "show":
            console.print("\n📊 [bold blue]Your Command Usage Patterns[/bold blue]")

            # Show command frequency
            freq = smart_defaults.usage_history.get('command_frequency', {})
            if freq:
                table = Table(title="Most Used Commands")
                table.add_column("Command", style="cyan")
                table.add_column("Usage Count", style="yellow")

                for cmd, count in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]:
                    table.add_row(cmd, str(count))

                console.print(table)
            else:
                console.print("No usage history available yet")

        elif action == "reset":
            if Confirm.ask("Reset all usage history and preferences?"):
                smart_defaults.usage_history = {
                    'command_frequency': {},
                    'flag_preferences': {},
                    'collection_names': [],
                    'file_paths': []
                }
                smart_defaults._save_usage_history()
                console.print("✅ Usage history reset")

        elif action == "configure":
            console.print("🔧 Smart defaults configuration coming soon!")

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("Available actions: show, reset, configure")

    return features_app


# Global instances
configuration_wizard = ConfigurationWizard()
smart_defaults = SmartDefaults()
command_suggestions = CommandSuggestionSystem()

# Export the features app
advanced_features_app = create_advanced_features_app()