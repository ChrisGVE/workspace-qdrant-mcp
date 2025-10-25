"""Enhanced Interactive Help System for wqm CLI.

This module provides interactive help and command discovery features
including fuzzy command matching, contextual help, and command suggestions.

Task 251: Create comprehensive unified CLI interface with interactive help system.
"""

import difflib
from dataclasses import dataclass
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


class HelpLevel(str, Enum):
    """Help detail levels."""
    BRIEF = "brief"
    DETAILED = "detailed"
    EXAMPLES = "examples"
    FULL = "full"


@dataclass
class CommandInfo:
    """Information about a CLI command."""
    name: str
    description: str
    usage: str
    examples: list[str]
    aliases: list[str]
    category: str
    subcommands: list[str]
    common_flags: list[str]
    related_commands: list[str]


class InteractiveHelpSystem:
    """Enhanced interactive help system with command discovery."""

    def __init__(self):
        """Initialize the help system."""
        self._commands = self._initialize_commands()
        self._categories = self._build_categories()
        self._command_tree = self._build_command_tree()

    def _initialize_commands(self) -> dict[str, CommandInfo]:
        """Initialize command information database."""
        return {
            "memory": CommandInfo(
                name="memory",
                description="Memory rules and LLM behavior management",
                usage="wqm memory [COMMAND] [OPTIONS]",
                examples=[
                    "wqm memory list",
                    "wqm memory add 'Use uv for Python'",
                    "wqm memory remove 'old rule'",
                    "wqm memory clear",
                ],
                aliases=["mem"],
                category="Core",
                subcommands=["list", "add", "remove", "clear", "search"],
                common_flags=["--help", "--verbose"],
                related_commands=["search", "admin"],
            ),
            "admin": CommandInfo(
                name="admin",
                description="System administration and configuration",
                usage="wqm admin [COMMAND] [OPTIONS]",
                examples=[
                    "wqm admin status",
                    "wqm admin collections",
                    "wqm admin diagnostics",
                    "wqm admin reset --confirm",
                ],
                aliases=["adm"],
                category="System",
                subcommands=["status", "collections", "diagnostics", "reset"],
                common_flags=["--help", "--verbose", "--force"],
                related_commands=["config", "service"],
            ),
            "search": CommandInfo(
                name="search",
                description="Command-line search interface",
                usage="wqm search [TYPE] [QUERY] [OPTIONS]",
                examples=[
                    "wqm search project 'rust patterns'",
                    "wqm search global 'machine learning'",
                    "wqm search collection books 'algorithms'",
                    "wqm search --fuzzy 'fuzzy query'",
                ],
                aliases=["find", "query"],
                category="Core",
                subcommands=["project", "global", "collection", "hybrid"],
                common_flags=["--help", "--limit", "--fuzzy", "--format"],
                related_commands=["memory", "library"],
            ),
            "config": CommandInfo(
                name="config",
                description="Configuration management with live updates",
                usage="wqm config [COMMAND] [OPTIONS]",
                examples=[
                    "wqm config show",
                    "wqm config get qdrant.url",
                    "wqm config set qdrant.url http://localhost:6334",
                    "wqm config edit",
                    "wqm config validate",
                ],
                aliases=["cfg"],
                category="System",
                subcommands=["show", "get", "set", "edit", "validate", "watch"],
                common_flags=["--help", "--format", "--file"],
                related_commands=["admin", "service"],
            ),
            "ingest": CommandInfo(
                name="ingest",
                description="Manual document processing and ingestion",
                usage="wqm ingest [TYPE] [SOURCE] [OPTIONS]",
                examples=[
                    "wqm ingest file document.pdf",
                    "wqm ingest folder ~/docs --collection=docs",
                    "wqm ingest url https://example.com/article",
                    "wqm ingest github repo/project",
                ],
                aliases=["add", "import"],
                category="Content",
                subcommands=["file", "folder", "url", "github", "batch"],
                common_flags=["--help", "--collection", "--force", "--format"],
                related_commands=["library", "watch"],
            ),
            "library": CommandInfo(
                name="library",
                description="Library collection management",
                usage="wqm library [COMMAND] [OPTIONS]",
                examples=[
                    "wqm library create technical-books",
                    "wqm library list",
                    "wqm library info my-collection",
                    "wqm library delete old-collection --confirm",
                ],
                aliases=["lib", "collections"],
                category="Content",
                subcommands=["create", "list", "info", "delete", "rename"],
                common_flags=["--help", "--verbose", "--force"],
                related_commands=["ingest", "search"],
            ),
            "watch": CommandInfo(
                name="watch",
                description="Folder watching configuration and management",
                usage="wqm watch [COMMAND] [OPTIONS]",
                examples=[
                    "wqm watch add ~/docs --collection=docs",
                    "wqm watch list",
                    "wqm watch remove ~/old-docs",
                    "wqm watch status",
                ],
                aliases=["monitor"],
                category="Content",
                subcommands=["add", "remove", "list", "status", "pause", "resume"],
                common_flags=["--help", "--collection", "--recursive"],
                related_commands=["ingest", "service"],
            ),
            "service": CommandInfo(
                name="service",
                description="User service management for memexd daemon",
                usage="wqm service [COMMAND] [OPTIONS]",
                examples=[
                    "wqm service install",
                    "wqm service start",
                    "wqm service stop",
                    "wqm service status",
                    "wqm service restart",
                ],
                aliases=["svc", "daemon"],
                category="System",
                subcommands=["install", "start", "stop", "status", "restart", "logs"],
                common_flags=["--help", "--verbose", "--force"],
                related_commands=["admin", "config"],
            ),
            "status": CommandInfo(
                name="status",
                description="Processing status and user feedback system",
                usage="wqm status [OPTIONS]",
                examples=[
                    "wqm status",
                    "wqm status --live",
                    "wqm status --interval 5",
                    "wqm status --format json",
                ],
                aliases=["stat"],
                category="Monitoring",
                subcommands=[],
                common_flags=["--help", "--live", "--interval", "--format"],
                related_commands=["admin", "observability"],
            ),
            "observability": CommandInfo(
                name="observability",
                description="Observability, monitoring, and health checks",
                usage="wqm observability [COMMAND] [OPTIONS]",
                examples=[
                    "wqm observability health",
                    "wqm observability metrics",
                    "wqm observability logs",
                    "wqm observability trace",
                ],
                aliases=["obs", "monitor"],
                category="Monitoring",
                subcommands=["health", "metrics", "logs", "trace"],
                common_flags=["--help", "--format", "--follow"],
                related_commands=["status", "admin"],
            ),
            "init": CommandInfo(
                name="init",
                description="Initialize shell completion scripts",
                usage="wqm init [SHELL] [OPTIONS]",
                examples=[
                    'eval "$(wqm init bash)"',
                    'eval "$(wqm init zsh)"',
                    "wqm init fish | source",
                    "wqm init help",
                ],
                aliases=["completion"],
                category="Setup",
                subcommands=["bash", "zsh", "fish", "help"],
                common_flags=["--help", "--prog-name"],
                related_commands=["config"],
            ),
            "lsp": CommandInfo(
                name="lsp",
                description="LSP server management and monitoring",
                usage="wqm lsp [COMMAND] [OPTIONS]",
                examples=[
                    "wqm lsp status",
                    "wqm lsp list",
                    "wqm lsp restart python",
                    "wqm lsp logs typescript",
                ],
                aliases=["language"],
                category="Development",
                subcommands=["status", "list", "restart", "logs", "config"],
                common_flags=["--help", "--language", "--verbose"],
                related_commands=["config", "observability"],
            ),
        }

    def _build_categories(self) -> dict[str, list[str]]:
        """Build command categories."""
        categories = {}
        for cmd_name, cmd_info in self._commands.items():
            category = cmd_info.category
            if category not in categories:
                categories[category] = []
            categories[category].append(cmd_name)
        return categories

    def _build_command_tree(self) -> Tree:
        """Build a rich tree of commands organized by category."""
        tree = Tree("🔧 [bold blue]wqm Commands[/bold blue]")

        for category, commands in self._categories.items():
            category_branch = tree.add(f"📁 [bold green]{category}[/bold green]")

            for cmd_name in sorted(commands):
                cmd_info = self._commands[cmd_name]
                cmd_branch = category_branch.add(
                    f"⚡ [cyan]{cmd_name}[/cyan] - [dim]{cmd_info.description}[/dim]"
                )

                if cmd_info.subcommands:
                    sub_branch = cmd_branch.add("[dim]Subcommands:[/dim]")
                    for sub in cmd_info.subcommands[:5]:  # Limit to 5 for display
                        sub_branch.add(f"• [yellow]{sub}[/yellow]")
                    if len(cmd_info.subcommands) > 5:
                        sub_branch.add(f"• [dim]... and {len(cmd_info.subcommands) - 5} more[/dim]")

        return tree

    def show_command_discovery(self) -> None:
        """Show interactive command discovery interface."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold blue]🚀 wqm Command Discovery[/bold blue]\n"
            "Explore all available commands organized by category",
            border_style="blue"
        ))
        console.print()
        console.print(self._command_tree)
        console.print()

        # Show quick tips
        tips = [
            "💡 Use [cyan]wqm [command] --help[/cyan] for detailed help",
            "🔍 Use [cyan]wqm help [command][/cyan] for interactive help",
            "📝 Use [cyan]wqm init[/cyan] to enable tab completion",
        ]

        tip_panel = Panel(
            "\n".join(tips),
            title="💡 Quick Tips",
            border_style="green"
        )
        console.print(tip_panel)

    def suggest_commands(self, partial_command: str, limit: int = 5) -> list[tuple[str, float]]:
        """Suggest commands based on partial input using fuzzy matching."""
        suggestions = []

        # Get all possible command names (including subcommands)
        all_commands = []
        for cmd_name, cmd_info in self._commands.items():
            all_commands.append(cmd_name)
            all_commands.extend(cmd_info.aliases)
            for sub in cmd_info.subcommands:
                all_commands.append(f"{cmd_name} {sub}")

        # Use difflib for fuzzy matching
        if limit <= 0:
            return []

        matches = difflib.get_close_matches(
            partial_command, all_commands, n=limit, cutoff=0.3
        )

        for match in matches:
            # Calculate similarity score
            similarity = difflib.SequenceMatcher(None, partial_command, match).ratio()
            suggestions.append((match, similarity))

        return sorted(suggestions, key=lambda x: x[1], reverse=True)

    def show_command_help(
        self,
        command: str,
        subcommand: str | None = None,
        level: HelpLevel = HelpLevel.DETAILED
    ) -> None:
        """Show enhanced help for a specific command."""
        if command not in self._commands:
            # Try to find similar commands
            suggestions = self.suggest_commands(command, limit=3)

            console.print(f"\n[red]❌ Command '{command}' not found.[/red]\n")

            if suggestions:
                console.print("🤔 Did you mean one of these?")
                for suggestion, score in suggestions:
                    console.print(f"  • [cyan]{suggestion}[/cyan] (similarity: {score:.2f})")
                console.print()

            console.print("📋 Use [cyan]wqm help discover[/cyan] to explore all commands")
            return

        cmd_info = self._commands[command]

        # Create main command panel
        title = f"🔧 {cmd_info.name}"
        if subcommand:
            title += f" {subcommand}"

        content = []

        # Description
        content.append(f"[bold]Description:[/bold] {cmd_info.description}")

        # Usage
        usage = cmd_info.usage
        if subcommand:
            usage = usage.replace("[COMMAND]", subcommand)
        content.append(f"[bold]Usage:[/bold] [cyan]{usage}[/cyan]")

        # Aliases
        if cmd_info.aliases:
            aliases_str = ", ".join(f"[cyan]{alias}[/cyan]" for alias in cmd_info.aliases)
            content.append(f"[bold]Aliases:[/bold] {aliases_str}")

        # Subcommands
        if cmd_info.subcommands and not subcommand:
            if level in [HelpLevel.DETAILED, HelpLevel.FULL]:
                content.append("[bold]Subcommands:[/bold]")
                for sub in cmd_info.subcommands:
                    content.append(f"  • [yellow]{sub}[/yellow]")

        # Common flags
        if cmd_info.common_flags and level in [HelpLevel.DETAILED, HelpLevel.FULL]:
            content.append("[bold]Common Flags:[/bold]")
            for flag in cmd_info.common_flags:
                content.append(f"  • [green]{flag}[/green]")

        # Examples
        if level in [HelpLevel.EXAMPLES, HelpLevel.FULL]:
            content.append("[bold]Examples:[/bold]")
            for example in cmd_info.examples:
                content.append(f"  [dim]$[/dim] [cyan]{example}[/cyan]")

        # Related commands
        if cmd_info.related_commands and level == HelpLevel.FULL:
            related_str = ", ".join(f"[cyan]{rel}[/cyan]" for rel in cmd_info.related_commands)
            content.append(f"[bold]Related Commands:[/bold] {related_str}")

        # Display the panel
        help_panel = Panel(
            "\n\n".join(content),
            title=title,
            border_style="blue",
            padding=(1, 2)
        )

        console.print("\n")
        console.print(help_panel)

        # Show additional tips for the command
        self._show_command_tips(command, subcommand)

    def _show_command_tips(self, command: str, subcommand: str | None = None) -> None:
        """Show contextual tips for a command."""
        tips = []

        if command == "memory":
            tips.extend([
                "💡 Memory rules are persistent and affect LLM behavior",
                "🔍 Use fuzzy search to find existing rules quickly",
                "⚡ Changes take effect immediately without restart"
            ])
        elif command == "search":
            tips.extend([
                "🎯 Use hybrid search for best semantic + keyword results",
                "🔍 Project search is scoped to current Git repository",
                "📊 Use --limit to control result count"
            ])
        elif command == "config":
            tips.extend([
                "🔧 Configuration changes may require service restart",
                "👀 Use 'config watch' to monitor changes live",
                "🔒 Environment variables override config file settings"
            ])
        elif command == "service":
            tips.extend([
                "🚀 Install service first before starting",
                "📋 Check logs if service fails to start",
                "🔄 Use restart for configuration changes"
            ])

        if tips:
            tip_content = "\n".join(tips)
            tip_panel = Panel(
                tip_content,
                title=f"💡 Tips for {command}",
                border_style="green",
                padding=(0, 2)
            )
            console.print()
            console.print(tip_panel)

    def show_category_help(self, category: str) -> None:
        """Show help for all commands in a category."""
        if category not in self._categories:
            console.print(f"[red]❌ Category '{category}' not found.[/red]")
            console.print("📋 Available categories:", ", ".join(self._categories.keys()))
            return

        commands = self._categories[category]

        console.print(f"\n📁 [bold blue]{category} Commands[/bold blue]\n")

        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Key Subcommands", style="yellow")

        for cmd_name in sorted(commands):
            cmd_info = self._commands[cmd_name]
            subcommands_str = ", ".join(cmd_info.subcommands[:3])
            if len(cmd_info.subcommands) > 3:
                subcommands_str += "..."

            table.add_row(cmd_name, cmd_info.description, subcommands_str)

        console.print(table)
        console.print()

    def show_quick_reference(self) -> None:
        """Show a quick reference card."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold blue]📚 wqm Quick Reference[/bold blue]",
            border_style="blue"
        ))

        # Most common commands
        common = [
            ("wqm admin status", "Check system health"),
            ("wqm memory add 'rule'", "Add memory rule"),
            ("wqm search project 'query'", "Search current project"),
            ("wqm ingest file doc.pdf", "Process document"),
            ("wqm config show", "Show configuration"),
        ]

        console.print("\n🚀 [bold]Most Common Commands:[/bold]")
        for cmd, desc in common:
            console.print(f"  [cyan]{cmd:<30}[/cyan] {desc}")

        # Categories overview
        console.print("\n📁 [bold]Command Categories:[/bold]")
        for category, commands in self._categories.items():
            console.print(f"  [green]{category:<12}[/green] {', '.join(commands[:3])}"
                         f"{'...' if len(commands) > 3 else ''}")

        console.print("\n💡 [bold]Getting Help:[/bold]")
        help_commands = [
            ("wqm help discover", "Explore all commands"),
            ("wqm [cmd] --help", "Command-specific help"),
            ("wqm help [cmd]", "Interactive command help"),
            ("wqm init", "Enable tab completion"),
        ]

        for cmd, desc in help_commands:
            console.print(f"  [cyan]{cmd:<20}[/cyan] {desc}")

        console.print()


# Global help system instance
help_system = InteractiveHelpSystem()


def create_help_app() -> typer.Typer:
    """Create the help command application."""
    help_app = typer.Typer(
        name="help",
        help="Interactive help and command discovery system",
        no_args_is_help=False
    )

    @help_app.command(name="info")
    def help_info(
        command: str = typer.Argument(..., help="Command to get help for"),
        level: HelpLevel = typer.Option(
            HelpLevel.DETAILED, "--level", "-l",
            help="Help detail level"
        ),
        examples: bool = typer.Option(
            False, "--examples", "-e",
            help="Show examples"
        )
    ) -> None:
        """Show detailed help for a specific command."""
        actual_level = HelpLevel.EXAMPLES if examples else level
        help_system.show_command_help(command, level=actual_level)

    @help_app.callback(invoke_without_command=True)
    def help_main(ctx: typer.Context) -> None:
        """Interactive help system for wqm commands."""
        if not ctx.invoked_subcommand:
            # Show quick reference when no subcommand is provided
            help_system.show_quick_reference()

    @help_app.command("discover")
    def discover_commands() -> None:
        """Discover all available commands interactively."""
        help_system.show_command_discovery()

    @help_app.command("category")
    def category_help(
        category: str = typer.Argument(..., help="Category to show help for")
    ) -> None:
        """Show help for commands in a specific category."""
        help_system.show_category_help(category)

    @help_app.command("suggest")
    def suggest_commands_cmd(
        partial: str = typer.Argument(..., help="Partial command to find suggestions for"),
        limit: int = typer.Option(5, "--limit", "-n", help="Maximum suggestions to show")
    ) -> None:
        """Get command suggestions based on partial input."""
        suggestions = help_system.suggest_commands(partial, limit)

        if suggestions:
            console.print(f"\n🔍 Suggestions for '[cyan]{partial}[/cyan]':")
            for suggestion, score in suggestions:
                console.print(f"  • [green]{suggestion}[/green] (match: {score:.1%})")
        else:
            console.print(f"[yellow]No suggestions found for '{partial}'[/yellow]")
            console.print("💡 Try [cyan]wqm help discover[/cyan] to explore commands")
        console.print()

    return help_app


# Export the help app for integration with main CLI
help_app = create_help_app()
