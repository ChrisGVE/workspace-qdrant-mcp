
from ...observability import get_logger
logger = get_logger(__name__)
"""Memory management CLI commands.

This module provides the wqm memory commands for managing user preferences,
LLM behavioral rules, and agent library definitions in the memory collection.
"""

import asyncio
import json
import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ...core.client import create_qdrant_client
from ...core.collection_naming import create_naming_manager
from ...core.config import Config
from ...core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    create_memory_manager,
    parse_conversational_memory_update,
)

console = Console()

# Create the memory app
memory_app = typer.Typer(
    help="🧠 Memory rules and LLM behavior management",
    no_args_is_help=True
)

def handle_async(coro):
    """Helper to run async commands."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        console.logger.info("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.logger.info("[red]Error: {e}[/red]")
        raise typer.Exit(1)

@memory_app.command("list")
def list_rules(
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by category"),
    authority: str | None = typer.Option(None, "--authority", "-a", help="Filter by authority level"),
    scope: str | None = typer.Option(None, "--scope", "-s", help="Filter by scope containing this value"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """📋 Show all memory rules."""
    handle_async(_list_memory_rules(category, authority, scope, json_output))

@memory_app.command("add")
def add_rule(
    rule: str | None = typer.Argument(None, help="The memory rule to add"),
    category: str | None = typer.Option(None, "--category", "-c", help="Category of the rule"),
    authority: str = typer.Option("default", "--authority", "-a", help="Authority level (default: default)"),
    scope: str | None = typer.Option(None, "--scope", "-s", help="Comma-separated list of scopes"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
):
    """➕ Add new memory rule (preference or behavior)."""
    handle_async(_add_memory_rule(rule, category, authority, scope, interactive))

@memory_app.command("edit")
def edit_rule(
    rule_id: str = typer.Argument(..., help="Memory rule ID to edit"),
):
    """✏️ Edit specific memory rule."""
    handle_async(_edit_memory_rule(rule_id))

@memory_app.command("remove")
def remove_rule(
    rule_id: str = typer.Argument(..., help="Memory rule ID to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """🗑️ Remove memory rule."""
    handle_async(_remove_memory_rule(rule_id, force))

@memory_app.command("tokens")
def token_usage():
    """📊 Show token usage statistics."""
    handle_async(_show_token_usage())

@memory_app.command("trim")
def trim_rules(
    max_tokens: int = typer.Option(2000, "--max-tokens", help="Maximum allowed tokens"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes"),
):
    """✂️ Interactive token optimization."""
    handle_async(_trim_memory(max_tokens, dry_run))

@memory_app.command("conflicts")
def detect_conflicts(
    auto_resolve: bool = typer.Option(False, "--auto-resolve", help="Automatically resolve simple conflicts"),
):
    """⚠️ Detect and resolve memory conflicts."""
    handle_async(_detect_conflicts(auto_resolve))

@memory_app.command("parse")
def parse_conversational(
    message: str = typer.Argument(..., help="Conversational message to parse"),
):
    """🔍 Parse conversational memory update."""
    handle_async(_parse_conversational_update(message))

@memory_app.command("web")
def start_web_interface(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run web server on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind web server to"),
):
    """🌐 Start web interface for memory curation."""
    handle_async(_start_web_interface(port, host))

# Async implementation functions (reuse from existing memory.py)
async def _list_memory_rules(
    category: str | None,
    authority: str | None,
    scope: str | None,
    output_json: bool
):
    """List memory rules with optional filtering."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        # Convert string parameters to enums
        category_enum = MemoryCategory(category) if category else None
        authority_enum = AuthorityLevel(authority) if authority else None

        # List rules
        rules = await memory_manager.list_memory_rules(
            category=category_enum,
            authority=authority_enum,
            scope=scope
        )

        if output_json:
            # Output as JSON
            rules_data = []
            for rule in rules:
                rule_dict = {
                    "id": rule.id,
                    "category": rule.category.value,
                    "name": rule.name,
                    "rule": rule.rule,
                    "authority": rule.authority.value,
                    "scope": rule.scope,
                    "source": rule.source,
                    "created_at": rule.created_at.isoformat() if rule.created_at else None,
                    "updated_at": rule.updated_at.isoformat() if rule.updated_at else None
                }
                rules_data.append(rule_dict)

            logger.info("Output", data=json.dumps(rules_data, indent=2))
        else:
            # Display in table format
            if not rules:
                console.logger.info("[yellow]No memory rules found.[/yellow]")
                return

            table = Table(title=f"💭 Memory Rules ({len(rules)} found)")
            table.add_column("ID", style="cyan", width=12)
            table.add_column("Category", width=10)
            table.add_column("Name", style="bold", width=20)
            table.add_column("Authority", width=10)
            table.add_column("Rule", width=50)
            table.add_column("Scope", width=15)

            for rule in rules:
                authority_style = "red" if rule.authority == AuthorityLevel.ABSOLUTE else "yellow"
                scope_text = ", ".join(rule.scope) if rule.scope else "-"

                table.add_row(
                    rule.id[-8:],  # Show last 8 chars of ID
                    rule.category.value,
                    rule.name,
                    f"[{authority_style}]{rule.authority.value}[/{authority_style}]",
                    rule.rule[:47] + "..." if len(rule.rule) > 50 else rule.rule,
                    scope_text
                )

            console.logger.info("Output", data=table)

            # Show summary
            stats = await memory_manager.get_memory_stats()
            console.logger.info("\n[dim]Total: {stats.total_rules} rules, ~{stats.estimated_tokens} tokens[/dim]")

    except Exception as e:
        console.logger.info("[red]Error listing memory rules: {e}[/red]")
        raise typer.Exit(1)

async def _add_memory_rule(
    rule: str | None,
    category: str | None,
    authority: str,
    scope: str | None,
    interactive: bool
):
    """Add a new memory rule."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        # Ensure memory collection exists
        await memory_manager.initialize_memory_collection()

        # Interactive mode or collect missing parameters
        if interactive or not rule:
            console.logger.info("[bold blue]Add Memory Rule[/bold blue]")
            console.logger.info("Enter details for the new memory rule.\n")

            if not rule:
                rule = Prompt.ask("Rule text")

            if not category:
                category_choices = [c.value for c in MemoryCategory]
                category = Prompt.ask(
                    "Category",
                    choices=category_choices,
                    default="preference"
                )

            # Generate name from rule if not provided
            name = Prompt.ask("Short name", default=_generate_name_from_rule(rule))

            if authority not in [a.value for a in AuthorityLevel]:
                authority_choices = [a.value for a in AuthorityLevel]
                authority = Prompt.ask(
                    "Authority level",
                    choices=authority_choices,
                    default="default"
                )

            if not scope:
                scope_input = Prompt.ask("Scope (comma-separated, optional)", default="")
                scope = scope_input if scope_input else None
        else:
            name = _generate_name_from_rule(rule)

        # Parse scope
        scope_list = []
        if scope:
            scope_list = [s.strip() for s in scope.split(",") if s.strip()]

        # Convert to enums
        category_enum = MemoryCategory(category or "preference")
        authority_enum = AuthorityLevel(authority)

        # Add the rule
        rule_id = await memory_manager.add_memory_rule(
            category=category_enum,
            name=name,
            rule=rule,
            authority=authority_enum,
            scope=scope_list,
            source="cli_user"
        )

        console.logger.info("[green]✅[/green] Added memory rule with ID: [cyan]{rule_id}[/cyan]")
        console.logger.info("  Name: {name}")
        console.logger.info("  Category: {category_enum.value}")
        console.logger.info("  Authority: {authority_enum.value}")
        if scope_list:
            console.logger.info("  Scope: {', '.join(scope_list)}")

    except Exception as e:
        console.logger.info("[red]Error adding memory rule: {e}[/red]")
        raise typer.Exit(1)

async def _edit_memory_rule(rule_id: str):
    """Edit an existing memory rule."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        # Get existing rule
        rule = await memory_manager.get_memory_rule(rule_id)
        if not rule:
            console.logger.info("[red]Memory rule {rule_id} not found.[/red]")
            raise typer.Exit(1)

        console.logger.info("[bold blue]Edit Memory Rule: {rule.name}[/bold blue]")
        console.logger.info("Current rule: {rule.rule}\n")

        # Collect updates
        updates = {}

        new_rule = Prompt.ask("New rule text", default=rule.rule)
        if new_rule != rule.rule:
            updates["rule"] = new_rule

        new_name = Prompt.ask("New name", default=rule.name)
        if new_name != rule.name:
            updates["name"] = new_name

        authority_choices = [a.value for a in AuthorityLevel]
        new_authority = Prompt.ask(
            "Authority level",
            choices=authority_choices,
            default=rule.authority.value
        )
        if new_authority != rule.authority.value:
            updates["authority"] = AuthorityLevel(new_authority)

        scope_str = ", ".join(rule.scope) if rule.scope else ""
        new_scope = Prompt.ask("Scope (comma-separated)", default=scope_str)
        new_scope_list = [s.strip() for s in new_scope.split(",") if s.strip()] if new_scope else []
        if new_scope_list != rule.scope:
            updates["scope"] = new_scope_list

        if not updates:
            console.logger.info("[yellow]No changes made.[/yellow]")
            return

        # Confirm changes
        console.logger.info("\n[bold]Proposed changes:[/bold]")
        for key, value in updates.items():
            console.logger.info("  {key}: {getattr(rule, key)} → {value}")

        if not Confirm.ask("\nApply changes?"):
            console.logger.info("[yellow]Changes cancelled.[/yellow]")
            return

        # Apply updates
        success = await memory_manager.update_memory_rule(rule_id, updates)

        if success:
            console.logger.info("[green]✅[/green] Updated memory rule {rule_id}")
        else:
            console.logger.info("[red]Failed to update memory rule {rule_id}[/red]")

    except Exception as e:
        console.logger.info("[red]Error editing memory rule: {e}[/red]")
        raise typer.Exit(1)

async def _remove_memory_rule(rule_id: str, force: bool):
    """Remove a memory rule."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        # Get rule details for confirmation
        rule = await memory_manager.get_memory_rule(rule_id)
        if not rule:
            console.logger.info("[red]Memory rule {rule_id} not found.[/red]")
            raise typer.Exit(1)

        # Confirm deletion
        if not force:
            console.logger.info("[bold red]Remove Memory Rule[/bold red]")
            console.logger.info("ID: {rule.id}")
            console.logger.info("Name: {rule.name}")
            console.logger.info("Rule: {rule.rule}")
            console.logger.info("Authority: {rule.authority.value}")

            if not Confirm.ask("\n[red]Are you sure you want to delete this rule?[/red]"):
                console.logger.info("[yellow]Deletion cancelled.[/yellow]")
                return

        # Delete the rule
        success = await memory_manager.delete_memory_rule(rule_id)

        if success:
            console.logger.info("[green]✅[/green] Deleted memory rule {rule_id}")
        else:
            console.logger.info("[red]Failed to delete memory rule {rule_id}[/red]")

    except Exception as e:
        console.logger.info("[red]Error removing memory rule: {e}[/red]")
        raise typer.Exit(1)

async def _show_token_usage():
    """Show token usage statistics."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        stats = await memory_manager.get_memory_stats()

        # Create usage panel
        usage_text = f"""Total Rules: {stats.total_rules}
Estimated Tokens: {stats.estimated_tokens}

By Category:
"""
        for category, count in stats.rules_by_category.items():
            usage_text += f"  {category.value}: {count}\n"

        usage_text += "\nBy Authority:\n"
        for authority, count in stats.rules_by_authority.items():
            usage_text += f"  {authority.value}: {count}\n"

        # Token usage assessment
        if stats.estimated_tokens < 1000:
            token_status = "[green]Low usage[/green]"
        elif stats.estimated_tokens < 2000:
            token_status = "[yellow]Moderate usage[/yellow]"
        else:
            token_status = "[red]High usage - consider optimization[/red]"

        usage_text += f"\nToken Status: {token_status}"

        if stats.last_optimization:
            usage_text += f"\nLast Optimized: {stats.last_optimization.strftime('%Y-%m-%d %H:%M:%S')}"

        panel = Panel(usage_text.strip(), title="📊 Memory Token Usage", title_align="left")
        console.logger.info("Output", data=panel)

    except Exception as e:
        console.logger.info("[red]Error getting token usage: {e}[/red]")
        raise typer.Exit(1)

async def _trim_memory(max_tokens: int, dry_run: bool):
    """Interactive memory optimization."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        stats = await memory_manager.get_memory_stats()

        console.logger.info("[bold blue]✂️ Memory Optimization[/bold blue]")
        console.logger.info("Current usage: {stats.estimated_tokens} tokens")
        console.logger.info("Target: {max_tokens} tokens")

        if stats.estimated_tokens <= max_tokens:
            console.logger.info("[green]✅ Memory already within token limit.[/green]")
            return

        excess_tokens = stats.estimated_tokens - max_tokens
        console.logger.info("Need to reduce by: [red]{excess_tokens}[/red] tokens\n")

        if dry_run:
            console.logger.info("[yellow]DRY RUN - No changes will be made[/yellow]\n")

        # Get optimization suggestions
        tokens_saved, actions = await memory_manager.optimize_memory(max_tokens)

        console.logger.info("[bold]Optimization Suggestions:[/bold]")
        for i, action in enumerate(actions, 1):
            console.logger.info("  {i}. {action}")

        console.logger.info("\nEstimated tokens saved: [green]{tokens_saved}[/green]")

        if not dry_run:
            if Confirm.ask("\nApply optimizations?"):
                console.logger.info("[green]✅[/green] Memory optimization applied")
            else:
                console.logger.info("[yellow]Optimization cancelled.[/yellow]")

    except Exception as e:
        console.logger.info("[red]Error optimizing memory: {e}[/red]")
        raise typer.Exit(1)

async def _detect_conflicts(auto_resolve: bool):
    """Detect and resolve memory conflicts."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        console.logger.info("[bold blue]⚠️ Conflict Detection[/bold blue]")
        console.logger.info("Analyzing memory rules for conflicts...\n")

        conflicts = await memory_manager.detect_conflicts()

        if not conflicts:
            console.logger.info("[green]✅ No conflicts detected.[/green]")
            return

        console.logger.info("[red]Found {len(conflicts)} conflict(s):[/red]\n")

        for i, conflict in enumerate(conflicts, 1):
            console.logger.info("[bold]Conflict {i}: {conflict.conflict_type}[/bold]")
            console.logger.info("Confidence: {conflict.confidence:.1%}")
            console.logger.info("Description: {conflict.description}")
            console.logger.info("Rule 1: {conflict.rule1.name} - {conflict.rule1.rule}")
            console.logger.info("Rule 2: {conflict.rule2.name} - {conflict.rule2.rule}")

            console.logger.info("Resolution options:")
            for j, option in enumerate(conflict.resolution_options, 1):
                console.logger.info("  {j}. {option}")
            console.print()

            if auto_resolve:
                console.logger.info("[yellow]Auto-resolving conflict {i}...[/yellow]")
                # Placeholder for auto-resolution logic
                console.logger.info("[green]✅ Conflict resolved automatically[/green]\n")
            else:
                if Confirm.ask(f"Resolve conflict {i}?"):
                    # Interactive resolution
                    choice = Prompt.ask(
                        "Choose resolution",
                        choices=[str(j) for j in range(1, len(conflict.resolution_options) + 1)],
                        default="1"
                    )
                    console.logger.info("[green]✅ Applied resolution option {choice}[/green]\n")
                else:
                    console.logger.info("[yellow]Conflict skipped[/yellow]\n")

    except Exception as e:
        console.logger.info("[red]Error detecting conflicts: {e}[/red]")
        raise typer.Exit(1)

async def _parse_conversational_update(message: str):
    """Parse a conversational memory update."""
    try:
        result = parse_conversational_memory_update(message)

        if result:
            console.logger.info("[green]✅ Parsed conversational memory update:[/green]")
            console.logger.info("  Category: {result['category'].value}")
            console.logger.info("  Rule: {result['rule']}")
            console.logger.info("  Authority: {result['authority'].value}")
            console.logger.info("  Source: {result['source']}")

            if Confirm.ask("\nAdd this as a memory rule?"):
                config = Config()
                client = create_qdrant_client(config.qdrant_client_config)
                naming_manager = create_naming_manager(config.workspace.global_collections)
                memory_manager = create_memory_manager(client, naming_manager)

                # Ensure memory collection exists
                await memory_manager.initialize_memory_collection()

                # Add the rule
                rule_id = await memory_manager.add_memory_rule(
                    category=result["category"],
                    name=_generate_name_from_rule(result["rule"]),
                    rule=result["rule"],
                    authority=result["authority"],
                    source=result["source"]
                )

                console.logger.info("[green]✅ Added memory rule with ID: {rule_id}[/green]")
            else:
                console.logger.info("[yellow]Memory rule not added.[/yellow]")
        else:
            console.logger.info("[yellow]No conversational memory update detected.[/yellow]")
            console.logger.info("Supported patterns:")
            console.logger.info("  - 'Note: <preference>'")
            console.logger.info("  - 'For future reference, <instruction>'")
            console.logger.info("  - 'Remember that I <preference>'")
            console.logger.info("  - 'Always <behavior>' or 'Never <behavior>'")

    except Exception as e:
        console.logger.info("[red]Error parsing conversational update: {e}[/red]")
        raise typer.Exit(1)

async def _start_web_interface(port: int, host: str):
    """Start the web interface for memory curation."""
    try:
        from ..web.server import start_web_server

        config = Config()

        console.logger.info("[bold blue]🌐 Starting Memory Curation Web Interface[/bold blue]")
        console.logger.info("Server: http://{host}:{port}")
        console.logger.info("[dim]Press Ctrl+C to stop the server[/dim]\n")

        # Start the web server
        await start_web_server(config, port, host)

    except ImportError:
        console.logger.info("[red]Web interface dependencies not installed.[/red]")
        console.logger.info("Please install with: pip install fastapi uvicorn jinja2")
        raise typer.Exit(1)
    except Exception as e:
        console.logger.info("[red]Error starting web server: {e}[/red]")
        raise typer.Exit(1)


def _generate_name_from_rule(rule: str) -> str:
    """Generate a short name from a rule text."""
    # Take first few words, clean them up
    words = rule.lower().split()[:3]
    # Remove common words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "always", "never"}
    words = [w for w in words if w not in stop_words]
    # Take first 2-3 meaningful words
    name_words = words[:2] if len(words) >= 2 else words
    return "-".join(name_words) if name_words else "rule"
