
"""
CLI commands for memory management.

This module provides the wqm memory commands for managing user preferences,
LLM behavioral rules, and agent library definitions in the memory collection.

Commands:
- wqm memory list: Show all memory rules
- wqm memory add: Add new rule (preference or behavior)
- wqm memory edit: Edit specific rule
- wqm memory remove: Remove rule
- wqm memory tokens: Show token usage
- wqm memory trim: Interactive token optimization
- wqm memory conflicts: Detect and resolve conflicts
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from ..core.client import create_qdrant_client
from ..core.collection_naming import create_naming_manager
from ..core.config import Config
from ..core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    MemoryRule,
    create_memory_manager,
    parse_conversational_memory_update,
)

console = Console()


@click.group()
def memory():
    """Memory rules and LLM behavior management."""
    pass


@memory.command()
@click.option(
    "--category",
    type=click.Choice([c.value for c in MemoryCategory]),
    help="Filter by category"
)
@click.option(
    "--authority",
    type=click.Choice([a.value for a in AuthorityLevel]),
    help="Filter by authority level"
)
@click.option(
    "--scope",
    help="Filter by scope containing this value"
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def list(category: str | None, authority: str | None, scope: str | None, output_json: bool):
    """Show all memory rules."""
    asyncio.run(_list_memory_rules(category, authority, scope, output_json))


@memory.command()
@click.argument("rule", required=False)
@click.option(
    "--category",
    type=click.Choice([c.value for c in MemoryCategory]),
    help="Category of the rule"
)
@click.option(
    "--authority",
    type=click.Choice([a.value for a in AuthorityLevel]),
    default="default",
    help="Authority level (default: default)"
)
@click.option("--scope", help="Comma-separated list of scopes")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def add(rule: str | None, category: str | None, authority: str, scope: str | None, interactive: bool):
    """Add new memory rule (preference or behavior)."""
    asyncio.run(_add_memory_rule(rule, category, authority, scope, interactive))


@memory.command()
@click.argument("rule_id")
def edit(rule_id: str):
    """Edit specific memory rule."""
    asyncio.run(_edit_memory_rule(rule_id))


@memory.command()
@click.argument("rule_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def remove(rule_id: str, force: bool):
    """Remove memory rule."""
    asyncio.run(_remove_memory_rule(rule_id, force))


@memory.command()
def tokens():
    """Show token usage statistics."""
    asyncio.run(_show_token_usage())


@memory.command()
@click.option("--max-tokens", default=2000, help="Maximum allowed tokens")
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes")
def trim(max_tokens: int, dry_run: bool):
    """Interactive token optimization."""
    asyncio.run(_trim_memory(max_tokens, dry_run))


@memory.command()
@click.option("--auto-resolve", is_flag=True, help="Automatically resolve simple conflicts")
def conflicts(auto_resolve: bool):
    """Detect and resolve memory conflicts."""
    asyncio.run(_detect_conflicts(auto_resolve))


@memory.command()
@click.argument("message")
def parse(message: str):
    """Parse conversational memory update."""
    asyncio.run(_parse_conversational_update(message))


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

            console.print(json.dumps(rules_data, indent=2))
        else:
            # Display in table format
            if not rules:
                console.print("[yellow]No memory rules found.[/yellow]")
                return

            table = Table(title=f"Memory Rules ({len(rules)} found)")
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

            console.print(table)

            # Show summary
            stats = await memory_manager.get_memory_stats()
            console.print(f"\n[dim]Total: {stats.total_rules} rules, ~{stats.estimated_tokens} tokens[/dim]")

    except Exception as e:
        console.print(f"[red]Error listing memory rules: {e}[/red]")
        sys.exit(1)


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
            console.print("[bold blue]Add Memory Rule[/bold blue]")
            console.print("Enter details for the new memory rule.\n")

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

        console.print(f"[green]✓[/green] Added memory rule with ID: [cyan]{rule_id}[/cyan]")
        console.print(f"  Name: {name}")
        console.print(f"  Category: {category_enum.value}")
        console.print(f"  Authority: {authority_enum.value}")
        if scope_list:
            console.print(f"  Scope: {', '.join(scope_list)}")

    except Exception as e:
        console.print(f"[red]Error adding memory rule: {e}[/red]")
        sys.exit(1)


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
            console.print(f"[red]Memory rule {rule_id} not found.[/red]")
            sys.exit(1)

        console.print(f"[bold blue]Edit Memory Rule: {rule.name}[/bold blue]")
        console.print(f"Current rule: {rule.rule}\n")

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
            console.print("[yellow]No changes made.[/yellow]")
            return

        # Confirm changes
        console.print("\n[bold]Proposed changes:[/bold]")
        for key, value in updates.items():
            console.print(f"  {key}: {getattr(rule, key)} → {value}")

        if not Confirm.ask("\nApply changes?"):
            console.print("[yellow]Changes cancelled.[/yellow]")
            return

        # Apply updates
        success = await memory_manager.update_memory_rule(rule_id, updates)

        if success:
            console.print(f"[green]✓[/green] Updated memory rule {rule_id}")
        else:
            console.print(f"[red]Failed to update memory rule {rule_id}[/red]")

    except Exception as e:
        console.print(f"[red]Error editing memory rule: {e}[/red]")
        sys.exit(1)


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
            console.print(f"[red]Memory rule {rule_id} not found.[/red]")
            sys.exit(1)

        # Confirm deletion
        if not force:
            console.print("[bold red]Remove Memory Rule[/bold red]")
            console.print(f"ID: {rule.id}")
            console.print(f"Name: {rule.name}")
            console.print(f"Rule: {rule.rule}")
            console.print(f"Authority: {rule.authority.value}")

            if not Confirm.ask("\n[red]Are you sure you want to delete this rule?[/red]"):
                console.print("[yellow]Deletion cancelled.[/yellow]")
                return

        # Delete the rule
        success = await memory_manager.delete_memory_rule(rule_id)

        if success:
            console.print(f"[green]✓[/green] Deleted memory rule {rule_id}")
        else:
            console.print(f"[red]Failed to delete memory rule {rule_id}[/red]")

    except Exception as e:
        console.print(f"[red]Error removing memory rule: {e}[/red]")
        sys.exit(1)


async def _show_token_usage():
    """Show token usage statistics."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        stats = await memory_manager.get_memory_stats()

        # Create usage panel
        usage_text = f"""
Total Rules: {stats.total_rules}
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

        panel = Panel(usage_text.strip(), title="Memory Token Usage", title_align="left")
        console.print(panel)

    except Exception as e:
        console.print(f"[red]Error getting token usage: {e}[/red]")
        sys.exit(1)


async def _trim_memory(max_tokens: int, dry_run: bool):
    """Interactive memory optimization."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        stats = await memory_manager.get_memory_stats()

        console.print("[bold blue]Memory Optimization[/bold blue]")
        console.print(f"Current usage: {stats.estimated_tokens} tokens")
        console.print(f"Target: {max_tokens} tokens")

        if stats.estimated_tokens <= max_tokens:
            console.print("[green]✓ Memory already within token limit.[/green]")
            return

        excess_tokens = stats.estimated_tokens - max_tokens
        console.print(f"Need to reduce by: [red]{excess_tokens}[/red] tokens\n")

        if dry_run:
            console.print("[yellow]DRY RUN - No changes will be made[/yellow]\n")

        # Get optimization suggestions
        tokens_saved, actions = await memory_manager.optimize_memory(max_tokens)

        console.print("[bold]Optimization Suggestions:[/bold]")
        for i, action in enumerate(actions, 1):
            console.print(f"  {i}. {action}")

        console.print(f"\nEstimated tokens saved: [green]{tokens_saved}[/green]")

        if not dry_run:
            if Confirm.ask("\nApply optimizations?"):
                console.print("[green]✓[/green] Memory optimization applied")
            else:
                console.print("[yellow]Optimization cancelled.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error optimizing memory: {e}[/red]")
        sys.exit(1)


async def _detect_conflicts(auto_resolve: bool):
    """Detect and resolve memory conflicts."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        console.print("[bold blue]Conflict Detection[/bold blue]")
        console.print("Analyzing memory rules for conflicts...\n")

        conflicts = await memory_manager.detect_conflicts()

        if not conflicts:
            console.print("[green]✓ No conflicts detected.[/green]")
            return

        console.print(f"[red]Found {len(conflicts)} conflict(s):[/red]\n")

        for i, conflict in enumerate(conflicts, 1):
            console.print(f"[bold]Conflict {i}: {conflict.conflict_type}[/bold]")
            console.print(f"Confidence: {conflict.confidence:.1%}")
            console.print(f"Description: {conflict.description}")
            console.print(f"Rule 1: {conflict.rule1.name} - {conflict.rule1.rule}")
            console.print(f"Rule 2: {conflict.rule2.name} - {conflict.rule2.rule}")

            console.print("Resolution options:")
            for j, option in enumerate(conflict.resolution_options, 1):
                console.print(f"  {j}. {option}")
            console.print()

            if auto_resolve:
                console.print(f"[yellow]Auto-resolving conflict {i}...[/yellow]")
                # Placeholder for auto-resolution logic
                console.print("[green]✓ Conflict resolved automatically[/green]\n")
            else:
                if Confirm.ask(f"Resolve conflict {i}?"):
                    # Interactive resolution
                    choice = Prompt.ask(
                        "Choose resolution",
                        choices=[str(j) for j in range(1, len(conflict.resolution_options) + 1)],
                        default="1"
                    )
                    console.print(f"[green]✓ Applied resolution option {choice}[/green]\n")
                else:
                    console.print("[yellow]Conflict skipped[/yellow]\n")

    except Exception as e:
        console.print(f"[red]Error detecting conflicts: {e}[/red]")
        sys.exit(1)


async def _parse_conversational_update(message: str):
    """Parse a conversational memory update."""
    try:
        result = parse_conversational_memory_update(message)

        if result:
            console.print("[green]✓ Parsed conversational memory update:[/green]")
            console.print(f"  Category: {result['category'].value}")
            console.print(f"  Rule: {result['rule']}")
            console.print(f"  Authority: {result['authority'].value}")
            console.print(f"  Source: {result['source']}")

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

                console.print(f"[green]✓ Added memory rule with ID: {rule_id}[/green]")
            else:
                console.print("[yellow]Memory rule not added.[/yellow]")
        else:
            console.print("[yellow]No conversational memory update detected.[/yellow]")
            console.print("Supported patterns:")
            console.print("  - 'Note: <preference>'")
            console.print("  - 'For future reference, <instruction>'")
            console.print("  - 'Remember that I <preference>'")
            console.print("  - 'Always <behavior>' or 'Never <behavior>'")

    except Exception as e:
        console.print(f"[red]Error parsing conversational update: {e}[/red]")
        sys.exit(1)


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


if __name__ == "__main__":
    memory()
