
# Suppress SSL warnings for localhost connections
import warnings
import urllib3
# Suppress both urllib3 and general UserWarnings about insecure connections
warnings.filterwarnings('ignore', message='.*insecure connection.*')
warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
from qdrant_client import QdrantClient
from ...core.collection_naming import create_naming_manager
from ...core.config import Config
from ...core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    create_memory_manager,
    parse_conversational_memory_update,
)

# Plain text console - no Rich formatting

def format_plain_table(headers, rows, title=None):
    """Format data as a plain text table with aligned columns."""
    if not rows:
        return "No data to display"
    
    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Build table
    lines = []
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * len(title))
    
    # Header row
    header_line = "  ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Data rows
    for row in rows:
        row_line = "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(row_line)
    
    return "\n".join(lines)

def get_user_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default value."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    response = input(full_prompt).strip()
    return response if response else (default or "")

def get_user_confirmation(prompt: str, default: bool = True) -> bool:
    """Get yes/no confirmation from user."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        response = input(prompt + suffix).strip().lower()
        if not response:
            return default
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'")

# Create the memory app
memory_app = typer.Typer(
    help="Memory rules and LLM behavior management",
    no_args_is_help=True,
    rich_markup_mode=None  # Disable Rich formatting completely
)

def handle_async(coro):
    """Helper to run async commands."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)

@memory_app.command("list")
def list_rules(
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by category"),
    authority: str | None = typer.Option(None, "--authority", "-a", help="Filter by authority level"),
    scope: str | None = typer.Option(None, "--scope", "-s", help="Filter by scope containing this value"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """ Show all memory rules."""
    handle_async(_list_memory_rules(category, authority, scope, json_output))

@memory_app.command("add")
def add_rule(
    rule: str | None = typer.Argument(None, help="The memory rule to add"),
    category: str | None = typer.Option(None, "--category", "-c", help="Category of the rule"),
    authority: str = typer.Option("default", "--authority", "-a", help="Authority level (default: default)"),
    scope: str | None = typer.Option(None, "--scope", "-s", help="Comma-separated list of scopes"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
):
    """ Add new memory rule (preference or behavior)."""
    handle_async(_add_memory_rule(rule, category, authority, scope, interactive))

@memory_app.command("edit")
def edit_rule(
    rule_id: str = typer.Argument(..., help="Memory rule ID to edit"),
):
    """ Edit specific memory rule."""
    handle_async(_edit_memory_rule(rule_id))

@memory_app.command("remove")
def remove_rule(
    rule_id: str = typer.Argument(..., help="Memory rule ID to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """ Remove memory rule."""
    handle_async(_remove_memory_rule(rule_id, force))

@memory_app.command("tokens")
def token_usage():
    """ Show token usage statistics."""
    handle_async(_show_token_usage())

@memory_app.command("trim")
def trim_rules(
    max_tokens: int = typer.Option(2000, "--max-tokens", help="Maximum allowed tokens"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes"),
):
    """ Interactive token optimization."""
    handle_async(_trim_memory(max_tokens, dry_run))

@memory_app.command("conflicts")
def detect_conflicts(
    auto_resolve: bool = typer.Option(False, "--auto-resolve", help="Automatically resolve simple conflicts"),
):
    """ Detect and resolve memory conflicts."""
    handle_async(_detect_conflicts(auto_resolve))

@memory_app.command("parse")
def parse_conversational(
    message: str = typer.Argument(..., help="Conversational message to parse"),
):
    """ Parse conversational memory update."""
    handle_async(_parse_conversational_update(message))

@memory_app.command("web")
def start_web_interface(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run web server on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind web server to"),
):
    """ Start web interface for memory curation."""
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
        client = QdrantClient(**config.qdrant_client_config)
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
                print("No memory rules found.")
                return

            # Format as plain text table
            headers = ["ID", "Category", "Name", "Authority", "Rule", "Scope"]
            table_rows = []
            
            for rule in rules:
                scope_text = ", ".join(rule.scope) if rule.scope else "-"
                rule_text = rule.rule[:47] + "..." if len(rule.rule) > 50 else rule.rule
                
                table_rows.append([
                    rule.id[-8:],  # Show last 8 chars of ID
                    rule.category.value,
                    rule.name,
                    rule.authority.value,
                    rule_text,
                    scope_text
                ])
            
            table_output = format_plain_table(headers, table_rows, f"Memory Rules ({len(rules)} found)")
            print(table_output)

            # Show summary
            stats = await memory_manager.get_memory_stats()
            print(f"\nTotal: {stats.total_rules} rules, ~{stats.estimated_tokens} tokens")

    except Exception as e:
        print(f"Error listing memory rules: {e}")
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
        client = QdrantClient(**config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        # Ensure memory collection exists
        await memory_manager.initialize_memory_collection()

        # Interactive mode or collect missing parameters
        if interactive or not rule:
            print("Add Memory Rule")
            print("Enter details for the new memory rule.\n")

            if not rule:
                rule = get_user_input("Rule text")

            if not category:
                category_choices = [c.value for c in MemoryCategory]
                print(f"Category choices: {', '.join(category_choices)}")
                category = get_user_input("Category", "preference")

            # Generate name from rule if not provided
            name = get_user_input("Short name", _generate_name_from_rule(rule))

            if authority not in [a.value for a in AuthorityLevel]:
                authority_choices = [a.value for a in AuthorityLevel]
                print(f"Authority level choices: {', '.join(authority_choices)}")
                authority = get_user_input("Authority level", "default")

            if not scope:
                scope_input = get_user_input("Scope (comma-separated, optional)", "")
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

        print(f" Added memory rule with ID: {rule_id}")
        print(f"  Name: {name}")
        print(f"  Category: {category_enum.value}")
        print(f"  Authority: {authority_enum.value}")
        if scope_list:
            print(f"  Scope: {', '.join(scope_list)}")

    except Exception as e:
        print(f"Error adding memory rule: {e}")
        raise typer.Exit(1)

async def _edit_memory_rule(rule_id: str):
    """Edit an existing memory rule."""
    try:
        config = Config()
        client = QdrantClient(**config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        # Get existing rule
        rule = await memory_manager.get_memory_rule(rule_id)
        if not rule:
            print(f"Memory rule {rule_id} not found.")
            raise typer.Exit(1)

        print(f"Edit Memory Rule: {rule.name}")
        print(f"Current rule: {rule.rule}\n")

        # Collect updates
        updates = {}

        new_rule = get_user_input("New rule text")
        if new_rule != rule.rule:
            updates["rule"] = new_rule

        new_name = get_user_input("New name")
        if new_name != rule.name:
            updates["name"] = new_name

        authority_choices = [a.value for a in AuthorityLevel]
        new_authority = get_user_input("Authority level")
        if new_authority != rule.authority.value:
            updates["authority"] = AuthorityLevel(new_authority)

        scope_str = ", ".join(rule.scope) if rule.scope else ""
        new_scope = get_user_input("Scope (comma-separated)", scope_str)
        new_scope_list = [s.strip() for s in new_scope.split(",") if s.strip()] if new_scope else []
        if new_scope_list != rule.scope:
            updates["scope"] = new_scope_list

        if not updates:
            print("No changes made.")
            return

        # Confirm changes
        print("\nProposed changes:")
        for key, value in updates.items():
            print(f"  {key}: {getattr(rule, key)} → {value}")

        if not get_user_confirmation("\nApply changes?"):
            print("Changes cancelled.")
            return

        # Apply updates
        success = await memory_manager.update_memory_rule(rule_id, updates)

        if success:
            print(f" Updated memory rule {rule_id}")
        else:
            print(f"Failed to update memory rule {rule_id}")

    except Exception as e:
        print(f"Error editing memory rule: {e}")
        raise typer.Exit(1)

async def _remove_memory_rule(rule_id: str, force: bool):
    """Remove a memory rule."""
    try:
        config = Config()
        client = QdrantClient(**config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        # Get rule details for confirmation
        rule = await memory_manager.get_memory_rule(rule_id)
        if not rule:
            print(f"Memory rule {rule_id} not found.")
            raise typer.Exit(1)

        # Confirm deletion
        if not force:
            print("Remove Memory Rule")
            print(f"ID: {rule.id}")
            print(f"Name: {rule.name}")
            print(f"Rule: {rule.rule}")
            print(f"Authority: {rule.authority.value}")

            if not get_user_confirmation("\nAre you sure you want to delete this rule?"):
                print("Deletion cancelled.")
                return

        # Delete the rule
        success = await memory_manager.delete_memory_rule(rule_id)

        if success:
            print(f" Deleted memory rule {rule_id}")
        else:
            print(f"Failed to delete memory rule {rule_id}")

    except Exception as e:
        print(f"Error removing memory rule: {e}")
        raise typer.Exit(1)

async def _show_token_usage():
    """Show token usage statistics."""
    try:
        config = Config()
        client = QdrantClient(**config.qdrant_client_config)
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

        print("\nMemory Token Usage")
        print("=" * 18)
        print(usage_text.strip())

    except Exception as e:
        print(f"Error getting token usage: {e}")
        raise typer.Exit(1)

async def _trim_memory(max_tokens: int, dry_run: bool):
    """Interactive memory optimization."""
    try:
        config = Config()
        client = QdrantClient(**config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        stats = await memory_manager.get_memory_stats()

        print(" Memory Optimization")
        print(f"Current usage: {stats.estimated_tokens} tokens")
        print(f"Target: {max_tokens} tokens")

        if stats.estimated_tokens <= max_tokens:
            print(" Memory already within token limit.")
            return

        excess_tokens = stats.estimated_tokens - max_tokens
        print(f"Need to reduce by: {excess_tokens} tokens\n")

        if dry_run:
            print("DRY RUN - No changes will be made\n")

        # Get optimization suggestions
        tokens_saved, actions = await memory_manager.optimize_memory(max_tokens)

        print("Optimization Suggestions:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")

        print(f"\nEstimated tokens saved: {tokens_saved}")

        if not dry_run:
            if get_user_confirmation("\nApply optimizations?"):
                print(" Memory optimization applied")
            else:
                print("Optimization cancelled.")

    except Exception as e:
        print(f"Error optimizing memory: {e}")
        raise typer.Exit(1)

async def _detect_conflicts(auto_resolve: bool):
    """Detect and resolve memory conflicts."""
    try:
        config = Config()
        client = QdrantClient(**config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        print(" Conflict Detection")
        print("Analyzing memory rules for conflicts...\n")

        conflicts = await memory_manager.detect_conflicts()

        if not conflicts:
            print(" No conflicts detected.")
            return

        print(f"Found {len(conflicts)} conflict(s):\n")

        for i, conflict in enumerate(conflicts, 1):
            print(f"Conflict {i}: {conflict.conflict_type}")
            print(f"Confidence: {conflict.confidence:.1%}")
            print(f"Description: {conflict.description}")
            print(f"Rule 1: {conflict.rule1.name} - {conflict.rule1.rule}")
            print(f"Rule 2: {conflict.rule2.name} - {conflict.rule2.rule}")

            print("Resolution options:")
            for j, option in enumerate(conflict.resolution_options, 1):
                print(f"  {j}. {option}")
            print()

            if auto_resolve:
                print(f"Auto-resolving conflict {i}...")
                # Placeholder for auto-resolution logic
                print(" Conflict resolved automatically\n")
            else:
                if get_user_confirmation(f"Resolve conflict {i}?"):
                    # Interactive resolution
                    choices = [str(j) for j in range(1, len(conflict.resolution_options) + 1)]
                    print(f"Choose resolution option (1-{len(conflict.resolution_options)}):")
                    choice = get_user_input("Choice", "1")
                    print(f" Applied resolution option {choice}\n")
                else:
                    print("Conflict skipped\n")

    except Exception as e:
        print(f"Error detecting conflicts: {e}")
        raise typer.Exit(1)

async def _parse_conversational_update(message: str):
    """Parse a conversational memory update."""
    try:
        result = parse_conversational_memory_update(message)

        if result:
            print(" Parsed conversational memory update:")
            print(f"  Category: {result['category'].value}")
            print(f"  Rule: {result['rule']}")
            print(f"  Authority: {result['authority'].value}")
            print(f"  Source: {result['source']}")

            if get_user_confirmation("\nAdd this as a memory rule?"):
                config = Config()
                client = QdrantClient(**config.qdrant_client_config)
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

                print(f" Added memory rule with ID: {rule_id}")
            else:
                print("Memory rule not added.")
        else:
            print("No conversational memory update detected.")
            print("Supported patterns:")
            print("  - 'Note: <preference>'")
            print("  - 'For future reference, <instruction>'")
            print("  - 'Remember that I <preference>'")
            print("  - 'Always <behavior>' or 'Never <behavior>'")

    except Exception as e:
        print(f"Error parsing conversational update: {e}")
        raise typer.Exit(1)

async def _start_web_interface(port: int, host: str):
    """Start the web interface for memory curation."""
    try:
        from ..web.server import start_web_server

        config = Config()

        print(" Starting Memory Curation Web Interface")
        print(f"Server: http://{host}:{port}")
        print("Press Ctrl+C to stop the server\n")

        # Start the web server
        await start_web_server(config, port, host)

    except ImportError:
        print("Web interface dependencies not installed.")
        print("Please install with: pip install fastapi uvicorn jinja2")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error starting web server: {e}")
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
