"""
Memory management CLI commands.

This module implements the wqm memory subcommands for managing memory rules
and LLM behavioral preferences.
"""

import asyncio
import json
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from ...core.config import Config
from ...memory.manager import MemoryManager
from ...memory.types import (
    MemoryRule, AuthorityLevel, MemoryCategory, MemoryContext, ClaudeCodeSession
)

console = Console()


class MemoryCommands:
    """Memory management CLI commands for wqm memory subcommands."""
    
    def __init__(self):
        self.app = typer.Typer(
            help="🧠 Memory rules and LLM behavior management",
            rich_markup_mode="rich"
        )
        self.config = Config()
        self.memory_manager = None
        
        # Register commands
        self.app.command("list")(self.list_rules)
        self.app.command("add")(self.add_rule)
        self.app.command("edit")(self.edit_rule)
        self.app.command("remove")(self.remove_rule)
        self.app.command("search")(self.search_rules)
        self.app.command("tokens")(self.token_usage)
        self.app.command("trim")(self.trim_rules)
        self.app.command("conflicts")(self.analyze_conflicts)
        self.app.command("stats")(self.show_stats)
        self.app.command("export")(self.export_rules)
        self.app.command("import")(self.import_rules)
        # Note: --web interface is future functionality
    
    async def get_memory_manager(self) -> MemoryManager:
        """Get initialized memory manager."""
        if not self.memory_manager:
            self.memory_manager = MemoryManager(self.config)
            await self.memory_manager.initialize()
        return self.memory_manager
    
    def list_rules(
        self,
        authority: Optional[str] = typer.Option(None, "--authority", "-a", help="Filter by authority level (absolute/default)"),
        category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
        source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source"),
        format_type: str = typer.Option("table", "--format", "-f", help="Output format: table, json, yaml"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
    ) -> None:
        """📋 List all memory rules with optional filtering."""
        
        async def _list_rules():
            try:
                memory_manager = await self.get_memory_manager()
                
                # Parse filters
                authority_filter = AuthorityLevel(authority) if authority else None
                category_filter = MemoryCategory(category) if category else None
                
                # Get rules
                rules = await memory_manager.list_rules(
                    authority_filter=authority_filter,
                    category_filter=category_filter,
                    source_filter=source
                )
                
                if format_type == "json":
                    console.print(json.dumps([rule.to_dict() for rule in rules], indent=2, default=str))
                    return
                
                if format_type == "yaml":
                    try:
                        import yaml
                        console.print(yaml.dump([rule.to_dict() for rule in rules], default_flow_style=False))
                    except ImportError:
                        console.print("[red]PyYAML not installed. Use: pip install pyyaml[/red]")
                        raise typer.Exit(1)
                    return
                
                # Table format (default)
                self._display_rules_table(rules, verbose)
                
            except Exception as e:
                console.print(f"[red]Error listing rules: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_list_rules())
    
    def add_rule(
        self,
        rule_text: str = typer.Argument(..., help="The memory rule to add"),
        authority: str = typer.Option("default", "--authority", "-a", help="Authority level: absolute, default"),
        category: str = typer.Option("preference", "--category", "-c", help="Rule category"),
        scope: Optional[List[str]] = typer.Option(None, "--scope", "-s", help="Rule scope (can be specified multiple times)"),
        tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags (can be specified multiple times)"),
        interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode for detailed configuration"),
        force: bool = typer.Option(False, "--force", help="Skip conflict checking")
    ) -> None:
        """➕ Add a new memory rule for LLM behavior."""
        
        async def _add_rule():
            try:
                memory_manager = await self.get_memory_manager()
                
                if interactive:
                    rule_text_input = Prompt.ask("Enter the memory rule", default=rule_text)
                    authority = Prompt.ask(
                        "Authority level",
                        choices=["absolute", "default"],
                        default=authority
                    )
                    category = Prompt.ask(
                        "Category",
                        choices=[c.value for c in MemoryCategory],
                        default=category
                    )
                    scope_input = Prompt.ask("Scope (comma-separated, or press Enter for global)", default="")
                    scope = [s.strip() for s in scope_input.split(",")] if scope_input else []
                    tags_input = Prompt.ask("Tags (comma-separated, or press Enter for none)", default="")
                    tags = [t.strip() for t in tags_input.split(",")] if tags_input else []
                else:
                    rule_text_input = rule_text
                
                # Validate inputs
                try:
                    authority_level = AuthorityLevel(authority)
                    category_enum = MemoryCategory(category)
                except ValueError as e:
                    console.print(f"[red]Invalid input: {e}[/red]")
                    raise typer.Exit(1)
                
                # Create memory rule
                memory_rule = MemoryRule(
                    rule=rule_text_input,
                    category=category_enum,
                    authority=authority_level,
                    scope=scope or [],
                    tags=tags or [],
                    source="user_cli"
                )
                
                # Add the rule with conflict checking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Adding memory rule...", total=None)
                    
                    rule_id, conflicts = await memory_manager.add_rule(
                        memory_rule, 
                        check_conflicts=not force
                    )
                
                console.print(f"[green]✅ Memory rule added successfully[/green]")
                console.print(f"[dim]Rule ID: {rule_id}[/dim]")
                
                # Show conflicts if any
                if conflicts and not force:
                    console.print(f"\n[yellow]⚠️  Found {len(conflicts)} potential conflicts:[/yellow]")
                    for conflict in conflicts[:3]:  # Show first 3 conflicts
                        severity_color = {
                            "low": "blue", 
                            "medium": "yellow", 
                            "high": "orange", 
                            "critical": "red"
                        }.get(conflict.severity, "white")
                        
                        console.print(f"  • [{severity_color}]{conflict.severity.upper()}[/{severity_color}]: {conflict.description}")
                        
                        if conflict.resolution_suggestion:
                            console.print(f"    [dim]Suggestion: {conflict.resolution_suggestion}[/dim]")
                    
                    if len(conflicts) > 3:
                        console.print(f"  ... and {len(conflicts) - 3} more conflicts")
                    
                    console.print(f"\n[dim]Use 'wqm memory conflicts' to analyze all conflicts[/dim]")
                
                # Show token usage impact
                token_usage = await memory_manager.get_token_usage()
                console.print(f"[dim]Total memory tokens: {token_usage.total_tokens} ({token_usage.percentage:.1f}% of context)[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error adding rule: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_add_rule())
    
    def search_rules(
        self,
        query: str = typer.Argument(..., help="Search query"),
        limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
        authority: Optional[str] = typer.Option(None, "--authority", "-a", help="Filter by authority level"),
        category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category")
    ) -> None:
        """🔍 Search memory rules using semantic similarity."""
        
        async def _search_rules():
            try:
                memory_manager = await self.get_memory_manager()
                
                # Parse filters
                filters = {}
                if authority:
                    filters['authority_filter'] = AuthorityLevel(authority)
                if category:
                    filters['category_filter'] = MemoryCategory(category)
                
                # Search rules
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Searching memory rules...", total=None)
                    
                    results = await memory_manager.search_rules(query, limit, **filters)
                
                if not results:
                    console.print(f"[yellow]No rules found matching '{query}'[/yellow]")
                    return
                
                # Display results
                table = Table(title=f"Search Results for '{query}'")
                table.add_column("Score", style="cyan", width=8)
                table.add_column("Authority", justify="center", width=10)
                table.add_column("Category", style="dim", width=12)
                table.add_column("Rule", style="white")
                
                for rule, score in results:
                    authority_style = "red" if rule.authority == AuthorityLevel.ABSOLUTE else "yellow"
                    
                    table.add_row(
                        f"{score:.3f}",
                        f"[{authority_style}]{rule.authority.value}[/{authority_style}]",
                        rule.category.value,
                        rule.rule[:80] + "..." if len(rule.rule) > 80 else rule.rule
                    )
                
                console.print(table)
                console.print(f"\n[dim]Found {len(results)} results[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error searching rules: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_search_rules())
    
    def token_usage(
        self,
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed breakdown")
    ) -> None:
        """📊 Show memory system token usage statistics."""
        
        async def _token_usage():
            try:
                memory_manager = await self.get_memory_manager()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Calculating token usage...", total=None)
                    
                    usage = await memory_manager.get_token_usage()
                
                # Main usage panel
                usage_text = f"""[bold]Total Memory Usage[/bold]

🧠 Total Tokens: [cyan]{usage.total_tokens:,}[/cyan]
📊 Context Usage: [yellow]{usage.percentage:.1f}%[/yellow]
📏 Rules Count: [green]{usage.rules_count}[/green]
💾 Remaining: [dim]{usage.remaining_tokens:,} tokens[/dim]"""
                
                usage_panel = Panel(
                    usage_text,
                    title="🧠 Memory Token Usage",
                    border_style="blue"
                )
                console.print(usage_panel)
                
                if verbose:
                    # Detailed breakdown
                    breakdown_table = Table(title="Token Usage Breakdown")
                    breakdown_table.add_column("Category", style="cyan")
                    breakdown_table.add_column("Tokens", justify="right", style="white")
                    breakdown_table.add_column("Percentage", justify="right", style="dim")
                    
                    categories = [
                        ("Preferences", usage.preference_tokens),
                        ("Behaviors", usage.behavior_tokens),
                        ("Agent Library", usage.agent_library_tokens),
                        ("Knowledge", usage.knowledge_tokens),
                        ("Context", usage.context_tokens),
                    ]
                    
                    for name, tokens in categories:
                        if tokens > 0:
                            pct = (tokens / usage.total_tokens) * 100 if usage.total_tokens > 0 else 0
                            breakdown_table.add_row(name, f"{tokens:,}", f"{pct:.1f}%")
                    
                    console.print(breakdown_table)
                    
                    # Authority breakdown
                    authority_table = Table(title="Authority Level Breakdown")
                    authority_table.add_column("Authority", style="cyan")
                    authority_table.add_column("Tokens", justify="right", style="white")
                    authority_table.add_column("Percentage", justify="right", style="dim")
                    
                    auth_data = [
                        ("Absolute", usage.absolute_tokens),
                        ("Default", usage.default_tokens),
                    ]
                    
                    for name, tokens in auth_data:
                        if tokens > 0:
                            pct = (tokens / usage.total_tokens) * 100 if usage.total_tokens > 0 else 0
                            color = "red" if name == "Absolute" else "yellow"
                            authority_table.add_row(f"[{color}]{name}[/{color}]", f"{tokens:,}", f"{pct:.1f}%")
                    
                    console.print(authority_table)
                
                # Usage warnings
                if usage.percentage > 80:
                    console.print("[red]⚠️  High memory usage! Consider using 'wqm memory trim' to optimize.[/red]")
                elif usage.percentage > 60:
                    console.print("[yellow]⚠️  Memory usage is getting high. Monitor for optimization needs.[/yellow]")
                
            except Exception as e:
                console.print(f"[red]Error calculating token usage: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_token_usage())
    
    def trim_rules(
        self,
        target_tokens: int = typer.Option(3000, "--target", "-t", help="Target token count"),
        interactive: bool = typer.Option(True, "--interactive/--auto", help="Interactive optimization mode"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show suggestions without making changes")
    ) -> None:
        """✂️ Optimize memory rules to reduce token usage."""
        
        async def _trim_rules():
            try:
                memory_manager = await self.get_memory_manager()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Analyzing memory usage...", total=None)
                    
                    suggestions = await memory_manager.suggest_optimizations(target_tokens)
                
                if not suggestions["optimization_needed"]:
                    console.print("[green]✅ Memory usage is already within target limits[/green]")
                    return
                
                # Show current usage and suggestions
                current_tokens = suggestions["current_tokens"]
                tokens_to_reduce = suggestions["tokens_to_reduce"]
                
                console.print(f"[yellow]Current usage: {current_tokens:,} tokens[/yellow]")
                console.print(f"[blue]Target usage: {target_tokens:,} tokens[/blue]")
                console.print(f"[red]Need to reduce: {tokens_to_reduce:,} tokens[/red]")
                console.print()
                
                # Display suggestions
                console.print("[bold]Optimization Suggestions:[/bold]")
                for i, suggestion in enumerate(suggestions["suggestions"], 1):
                    console.print(f"{i}. {suggestion}")
                console.print()
                
                if dry_run:
                    console.print("[dim]--dry-run mode: No changes will be made[/dim]")
                    return
                
                if interactive:
                    console.print("[dim]Interactive optimization not yet implemented.[/dim]")
                    console.print("[dim]Use 'wqm memory list' to identify rules to remove manually.[/dim]")
                else:
                    console.print("[dim]Automatic optimization not yet implemented.[/dim]")
                    console.print("[dim]Please review suggestions and make changes manually.[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error optimizing rules: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_trim_rules())
    
    def analyze_conflicts(self,
                         verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed conflict information")
    ) -> None:
        """⚠️ Analyze conflicts between memory rules."""
        
        async def _analyze_conflicts():
            try:
                memory_manager = await self.get_memory_manager()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Analyzing rule conflicts...", total=None)
                    
                    conflicts = await memory_manager.analyze_all_conflicts()
                
                if not conflicts:
                    console.print("[green]✅ No conflicts found between memory rules[/green]")
                    return
                
                # Show conflict summary
                summary = memory_manager.conflict_detector.get_conflict_summary(conflicts)
                
                console.print(f"[red]Found {summary['total']} conflicts:[/red]")
                console.print()
                
                # Severity breakdown
                for severity, count in summary["by_severity"].items():
                    if count > 0:
                        color = {"critical": "red", "high": "orange", "medium": "yellow", "low": "blue"}[severity]
                        console.print(f"  [{color}]{severity.upper()}[/{color}]: {count} conflicts")
                
                console.print()
                
                # Show recommendations
                if summary["recommendations"]:
                    console.print("[bold]Recommendations:[/bold]")
                    for rec in summary["recommendations"]:
                        console.print(f"  • {rec}")
                    console.print()
                
                # Show detailed conflicts if verbose
                if verbose:
                    for i, conflict in enumerate(conflicts[:10], 1):  # Show first 10
                        severity_color = {
                            "critical": "red", 
                            "high": "orange", 
                            "medium": "yellow", 
                            "low": "blue"
                        }[conflict.severity]
                        
                        console.print(f"[bold]{i}. [{severity_color}]{conflict.severity.upper()}[/{severity_color}] {conflict.conflict_type.upper()} Conflict[/bold]")
                        console.print(f"   Confidence: {conflict.confidence:.2f}")
                        console.print(f"   Description: {conflict.description}")
                        console.print(f"   Rule 1: {conflict.rule1.rule}")
                        console.print(f"   Rule 2: {conflict.rule2.rule}")
                        
                        if conflict.resolution_suggestion:
                            console.print(f"   [dim]Suggestion: {conflict.resolution_suggestion}[/dim]")
                        
                        console.print()
                    
                    if len(conflicts) > 10:
                        console.print(f"[dim]... and {len(conflicts) - 10} more conflicts[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error analyzing conflicts: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_analyze_conflicts())
    
    def show_stats(self) -> None:
        """📈 Show comprehensive memory system statistics."""
        
        async def _show_stats():
            try:
                memory_manager = await self.get_memory_manager()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Gathering statistics...", total=None)
                    
                    stats = await memory_manager.get_memory_stats()
                
                # Collection statistics
                collection_stats = stats["collection"]
                
                collection_text = f"""[bold]Collection Statistics[/bold]

📊 Total Rules: [cyan]{collection_stats.get('total_rules', 0)}[/cyan]
💾 Collection Size: [dim]{collection_stats.get('collection_size_bytes', 0)} bytes[/dim]"""
                
                # Add category breakdown
                categories = collection_stats.get("categories", {})
                if categories:
                    collection_text += "\n\n[bold]By Category:[/bold]"
                    for category, count in categories.items():
                        if count > 0:
                            collection_text += f"\n  {category}: [green]{count}[/green]"
                
                # Add authority breakdown  
                authorities = collection_stats.get("authorities", {})
                if authorities:
                    collection_text += "\n\n[bold]By Authority:[/bold]"
                    for authority, count in authorities.items():
                        if count > 0:
                            color = "red" if authority == "absolute" else "yellow"
                            collection_text += f"\n  [{color}]{authority}[/{color}]: [green]{count}[/green]"
                
                collection_panel = Panel(
                    collection_text,
                    title="📊 Memory Collection",
                    border_style="blue"
                )
                console.print(collection_panel)
                
                # Token usage summary
                token_stats = stats["token_usage"]
                token_text = f"""[bold]Token Usage[/bold]

🧠 Total Tokens: [cyan]{token_stats['total_tokens']:,}[/cyan]
📊 Context Usage: [yellow]{token_stats['context_window']['percentage']:.1f}%[/yellow]
💾 Remaining: [dim]{token_stats['context_window']['remaining']:,} tokens[/dim]"""
                
                token_panel = Panel(
                    token_text,
                    title="🧠 Token Usage",
                    border_style="green"
                )
                console.print(token_panel)
                
                # Conflict summary
                conflicts = stats["conflicts"]
                if conflicts["total"] > 0:
                    conflict_text = f"[red]⚠️  {conflicts['total']} conflicts detected[/red]\n\n"
                    
                    for severity, count in conflicts["by_severity"].items():
                        if count > 0:
                            color = {"critical": "red", "high": "orange", "medium": "yellow", "low": "blue"}[severity]
                            conflict_text += f"[{color}]{severity.upper()}[/{color}]: {count}\n"
                    
                    if conflicts["recommendations"]:
                        conflict_text += "\n[bold]Recommendations:[/bold]\n"
                        for rec in conflicts["recommendations"][:3]:
                            conflict_text += f"• {rec}\n"
                else:
                    conflict_text = "[green]✅ No conflicts detected[/green]"
                
                conflict_panel = Panel(
                    conflict_text.strip(),
                    title="⚠️ Conflicts",
                    border_style="yellow" if conflicts["total"] > 0 else "green"
                )
                console.print(conflict_panel)
                
                console.print(f"[dim]Last updated: {stats['last_updated']}[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error showing statistics: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_show_stats())
    
    def remove_rule(
        self,
        rule_id: str = typer.Argument(..., help="Rule ID to remove"),
        force: bool = typer.Option(False, "--force", help="Skip confirmation prompt")
    ) -> None:
        """🗑️ Remove a memory rule."""
        
        async def _remove_rule():
            try:
                memory_manager = await self.get_memory_manager()
                
                # Get rule details first
                rule = await memory_manager.get_rule(rule_id)
                if not rule:
                    console.print(f"[red]Rule not found: {rule_id}[/red]")
                    raise typer.Exit(1)
                
                # Show rule details
                console.print(f"[yellow]Rule to remove:[/yellow]")
                console.print(f"  ID: {rule.id}")
                console.print(f"  Rule: {rule.rule}")
                console.print(f"  Authority: {rule.authority.value}")
                console.print(f"  Category: {rule.category.value}")
                
                # Confirm deletion unless forced
                if not force:
                    if not Confirm.ask("Are you sure you want to remove this rule?"):
                        console.print("[yellow]Deletion cancelled[/yellow]")
                        return
                
                # Delete the rule
                success = await memory_manager.delete_rule(rule_id)
                
                if success:
                    console.print(f"[green]✅ Memory rule removed successfully[/green]")
                else:
                    console.print(f"[red]❌ Failed to remove rule[/red]")
                    raise typer.Exit(1)
                
            except Exception as e:
                console.print(f"[red]Error removing rule: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_remove_rule())
    
    def edit_rule(
        self,
        rule_id: str = typer.Argument(..., help="Rule ID to edit")
    ) -> None:
        """✏️ Edit an existing memory rule."""
        
        async def _edit_rule():
            try:
                memory_manager = await self.get_memory_manager()
                
                # Get existing rule
                rule = await memory_manager.get_rule(rule_id)
                if not rule:
                    console.print(f"[red]Rule not found: {rule_id}[/red]")
                    raise typer.Exit(1)
                
                console.print(f"[blue]Editing rule: {rule_id}[/blue]")
                console.print()
                
                # Interactive editing
                new_rule_text = Prompt.ask("Rule text", default=rule.rule)
                new_authority = Prompt.ask(
                    "Authority level",
                    choices=["absolute", "default"],
                    default=rule.authority.value
                )
                new_category = Prompt.ask(
                    "Category",
                    choices=[c.value for c in MemoryCategory],
                    default=rule.category.value
                )
                
                scope_str = ", ".join(rule.scope) if rule.scope else ""
                new_scope_input = Prompt.ask("Scope (comma-separated)", default=scope_str)
                new_scope = [s.strip() for s in new_scope_input.split(",")] if new_scope_input else []
                
                tags_str = ", ".join(rule.tags) if rule.tags else ""
                new_tags_input = Prompt.ask("Tags (comma-separated)", default=tags_str)
                new_tags = [t.strip() for t in new_tags_input.split(",")] if new_tags_input else []
                
                # Update rule
                rule.rule = new_rule_text
                rule.authority = AuthorityLevel(new_authority)
                rule.category = MemoryCategory(new_category)
                rule.scope = new_scope
                rule.tags = new_tags
                
                # Save changes
                success = await memory_manager.update_rule(rule)
                
                if success:
                    console.print(f"[green]✅ Rule updated successfully[/green]")
                else:
                    console.print(f"[red]❌ Failed to update rule[/red]")
                    raise typer.Exit(1)
                
            except Exception as e:
                console.print(f"[red]Error editing rule: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_edit_rule())
    
    def export_rules(
        self,
        output_file: str = typer.Argument(..., help="Output file path"),
        format_type: str = typer.Option("json", "--format", "-f", help="Export format: json, yaml")
    ) -> None:
        """📤 Export memory rules to file."""
        
        async def _export_rules():
            try:
                memory_manager = await self.get_memory_manager()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Exporting rules...", total=None)
                    
                    rules_data = await memory_manager.export_rules()
                
                output_path = Path(output_file)
                
                if format_type == "json":
                    with open(output_path, 'w') as f:
                        json.dump(rules_data, f, indent=2, default=str)
                elif format_type == "yaml":
                    try:
                        import yaml
                        with open(output_path, 'w') as f:
                            yaml.dump(rules_data, f, default_flow_style=False)
                    except ImportError:
                        console.print("[red]PyYAML not installed. Use: pip install pyyaml[/red]")
                        raise typer.Exit(1)
                else:
                    console.print(f"[red]Unsupported format: {format_type}[/red]")
                    raise typer.Exit(1)
                
                console.print(f"[green]✅ Exported {len(rules_data)} rules to {output_path}[/green]")
                
            except Exception as e:
                console.print(f"[red]Error exporting rules: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_export_rules())
    
    def import_rules(
        self,
        input_file: str = typer.Argument(..., help="Input file path"),
        overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing rules"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be imported without making changes")
    ) -> None:
        """📥 Import memory rules from file."""
        
        async def _import_rules():
            try:
                input_path = Path(input_file)
                
                if not input_path.exists():
                    console.print(f"[red]File not found: {input_path}[/red]")
                    raise typer.Exit(1)
                
                # Load rules data
                if input_path.suffix.lower() == '.json':
                    with open(input_path) as f:
                        rules_data = json.load(f)
                elif input_path.suffix.lower() in ['.yaml', '.yml']:
                    try:
                        import yaml
                        with open(input_path) as f:
                            rules_data = yaml.safe_load(f)
                    except ImportError:
                        console.print("[red]PyYAML not installed. Use: pip install pyyaml[/red]")
                        raise typer.Exit(1)
                else:
                    console.print(f"[red]Unsupported file format: {input_path.suffix}[/red]")
                    raise typer.Exit(1)
                
                if not isinstance(rules_data, list):
                    console.print("[red]Invalid file format: expected list of rules[/red]")
                    raise typer.Exit(1)
                
                console.print(f"[blue]Found {len(rules_data)} rules to import[/blue]")
                
                if dry_run:
                    console.print("[dim]--dry-run mode: No changes will be made[/dim]")
                    for rule_dict in rules_data[:5]:  # Show first 5
                        console.print(f"  - {rule_dict.get('rule', 'Unknown rule')}")
                    if len(rules_data) > 5:
                        console.print(f"  ... and {len(rules_data) - 5} more")
                    return
                
                memory_manager = await self.get_memory_manager()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Importing rules...", total=None)
                    
                    imported, skipped, errors = await memory_manager.import_rules(
                        rules_data, overwrite_existing=overwrite
                    )
                
                console.print(f"[green]✅ Import completed:[/green]")
                console.print(f"  Imported: {imported}")
                console.print(f"  Skipped: {skipped}")
                
                if errors:
                    console.print(f"  [red]Errors: {len(errors)}[/red]")
                    for error in errors[:3]:  # Show first 3 errors
                        console.print(f"    • {error}")
                    if len(errors) > 3:
                        console.print(f"    ... and {len(errors) - 3} more errors")
                
            except Exception as e:
                console.print(f"[red]Error importing rules: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(_import_rules())
    
    def _display_rules_table(self, rules: List[MemoryRule], verbose: bool = False):
        """Display rules in a formatted table."""
        
        if not rules:
            console.print("[yellow]No memory rules found[/yellow]")
            return
        
        table = Table(title="💭 Memory Rules")
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Authority", justify="center", width=10)
        table.add_column("Category", style="dim", width=12)
        table.add_column("Rule", style="white")
        
        if verbose:
            table.add_column("Scope", style="dim", width=15)
            table.add_column("Source", style="dim", width=10)
            table.add_column("Usage", style="green", width=8)
        
        for rule in rules:
            authority_style = "red" if rule.authority == AuthorityLevel.ABSOLUTE else "yellow"
            rule_text = rule.rule[:60] + "..." if len(rule.rule) > 60 else rule.rule
            
            row = [
                rule.id[:8],
                f"[{authority_style}]{rule.authority.value}[/{authority_style}]",
                rule.category.value,
                rule_text
            ]
            
            if verbose:
                scope_text = ", ".join(rule.scope[:2]) if rule.scope else "global"
                if len(rule.scope) > 2:
                    scope_text += "..."
                
                row.extend([
                    scope_text,
                    rule.source[:10],
                    str(rule.use_count)
                ])
            
            table.add_row(*row)
        
        console.print(table)
        console.print(f"\n[dim]Total: {len(rules)} rules[/dim]")
        
        if not verbose and len(rules) > 0:
            console.print("[dim]Use --verbose for detailed information[/dim]")