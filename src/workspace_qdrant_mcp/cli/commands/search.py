
from ...observability import get_logger
logger = get_logger(__name__)
"""Search CLI commands.

This module provides command-line search interface with different
context modes and research capabilities.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from ...core.client import create_qdrant_client
from ...core.config import Config
from ...tools.search import search_workspace
from ...utils.project_detection import ProjectDetector

def _print_table_as_text(table_data):
    """Helper to print table data as plain text."""
    # This helper is no longer needed as we've replaced all Rich tables
    # with direct plain text formatting
    pass

# Create the search app
search_app = typer.Typer(
    help="Command-line search interface",
    no_args_is_help=True
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

@search_app.command("project")
def search_project(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Minimum similarity threshold"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, detailed"),
    collections: list[str] | None = typer.Option(None, "--collection", "-c", help="Specific collections to search"),
    include_content: bool = typer.Option(False, "--content", help="Include document content in results"),
):
    """Search current project collections."""
    handle_async(_search_project(query, limit, threshold, format, collections, include_content))

@search_app.command("collection")
def search_collection(
    query: str = typer.Argument(..., help="Search query"),
    collection: str = typer.Option(..., "--collection", "-c", help="Collection name to search"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Minimum similarity threshold"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, detailed"),
    include_content: bool = typer.Option(False, "--content", help="Include document content in results"),
    with_vectors: bool = typer.Option(False, "--vectors", help="Include vector information"),
):
    """Search specific collection."""
    handle_async(_search_collection(query, collection, limit, threshold, format, include_content, with_vectors))

@search_app.command("global")
def search_global(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Minimum similarity threshold"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, detailed"),
    exclude_projects: bool = typer.Option(False, "--exclude-projects", help="Search only global/library collections"),
    include_content: bool = typer.Option(False, "--content", help="Include document content in results"),
):
    """Search global collections (library and system)."""
    handle_async(_search_global(query, limit, threshold, format, exclude_projects, include_content))

@search_app.command("all")
def search_all(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Minimum similarity threshold"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, detailed"),
    group_by_collection: bool = typer.Option(True, "--group/--no-group", help="Group results by collection"),
    include_content: bool = typer.Option(False, "--content", help="Include document content in results"),
):
    """Search all collections."""
    handle_async(_search_all(query, limit, threshold, format, group_by_collection, include_content))

@search_app.command("memory")
def search_memory(
    query: str = typer.Argument(..., help="Search query for memory/knowledge graph"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    category: str | None = typer.Option(None, "--category", help="Filter by memory category"),
    authority: str | None = typer.Option(None, "--authority", help="Filter by authority level"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
):
    """Search memory rules and knowledge graph."""
    handle_async(_search_memory(query, limit, category, authority, format))

@search_app.command("research")
def research_query(
    query: str = typer.Argument(..., help="Research query"),
    mode: str = typer.Option("comprehensive", "--mode", "-m", help="Research mode: quick, standard, comprehensive, deep"),
    collections: list[str] | None = typer.Option(None, "--collection", "-c", help="Collections to include in research"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Save research report to file"),
    format: str = typer.Option("report", "--format", "-f", help="Output format: report, json, markdown"),
):
    """Advanced research mode with analysis."""
    handle_async(_research_query(query, mode, collections, output_file, format))

# Async implementation functions
async def _search_project(
    query: str,
    limit: int,
    threshold: float,
    format: str,
    collections: list[str] | None,
    include_content: bool
):
    """Search current project collections."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        # Detect current project
        detector = ProjectDetector(config.workspace.github_user if hasattr(config, 'workspace') else None)
        projects = detector.detect_projects([Path.cwd()])
        current_project = projects[0].name if projects else "unknown"

        print(f"Searching project: {current_project}")

        # Get project collections
        all_collections = await client.list_collections()
        project_prefix = f"{config.workspace.collection_prefix}{current_project}_" if hasattr(config, 'workspace') else f"{current_project}_"

        if collections:
            # User specified collections - validate they belong to project
            project_collections = []
            for col_name in collections:
                if col_name.startswith(project_prefix) or col_name in [c.get("name") for c in all_collections]:
                    project_collections.append(col_name)
                else:
                    print(f"Warning: Collection not found: {col_name}")
        else:
            # Find all project collections
            project_collections = [
                col.get("name") for col in all_collections
                if col.get("name", "").startswith(project_prefix)
            ]

        if not project_collections:
            print(f"No collections found for project '{current_project}'")
            print("Try running 'wqm ingest folder' to create collections first")
            return

        # Search across project collections
        all_results = []
        for collection_name in project_collections:
            try:
                results = await search_workspace(
                    query=query,
                    collection_name=collection_name,
                    limit=limit,
                    threshold=threshold
                )

                for result in results:
                    result["collection"] = collection_name
                    all_results.append(result)

            except Exception as e:
                print(f"Error: Search failed for {collection_name}: {e}")
                continue

        # Sort by score
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        all_results = all_results[:limit]

        _display_search_results(all_results, query, format, include_content)

    except Exception as e:
        print(f"Error: Project search failed: {e}")
        raise typer.Exit(1)

async def _search_collection(
    query: str,
    collection: str,
    limit: int,
    threshold: float,
    format: str,
    include_content: bool,
    with_vectors: bool
):
    """Search specific collection."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        print(f"Searching collection: {collection}")

        # Verify collection exists
        all_collections = await client.list_collections()
        collection_names = [col.get("name") for col in all_collections]

        if collection not in collection_names:
            print(f"Error: Collection not found: {collection}")
            print(f"Available collections: {', '.join(collection_names)}")
            raise typer.Exit(1)

        # Perform search
        results = await search_workspace(
            query=query,
            collection_name=collection,
            limit=limit,
            threshold=threshold
        )

        # Add collection info to results
        for result in results:
            result["collection"] = collection

        _display_search_results(results, query, format, include_content, with_vectors)

    except Exception as e:
        print(f"Error: Collection search failed: {e}")
        raise typer.Exit(1)

async def _search_global(
    query: str,
    limit: int,
    threshold: float,
    format: str,
    exclude_projects: bool,
    include_content: bool
):
    """Search global collections."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        print("Searching global collections")

        # Get all collections
        all_collections = await client.list_collections()

        # Filter for global collections
        global_collections = []
        for col in all_collections:
            name = col.get("name", "")

            # Include library collections (start with _)
            if name.startswith("_"):
                global_collections.append(name)
            # Include system collections if not excluding projects
            elif not exclude_projects and not any(name.startswith(f"{prefix}_") for prefix in ["docs", "code", "notes"]):
                global_collections.append(name)

        if not global_collections:
            print("No global collections found")
            return

        print(f"Searching {len(global_collections)} global collections")

        # Search across global collections
        all_results = []
        for collection_name in global_collections:
            try:
                results = await search_workspace(
                    query=query,
                    collection_name=collection_name,
                    limit=limit,
                    threshold=threshold
                )

                for result in results:
                    result["collection"] = collection_name
                    all_results.append(result)

            except Exception as e:
                print(f"Warning: Search failed for {collection_name}: {e}")
                continue

        # Sort by score
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        all_results = all_results[:limit]

        _display_search_results(all_results, query, format, include_content)

    except Exception as e:
        print(f"Error: Global search failed: {e}")
        raise typer.Exit(1)

async def _search_all(
    query: str,
    limit: int,
    threshold: float,
    format: str,
    group_by_collection: bool,
    include_content: bool
):
    """Search all collections."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        print("Searching all collections")

        # Get all collections
        all_collections = await client.list_collections()
        collection_names = [col.get("name") for col in all_collections]

        if not collection_names:
            print("No collections found")
            return

        print(f"Searching {len(collection_names)} collections")

        # Search across all collections
        all_results = []
        for collection_name in collection_names:
            try:
                results = await search_workspace(
                    query=query,
                    collection_name=collection_name,
                    limit=limit,
                    threshold=threshold
                )

                for result in results:
                    result["collection"] = collection_name
                    all_results.append(result)

            except Exception as e:
                print(f"Warning: Search failed for {collection_name}: {e}")
                continue

        # Sort by score
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        all_results = all_results[:limit]

        if group_by_collection:
            _display_grouped_search_results(all_results, query, format, include_content)
        else:
            _display_search_results(all_results, query, format, include_content)

    except Exception as e:
        print(f"Error: Search all failed: {e}")
        raise typer.Exit(1)

async def _search_memory(
    query: str,
    limit: int,
    category: str | None,
    authority: str | None,
    format: str
):
    """Search memory rules and knowledge graph."""
    try:
        from ...core.collection_naming import create_naming_manager
        from ...core.memory import AuthorityLevel, MemoryCategory, create_memory_manager

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        naming_manager = create_naming_manager(config.workspace.global_collections)
        memory_manager = create_memory_manager(client, naming_manager)

        print("Searching memory rules")

        # Convert filters to enums if provided
        category_enum = MemoryCategory(category) if category else None
        authority_enum = AuthorityLevel(authority) if authority else None

        # Search memory rules
        matching_rules = await memory_manager.search_memory_rules(
            query=query,
            limit=limit,
            category=category_enum,
            authority=authority_enum
        )

        if not matching_rules:
            print(f"No memory rules found matching '{query}'")
            return

        if format == "json":
            rules_data = []
            for rule in matching_rules:
                rules_data.append({
                    "id": rule.id,
                    "name": rule.name,
                    "rule": rule.rule,
                    "category": rule.category.value,
                    "authority": rule.authority.value,
                    "scope": rule.scope,
                    "source": rule.source,
                    "relevance": rule.get("relevance", 1.0)
                })
            print(json.dumps(rules_data, indent=2))
        else:
            # Display as table
            print(f"Memory Search Results ({len(matching_rules)} found)")
            print("=" * 80)
            print(f"{'ID':<8} {'Name':<20} {'Rule':<30} {'Category':<12} {'Authority':<10}")
            print("-" * 80)

            for rule in matching_rules:
                rule_id = rule.id[-8:]
                name = rule.name[:19]
                rule_text = rule.rule[:27] + "..." if len(rule.rule) > 30 else rule.rule
                category = rule.category.value[:11]
                authority = rule.authority.value

                print(f"{rule_id:<8} {name:<20} {rule_text:<30} {category:<12} {authority:<10}")

    except Exception as e:
        print(f"Error: Memory search failed: {e}")
        raise typer.Exit(1)

async def _research_query(
    query: str,
    mode: str,
    collections: list[str] | None,
    output_file: str | None,
    format: str
):
    """Advanced research mode with analysis."""
    try:
        print(f"Research Mode: {mode.upper()}")
        print(f"Query: {query}")

        # TODO: Implement comprehensive research functionality
        # This will be part of Task 13: Advanced search modes implementation

        research_modes = {
            "quick": "Fast overview with top 5 results per collection",
            "standard": "Balanced search with analysis and 10 results per collection",
            "comprehensive": "Deep search with cross-references and 20 results per collection",
            "deep": "Exhaustive analysis with relationships and unlimited results"
        }

        mode_description = research_modes.get(mode, "Unknown mode")
        print(f"Mode: {mode_description}")

        if collections:
            print(f"Collections: {', '.join(collections)}")

        print("\nWarning: Research modes are not yet fully implemented")
        print("This advanced functionality will be available in Task 13: Advanced search modes")
        print("For now, use the basic search commands: project, collection, global, all")

        # Basic search as fallback
        print("\nPerforming basic search as fallback...")
        await _search_all(query, 20, 0.5, format, True, False)

    except Exception as e:
        print(f"Error: Research query failed: {e}")
        raise typer.Exit(1)

def _display_search_results(
    results: list[dict[str, Any]],
    query: str,
    format: str,
    include_content: bool,
    with_vectors: bool = False
):
    """Display search results in specified format."""

    if not results:
        print(f"No results found for: '{query}'")
        return

    if format == "json":
        # Clean results for JSON output
        json_results = []
        for result in results:
            clean_result = {
                "score": result.get("score", 0),
                "collection": result.get("collection", "unknown"),
                "title": result.get("title", "Untitled"),
                "content_preview": result.get("content", "")[:200] + "..." if result.get("content") else "",
            }
            if include_content:
                clean_result["full_content"] = result.get("content", "")
            if with_vectors and "vector" in result:
                clean_result["vector"] = result["vector"]

            json_results.append(clean_result)

        print(json.dumps(json_results, indent=2))
        return

    if format == "detailed":
        _display_detailed_results(results, query, include_content)
        return

    # Plain text table format
    print(f"Search Results for: '{query}' ({len(results)} found)")
    print("=" * 80)
    print(f"{'Score':<8} {'Collection':<15} {'Title':<25} {'Preview':<30}")
    print("-" * 80)

    for result in results:
        score = f"{result.get('score', 0):.3f}"
        collection = result.get("collection", "unknown")[:14]
        title = result.get("title", "Untitled")[:24]
        
        content = result.get("content", "")
        preview = content[:29] if len(content) > 30 else content
        preview = preview.replace('\n', ' ').replace('\t', ' ')

        print(f"{score:<8} {collection:<15} {title:<25} {preview:<30}")

    # Summary
    avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    print(f"\nAverage score: {avg_score:.3f} | Best match: {results[0].get('score', 0):.3f}")

def _display_detailed_results(results: list[dict[str, Any]], query: str, include_content: bool):
    """Display detailed search results with full content."""

    print(f"Detailed Results for: '{query}'")
    print(f"{len(results)} results found\n")

    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        collection = result.get("collection", "unknown")
        title = result.get("title", "Untitled")

        # Result header
        result_header = f"[bold cyan]#{i}[/bold cyan] [bold]{title}[/bold]"
        result_info = f"[dim]Collection: {collection} | Score: {score:.3f}[/dim]"

        print(result_header)
        print(result_info)

        # Content
        content = result.get("content", "")
        if include_content and content:
            # Try to syntax highlight if it looks like code
            if any(indicator in content.lower() for indicator in ["def ", "class ", "import ", "function", "var ", "const "]):
                try:
                    syntax = Syntax(content[:500], "python", theme="monokai", line_numbers=False)
                    # Code syntax highlighting not available in plain text mode
                    print(f"Code content (first 500 chars): {content[:500]}")
                except:
                    print(f"Content (first 500 chars): {content[:500]}")
            else:
                print(f"Content (first 500 chars): {content[:500]}")

            if len(content) > 500:
                print("... (truncated)")
        else:
            # Show preview
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"{preview}")

        if i < len(results):
            print()  # Spacing between results

def _display_grouped_search_results(results: list[dict[str, Any]], query: str, format: str, include_content: bool):
    """Display search results grouped by collection."""

    if not results:
        print(f"No results found for: '{query}'")
        return

    # Group results by collection
    grouped = {}
    for result in results:
        collection = result.get("collection", "unknown")
        if collection not in grouped:
            grouped[collection] = []
        grouped[collection].append(result)

    print(f"Search Results for: '{query}'")
    print(f"Found results in {len(grouped)} collections\n")

    for collection, collection_results in grouped.items():
        # Collection header
        collection_type = "Library" if collection.startswith("_") else "Project"
        print(f"Collection: {collection} ({collection_type}) - {len(collection_results)} results")

        # Results table for this collection
        # Display results in plain text format
        table.add_column("Score", justify="right", style="green", width=8)
        table.add_column("Title", style="bold", width=25)
        table.add_column("Preview", width=60)

        for result in collection_results[:5]:  # Show top 5 per collection
            score = f"{result.get('score', 0):.3f}"
            title = result.get("title", "Untitled")[:23] + "..." if len(result.get("title", "")) > 25 else result.get("title", "Untitled")

            content = result.get("content", "")
            preview = content[:57] + "..." if len(content) > 60 else content

            table.add_row(score, title, preview)

        # Convert Rich table to plain text format
    print(f"{'Score':<8} {'Collection':<20} {'Content Preview':<50}")
    print("-" * 78)
    
    for i, result in enumerate(results[:limit] if limit else results):
        score = f"{result.get('score', 0):.3f}"
        collection = result.get('collection_name', 'unknown')[:18]
        preview = str(result.get('content', ''))[:47] if result.get('content') else 'N/A'
        preview = preview.replace('\n', ' ').replace('\t', ' ')
        print(f"{score:<8} {collection:<20} {preview:<50}")

        if len(collection_results) > 5:
            print(f"... and {len(collection_results) - 5} more results in this collection")

        print()  # Spacing between collections
