"""
Example usage of the memory system.

This example demonstrates how to:
1. Initialize the memory system
2. Add memory rules for different categories
3. Handle conversational updates
4. Search memory rules
5. Detect and resolve conflicts
6. Integrate with Claude Code sessions
"""

import asyncio
import logging
from datetime import datetime

from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.core.client import create_qdrant_client
from workspace_qdrant_mcp.core.collection_naming import create_naming_manager
from workspace_qdrant_mcp.core.memory import (
    create_memory_manager,
    MemoryCategory,
    AuthorityLevel,
    parse_conversational_memory_update
)
from workspace_qdrant_mcp.core.claude_integration import create_claude_integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def memory_system_example():
    """
    Comprehensive example of memory system usage.
    """
    print("=== Memory System Example ===\n")
    
    # Initialize the system
    config = Config()
    client = create_qdrant_client(config.qdrant_client_config)
    naming_manager = create_naming_manager(config.workspace.global_collections)
    memory_manager = create_memory_manager(client, naming_manager)
    claude_integration = create_claude_integration(memory_manager)
    
    print("1. Initializing memory collection...")
    await memory_manager.initialize_memory_collection()
    print("✓ Memory collection initialized\n")
    
    # Add some example memory rules
    print("2. Adding memory rules...")
    
    # User preference
    pref_rule_id = await memory_manager.add_memory_rule(
        category=MemoryCategory.PREFERENCE,
        name="python-package-manager",
        rule="Use uv exclusively for Python package management",
        authority=AuthorityLevel.DEFAULT,
        scope=["python", "development"],
        source="user_explicit"
    )
    print(f"✓ Added preference rule: {pref_rule_id}")
    
    # LLM behavior rule (absolute)
    behavior_rule_id = await memory_manager.add_memory_rule(
        category=MemoryCategory.BEHAVIOR,
        name="atomic-commits",
        rule="Always make atomic commits following conventional commit format",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["git", "development"],
        source="user_explicit"
    )
    print(f"✓ Added behavior rule: {behavior_rule_id}")
    
    # Agent definition
    agent_rule_id = await memory_manager.add_memory_rule(
        category=MemoryCategory.AGENT,
        name="python-pro",
        rule="Expert Python developer with modern practices. Capabilities: fastapi, async, testing, type-hints. Deploy cost: medium",
        authority=AuthorityLevel.DEFAULT,
        scope=["python", "development"],
        source="user_explicit"
    )
    print(f"✓ Added agent rule: {agent_rule_id}")
    print()
    
    # List all rules
    print("3. Listing all memory rules...")
    rules = await memory_manager.list_memory_rules()
    for rule in rules:
        authority_symbol = "🔒" if rule.authority == AuthorityLevel.ABSOLUTE else "⚙️"
        print(f"   {authority_symbol} [{rule.category.value}] {rule.name}: {rule.rule[:50]}...")
    print()
    
    # Get memory statistics
    print("4. Memory usage statistics...")
    stats = await memory_manager.get_memory_stats()
    print(f"   Total rules: {stats.total_rules}")
    print(f"   Estimated tokens: {stats.estimated_tokens}")
    print(f"   By category: {dict(stats.rules_by_category)}")
    print(f"   By authority: {dict(stats.rules_by_authority)}")
    print()
    
    # Test conversational updates
    print("5. Testing conversational memory updates...")
    conversational_messages = [
        "Note: call me Chris",
        "For future reference, always use TypeScript strict mode",
        "Remember that I prefer pytest over unittest",
        "Always validate input parameters",
        "Never ignore type errors"
    ]
    
    for message in conversational_messages:
        result = await claude_integration.handle_conversational_update(message)
        if result["detected"]:
            print(f"   ✓ Detected and added: {message}")
            print(f"     → Rule: {result['rule_text']}")
            print(f"     → Category: {result['category']}, Authority: {result['authority']}")
        else:
            print(f"   ○ No pattern detected: {message}")
    print()
    
    # Search memory rules
    print("6. Searching memory rules...")
    search_queries = [
        "python package management",
        "commit guidelines",
        "testing preferences"
    ]
    
    for query in search_queries:
        results = await memory_manager.search_memory_rules(query, limit=3)
        print(f"   Query: '{query}'")
        if results:
            for rule, score in results:
                print(f"     • {rule.name} (score: {score:.3f}): {rule.rule[:40]}...")
        else:
            print("     (no results)")
    print()
    
    # Detect conflicts
    print("7. Detecting memory conflicts...")
    conflicts = await memory_manager.detect_conflicts()
    if conflicts:
        print(f"   Found {len(conflicts)} conflict(s):")
        for i, conflict in enumerate(conflicts, 1):
            print(f"     Conflict {i}: {conflict.description}")
            print(f"       Rule 1: {conflict.rule1.name}")
            print(f"       Rule 2: {conflict.rule2.name}")
            print(f"       Confidence: {conflict.confidence:.1%}")
    else:
        print("   ✓ No conflicts detected")
    print()
    
    # Claude Code session initialization
    print("8. Claude Code session initialization...")
    session_data = await claude_integration.initialize_session()
    print(f"   Status: {session_data['status']}")
    print(f"   Total rules loaded: {session_data['memory_stats']['total_rules']}")
    print(f"   Estimated tokens: {session_data['memory_stats']['estimated_tokens']}")
    print(f"   Conflicts detected: {session_data['conflicts_detected']}")
    
    if session_data["status"] == "conflicts_detected":
        print("   ⚠️  Conflicts require resolution before session can start")
    else:
        print("   ✓ Session ready for rule injection")
    print()
    
    # Show formatted system context (preview)
    print("9. System context for Claude Code injection (preview)...")
    system_context = await claude_integration.format_system_rules_for_injection()
    # Show first few lines
    context_lines = system_context.split('\n')[:10]
    for line in context_lines:
        print(f"   {line}")
    print(f"   ... ({len(context_lines)} more lines)")
    print()
    
    # Demonstrate rule updates
    print("10. Updating memory rules...")
    updated_rules = await memory_manager.list_memory_rules(category=MemoryCategory.PREFERENCE)
    if updated_rules:
        rule_to_update = updated_rules[0]
        success = await memory_manager.update_memory_rule(
            rule_to_update.id,
            {"rule": f"{rule_to_update.rule} (updated at {datetime.now().strftime('%H:%M')})"}
        )
        if success:
            print(f"   ✓ Updated rule: {rule_to_update.name}")
        else:
            print(f"   ✗ Failed to update rule: {rule_to_update.name}")
    print()
    
    # Final statistics
    print("11. Final memory statistics...")
    final_stats = await memory_manager.get_memory_stats()
    print(f"    Final rule count: {final_stats.total_rules}")
    print(f"    Final token estimate: {final_stats.estimated_tokens}")
    print(f"    Memory system ready for production use! 🎉")


async def conversational_parsing_examples():
    """
    Examples of conversational memory update parsing.
    """
    print("\n=== Conversational Parsing Examples ===\n")
    
    test_messages = [
        # Note patterns
        "Note: call me Chris",
        "Note: I prefer dark mode",
        "Reminder: use semantic versioning",
        
        # Future reference patterns  
        "For future reference, always use TypeScript strict mode",
        "For future reference use pytest for new projects",
        
        # Remember patterns
        "Remember that I prefer uv for Python",
        "Remember I like to use atomic commits",
        
        # Always/Never patterns
        "Always make atomic commits",
        "Never ignore TypeScript errors",
        "Always validate input parameters",
        
        # Non-memory messages
        "This is just a regular message",
        "Can you help me with this code?",
        "What's the weather like today?"
    ]
    
    for message in test_messages:
        result = parse_conversational_memory_update(message)
        if result:
            print(f"✓ '{message}'")
            print(f"   → Category: {result['category'].value}")
            print(f"   → Rule: {result['rule']}")
            print(f"   → Authority: {result['authority'].value}")
            print(f"   → Source: {result['source']}")
        else:
            print(f"○ '{message}' - no pattern detected")
        print()


async def main():
    """
    Main example function.
    """
    try:
        await memory_system_example()
        await conversational_parsing_examples()
        
    except Exception as e:
        print(f"Error running example: {e}")
        logger.exception("Example failed")


if __name__ == "__main__":
    asyncio.run(main())