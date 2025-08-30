#!/usr/bin/env python3
"""
Memory System Demonstration

This script demonstrates the key features of the memory system including:
- Creating and managing memory rules
- Detecting conflicts between rules
- Token usage optimization
- Conversational memory updates
- Claude Code session integration
"""

import asyncio
import os
from pathlib import Path
from typing import List

# Add src to path for demo
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workspace_qdrant_mcp.memory.manager import MemoryManager
from workspace_qdrant_mcp.memory.types import (
    MemoryRule, AuthorityLevel, MemoryCategory, 
    MemoryContext, ClaudeCodeSession
)
from workspace_qdrant_mcp.core.config import Config


async def demo_memory_system():
    """Demonstrate the memory system capabilities."""
    
    print("🧠 Memory System Demonstration")
    print("=" * 50)
    
    # Note: This demo uses mock components since we don't have a real Qdrant instance
    # In practice, you would connect to a running Qdrant server
    
    try:
        # Initialize memory manager (with mocked components for demo)
        config = Config()
        memory_manager = MemoryManager(config)
        
        print("\n📋 Step 1: Creating Memory Rules")
        print("-" * 30)
        
        # Create some example memory rules
        rules = [
            MemoryRule(
                rule="Always use uv for Python package management",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["python"],
                tags=["python", "tooling"],
                source="user_cli"
            ),
            MemoryRule(
                rule="Make atomic commits with conventional format",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["git", "development"],
                tags=["git", "commits", "best-practices"],
                source="user_cli"
            ),
            MemoryRule(
                rule="Prefer FastAPI over Flask for new projects",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                scope=["python", "web"],
                tags=["python", "web", "framework"],
                source="user_cli"
            ),
            MemoryRule(
                rule="User name is Chris",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["user_preference"],
                tags=["user", "identity"],
                source="conversation"
            ),
        ]
        
        # Display the rules
        for i, rule in enumerate(rules, 1):
            authority_symbol = "🔴" if rule.authority == AuthorityLevel.ABSOLUTE else "🟡"
            print(f"{i}. {authority_symbol} [{rule.category.value.upper()}] {rule.rule}")
            if rule.scope:
                print(f"   Scope: {', '.join(rule.scope)}")
            if rule.tags:
                print(f"   Tags: {', '.join(rule.tags)}")
            print()
        
        print("\n🔍 Step 2: Token Usage Analysis")
        print("-" * 30)
        
        # Demonstrate token counting
        from workspace_qdrant_mcp.memory.token_counter import TokenCounter
        
        token_counter = TokenCounter()
        token_usage = token_counter.count_rules_tokens(rules)
        
        print(f"Total Rules: {token_usage.rules_count}")
        print(f"Total Tokens: {token_usage.total_tokens}")
        print(f"Context Usage: {token_usage.percentage:.1f}%")
        print(f"Remaining: {token_usage.remaining_tokens:,} tokens")
        
        print("\nBy Category:")
        print(f"  Preferences: {token_usage.preference_tokens} tokens")
        print(f"  Behaviors: {token_usage.behavior_tokens} tokens")
        print(f"  Knowledge: {token_usage.knowledge_tokens} tokens")
        
        print("\nBy Authority:")
        print(f"  🔴 Absolute: {token_usage.absolute_tokens} tokens")
        print(f"  🟡 Default: {token_usage.default_tokens} tokens")
        
        print("\n⚠️ Step 3: Conflict Detection")
        print("-" * 30)
        
        # Demonstrate conflict detection with rule-based analysis
        from workspace_qdrant_mcp.memory.conflict_detector import ConflictDetector
        
        conflict_detector = ConflictDetector(enable_ai_analysis=False)  # Disable AI for demo
        
        # Create a conflicting rule
        conflicting_rule = MemoryRule(
            rule="Never use uv, always use pip for Python",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"],
            tags=["python", "tooling"],
            source="user_cli"
        )
        
        print("Adding conflicting rule:")
        print(f"🔴 [PREFERENCE] {conflicting_rule.rule}")
        
        # Detect conflicts
        conflicts = await conflict_detector.detect_conflicts(conflicting_rule, rules)
        
        if conflicts:
            print(f"\n⚠️ Found {len(conflicts)} conflicts:")
            for conflict in conflicts:
                severity_symbol = {"low": "🔵", "medium": "🟡", "high": "🟠", "critical": "🔴"}[conflict.severity]
                print(f"{severity_symbol} {conflict.severity.upper()}: {conflict.description}")
                if conflict.resolution_suggestion:
                    print(f"   💡 Suggestion: {conflict.resolution_suggestion}")
        else:
            print("✅ No conflicts detected")
        
        print("\n🎯 Step 4: Context-Aware Rule Selection")
        print("-" * 30)
        
        # Demonstrate context-aware rule optimization
        python_context = MemoryContext(
            session_id="demo-session",
            project_name="python-web-app",
            user_name="chris",
            active_scopes=["python", "web"]
        )
        
        print(f"Context: Python web development")
        print(f"Active scopes: {python_context.to_scope_list()}")
        
        # Filter rules by context relevance
        relevant_rules = [
            rule for rule in rules 
            if rule.matches_scope(python_context.to_scope_list())
        ]
        
        print(f"\nRelevant rules ({len(relevant_rules)} of {len(rules)}):")
        for rule in relevant_rules:
            authority_symbol = "🔴" if rule.authority == AuthorityLevel.ABSOLUTE else "🟡"
            print(f"  {authority_symbol} {rule.rule}")
        
        # Optimize for token budget
        selected_rules, optimized_usage = token_counter.optimize_rules_for_context(
            relevant_rules, max_tokens=200, preserve_absolute=True
        )
        
        print(f"\nOptimized for 200 token budget:")
        print(f"Selected: {len(selected_rules)} rules ({optimized_usage.total_tokens} tokens)")
        for rule in selected_rules:
            authority_symbol = "🔴" if rule.authority == AuthorityLevel.ABSOLUTE else "🟡"
            print(f"  {authority_symbol} {rule.rule}")
        
        print("\n💬 Step 5: Conversational Memory Updates")
        print("-" * 30)
        
        # Demonstrate conversational update detection
        from workspace_qdrant_mcp.memory.claude_integration import ClaudeCodeIntegration
        
        claude_integration = ClaudeCodeIntegration(token_counter)
        
        conversational_texts = [
            "Note: call me Chris from now on",
            "Remember: I prefer using TypeScript strict mode",
            "Always use pytest for Python testing",
            "I work on project workspace-qdrant-mcp"
        ]
        
        print("Processing conversational updates:")
        for text in conversational_texts:
            print(f"\n📝 \"{text}\"")
            
            updates = claude_integration.detect_conversational_updates(text, python_context)
            
            if updates:
                for update in updates:
                    if update.is_valid():
                        print(f"   ✅ Extracted: {update.extracted_rule}")
                        print(f"   📂 Category: {update.category.value}")
                        print(f"   🎯 Authority: {update.authority.value}")
                        print(f"   🔍 Confidence: {update.confidence:.1f}")
                    else:
                        print(f"   ❌ Invalid update (confidence: {update.confidence:.1f})")
            else:
                print("   ℹ️ No memory updates detected")
        
        print("\n🚀 Step 6: Claude Code Session Integration")
        print("-" * 30)
        
        # Demonstrate Claude Code session initialization
        session = ClaudeCodeSession(
            session_id="demo-claude-session",
            workspace_path="/path/to/project",
            user_name="chris",
            project_name="workspace-qdrant-mcp",
            active_files=["main.py", "config.py", "tests/test_memory.py"],
            context_window_size=200000
        )
        
        print(f"Claude Code Session:")
        print(f"  Project: {session.project_name}")
        print(f"  User: {session.user_name}")
        print(f"  Active files: {len(session.active_files)}")
        print(f"  Context window: {session.context_window_size:,} tokens")
        
        # Generate system prompt injection
        system_prompt = claude_integration.create_system_prompt_injection(
            selected_rules, python_context
        )
        
        print(f"\nGenerated system prompt injection:")
        print("─" * 50)
        print(system_prompt)
        print("─" * 50)
        
        print("\n✅ Memory System Demo Complete!")
        print("\nThe memory system provides:")
        print("  🧠 Rule-based LLM behavior management")
        print("  ⚖️ Authority levels (absolute vs default)")
        print("  🔍 Semantic conflict detection")
        print("  🎯 Context-aware rule selection")
        print("  📊 Token usage optimization")
        print("  💬 Conversational memory updates")
        print("  🚀 Claude Code SDK integration")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_memory_system())