"""
Tests for memory system types and data structures.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from src.workspace_qdrant_mcp.memory.types import (
    MemoryRule, AuthorityLevel, MemoryCategory, MemoryRuleConflict,
    MemoryContext, ConversationalUpdate, AgentCapability, AgentDefinition,
    ClaudeCodeSession, MemoryInjectionResult
)


class TestMemoryRule:
    """Test MemoryRule data structure."""
    
    def test_create_basic_rule(self):
        """Test creating a basic memory rule."""
        rule = MemoryRule(
            rule="Always use uv for Python package management",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
        )
        
        assert rule.rule == "Always use uv for Python package management"
        assert rule.category == MemoryCategory.PREFERENCE
        assert rule.authority == AuthorityLevel.ABSOLUTE
        assert rule.scope == []
        assert rule.tags == []
        assert rule.source == "user_cli"
        assert rule.use_count == 0
        assert isinstance(rule.created_at, datetime)
        assert rule.updated_at is None
        assert rule.last_used is None
    
    def test_rule_validation(self):
        """Test rule validation on creation."""
        # Empty rule should raise ValueError
        with pytest.raises(ValueError, match="Rule text cannot be empty"):
            MemoryRule(
                rule="",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            )
        
        # Whitespace-only rule should raise ValueError
        with pytest.raises(ValueError, match="Rule text cannot be empty"):
            MemoryRule(
                rule="   ",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            )
    
    def test_rule_serialization(self):
        """Test rule to_dict and from_dict methods."""
        original_rule = MemoryRule(
            rule="Use TypeScript strict mode",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["typescript", "project:webapp"],
            tags=["typescript", "quality"],
            source="conversation",
        )
        
        # Serialize to dict
        rule_dict = original_rule.to_dict()
        
        assert rule_dict["rule"] == "Use TypeScript strict mode"
        assert rule_dict["category"] == "behavior"
        assert rule_dict["authority"] == "default"
        assert rule_dict["scope"] == ["typescript", "project:webapp"]
        assert rule_dict["tags"] == ["typescript", "quality"]
        assert rule_dict["source"] == "conversation"
        
        # Deserialize from dict
        restored_rule = MemoryRule.from_dict(rule_dict)
        
        assert restored_rule.id == original_rule.id
        assert restored_rule.rule == original_rule.rule
        assert restored_rule.category == original_rule.category
        assert restored_rule.authority == original_rule.authority
        assert restored_rule.scope == original_rule.scope
        assert restored_rule.tags == original_rule.tags
        assert restored_rule.source == original_rule.source
    
    def test_rule_usage_tracking(self):
        """Test rule usage tracking functionality."""
        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
        )
        
        assert rule.use_count == 0
        assert rule.last_used is None
        
        # Update usage
        before_update = datetime.utcnow()
        rule.update_usage()
        after_update = datetime.utcnow()
        
        assert rule.use_count == 1
        assert rule.last_used is not None
        assert before_update <= rule.last_used <= after_update
        assert rule.updated_at is not None
        assert before_update <= rule.updated_at <= after_update
    
    def test_scope_matching(self):
        """Test rule scope matching functionality."""
        # Global rule (empty scope)
        global_rule = MemoryRule(
            rule="Global rule",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
        )
        
        # Scoped rule
        scoped_rule = MemoryRule(
            rule="Scoped rule",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "project:webapp"]
        )
        
        # Test global rule matches any context
        assert global_rule.matches_scope([])
        assert global_rule.matches_scope(["python"])
        assert global_rule.matches_scope(["javascript", "node"])
        
        # Test scoped rule matching
        assert scoped_rule.matches_scope(["python"])  # Matches one scope
        assert scoped_rule.matches_scope(["python", "javascript"])  # Matches one scope
        assert scoped_rule.matches_scope(["project:webapp"])  # Matches other scope
        assert not scoped_rule.matches_scope(["javascript"])  # No match
        assert not scoped_rule.matches_scope([])  # No context


class TestMemoryContext:
    """Test MemoryContext functionality."""
    
    def test_create_basic_context(self):
        """Test creating basic memory context."""
        context = MemoryContext(
            session_id="test-session-123",
            project_name="my-project",
            user_name="chris"
        )
        
        assert context.session_id == "test-session-123"
        assert context.project_name == "my-project"
        assert context.user_name == "chris"
        assert context.project_path is None
        assert context.agent_type is None
        assert context.conversation_context == []
        assert context.active_scopes == []
    
    def test_scope_list_generation(self):
        """Test converting context to scope list."""
        context = MemoryContext(
            session_id="test-session",
            project_name="webapp",
            user_name="chris",
            agent_type="python-pro",
            conversation_context=["debugging", "optimization"],
            active_scopes=["urgent"]
        )
        
        scopes = context.to_scope_list()
        
        expected_scopes = [
            "project:webapp",
            "agent:python-pro",
            "user:chris",
            "debugging",
            "optimization",
            "urgent"
        ]
        
        assert set(scopes) == set(expected_scopes)


class TestConversationalUpdate:
    """Test conversational update detection."""
    
    def test_create_valid_update(self):
        """Test creating valid conversational update."""
        update = ConversationalUpdate(
            text="Note: call me Chris",
            extracted_rule="User name is Chris",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["user_preference"],
            confidence=0.9
        )
        
        assert update.is_valid()
        assert update.text == "Note: call me Chris"
        assert update.extracted_rule == "User name is Chris"
        assert update.category == MemoryCategory.PREFERENCE
        assert update.authority == AuthorityLevel.ABSOLUTE
        assert update.confidence == 0.9
    
    def test_invalid_update(self):
        """Test invalid conversational updates."""
        # No extracted rule
        update1 = ConversationalUpdate(
            text="Some text",
            confidence=0.8
        )
        assert not update1.is_valid()
        
        # Low confidence
        update2 = ConversationalUpdate(
            text="Some text",
            extracted_rule="Some rule",
            category=MemoryCategory.PREFERENCE,
            confidence=0.3  # Below threshold
        )
        assert not update2.is_valid()
        
        # Empty extracted rule
        update3 = ConversationalUpdate(
            text="Some text",
            extracted_rule="",
            category=MemoryCategory.PREFERENCE,
            confidence=0.8
        )
        assert not update3.is_valid()


class TestAgentDefinition:
    """Test agent library functionality."""
    
    def test_create_agent_capability(self):
        """Test creating agent capability."""
        capability = AgentCapability(
            name="Python Development",
            description="Expert Python programming and development",
            examples=["FastAPI applications", "Data analysis with pandas"],
            limitations=["No frontend JavaScript"]
        )
        
        assert capability.name == "Python Development"
        assert capability.description == "Expert Python programming and development"
        assert len(capability.examples) == 2
        assert len(capability.limitations) == 1
    
    def test_agent_capability_serialization(self):
        """Test agent capability serialization."""
        capability = AgentCapability(
            name="Testing",
            description="Test automation",
            examples=["pytest", "unittest"],
        )
        
        # Serialize
        data = capability.to_dict()
        assert data["name"] == "Testing"
        assert data["description"] == "Test automation"
        assert data["examples"] == ["pytest", "unittest"]
        assert data["limitations"] == []
        
        # Deserialize
        restored = AgentCapability.from_dict(data)
        assert restored.name == capability.name
        assert restored.description == capability.description
        assert restored.examples == capability.examples
        assert restored.limitations == capability.limitations
    
    def test_create_agent_definition(self):
        """Test creating agent definition."""
        capability = AgentCapability(
            name="Python Development",
            description="Python programming"
        )
        
        agent = AgentDefinition(
            name="python-pro",
            display_name="Python Professional",
            description="Expert Python developer",
            capabilities=[capability],
            specializations=["FastAPI", "Django", "Data Science"],
            tools=["pytest", "mypy", "black"],
            frameworks=["FastAPI", "Django"],
            default_rules=["Use type hints", "Write comprehensive tests"]
        )
        
        assert agent.name == "python-pro"
        assert agent.display_name == "Python Professional"
        assert len(agent.capabilities) == 1
        assert len(agent.specializations) == 3
        assert len(agent.tools) == 3
        assert len(agent.default_rules) == 2
    
    def test_agent_definition_serialization(self):
        """Test agent definition serialization."""
        capability = AgentCapability(name="Test", description="Test capability")
        agent = AgentDefinition(
            name="test-agent",
            display_name="Test Agent",
            description="Test agent",
            capabilities=[capability]
        )
        
        # Serialize
        data = agent.to_dict()
        assert data["name"] == "test-agent"
        assert data["display_name"] == "Test Agent"
        assert len(data["capabilities"]) == 1
        
        # Deserialize
        restored = AgentDefinition.from_dict(data)
        assert restored.name == agent.name
        assert restored.display_name == agent.display_name
        assert len(restored.capabilities) == 1
        assert restored.capabilities[0].name == "Test"


class TestClaudeCodeSession:
    """Test Claude Code session integration types."""
    
    def test_create_session(self):
        """Test creating Claude Code session."""
        session = ClaudeCodeSession(
            session_id="session-123",
            workspace_path="/path/to/project",
            user_name="chris",
            project_name="my-project",
            active_files=["main.py", "config.py"],
            context_window_size=200000
        )
        
        assert session.session_id == "session-123"
        assert session.workspace_path == "/path/to/project"
        assert session.user_name == "chris"
        assert session.project_name == "my-project"
        assert len(session.active_files) == 2
        assert session.context_window_size == 200000
    
    def test_session_serialization(self):
        """Test session serialization."""
        session = ClaudeCodeSession(
            session_id="test-session",
            workspace_path="/test/path"
        )
        
        data = session.to_dict()
        assert data["session_id"] == "test-session"
        assert data["workspace_path"] == "/test/path"
        assert data["user_name"] is None
        assert data["context_window_size"] == 200000


class TestMemoryInjectionResult:
    """Test memory injection result."""
    
    def test_create_successful_result(self):
        """Test creating successful injection result."""
        result = MemoryInjectionResult(
            success=True,
            rules_injected=5,
            total_tokens_used=1500,
            remaining_context_tokens=198500,
            skipped_rules=["Rule that was too long..."],
            errors=[]
        )
        
        assert result.success is True
        assert result.rules_injected == 5
        assert result.total_tokens_used == 1500
        assert result.remaining_context_tokens == 198500
        assert len(result.skipped_rules) == 1
        assert len(result.errors) == 0
    
    def test_create_failed_result(self):
        """Test creating failed injection result."""
        result = MemoryInjectionResult(
            success=False,
            rules_injected=0,
            total_tokens_used=0,
            remaining_context_tokens=200000,
            skipped_rules=[],
            errors=["Connection failed", "Invalid configuration"]
        )
        
        assert result.success is False
        assert result.rules_injected == 0
        assert len(result.errors) == 2
    
    def test_result_serialization(self):
        """Test injection result serialization."""
        result = MemoryInjectionResult(
            success=True,
            rules_injected=3,
            total_tokens_used=800,
            remaining_context_tokens=199200
        )
        
        data = result.to_dict()
        assert data["success"] is True
        assert data["rules_injected"] == 3
        assert data["total_tokens_used"] == 800
        assert data["remaining_context_tokens"] == 199200
        assert data["skipped_rules"] == []
        assert data["errors"] == []