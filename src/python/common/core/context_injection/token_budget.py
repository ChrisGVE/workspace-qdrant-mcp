"""
Token budget management for memory rule context injection.

This module provides token budget allocation and optimization strategies
to ensure memory rules fit within LLM context windows. Includes support
for accurate tokenization using tiktoken and transformers libraries.
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from cachetools import LRUCache, TTLCache

from ..memory import AuthorityLevel, MemoryRule


class AllocationStrategy(Enum):
    """Token allocation strategies."""

    PRIORITY_BASED = "priority"  # Prioritize by authority + priority
    ROUND_ROBIN = "round_robin"  # Distribute equally across categories
    EQUAL = "equal"  # Fixed allocation per rule


class CompressionStrategy(Enum):
    """Rule compression strategies."""

    NONE = "none"  # No compression
    SIMPLE = "simple"  # Remove optional fields
    INTELLIGENT = "intelligent"  # Smart summarization


@dataclass
class BudgetAllocation:
    """
    Token budget allocation result.

    Attributes:
        total_budget: Total available tokens
        absolute_tokens: Tokens used by absolute authority rules
        default_tokens: Tokens used by default authority rules
        overhead_tokens: Tokens reserved for formatting overhead
        rules_included: List of rules included in allocation
        rules_skipped: List of rules skipped due to budget constraints
        compression_applied: Whether compression was applied
        allocation_stats: Additional allocation statistics
    """

    total_budget: int
    absolute_tokens: int
    default_tokens: int
    overhead_tokens: int
    rules_included: List[MemoryRule]
    rules_skipped: List[MemoryRule]
    compression_applied: bool
    allocation_stats: Dict[str, Any]


@dataclass
class CacheStatistics:
    """
    Token count cache statistics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        hit_rate: Cache hit rate (hits / total_requests)
        size: Current cache size
        max_size: Maximum cache size
        evictions: Number of cache evictions
    """

    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    size: int = 0
    max_size: int = 0
    evictions: int = 0

    def update_hit_rate(self) -> None:
        """Update calculated hit rate."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


@dataclass
class PerformanceMetrics:
    """
    Token counting performance metrics.

    Attributes:
        cache_enabled: Whether caching is enabled
        batch_size: Batch size used for counting
        total_counts: Total number of count operations
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        avg_count_time_ms: Average time per count (milliseconds)
        total_time_ms: Total time for all counts (milliseconds)
    """

    cache_enabled: bool = False
    batch_size: int = 1
    total_counts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_count_time_ms: float = 0.0
    total_time_ms: float = 0.0


class TokenizerType(Enum):
    """Available tokenizer implementations."""

    TIKTOKEN = "tiktoken"  # OpenAI's tiktoken (fast, accurate for GPT models)
    TRANSFORMERS = "transformers"  # HuggingFace transformers (broad support)
    ESTIMATION = "estimation"  # Character/word-based estimation (fallback)


class TokenizerFactory:
    """
    Factory for creating and caching tokenizer instances.

    Manages tokenizer creation with automatic library availability detection
    and instance caching for performance.
    """

    # Tokenizer instance cache (expensive to create)
    _cache: LRUCache = LRUCache(maxsize=10)

    # Library availability flags (checked once)
    _tiktoken_available: Optional[bool] = None
    _transformers_available: Optional[bool] = None

    @classmethod
    def is_tiktoken_available(cls) -> bool:
        """
        Check if tiktoken library is available.

        Returns:
            True if tiktoken can be imported
        """
        if cls._tiktoken_available is None:
            try:
                import tiktoken  # noqa: F401

                cls._tiktoken_available = True
            except ImportError:
                cls._tiktoken_available = False
        return cls._tiktoken_available

    @classmethod
    def is_transformers_available(cls) -> bool:
        """
        Check if transformers library is available.

        Returns:
            True if transformers can be imported
        """
        if cls._transformers_available is None:
            try:
                import transformers  # noqa: F401

                cls._transformers_available = True
            except ImportError:
                cls._transformers_available = False
        return cls._transformers_available

    @classmethod
    def get_tiktoken_encoding(cls, encoding_name: str = "cl100k_base") -> Any:
        """
        Get tiktoken encoding instance with caching.

        Args:
            encoding_name: Tiktoken encoding name (default: cl100k_base for GPT-4)

        Returns:
            Tiktoken encoding instance

        Raises:
            ImportError: If tiktoken is not available
        """
        cache_key = ("tiktoken", encoding_name)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        if not cls.is_tiktoken_available():
            raise ImportError(
                "tiktoken library not available. "
                "Install with: pip install tiktoken or uv pip install 'workspace-qdrant-mcp[tokenizers]'"
            )

        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        cls._cache[cache_key] = encoding
        return encoding

    @classmethod
    def get_transformers_tokenizer(cls, model_name: str) -> Any:
        """
        Get transformers tokenizer instance with caching.

        Args:
            model_name: HuggingFace model name or path

        Returns:
            Transformers tokenizer instance

        Raises:
            ImportError: If transformers is not available
        """
        cache_key = ("transformers", model_name)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        if not cls.is_transformers_available():
            raise ImportError(
                "transformers library not available. "
                "Install with: pip install transformers or uv pip install 'workspace-qdrant-mcp[tokenizers]'"
            )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        cls._cache[cache_key] = tokenizer
        return tokenizer

    @classmethod
    def clear_cache(cls) -> None:
        """Clear tokenizer instance cache."""
        cls._cache.clear()
        # Reset availability flags to force re-check
        cls._tiktoken_available = None
        cls._transformers_available = None

    @classmethod
    def get_preferred_tokenizer_type(cls, tool_name: str) -> TokenizerType:
        """
        Get preferred tokenizer type for a tool.

        Args:
            tool_name: Tool name (claude, codex, gemini, etc.)

        Returns:
            Preferred tokenizer type based on availability
        """
        # Map tools to their preferred tokenizer libraries
        if tool_name in ("codex", "gpt", "openai"):
            if cls.is_tiktoken_available():
                return TokenizerType.TIKTOKEN
        elif tool_name in ("gemini", "palm"):
            if cls.is_transformers_available():
                return TokenizerType.TRANSFORMERS
        elif tool_name == "claude":
            # Claude uses a custom tokenizer, tiktoken cl100k_base is a good approximation
            if cls.is_tiktoken_available():
                return TokenizerType.TIKTOKEN

        # Fallback to estimation if no suitable tokenizer available
        return TokenizerType.ESTIMATION


class TokenCountCache:
    """
    Content-based token count cache with TTL support.

    Caches token counts for frequently counted content to avoid
    redundant tokenization operations. Uses content hashing for
    cache keys and supports TTL-based expiration.

    Performance impact:
        - Cache hit: O(1) lookup, ~10x faster than tokenization
        - Cache miss: Tokenization + O(1) cache insertion
        - Memory overhead: ~100 bytes per cached entry
    """

    def __init__(
        self,
        maxsize: int = 1000,
        ttl: float = 300.0,
        enabled: bool = True,
    ):
        """
        Initialize token count cache.

        Args:
            maxsize: Maximum number of cached entries (LRU eviction)
            ttl: Time-to-live in seconds (0 = no TTL)
            enabled: Whether caching is enabled
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.enabled = enabled

        # Use TTL cache if TTL > 0, otherwise LRU cache
        if ttl > 0:
            self._cache: Any = TTLCache(maxsize=maxsize, ttl=ttl)
        else:
            self._cache = LRUCache(maxsize=maxsize)

        # Thread-safe cache access
        self._lock = Lock()

        # Statistics tracking
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _make_key(self, text: str, tool_name: str) -> str:
        """
        Generate cache key from text and tool name.

        Uses MD5 hash for efficient key generation and collision avoidance.

        Args:
            text: Text content
            tool_name: Tool name for tokenization

        Returns:
            Cache key string
        """
        # Combine text and tool_name for unique key
        content = f"{tool_name}:{text}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get(self, text: str, tool_name: str) -> Optional[int]:
        """
        Get cached token count.

        Args:
            text: Text content
            tool_name: Tool name

        Returns:
            Cached token count or None if not found
        """
        if not self.enabled:
            return None

        key = self._make_key(text, tool_name)

        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None

    def set(self, text: str, tool_name: str, count: int) -> None:
        """
        Store token count in cache.

        Args:
            text: Text content
            tool_name: Tool name
            count: Token count to cache
        """
        if not self.enabled:
            return

        key = self._make_key(text, tool_name)

        with self._lock:
            # Track eviction if cache is full
            if len(self._cache) >= self.maxsize and key not in self._cache:
                self._evictions += 1

            self._cache[key] = count

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_statistics(self) -> CacheStatistics:
        """
        Get cache statistics.

        Returns:
            CacheStatistics with current metrics
        """
        with self._lock:
            stats = CacheStatistics(
                hits=self._hits,
                misses=self._misses,
                size=len(self._cache),
                max_size=self.maxsize,
                evictions=self._evictions,
            )
            stats.update_hit_rate()
            return stats

    def disable(self) -> None:
        """Disable caching and clear cache."""
        self.enabled = False
        self.clear()

    def enable(self) -> None:
        """Enable caching."""
        self.enabled = True


class TokenCounter:
    """
    Tool-specific token counting utilities with multi-tokenizer support.

    Provides accurate token counting using tiktoken or transformers libraries
    when available, with fallback to character/word-based estimation.

    Performance optimizations:
        - Content-based caching for repeated counts
        - Batch processing for multiple texts
        - Async counting for concurrent operations
    """

    # Model name mappings for better accuracy
    MODEL_MAPPINGS = {
        "claude": "cl100k_base",  # Approximate Claude with GPT-4 encoding
        "codex": "cl100k_base",  # GPT-4 encoding
        "gpt": "cl100k_base",  # GPT-4 encoding
        "openai": "cl100k_base",  # GPT-4 encoding
        "gemini": "google/gemma-2b",  # Gemini approximation
        "palm": "google/gemma-2b",  # PaLM approximation
    }

    # Global cache instance (shared across all counters)
    _cache: Optional[TokenCountCache] = None
    _cache_lock = Lock()

    @classmethod
    def get_cache(cls) -> Optional[TokenCountCache]:
        """
        Get global token count cache.

        Returns:
            TokenCountCache instance or None if not initialized
        """
        with cls._cache_lock:
            return cls._cache

    @classmethod
    def initialize_cache(
        cls,
        maxsize: int = 1000,
        ttl: float = 300.0,
        enabled: bool = True,
    ) -> TokenCountCache:
        """
        Initialize global token count cache.

        Args:
            maxsize: Maximum number of cached entries
            ttl: Time-to-live in seconds
            enabled: Whether caching is enabled

        Returns:
            Initialized TokenCountCache instance
        """
        with cls._cache_lock:
            cls._cache = TokenCountCache(maxsize=maxsize, ttl=ttl, enabled=enabled)
            return cls._cache

    @classmethod
    def clear_cache(cls) -> None:
        """Clear global token count cache."""
        with cls._cache_lock:
            if cls._cache:
                cls._cache.clear()

    @classmethod
    def get_cache_statistics(cls) -> Optional[CacheStatistics]:
        """
        Get cache statistics.

        Returns:
            CacheStatistics or None if cache not initialized
        """
        cache = cls.get_cache()
        return cache.get_statistics() if cache else None

    @staticmethod
    def count_claude_tokens(text: str, use_tokenizer: bool = True) -> int:
        """
        Count tokens for Claude.

        Uses tiktoken cl100k_base encoding when available (good approximation),
        falls back to character-based estimation (~1 token per 4 characters).

        Args:
            text: Text to count tokens for
            use_tokenizer: If True, use actual tokenizer when available

        Returns:
            Token count (minimum 1)
        """
        if use_tokenizer and TokenizerFactory.is_tiktoken_available():
            try:
                encoding = TokenizerFactory.get_tiktoken_encoding("cl100k_base")
                return max(1, len(encoding.encode(text)))
            except Exception:
                pass  # Fall back to estimation

        # Estimation fallback
        return max(1, len(text) // 4)

    @staticmethod
    def count_codex_tokens(text: str, use_tokenizer: bool = True) -> int:
        """
        Count tokens for Codex (GPT-based).

        Uses tiktoken cl100k_base encoding when available,
        falls back to word-based estimation (~1.3 tokens per word).

        Args:
            text: Text to count tokens for
            use_tokenizer: If True, use actual tokenizer when available

        Returns:
            Token count (minimum 1)
        """
        if use_tokenizer and TokenizerFactory.is_tiktoken_available():
            try:
                encoding = TokenizerFactory.get_tiktoken_encoding("cl100k_base")
                return max(1, len(encoding.encode(text)))
            except Exception:
                pass  # Fall back to estimation

        # Estimation fallback
        words = len(text.split())
        return max(1, int(words * 1.3))

    @staticmethod
    def count_gemini_tokens(text: str, use_tokenizer: bool = True) -> int:
        """
        Count tokens for Gemini.

        Uses transformers tokenizer when available,
        falls back to character-based estimation (~1 token per 4 characters).

        Args:
            text: Text to count tokens for
            use_tokenizer: If True, use actual tokenizer when available

        Returns:
            Token count (minimum 1)
        """
        if use_tokenizer and TokenizerFactory.is_transformers_available():
            try:
                tokenizer = TokenizerFactory.get_transformers_tokenizer(
                    "google/gemma-2b"
                )
                return max(1, len(tokenizer.encode(text)))
            except Exception:
                pass  # Fall back to estimation

        # Estimation fallback
        return max(1, len(text) // 4)

    @staticmethod
    def count_tokens(
        text: str,
        tool_name: str,
        use_tokenizer: bool = True,
        use_cache: bool = True,
    ) -> int:
        """
        Count tokens using tool-specific counter with optional caching.

        Args:
            text: Text to count tokens for
            tool_name: Target tool name ("claude", "codex", "gemini")
            use_tokenizer: If True, use actual tokenizer when available
            use_cache: If True, use cache for repeated content

        Returns:
            Token count (minimum 1)
        """
        # Check cache first
        if use_cache:
            cache = TokenCounter.get_cache()
            if cache:
                cached_count = cache.get(text, tool_name)
                if cached_count is not None:
                    return cached_count

        # Count tokens
        counters = {
            "claude": TokenCounter.count_claude_tokens,
            "codex": TokenCounter.count_codex_tokens,
            "gemini": TokenCounter.count_gemini_tokens,
            "gpt": TokenCounter.count_codex_tokens,  # Alias for codex
            "openai": TokenCounter.count_codex_tokens,  # Alias for codex
        }
        counter = counters.get(tool_name, TokenCounter.count_claude_tokens)
        count = counter(text, use_tokenizer=use_tokenizer)

        # Store in cache
        if use_cache:
            cache = TokenCounter.get_cache()
            if cache:
                cache.set(text, tool_name, count)

        return count

    @staticmethod
    def batch_count_tokens(
        texts: List[str],
        tool_name: str,
        use_tokenizer: bool = True,
        use_cache: bool = True,
    ) -> List[int]:
        """
        Count tokens for multiple texts in batch.

        More efficient than counting individually due to tokenizer reuse
        and reduced cache locking overhead.

        Args:
            texts: List of texts to count tokens for
            tool_name: Target tool name
            use_tokenizer: If True, use actual tokenizer when available
            use_cache: If True, use cache for repeated content

        Returns:
            List of token counts (same order as input texts)
        """
        if not texts:
            return []

        counts = []
        cache = TokenCounter.get_cache() if use_cache else None

        # Process each text with cache lookup/update
        for text in texts:
            # Check cache
            cached_count = None
            if cache:
                cached_count = cache.get(text, tool_name)

            if cached_count is not None:
                counts.append(cached_count)
            else:
                # Count and cache
                count = TokenCounter.count_tokens(
                    text, tool_name, use_tokenizer=use_tokenizer, use_cache=False
                )
                counts.append(count)

                if cache:
                    cache.set(text, tool_name, count)

        return counts

    @staticmethod
    async def async_count_tokens(
        text: str,
        tool_name: str,
        use_tokenizer: bool = True,
        use_cache: bool = True,
    ) -> int:
        """
        Asynchronously count tokens for a single text.

        Runs counting in thread pool to avoid blocking async event loop.

        Args:
            text: Text to count tokens for
            tool_name: Target tool name
            use_tokenizer: If True, use actual tokenizer when available
            use_cache: If True, use cache for repeated content

        Returns:
            Token count (minimum 1)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            TokenCounter.count_tokens,
            text,
            tool_name,
            use_tokenizer,
            use_cache,
        )

    @staticmethod
    async def async_batch_count_tokens(
        texts: List[str],
        tool_name: str,
        use_tokenizer: bool = True,
        use_cache: bool = True,
    ) -> List[int]:
        """
        Asynchronously count tokens for multiple texts concurrently.

        Uses asyncio.gather to process texts in parallel, significantly
        faster than sequential counting for large batches.

        Args:
            texts: List of texts to count tokens for
            tool_name: Target tool name
            use_tokenizer: If True, use actual tokenizer when available
            use_cache: If True, use cache for repeated content

        Returns:
            List of token counts (same order as input texts)
        """
        if not texts:
            return []

        # Count all texts concurrently
        tasks = [
            TokenCounter.async_count_tokens(text, tool_name, use_tokenizer, use_cache)
            for text in texts
        ]

        return await asyncio.gather(*tasks)

    @staticmethod
    def count_tokens_with_model(
        text: str,
        model_name: str,
        tokenizer_type: Optional[TokenizerType] = None,
    ) -> int:
        """
        Count tokens for a specific model using explicit tokenizer type.

        Args:
            text: Text to count tokens for
            model_name: Model name or encoding name
            tokenizer_type: Specific tokenizer type to use (auto-detect if None)

        Returns:
            Token count (minimum 1)
        """
        if tokenizer_type is None:
            # Auto-detect based on model name
            if "gpt" in model_name.lower() or "openai" in model_name.lower():
                tokenizer_type = TokenizerType.TIKTOKEN
            elif "gemini" in model_name.lower() or "palm" in model_name.lower():
                tokenizer_type = TokenizerType.TRANSFORMERS
            else:
                tokenizer_type = TokenizerType.ESTIMATION

        if tokenizer_type == TokenizerType.TIKTOKEN:
            try:
                # Try to use model_name as encoding name, fall back to cl100k_base
                try:
                    encoding = TokenizerFactory.get_tiktoken_encoding(model_name)
                except Exception:
                    encoding = TokenizerFactory.get_tiktoken_encoding(
                        "cl100k_base"
                    )
                return max(1, len(encoding.encode(text)))
            except ImportError:
                pass  # Fall through to estimation

        elif tokenizer_type == TokenizerType.TRANSFORMERS:
            try:
                tokenizer = TokenizerFactory.get_transformers_tokenizer(model_name)
                return max(1, len(tokenizer.encode(text)))
            except (ImportError, Exception):
                pass  # Fall through to estimation

        # Estimation fallback
        return max(1, len(text) // 4)


class TokenBudgetManager:
    """
    Manages token budget allocation and optimization for memory rules.

    Provides intelligent budget allocation strategies to fit memory rules
    within LLM context windows while protecting high-priority rules.
    """

    def __init__(
        self,
        allocation_strategy: AllocationStrategy = AllocationStrategy.PRIORITY_BASED,
        compression_strategy: CompressionStrategy = CompressionStrategy.NONE,
        absolute_rules_protected: bool = True,
        overhead_percentage: float = 0.05,
        use_accurate_counting: bool = True,
        enable_cache: bool = True,
        cache_maxsize: int = 1000,
        cache_ttl: float = 300.0,
    ):
        """
        Initialize token budget manager.

        Args:
            allocation_strategy: Strategy for allocating budget to rules
            compression_strategy: Strategy for compressing rules if needed
            absolute_rules_protected: If True, absolute rules always included
            overhead_percentage: Percentage of budget reserved for formatting overhead
            use_accurate_counting: If True, use tiktoken/transformers when available
            enable_cache: If True, enable token count caching
            cache_maxsize: Maximum cache size
            cache_ttl: Cache TTL in seconds
        """
        self.allocation_strategy = allocation_strategy
        self.compression_strategy = compression_strategy
        self.absolute_rules_protected = absolute_rules_protected
        self.overhead_percentage = overhead_percentage
        self.use_accurate_counting = use_accurate_counting

        # Initialize cache if enabled and not already initialized
        if enable_cache and TokenCounter.get_cache() is None:
            TokenCounter.initialize_cache(
                maxsize=cache_maxsize,
                ttl=cache_ttl,
                enabled=True,
            )

    def allocate_budget(
        self,
        rules: List[MemoryRule],
        total_budget: int,
        tool_name: str,
    ) -> BudgetAllocation:
        """
        Allocate token budget across rules.

        Algorithm:
        1. Separate absolute vs default rules
        2. Allocate tokens to absolute rules (protected)
        3. Calculate remaining budget after overhead
        4. Allocate remaining budget to default rules per strategy
        5. Apply compression if needed
        6. Return allocation result with statistics

        Args:
            rules: List of memory rules to allocate
            total_budget: Total available token budget
            tool_name: Target LLM tool name for token counting

        Returns:
            BudgetAllocation with selected rules and statistics
        """
        # Separate by authority
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        # Count absolute rule tokens
        absolute_tokens = sum(
            self._estimate_tokens(r, tool_name) for r in absolute_rules
        )

        # Reserve budget for overhead (headers, footers, formatting)
        overhead_tokens = int(total_budget * self.overhead_percentage)
        available_budget = total_budget - overhead_tokens

        # Check if absolute rules fit
        if absolute_tokens > available_budget:
            if self.absolute_rules_protected:
                # Include all absolute rules even if over budget
                # This ensures critical rules are never skipped
                return BudgetAllocation(
                    total_budget=total_budget,
                    absolute_tokens=absolute_tokens,
                    default_tokens=0,
                    overhead_tokens=overhead_tokens,
                    rules_included=absolute_rules,
                    rules_skipped=default_rules,
                    compression_applied=False,
                    allocation_stats={
                        "over_budget": True,
                        "absolute_count": len(absolute_rules),
                        "default_count": 0,
                        "strategy": self.allocation_strategy.value,
                        "tokenizer_used": self._get_tokenizer_type(tool_name).value,
                    },
                )
            else:
                # Truncate absolute rules to fit budget
                absolute_rules = self._truncate_rules(
                    absolute_rules, available_budget, tool_name
                )
                absolute_tokens = sum(
                    self._estimate_tokens(r, tool_name) for r in absolute_rules
                )

        # Allocate remaining budget to default rules
        remaining_budget = available_budget - absolute_tokens
        selected_default_rules = self._allocate_to_default_rules(
            default_rules, remaining_budget, tool_name
        )

        default_tokens = sum(
            self._estimate_tokens(r, tool_name) for r in selected_default_rules
        )

        all_included = absolute_rules + selected_default_rules
        all_skipped = [r for r in default_rules if r not in selected_default_rules]

        return BudgetAllocation(
            total_budget=total_budget,
            absolute_tokens=absolute_tokens,
            default_tokens=default_tokens,
            overhead_tokens=overhead_tokens,
            rules_included=all_included,
            rules_skipped=all_skipped,
            compression_applied=False,
            allocation_stats={
                "over_budget": False,
                "absolute_count": len(absolute_rules),
                "default_count": len(selected_default_rules),
                "skipped_count": len(all_skipped),
                "strategy": self.allocation_strategy.value,
                "utilization": (absolute_tokens + default_tokens) / total_budget
                if total_budget > 0
                else 0,
                "tokenizer_used": self._get_tokenizer_type(tool_name).value,
            },
        )

    def _allocate_to_default_rules(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Allocate budget to default rules based on strategy.

        Args:
            rules: Default authority rules
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules that fit within budget
        """
        if self.allocation_strategy == AllocationStrategy.PRIORITY_BASED:
            return self._priority_based_allocation(rules, budget, tool_name)
        elif self.allocation_strategy == AllocationStrategy.ROUND_ROBIN:
            return self._round_robin_allocation(rules, budget, tool_name)
        else:  # EQUAL
            return self._equal_allocation(rules, budget, tool_name)

    def _priority_based_allocation(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Allocate by priority (highest priority first).

        Sorts rules by metadata priority and creation time, then
        includes rules sequentially until budget is exhausted.

        Args:
            rules: Rules to allocate
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules
        """
        sorted_rules = sorted(
            rules,
            key=lambda r: (
                getattr(r.metadata, "priority", 50) if r.metadata else 50,
                -r.created_at.timestamp(),
            ),
            reverse=True,
        )

        selected = []
        used_tokens = 0

        for rule in sorted_rules:
            rule_tokens = self._estimate_tokens(rule, tool_name)
            if used_tokens + rule_tokens <= budget:
                selected.append(rule)
                used_tokens += rule_tokens
            else:
                break

        return selected

    def _round_robin_allocation(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Distribute tokens equally across categories.

        Splits budget evenly across rule categories, then applies
        priority-based allocation within each category.

        Args:
            rules: Rules to allocate
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules
        """
        by_category = defaultdict(list)
        for rule in rules:
            by_category[rule.category].append(rule)

        budget_per_category = budget // len(by_category) if by_category else 0
        selected = []

        for category_rules in by_category.values():
            category_selected = self._priority_based_allocation(
                category_rules, budget_per_category, tool_name
            )
            selected.extend(category_selected)

        return selected

    def _equal_allocation(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Fixed token allocation per rule.

        Divides budget equally among rules, including only rules
        that fit within per-rule allocation.

        Args:
            rules: Rules to allocate
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules
        """
        if not rules:
            return []

        tokens_per_rule = budget // len(rules)
        selected = []

        for rule in rules:
            rule_tokens = self._estimate_tokens(rule, tool_name)
            if rule_tokens <= tokens_per_rule:
                selected.append(rule)

        return selected

    def _truncate_rules(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Truncate rules to fit within budget.

        Uses priority-based allocation to select highest priority
        rules that fit within budget.

        Args:
            rules: Rules to truncate
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules
        """
        return self._priority_based_allocation(rules, budget, tool_name)

    def _estimate_tokens(self, rule: MemoryRule, tool_name: str) -> int:
        """
        Estimate token count for a rule (tool-specific).

        Includes rule text, scope, and basic metadata in estimation.

        Args:
            rule: Memory rule to estimate
            tool_name: Target tool name

        Returns:
            Token count
        """
        # Build complete rule text for estimation
        text = rule.rule

        if rule.scope:
            text += " " + " ".join(rule.scope)

        if rule.source:
            text += f" source:{rule.source}"

        # Use tool-specific counter with accurate tokenization if enabled
        return TokenCounter.count_tokens(
            text, tool_name, use_tokenizer=self.use_accurate_counting
        )

    def _get_tokenizer_type(self, tool_name: str) -> TokenizerType:
        """
        Get tokenizer type being used for a tool.

        Args:
            tool_name: Tool name

        Returns:
            Tokenizer type (for reporting)
        """
        if not self.use_accurate_counting:
            return TokenizerType.ESTIMATION

        return TokenizerFactory.get_preferred_tokenizer_type(tool_name)
