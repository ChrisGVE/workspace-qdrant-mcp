#!/usr/bin/env python3
"""
A comprehensive Python module for testing code parsing and analysis.

This module demonstrates various Python constructs including:
- Classes and inheritance
- Async/await patterns
- Type hints
- Decorators
- Context managers
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Generic, TypeVar
from contextlib import asynccontextmanager


T = TypeVar('T')


@dataclass
class DataPoint:
    """Represents a single data point with metadata."""
    id: str
    value: float
    timestamp: int
    metadata: Dict[str, str]

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Value must be non-negative")


class BaseProcessor(ABC, Generic[T]):
    """Abstract base class for data processors."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"processor.{name}")

    @abstractmethod
    async def process(self, data: T) -> T:
        """Process a single data item."""
        pass

    @abstractmethod
    def validate(self, data: T) -> bool:
        """Validate a data item."""
        pass


class DocumentProcessor(BaseProcessor[str]):
    """Concrete processor for text documents."""

    def __init__(self, name: str, max_length: int = 1000):
        super().__init__(name)
        self.max_length = max_length

    async def process(self, data: str) -> str:
        """Process a document by cleaning and truncating."""
        self.logger.info(f"Processing document of length {len(data)}")

        # Simulate async processing
        await asyncio.sleep(0.01)

        # Clean and truncate
        cleaned = data.strip().replace('\n\n', '\n')
        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length] + "..."

        return cleaned

    def validate(self, data: str) -> bool:
        """Validate that the document is not empty."""
        return bool(data and data.strip())

    @asynccontextmanager
    async def batch_context(self):
        """Context manager for batch processing."""
        self.logger.info("Starting batch processing")
        try:
            yield self
        finally:
            self.logger.info("Finished batch processing")


async def process_documents(
    processor: DocumentProcessor,
    documents: List[str]
) -> List[str]:
    """Process multiple documents concurrently."""
    async with processor.batch_context():
        tasks = [processor.process(doc) for doc in documents]
        return await asyncio.gather(*tasks)


def main():
    """Main function demonstrating the processor."""
    logging.basicConfig(level=logging.INFO)

    processor = DocumentProcessor("test_processor")
    test_doc = "This is a test document with some content."

    async def run():
        result = await processor.process(test_doc)
        print(f"Processed: {result}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
