"""
Integration layer between web crawling and document processing pipeline.

This module provides seamless integration between the enhanced web crawler
and the existing document processing pipeline for vector database ingestion.
"""

import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from .advanced_retry import RetryConfig
from .enhanced_web_crawler import EnhancedCrawlResult, EnhancedWebCrawler
from .web_cache import CacheConfig
from .web_crawler import SecurityConfig

try:
    # Import document processing pipeline components
    from src.python.common.core.client import QdrantClient
    from src.python.common.core.embeddings import EmbeddingManager
    from src.python.common.memory.manager import MemoryManager
except ImportError:
    logger.warning("Document processing pipeline components not available for import")


class WebCrawlSession:
    """Manages a web crawling session with progress tracking."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.end_time: float | None = None
        self.status = "running"
        self.urls_processed = 0
        self.urls_successful = 0
        self.urls_failed = 0
        self.total_content_length = 0
        self.total_processing_time = 0.0
        self.errors: list[dict[str, Any]] = []
        self.results: list[EnhancedCrawlResult] = []

    def add_result(self, result: EnhancedCrawlResult) -> None:
        """Add crawl result to session."""
        self.results.append(result)
        self.urls_processed += 1

        if result.success:
            self.urls_successful += 1
            if result.content:
                self.total_content_length += len(result.content)
        else:
            self.urls_failed += 1
            self.errors.append({
                'url': result.url,
                'error': result.error,
                'timestamp': time.time()
            })

        if result.processing_time:
            self.total_processing_time += result.processing_time

    def finish(self, status: str = "completed") -> None:
        """Mark session as finished."""
        self.end_time = time.time()
        self.status = status

    def get_summary(self) -> dict[str, Any]:
        """Get session summary."""
        duration = (self.end_time or time.time()) - self.start_time

        return {
            'session_id': self.session_id,
            'status': self.status,
            'duration': duration,
            'urls_processed': self.urls_processed,
            'urls_successful': self.urls_successful,
            'urls_failed': self.urls_failed,
            'success_rate': self.urls_successful / max(1, self.urls_processed),
            'total_content_length': self.total_content_length,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.total_processing_time / max(1, self.urls_processed),
            'error_count': len(self.errors),
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class WebContentProcessor:
    """Processes web crawl results for document pipeline ingestion."""

    def __init__(self, qdrant_client: Any | None = None, embedding_manager: Any | None = None):
        self.qdrant_client = qdrant_client
        self.embedding_manager = embedding_manager

    def prepare_document(self, result: EnhancedCrawlResult) -> dict[str, Any]:
        """Prepare crawl result for document processing pipeline."""
        # Extract URL components
        parsed_url = urlparse(result.url)

        # Create document metadata
        metadata = {
            'source': 'web_crawl',
            'url': result.url,
            'domain': parsed_url.netloc,
            'path': parsed_url.path,
            'status_code': result.status_code,
            'content_type': result.content_type,
            'crawl_timestamp': result.timestamp,
            'content_quality_score': result.content_quality_score,
            'processing_time': result.processing_time,
            'cache_hit': result.cache_hit,
            'retry_attempts': result.retry_attempts
        }

        # Add extracted metadata from content
        if result.extracted_content.get('metadata'):
            metadata.update({
                f'page_{k}': v for k, v in result.extracted_content['metadata'].items()
            })

        # Add structured data if available
        if result.structured_data:
            metadata['has_structured_data'] = True
            metadata['structured_data_types'] = list(result.structured_data.keys())

        # Add media information
        if result.media_links:
            metadata['has_images'] = len(result.media_links.get('images', []))
            metadata['has_videos'] = len(result.media_links.get('videos', []))
            metadata['has_audio'] = len(result.media_links.get('audio', []))
            metadata['has_documents'] = len(result.media_links.get('documents', []))

        # Add quality metrics
        if result.extracted_content.get('quality_metrics'):
            quality = result.extracted_content['quality_metrics']
            metadata.update({
                f'quality_{k}': v for k, v in quality.items()
            })

        # Add deduplication information
        if result.deduplication_key:
            metadata['deduplication_key'] = result.deduplication_key

        if result.metadata.get('duplicate_urls'):
            metadata['has_duplicates'] = True
            metadata['duplicate_count'] = len(result.metadata['duplicate_urls'])

        # Create document for processing
        document = {
            'content': result.content or '',
            'metadata': metadata,
            'file_path': result.url,  # Use URL as file path
            'file_type': 'web_page',
            'word_count': result.extracted_content.get('word_count', 0),
            'char_count': result.extracted_content.get('char_count', 0)
        }

        return document

    def chunk_content(self, document: dict[str, Any], chunk_size: int = 1000) -> list[dict[str, Any]]:
        """Split document content into chunks for processing."""
        content = document['content']
        if not content:
            return []

        chunks = []
        words = content.split()

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = ' '.join(chunk_words)

            chunk = {
                'content': chunk_content,
                'metadata': document['metadata'].copy(),
                'chunk_index': len(chunks),
                'chunk_size': len(chunk_words),
                'is_chunk': True,
                'parent_url': document['file_path']
            }

            chunks.append(chunk)

        return chunks

    async def process_for_vector_db(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        """Process document for vector database ingestion."""
        try:
            # Chunk the document if it's large
            if document.get('word_count', 0) > 1000:
                chunks = self.chunk_content(document)
                documents_to_process = chunks
            else:
                documents_to_process = [document]

            processed_docs = []

            for doc in documents_to_process:
                # Generate embeddings if embedding manager available
                if self.embedding_manager:
                    try:
                        embedding = await self.embedding_manager.embed_text(doc['content'])
                        doc['embedding'] = embedding
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding: {e}")

                processed_docs.append(doc)

            return processed_docs

        except Exception as e:
            logger.error(f"Failed to process document for vector DB: {e}")
            return []


class WebCrawlPipeline:
    """Complete web crawling and document processing pipeline."""

    def __init__(
        self,
        security_config: SecurityConfig | None = None,
        cache_config: CacheConfig | None = None,
        retry_config: RetryConfig | None = None,
        qdrant_client: Any | None = None,
        embedding_manager: Any | None = None,
        collection_name: str | None = None
    ):
        self.crawler = EnhancedWebCrawler(security_config, cache_config, retry_config)
        self.processor = WebContentProcessor(qdrant_client, embedding_manager)
        self.collection_name = collection_name
        self.active_sessions: dict[str, WebCrawlSession] = {}

    async def crawl_and_process(
        self,
        urls: str | list[str],
        process_for_vector_db: bool = True,
        **crawl_options
    ) -> WebCrawlSession:
        """Crawl URLs and process results through the document pipeline."""
        if isinstance(urls, str):
            urls = [urls]

        # Create new session
        session_id = f"crawl_{int(time.time())}_{len(urls)}"
        session = WebCrawlSession(session_id)
        self.active_sessions[session_id] = session

        logger.info(f"Starting web crawl session {session_id} with {len(urls)} URLs")

        try:
            async with self.crawler:
                for url in urls:
                    try:
                        # Crawl URL
                        result = await self.crawler.crawl_url(url, **crawl_options)
                        session.add_result(result)

                        # Process for vector DB if successful and requested
                        if result.success and process_for_vector_db:
                            await self._process_result_for_vector_db(result)

                        logger.info(f"Processed {url}: success={result.success}, quality={result.content_quality_score:.2f}")

                    except Exception as e:
                        error_result = EnhancedCrawlResult(url, success=False)
                        error_result.error = str(e)
                        session.add_result(error_result)
                        logger.error(f"Failed to process {url}: {e}")

            session.finish("completed")

        except Exception as e:
            session.finish("failed")
            logger.error(f"Crawl session {session_id} failed: {e}")

        logger.info(f"Crawl session {session_id} completed: {session.get_summary()}")
        return session

    async def crawl_recursive_and_process(
        self,
        start_url: str,
        process_for_vector_db: bool = True,
        **crawl_options
    ) -> WebCrawlSession:
        """Perform recursive crawl and process all results."""
        session_id = f"recursive_crawl_{int(time.time())}"
        session = WebCrawlSession(session_id)
        self.active_sessions[session_id] = session

        logger.info(f"Starting recursive crawl session {session_id} from {start_url}")

        try:
            async with self.crawler:
                # Perform recursive crawl
                results = await self.crawler.crawl_recursive(start_url, **crawl_options)

                # Process all results
                for result in results:
                    session.add_result(result)

                    if result.success and process_for_vector_db:
                        await self._process_result_for_vector_db(result)

            session.finish("completed")

        except Exception as e:
            session.finish("failed")
            logger.error(f"Recursive crawl session {session_id} failed: {e}")

        logger.info(f"Recursive crawl session {session_id} completed: {session.get_summary()}")
        return session

    async def _process_result_for_vector_db(self, result: EnhancedCrawlResult) -> None:
        """Process single crawl result for vector database."""
        try:
            # Prepare document
            document = self.processor.prepare_document(result)

            # Process for vector DB
            processed_docs = await self.processor.process_for_vector_db(document)

            # Store in vector DB if client available
            if self.crawler.client and self.collection_name and processed_docs:
                for _doc in processed_docs:
                    try:
                        # This would integrate with the actual Qdrant client
                        # await self.crawler.client.upsert_document(
                        #     collection_name=self.collection_name,
                        #     document=doc
                        # )
                        pass
                    except Exception as e:
                        logger.error(f"Failed to store document in vector DB: {e}")

        except Exception as e:
            logger.error(f"Failed to process result for vector DB: {e}")

    def get_session(self, session_id: str) -> WebCrawlSession | None:
        """Get crawl session by ID."""
        return self.active_sessions.get(session_id)

    def get_active_sessions(self) -> list[dict[str, Any]]:
        """Get all active session summaries."""
        return [session.get_summary() for session in self.active_sessions.values()]

    async def export_session_results(self, session_id: str, output_path: Path) -> bool:
        """Export session results to file."""
        session = self.get_session(session_id)
        if not session:
            return False

        try:
            import json

            export_data = {
                'session_summary': session.get_summary(),
                'crawler_stats': self.crawler.get_session_stats(),
                'results': [
                    {
                        'url': result.url,
                        'success': result.success,
                        'status_code': result.status_code,
                        'content_length': len(result.content) if result.content else 0,
                        'quality_score': result.content_quality_score,
                        'processing_time': result.processing_time,
                        'cache_hit': result.cache_hit,
                        'retry_attempts': result.retry_attempts,
                        'error': result.error,
                        'metadata_keys': list(result.metadata.keys()) if result.metadata else [],
                        'has_structured_data': bool(result.structured_data),
                        'has_media_links': bool(result.media_links),
                        'link_count': len(result.text_links),
                        'deduplication_key': result.deduplication_key
                    }
                    for result in session.results
                ]
            }

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Session {session_id} results exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export session results: {e}")
            return False

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    async def close(self) -> None:
        """Close the pipeline and cleanup resources."""
        await self.crawler.close()
        self.active_sessions.clear()


# Utility functions for common crawling patterns

async def crawl_website_sitemap(
    sitemap_url: str,
    pipeline: WebCrawlPipeline,
    **options
) -> WebCrawlSession:
    """Crawl a website using its sitemap."""
    # This would parse sitemap XML and extract URLs
    # For now, just crawl the sitemap URL itself
    return await pipeline.crawl_and_process([sitemap_url], **options)


async def crawl_domain_with_depth(
    start_url: str,
    max_depth: int,
    pipeline: WebCrawlPipeline,
    **options
) -> WebCrawlSession:
    """Crawl a domain with specified depth limit."""
    options.update({
        'max_depth': max_depth,
        'same_domain_only': True
    })

    return await pipeline.crawl_recursive_and_process(start_url, **options)


async def crawl_url_list_from_file(
    file_path: Path,
    pipeline: WebCrawlPipeline,
    **options
) -> WebCrawlSession:
    """Crawl URLs from a text file."""
    try:
        with open(file_path) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        return await pipeline.crawl_and_process(urls, **options)

    except Exception as e:
        logger.error(f"Failed to read URL list from {file_path}: {e}")
        raise
