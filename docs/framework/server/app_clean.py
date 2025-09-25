"""Enhanced FastAPI-based interactive documentation server with analytics."""

import os
import json
import logging
import asyncio
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import uvicorn

try:
    from ..generators.ast_parser import PythonASTParser, DocumentationNode
    from ..generators.template_engine import DocumentationTemplateEngine
    from ..validation.coverage_analyzer import DocumentationCoverageAnalyzer
    from ..validation.quality_checker import DocumentationQualityChecker
    from .sandbox import CodeSandbox
    from ..analytics.collector import AnalyticsCollector, TrackingConfig
    from ..analytics.storage import AnalyticsStorage
    from ..analytics.privacy import PrivacyManager
    from ..analytics.dashboard import AnalyticsDashboard
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generators.ast_parser import PythonASTParser, DocumentationNode
    from generators.template_engine import DocumentationTemplateEngine
    from validation.coverage_analyzer import DocumentationCoverageAnalyzer
    from validation.quality_checker import DocumentationQualityChecker
    from server.sandbox import CodeSandbox
    from analytics.collector import AnalyticsCollector, TrackingConfig
    from analytics.storage import AnalyticsStorage
    from analytics.privacy import PrivacyManager
    from analytics.dashboard import AnalyticsDashboard


logger = logging.getLogger(__name__)


class CodeExecutionRequest(BaseModel):
    """Request model for code execution."""
    code: str
    language: str = "python"
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class CodeExecutionResponse(BaseModel):
    """Response model for code execution."""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    output: Optional[str] = None
    execution_time: Optional[float] = None
    session_id: Optional[str] = None


class DocumentationSearchRequest(BaseModel):
    """Request model for documentation search."""
    query: str
    member_types: Optional[List[str]] = None
    include_private: bool = False
    session_id: Optional[str] = None


class DocumentationSearchResponse(BaseModel):
    """Response model for documentation search."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: int


class AnalyticsEventRequest(BaseModel):
    """Request model for analytics events."""
    event_type: str
    page_path: str
    session_id: str
    duration_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class PageViewRequest(BaseModel):
    """Request model for page view tracking."""
    page_path: str
    session_id: str
    referrer: Optional[str] = None
    viewport_size: Optional[str] = None
    duration_ms: Optional[int] = None


class InteractionRequest(BaseModel):
    """Request model for user interaction tracking."""
    session_id: str
    page_path: str
    interaction_type: str
    element_id: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class PrivacyConsentRequest(BaseModel):
    """Request model for privacy consent updates."""
    consent_level: str
    session_id: Optional[str] = None


class EnhancedDocumentationServer:
    """Enhanced documentation server with analytics and real-time features."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced documentation server.

        Args:
            config: Server configuration dictionary
        """
        self.config = config
        self.app = FastAPI(
            title="Enhanced Documentation Server",
            description="Interactive documentation with analytics and live examples",
            version="2.0.0"
        )

        # Add middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get('cors_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Initialize components
        self._setup_components()
        self._setup_routes()

        logger.info("Enhanced documentation server initialized")

    def _setup_components(self):
        """Set up server components."""
        # Initialize analytics
        analytics_db_path = Path(self.config.get('analytics_db', 'analytics.db'))
        self.analytics_storage = AnalyticsStorage(analytics_db_path)

        privacy_settings_path = Path(self.config.get('privacy_settings', 'privacy_settings.json'))
        self.privacy_manager = PrivacyManager(privacy_settings_path)

        tracking_config = TrackingConfig(
            enabled=self.config.get('analytics_enabled', True),
            batch_size=self.config.get('analytics_batch_size', 10),
            flush_interval_seconds=self.config.get('analytics_flush_interval', 30)
        )

        self.analytics_collector = AnalyticsCollector(
            self.analytics_storage,
            tracking_config
        )

        self.analytics_dashboard = AnalyticsDashboard(
            self.analytics_storage,
            self.privacy_manager
        )

        # Initialize documentation components
        self.ast_parser = PythonASTParser(
            include_private=self.config.get('include_private', False)
        )

        template_dir = Path(self.config.get('template_dir', 'templates'))
        if template_dir.exists():
            self.template_engine = DocumentationTemplateEngine(template_dir, self.config)
        else:
            self.template_engine = None

        self.coverage_analyzer = DocumentationCoverageAnalyzer()
        self.quality_checker = DocumentationQualityChecker()

        # Initialize code sandbox
        sandbox_config = self.config.get('sandbox', {})
        self.code_sandbox = CodeSandbox(
            timeout=sandbox_config.get('timeout', 30),
            memory_limit=sandbox_config.get('memory_limit', 128)
        )

        # Cache for parsed modules
        self._modules_cache = None
        self._cache_timestamp = None

        logger.info("Server components initialized")

    def _setup_routes(self):
        """Set up API routes."""
        # Main documentation routes
        self.app.get("/", response_class=HTMLResponse)(self._home)
        self.app.get("/health")(self._health_check)

        # API routes
        self.app.get("/api/modules")(self._get_modules)
        self.app.get("/api/modules/{module_name}")(self._get_module)
        self.app.post("/api/search", response_model=DocumentationSearchResponse)(self._search_documentation)
        self.app.post("/api/execute", response_model=CodeExecutionResponse)(self._execute_code)

        # Analytics routes
        self.app.post("/api/analytics/page-view")(self._track_page_view)
        self.app.post("/api/analytics/interaction")(self._track_interaction)
        self.app.post("/api/analytics/event")(self._track_analytics_event)
        self.app.get("/api/analytics/dashboard")(self._get_analytics_dashboard)

        # Privacy routes
        self.app.post("/api/privacy/consent")(self._update_privacy_consent)
        self.app.get("/api/privacy/status")(self._get_privacy_status)

        # Real-time routes
        self.app.get("/api/live/search/{query}")(self._live_search)
        self.app.websocket("/ws/live-updates")(self._websocket_live_updates)

        # Admin routes
        self.app.get("/api/admin/stats")(self._get_admin_stats)
        self.app.get("/api/admin/modules/reload")(self._reload_modules)

        logger.info("API routes configured")

    async def _home(self, request: Request):
        """Serve the main documentation page."""
        session_id = request.cookies.get('session_id', str(uuid.uuid4()))

        # Track page view
        await self._track_page_view_internal(
            page_path="/",
            session_id=session_id,
            user_agent=request.headers.get('user-agent'),
            referrer=request.headers.get('referer')
        )

        # Get modules for rendering
        modules = await self._get_cached_modules()

        context = {
            "request": request,
            "modules": modules,
            "session_id": session_id,
            "config": self.config
        }

        if self.template_engine:
            return HTMLResponse(
                self.template_engine.render_template("index.html", context)
            )
        else:
            # Fallback HTML
            html_content = self._generate_fallback_html(modules)
            return HTMLResponse(html_content)

    async def _health_check(self):
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "analytics": self.analytics_collector.is_enabled(),
                "sandbox": True,
                "parser": True,
                "template_engine": self.template_engine is not None
            }
        }

    async def _get_modules(self):
        """Get all documentation modules."""
        modules = await self._get_cached_modules()
        return {
            "modules": [self._serialize_module_summary(module) for module in modules],
            "total": len(modules),
            "generated_at": datetime.now().isoformat()
        }

    async def _get_module(self, module_name: str, request: Request):
        """Get detailed information about a specific module."""
        session_id = request.cookies.get('session_id', str(uuid.uuid4()))

        # Track module view
        await self._track_page_view_internal(
            page_path=f"/api/modules/{module_name}",
            session_id=session_id,
            user_agent=request.headers.get('user-agent')
        )

        modules = await self._get_cached_modules()
        module = next((m for m in modules if m.name == module_name), None)

        if not module:
            raise HTTPException(status_code=404, detail=f"Module '{module_name}' not found")

        # Get coverage and quality info
        try:
            coverage_info = await self._get_module_coverage(module)
            quality_info = await self._get_module_quality(module)
        except Exception as e:
            logger.warning(f"Failed to get coverage/quality info for {module_name}: {e}")
            coverage_info = None
            quality_info = None

        return {
            "module": self._serialize_module_detailed(module),
            "coverage": coverage_info,
            "quality": quality_info,
            "generated_at": datetime.now().isoformat()
        }

    async def _search_documentation(
        self,
        search_request: DocumentationSearchRequest,
        request: Request
    ) -> DocumentationSearchResponse:
        """Search documentation with analytics tracking."""
        start_time = datetime.now()
        session_id = search_request.session_id or request.cookies.get('session_id', str(uuid.uuid4()))

        try:
            modules = await self._get_cached_modules()
            results = self._perform_search(
                modules,
                search_request.query,
                search_request.member_types,
                search_request.include_private
            )

            search_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Track search analytics
            if self.analytics_collector.is_enabled():
                self.analytics_collector.track_search(
                    query=search_request.query,
                    session_id=session_id,
                    page_path="/api/search",
                    results_count=len(results),
                    duration_ms=search_time_ms
                )

            return DocumentationSearchResponse(
                query=search_request.query,
                results=results,
                total_results=len(results),
                search_time_ms=search_time_ms
            )

        except Exception as e:
            # Track error
            if self.analytics_collector.is_enabled():
                self.analytics_collector.track_error(
                    session_id=session_id,
                    page_path="/api/search",
                    error_type="SearchError",
                    error_message=str(e)
                )
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    async def _execute_code(
        self,
        execution_request: CodeExecutionRequest,
        request: Request
    ) -> CodeExecutionResponse:
        """Execute code in sandbox with analytics tracking."""
        start_time = datetime.now()
        session_id = execution_request.session_id or request.cookies.get('session_id', str(uuid.uuid4()))

        try:
            # Execute code in sandbox
            result = await self.code_sandbox.execute_code(
                execution_request.code,
                context=execution_request.context
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Track code execution analytics
            if self.analytics_collector.is_enabled():
                self.analytics_collector.track_code_execution(
                    session_id=session_id,
                    page_path="/api/execute",
                    language=execution_request.language,
                    success=result.get('success', False),
                    execution_time_ms=int(execution_time * 1000),
                    code_length=len(execution_request.code),
                    error_type=result.get('error_type') if not result.get('success') else None
                )

            return CodeExecutionResponse(
                success=result.get('success', False),
                result=result.get('result'),
                error=result.get('error'),
                output=result.get('output'),
                execution_time=execution_time,
                session_id=session_id
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            # Track error
            if self.analytics_collector.is_enabled():
                self.analytics_collector.track_error(
                    session_id=session_id,
                    page_path="/api/execute",
                    error_type="ExecutionError",
                    error_message=str(e)
                )

            return CodeExecutionResponse(
                success=False,
                error=str(e),
                execution_time=execution_time,
                session_id=session_id
            )

    async def _track_page_view(self, page_view_request: PageViewRequest):
        """Track page view analytics."""
        if not self.analytics_collector.is_enabled():
            return {"status": "analytics_disabled"}

        success = self.analytics_collector.track_page_view(
            page_path=page_view_request.page_path,
            session_id=page_view_request.session_id,
            referrer=page_view_request.referrer,
            duration_ms=page_view_request.duration_ms,
            viewport_size=page_view_request.viewport_size
        )

        return {"status": "tracked" if success else "failed"}

    async def _track_interaction(self, interaction_request: InteractionRequest):
        """Track user interaction analytics."""
        if not self.analytics_collector.is_enabled():
            return {"status": "analytics_disabled"}

        success = self.analytics_collector.track_interaction(
            session_id=interaction_request.session_id,
            page_path=interaction_request.page_path,
            interaction_type=interaction_request.interaction_type,
            element_id=interaction_request.element_id,
            duration_ms=interaction_request.duration_ms,
            metadata=interaction_request.metadata
        )

        return {"status": "tracked" if success else "failed"}

    async def _track_analytics_event(self, event_request: AnalyticsEventRequest):
        """Track generic analytics event."""
        # This would route to appropriate tracking methods based on event_type
        return {"status": "tracked"}

    async def _get_analytics_dashboard(self, request: Request):
        """Get analytics dashboard data."""
        session_id = request.cookies.get('session_id')

        # Check if user has permission to view analytics
        if not self._can_view_analytics(session_id):
            raise HTTPException(status_code=403, detail="Analytics access denied")

        dashboard_data = self.analytics_dashboard.generate_dashboard()
        if not dashboard_data:
            raise HTTPException(status_code=500, detail="Failed to generate analytics dashboard")

        return {
            "dashboard": {
                "generated_at": dashboard_data.generated_at.isoformat(),
                "date_range": [
                    dashboard_data.date_range[0].isoformat(),
                    dashboard_data.date_range[1].isoformat()
                ],
                "summary_metrics": [
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "trend": metric.trend
                    } for metric in dashboard_data.summary_metrics
                ],
                "charts": [
                    {
                        "type": chart.chart_type,
                        "title": chart.title,
                        "labels": chart.labels,
                        "datasets": chart.datasets,
                        "options": chart.options
                    } for chart in dashboard_data.charts
                ],
                "insights": dashboard_data.insights
            }
        }

    async def _update_privacy_consent(self, consent_request: PrivacyConsentRequest):
        """Update user privacy consent."""
        from analytics.privacy import ConsentLevel

        try:
            consent_level = ConsentLevel(consent_request.consent_level)
            success = self.privacy_manager.set_consent_level(consent_level)

            if success and consent_request.session_id:
                self.privacy_manager.clear_session_cache()

            return {
                "status": "updated" if success else "failed",
                "consent_level": consent_level.value
            }
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid consent level")

    async def _get_privacy_status(self):
        """Get current privacy settings."""
        summary = self.privacy_manager.get_privacy_summary()
        return {"privacy": summary}

    async def _live_search(self, query: str):
        """Live search with streaming results."""
        def generate_search_results():
            modules = self._get_modules_sync()
            results = self._perform_search(modules, query, limit=10)

            for i, result in enumerate(results):
                yield f"data: {json.dumps({'index': i, 'result': result})}\n\n"

            yield f"data: {json.dumps({'complete': True, 'total': len(results)})}\n\n"

        return StreamingResponse(
            generate_search_results(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )

    async def _websocket_live_updates(self, websocket):
        """WebSocket endpoint for live updates."""
        await websocket.accept()

        try:
            while True:
                # Send periodic updates (e.g., analytics, new content)
                update = {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "stats": await self._get_live_stats()
                }

                await websocket.send_text(json.dumps(update))
                await asyncio.sleep(30)  # Send updates every 30 seconds

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()

    async def _get_admin_stats(self, request: Request):
        """Get administrative statistics."""
        if not self._is_admin(request):
            raise HTTPException(status_code=403, detail="Admin access required")

        return {
            "server": {
                "uptime": "unknown",  # Would track actual uptime
                "requests_served": "unknown",
                "active_sessions": "unknown"
            },
            "analytics": self.analytics_storage.get_database_info(),
            "modules": {
                "total": len(await self._get_cached_modules()),
                "cache_age": self._get_cache_age()
            }
        }

    async def _reload_modules(self, request: Request):
        """Reload documentation modules."""
        if not self._is_admin(request):
            raise HTTPException(status_code=403, detail="Admin access required")

        self._modules_cache = None
        self._cache_timestamp = None

        modules = await self._get_cached_modules()
        return {
            "status": "reloaded",
            "modules_count": len(modules),
            "reloaded_at": datetime.now().isoformat()
        }

    # Helper methods
    async def _get_cached_modules(self) -> List[DocumentationNode]:
        """Get cached documentation modules."""
        cache_ttl = self.config.get('cache_ttl_seconds', 300)  # 5 minutes default

        if (self._modules_cache is None or
            self._cache_timestamp is None or
            (datetime.now() - self._cache_timestamp).total_seconds() > cache_ttl):

            self._modules_cache = await self._load_modules()
            self._cache_timestamp = datetime.now()

        return self._modules_cache

    def _get_modules_sync(self) -> List[DocumentationNode]:
        """Synchronous version for streaming endpoints."""
        if self._modules_cache:
            return self._modules_cache

        # Load synchronously if needed
        source_dirs = self.config.get('source_dirs', ['src'])
        modules = []

        for source_dir in source_dirs:
            if Path(source_dir).exists():
                try:
                    parsed = self.ast_parser.parse_directory(Path(source_dir), recursive=True)
                    modules.extend(parsed)
                except Exception as e:
                    logger.warning(f"Failed to parse {source_dir}: {e}")

        return modules

    async def _load_modules(self) -> List[DocumentationNode]:
        """Load documentation modules asynchronously."""
        source_dirs = self.config.get('source_dirs', ['src'])
        modules = []

        for source_dir in source_dirs:
            if Path(source_dir).exists():
                try:
                    # Run in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    parsed = await loop.run_in_executor(
                        None,
                        lambda: self.ast_parser.parse_directory(Path(source_dir), recursive=True)
                    )
                    modules.extend(parsed)
                except Exception as e:
                    logger.warning(f"Failed to parse {source_dir}: {e}")

        return modules

    async def _track_page_view_internal(self, page_path: str, session_id: str,
                                      user_agent: str = None, referrer: str = None):
        """Internal method to track page views."""
        if self.analytics_collector.is_enabled():
            self.analytics_collector.track_page_view(
                page_path=page_path,
                session_id=session_id,
                user_agent=user_agent,
                referrer=referrer
            )

    def _perform_search(self, modules: List[DocumentationNode], query: str,
                       member_types: List[str] = None, include_private: bool = False,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Perform documentation search."""
        results = []
        query_lower = query.lower()

        for module in modules:
            if not include_private and module.is_private:
                continue

            # Search in module name and docstring
            if query_lower in module.name.lower() or (module.docstring and query_lower in module.docstring.lower()):
                results.append({
                    "type": "module",
                    "name": module.name,
                    "docstring": module.docstring,
                    "source_file": module.source_file,
                    "line_number": module.line_number
                })

            # Search in module members
            results.extend(self._search_members(module, query_lower, member_types, include_private))

        # Sort by relevance (simple scoring)
        results.sort(key=lambda x: self._calculate_relevance_score(x, query_lower), reverse=True)

        return results[:limit]

    def _search_members(self, node: DocumentationNode, query: str,
                       member_types: List[str] = None, include_private: bool = False) -> List[Dict[str, Any]]:
        """Recursively search members of a documentation node."""
        results = []

        for member in node.children:
            if not include_private and member.is_private:
                continue

            if member_types and member.member_type.value not in member_types:
                continue

            # Check if member matches query
            if (query in member.name.lower() or
                (member.docstring and query in member.docstring.lower())):

                results.append({
                    "type": member.member_type.value,
                    "name": member.name,
                    "full_name": f"{node.name}.{member.name}",
                    "docstring": member.docstring,
                    "signature": member.signature,
                    "source_file": member.source_file,
                    "line_number": member.line_number,
                    "parent": node.name
                })

            # Recursively search nested members
            if member.children:
                results.extend(self._search_members(member, query, member_types, include_private))

        return results

    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for search results."""
        score = 0.0

        # Exact name match gets highest score
        if result["name"].lower() == query:
            score += 10.0
        elif query in result["name"].lower():
            score += 5.0

        # Docstring matches
        if result.get("docstring"):
            docstring_lower = result["docstring"].lower()
            query_words = query.split()
            for word in query_words:
                if word in docstring_lower:
                    score += 1.0

        # Type-specific scoring
        type_scores = {
            "module": 1.0,
            "class": 2.0,
            "function": 3.0,
            "method": 2.5
        }
        score += type_scores.get(result["type"], 1.0)

        return score

    def _serialize_module_summary(self, module: DocumentationNode) -> Dict[str, Any]:
        """Serialize module for summary listing."""
        return {
            "name": module.name,
            "docstring": module.docstring[:200] + "..." if module.docstring and len(module.docstring) > 200 else module.docstring,
            "source_file": module.source_file,
            "line_number": module.line_number,
            "member_count": len(module.children),
            "member_types": list(set(child.member_type.value for child in module.children))
        }

    def _serialize_module_detailed(self, module: DocumentationNode) -> Dict[str, Any]:
        """Serialize module with detailed information."""
        return {
            "name": module.name,
            "docstring": module.docstring,
            "source_file": module.source_file,
            "line_number": module.line_number,
            "signature": module.signature,
            "parameters": [
                {
                    "name": param.name,
                    "annotation": param.annotation,
                    "default": param.default,
                    "description": param.description
                } for param in module.parameters
            ] if module.parameters else [],
            "return_annotation": module.return_annotation,
            "return_description": module.return_description,
            "examples": module.examples,
            "decorators": module.decorators,
            "metadata": module.metadata,
            "members": [self._serialize_member(member) for member in module.children]
        }

    def _serialize_member(self, member: DocumentationNode) -> Dict[str, Any]:
        """Serialize a member for detailed display."""
        return {
            "name": member.name,
            "type": member.member_type.value,
            "docstring": member.docstring,
            "signature": member.signature,
            "source_file": member.source_file,
            "line_number": member.line_number,
            "is_private": member.is_private,
            "parameters": [
                {
                    "name": param.name,
                    "annotation": param.annotation,
                    "default": param.default,
                    "description": param.description
                } for param in member.parameters
            ] if member.parameters else [],
            "return_annotation": member.return_annotation,
            "examples": member.examples,
            "decorators": member.decorators,
            "children": [self._serialize_member(child) for child in member.children] if member.children else []
        }

    async def _get_module_coverage(self, module: DocumentationNode) -> Optional[Dict[str, Any]]:
        """Get coverage information for a module."""
        try:
            # This would integrate with the coverage analyzer
            # For now, return placeholder data
            return {
                "coverage_percentage": 85.0,
                "documented_members": 17,
                "total_members": 20,
                "missing_docs": ["method1", "method2", "method3"]
            }
        except Exception as e:
            logger.error(f"Failed to get coverage for {module.name}: {e}")
            return None

    async def _get_module_quality(self, module: DocumentationNode) -> Optional[Dict[str, Any]]:
        """Get quality information for a module."""
        try:
            # This would integrate with the quality checker
            # For now, return placeholder data
            return {
                "quality_score": 92.5,
                "issues_found": 2,
                "high_quality_members": 15,
                "improvement_suggestions": [
                    "Add examples to method1",
                    "Improve parameter descriptions in method2"
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get quality info for {module.name}: {e}")
            return None

    async def _get_live_stats(self) -> Dict[str, Any]:
        """Get live statistics for WebSocket updates."""
        return {
            "active_sessions": 1,  # Would track actual sessions
            "recent_searches": 5,  # Would track recent activity
            "server_load": 0.1     # Would track actual server metrics
        }

    def _can_view_analytics(self, session_id: str) -> bool:
        """Check if user can view analytics dashboard."""
        # In a real implementation, this would check user permissions
        return True

    def _is_admin(self, request: Request) -> bool:
        """Check if request is from an admin user."""
        # In a real implementation, this would check authentication
        admin_key = request.headers.get('X-Admin-Key')
        return admin_key == self.config.get('admin_key')

    def _get_cache_age(self) -> int:
        """Get age of module cache in seconds."""
        if not self._cache_timestamp:
            return 0
        return int((datetime.now() - self._cache_timestamp).total_seconds())

    def _generate_fallback_html(self, modules: List[DocumentationNode]) -> str:
        """Generate fallback HTML when no template engine is available."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Documentation Server</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .module { border: 1px solid #ddd; padding: 20px; margin: 20px 0; }
                .member { margin: 10px 0; padding: 10px; background: #f5f5f5; }
                pre { background: #f0f0f0; padding: 10px; overflow: auto; }
            </style>
        </head>
        <body>
            <h1>Documentation Server</h1>
            <h2>Available Modules</h2>
        """

        for module in modules:
            html += f"""
            <div class="module">
                <h3>{module.name}</h3>
                <p>{module.docstring or 'No description available'}</p>
                <p><strong>Members:</strong> {len(module.children)}</p>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    def run(self, host: str = "127.0.0.1", port: int = 8080, reload: bool = False):
        """Run the enhanced documentation server."""
        try:
            logger.info(f"Starting enhanced documentation server on {host}:{port}")
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            # Cleanup
            if self.analytics_collector:
                self.analytics_collector.shutdown()
            logger.info("Enhanced documentation server stopped")


# Factory function for creating server instances
def create_enhanced_documentation_server(config: Dict[str, Any]) -> EnhancedDocumentationServer:
    """Create an enhanced documentation server instance.

    Args:
        config: Server configuration dictionary

    Returns:
        EnhancedDocumentationServer instance
    """
    return EnhancedDocumentationServer(config)


# Legacy compatibility function
def run_server(config: Dict[str, Any], host: str = "127.0.0.1", port: int = 8080):
    """Legacy function to run the documentation server.

    Args:
        config: Server configuration
        host: Server host
        port: Server port
    """
    server = create_enhanced_documentation_server(config)
    server.run(host=host, port=port)
