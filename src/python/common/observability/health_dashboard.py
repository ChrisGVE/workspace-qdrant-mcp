"""
Unified Health Dashboard Coordinator for workspace-qdrant-mcp.

Provides a comprehensive health dashboard that aggregates data from all health monitoring
components and presents a unified view of system health with real-time updates,
historical trends, and actionable insights.

Key Features:
    - Real-time health status aggregation across all four components
    - Interactive web dashboard with live updates via WebSocket
    - Historical health trend visualization and analysis
    - Alert timeline and management interface
    - Component dependency visualization
    - Performance metrics correlation analysis
    - Health action triggers and recovery management
    - Export capabilities for monitoring integration

Dashboard Components:
    - System overview with component status matrix
    - Real-time metrics charts and graphs
    - Alert management and escalation tracking
    - Health trend analysis and predictions
    - Component dependency health visualization
    - Recovery action logs and effectiveness tracking

Example:
    ```python
    from workspace_qdrant_mcp.observability.health_dashboard import HealthDashboard

    # Initialize health dashboard
    dashboard = HealthDashboard(port=8080)
    await dashboard.initialize()

    # Start dashboard server
    await dashboard.start_server()

    # Dashboard available at http://localhost:8080/health
    ```
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from aiohttp import WSMsgType, web
from loguru import logger

from .enhanced_alerting import AlertingManager, get_alerting_manager
from .grpc_health import GrpcHealthService, get_grpc_health_service
from .health import HealthChecker, get_health_checker
from .health_coordinator import HealthCoordinator, get_health_coordinator
from .metrics import metrics_instance


class HealthDashboard:
    """
    Unified Health Dashboard Coordinator.

    Provides a comprehensive web-based dashboard for monitoring system health
    with real-time updates, historical analysis, and management capabilities.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        dashboard_path: str = "/health",
        enable_websocket: bool = True,
        update_interval_seconds: float = 5.0
    ):
        """
        Initialize health dashboard.

        Args:
            host: Host address to bind the dashboard server
            port: Port number for the dashboard server
            dashboard_path: URL path for the dashboard
            enable_websocket: Enable WebSocket for real-time updates
            update_interval_seconds: Interval for pushing updates to connected clients
        """
        self.host = host
        self.port = port
        self.dashboard_path = dashboard_path
        self.enable_websocket = enable_websocket
        self.update_interval_seconds = update_interval_seconds

        # Web server
        self.app: web.Application | None = None
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        # Health monitoring components
        self.health_coordinator: HealthCoordinator | None = None
        self.alerting_manager: AlertingManager | None = None
        self.grpc_health_service: GrpcHealthService | None = None
        self.health_checker: HealthChecker | None = None

        # WebSocket connections
        self.websocket_connections: set[web.WebSocketResponse] = set()

        # Background tasks
        self.background_tasks: list[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

        # Dashboard state
        self.last_dashboard_data: dict[str, Any] | None = None
        self.dashboard_access_count = 0

        logger.info(
            "Health Dashboard initialized",
            host=host,
            port=port,
            dashboard_path=dashboard_path,
            websocket_enabled=enable_websocket
        )

    async def initialize(self) -> None:
        """Initialize the health dashboard and all monitoring components."""
        try:
            logger.info("Initializing Health Dashboard")

            # Initialize health monitoring components
            self.health_coordinator = await get_health_coordinator()
            self.alerting_manager = await get_alerting_manager()
            self.grpc_health_service = await get_grpc_health_service()
            self.health_checker = get_health_checker()

            # Start health monitoring if not already started
            await self.health_coordinator.start_monitoring()

            # Create web application
            self.app = web.Application()
            await self._setup_routes()

            logger.info("Health Dashboard initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Health Dashboard: {e}")
            raise

    async def start_server(self) -> None:
        """Start the health dashboard web server."""
        try:
            logger.info(f"Starting Health Dashboard server on {self.host}:{self.port}")

            # Create and start web server
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()

            # Start background tasks
            if self.enable_websocket:
                self.background_tasks.append(
                    asyncio.create_task(self._websocket_update_loop())
                )

            self.background_tasks.append(
                asyncio.create_task(self._dashboard_metrics_loop())
            )

            logger.info(
                "Health Dashboard started",
                url=f"http://{self.host}:{self.port}{self.dashboard_path}",
                websocket_enabled=self.enable_websocket
            )

        except Exception as e:
            logger.error(f"Failed to start Health Dashboard server: {e}")
            raise

    async def stop_server(self) -> None:
        """Stop the health dashboard web server."""
        logger.info("Stopping Health Dashboard server")

        # Signal shutdown
        self.shutdown_event.set()

        # Close WebSocket connections
        for ws in self.websocket_connections.copy():
            await ws.close()

        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()

        # Stop web server
        if self.site:
            await self.site.stop()

        if self.runner:
            await self.runner.cleanup()

        logger.info("Health Dashboard server stopped")

    async def _setup_routes(self) -> None:
        """Set up web routes for the dashboard."""
        # Main dashboard route
        self.app.router.add_get(self.dashboard_path, self._dashboard_handler)

        # API routes
        self.app.router.add_get(f"{self.dashboard_path}/api/status", self._api_status_handler)
        self.app.router.add_get(f"{self.dashboard_path}/api/dashboard", self._api_dashboard_handler)
        self.app.router.add_get(f"{self.dashboard_path}/api/alerts", self._api_alerts_handler)
        self.app.router.add_get(f"{self.dashboard_path}/api/metrics", self._api_metrics_handler)
        self.app.router.add_get(f"{self.dashboard_path}/api/components", self._api_components_handler)

        # WebSocket route
        if self.enable_websocket:
            self.app.router.add_get(f"{self.dashboard_path}/ws", self._websocket_handler)

        # Management actions
        self.app.router.add_post(f"{self.dashboard_path}/api/actions/acknowledge", self._action_acknowledge_handler)
        self.app.router.add_post(f"{self.dashboard_path}/api/actions/resolve", self._action_resolve_handler)
        self.app.router.add_post(f"{self.dashboard_path}/api/actions/recovery", self._action_recovery_handler)

        # Static assets (would serve from a proper static directory in production)
        self.app.router.add_get(f"{self.dashboard_path}/dashboard.js", self._static_js_handler)
        self.app.router.add_get(f"{self.dashboard_path}/dashboard.css", self._static_css_handler)

        logger.debug("Health Dashboard routes configured")

    async def _dashboard_handler(self, request: web.Request) -> web.Response:
        """Serve the main dashboard HTML page."""
        self.dashboard_access_count += 1

        # Record dashboard access
        metrics_instance.increment_counter("health_dashboard_views_total")

        html_content = self._generate_dashboard_html()

        return web.Response(
            text=html_content,
            content_type="text/html",
            headers={"Cache-Control": "no-cache"}
        )

    async def _api_status_handler(self, request: web.Request) -> web.Response:
        """API endpoint for basic health status."""
        try:
            unified_status = await self.health_coordinator.get_unified_health_status()

            response_data = {
                "status": "success",
                "data": {
                    "overall_status": unified_status.get("overall_status"),
                    "timestamp": unified_status.get("timestamp"),
                    "component_count": len(unified_status.get("component_health", {})),
                    "active_alerts": len(unified_status.get("active_alerts", {}).get("recent_alerts", [])),
                }
            }

            return web.json_response(response_data)

        except Exception as e:
            logger.error(f"API status handler error: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )

    async def _api_dashboard_handler(self, request: web.Request) -> web.Response:
        """API endpoint for complete dashboard data."""
        try:
            dashboard_data = await self.health_coordinator.get_health_dashboard_data()

            # Cache the data for WebSocket updates
            self.last_dashboard_data = dashboard_data

            return web.json_response({
                "status": "success",
                "data": dashboard_data
            })

        except Exception as e:
            logger.error(f"API dashboard handler error: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )

    async def _api_alerts_handler(self, request: web.Request) -> web.Response:
        """API endpoint for alert management."""
        try:
            # Get alert summary from alerting manager
            alert_data = {
                "active_alerts": [],
                "resolved_alerts": [],
                "alert_groups": [],
            }

            if self.alerting_manager:
                # Get recent alerts (implementation would depend on alerting manager API)
                alert_data["summary"] = {
                    "total_active": 0,
                    "by_severity": {},
                    "by_component": {},
                }

            return web.json_response({
                "status": "success",
                "data": alert_data
            })

        except Exception as e:
            logger.error(f"API alerts handler error: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )

    async def _api_metrics_handler(self, request: web.Request) -> web.Response:
        """API endpoint for metrics data."""
        try:
            # Get metrics summary
            metrics_data = metrics_instance.get_metrics_summary()

            # Add dashboard-specific metrics
            metrics_data["dashboard_metrics"] = {
                "access_count": self.dashboard_access_count,
                "websocket_connections": len(self.websocket_connections),
                "last_update": time.time(),
            }

            return web.json_response({
                "status": "success",
                "data": metrics_data
            })

        except Exception as e:
            logger.error(f"API metrics handler error: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )

    async def _api_components_handler(self, request: web.Request) -> web.Response:
        """API endpoint for component details."""
        try:
            # Get component status from gRPC health service
            component_data = {}

            if self.grpc_health_service:
                component_data = await self.grpc_health_service.get_all_service_statuses()

            return web.json_response({
                "status": "success",
                "data": component_data
            })

        except Exception as e:
            logger.error(f"API components handler error: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )

    async def _action_acknowledge_handler(self, request: web.Request) -> web.Response:
        """Handle alert acknowledgment actions."""
        try:
            data = await request.json()
            alert_id = data.get("alert_id")
            acknowledged_by = data.get("acknowledged_by", "dashboard_user")

            if not alert_id:
                return web.json_response(
                    {"status": "error", "message": "alert_id required"},
                    status=400
                )

            success = await self.alerting_manager.acknowledge_alert(alert_id, acknowledged_by)

            return web.json_response({
                "status": "success" if success else "error",
                "message": "Alert acknowledged" if success else "Failed to acknowledge alert"
            })

        except Exception as e:
            logger.error(f"Alert acknowledgment error: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )

    async def _action_resolve_handler(self, request: web.Request) -> web.Response:
        """Handle alert resolution actions."""
        try:
            data = await request.json()
            alert_id = data.get("alert_id")
            resolved_by = data.get("resolved_by", "dashboard_user")

            if not alert_id:
                return web.json_response(
                    {"status": "error", "message": "alert_id required"},
                    status=400
                )

            success = await self.alerting_manager.resolve_alert(alert_id, resolved_by)

            return web.json_response({
                "status": "success" if success else "error",
                "message": "Alert resolved" if success else "Failed to resolve alert"
            })

        except Exception as e:
            logger.error(f"Alert resolution error: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )

    async def _action_recovery_handler(self, request: web.Request) -> web.Response:
        """Handle component recovery actions."""
        try:
            data = await request.json()
            component = data.get("component")
            action = data.get("action", "restart")

            if not component:
                return web.json_response(
                    {"status": "error", "message": "component required"},
                    status=400
                )

            # Map component name to component type (simplified)
            from ..core.component_coordination import ComponentType

            component_mapping = {
                "rust_daemon": ComponentType.RUST_DAEMON,
                "python_mcp_server": ComponentType.PYTHON_MCP_SERVER,
                "cli_utility": ComponentType.CLI_UTILITY,
                "context_injector": ComponentType.CONTEXT_INJECTOR,
            }

            component_type = component_mapping.get(component)
            if not component_type:
                return web.json_response(
                    {"status": "error", "message": "invalid component"},
                    status=400
                )

            success = await self.health_coordinator.trigger_manual_recovery(component_type, action)

            return web.json_response({
                "status": "success" if success else "error",
                "message": f"Recovery action {action} {'initiated' if success else 'failed'} for {component}"
            })

        except Exception as e:
            logger.error(f"Recovery action error: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )

    async def _websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.websocket_connections.add(ws)
        logger.debug(f"WebSocket connected: {len(self.websocket_connections)} total connections")

        try:
            # Send initial dashboard data
            if self.last_dashboard_data:
                await ws.send_str(json.dumps({
                    "type": "dashboard_update",
                    "data": self.last_dashboard_data
                }))

            # Handle incoming messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON"
                        }))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")

        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self.websocket_connections.discard(ws)
            logger.debug(f"WebSocket disconnected: {len(self.websocket_connections)} total connections")

        return ws

    async def _handle_websocket_message(self, ws: web.WebSocketResponse, data: dict[str, Any]) -> None:
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")

        if message_type == "ping":
            await ws.send_str(json.dumps({"type": "pong", "timestamp": time.time()}))

        elif message_type == "subscribe":
            # Handle subscription to specific data streams
            await ws.send_str(json.dumps({
                "type": "subscription_confirmed",
                "streams": data.get("streams", [])
            }))

        else:
            await ws.send_str(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }))

    async def _websocket_update_loop(self) -> None:
        """Background task for sending WebSocket updates."""
        while not self.shutdown_event.is_set():
            try:
                if self.websocket_connections:
                    # Get latest dashboard data
                    dashboard_data = await self.health_coordinator.get_health_dashboard_data()

                    # Check if data has changed significantly
                    if self._should_send_update(dashboard_data):
                        self.last_dashboard_data = dashboard_data

                        # Send update to all connected clients
                        update_message = json.dumps({
                            "type": "dashboard_update",
                            "data": dashboard_data,
                            "timestamp": time.time()
                        })

                        # Send to all connections (remove failed ones)
                        failed_connections = []
                        for ws in self.websocket_connections.copy():
                            try:
                                await ws.send_str(update_message)
                            except Exception as e:
                                logger.debug(f"Failed to send WebSocket update: {e}")
                                failed_connections.append(ws)

                        # Remove failed connections
                        for ws in failed_connections:
                            self.websocket_connections.discard(ws)

                        if failed_connections:
                            logger.debug(f"Removed {len(failed_connections)} failed WebSocket connections")

                await asyncio.sleep(self.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket update loop error: {e}")
                await asyncio.sleep(self.update_interval_seconds)

    def _should_send_update(self, new_data: dict[str, Any]) -> bool:
        """Determine if dashboard update should be sent."""
        if not self.last_dashboard_data:
            return True

        # Check for significant changes
        old_status = self.last_dashboard_data.get("overall_status")
        new_status = new_data.get("overall_status")

        if old_status != new_status:
            return True

        # Check alert changes
        old_alerts = len(self.last_dashboard_data.get("active_alerts", {}).get("recent_alerts", []))
        new_alerts = len(new_data.get("active_alerts", {}).get("recent_alerts", []))

        if old_alerts != new_alerts:
            return True

        # Send periodic updates even if nothing changed
        last_update = self.last_dashboard_data.get("dashboard_metadata", {}).get("generated_at")
        if last_update:
            try:
                last_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                if (datetime.now(timezone.utc) - last_time).total_seconds() > 60:  # 1 minute
                    return True
            except Exception:
                return True

        return False

    async def _dashboard_metrics_loop(self) -> None:
        """Background task for recording dashboard metrics."""
        while not self.shutdown_event.is_set():
            try:
                # Record WebSocket connection metrics
                metrics_instance.set_gauge(
                    "health_dashboard_websocket_connections",
                    len(self.websocket_connections)
                )

                # Record dashboard access metrics
                metrics_instance.set_gauge(
                    "health_dashboard_access_count",
                    self.dashboard_access_count
                )

                await asyncio.sleep(30.0)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard metrics loop error: {e}")
                await asyncio.sleep(30.0)

    async def _static_js_handler(self, request: web.Request) -> web.Response:
        """Serve dashboard JavaScript."""
        js_content = self._generate_dashboard_js()
        return web.Response(
            text=js_content,
            content_type="application/javascript",
            headers={"Cache-Control": "max-age=300"}
        )

    async def _static_css_handler(self, request: web.Request) -> web.Response:
        """Serve dashboard CSS."""
        css_content = self._generate_dashboard_css()
        return web.Response(
            text=css_content,
            content_type="text/css",
            headers={"Cache-Control": "max-age=300"}
        )

    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML page."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workspace Qdrant MCP - Health Dashboard</title>
    <link rel="stylesheet" href="dashboard.css">
</head>
<body>
    <div id="app">
        <header class="dashboard-header">
            <h1>Workspace Qdrant MCP - Health Dashboard</h1>
            <div class="status-indicator" id="overall-status">
                <span class="status-text">Loading...</span>
            </div>
        </header>

        <main class="dashboard-main">
            <div class="dashboard-grid">
                <!-- System Overview -->
                <section class="dashboard-card">
                    <h2>System Overview</h2>
                    <div id="system-overview">
                        <div class="loading">Loading system status...</div>
                    </div>
                </section>

                <!-- Component Health -->
                <section class="dashboard-card">
                    <h2>Component Health</h2>
                    <div id="component-health">
                        <div class="loading">Loading component status...</div>
                    </div>
                </section>

                <!-- Active Alerts -->
                <section class="dashboard-card">
                    <h2>Active Alerts</h2>
                    <div id="active-alerts">
                        <div class="loading">Loading alerts...</div>
                    </div>
                </section>

                <!-- Performance Metrics -->
                <section class="dashboard-card">
                    <h2>Performance Metrics</h2>
                    <div id="performance-metrics">
                        <div class="loading">Loading metrics...</div>
                    </div>
                </section>

                <!-- Health Trends -->
                <section class="dashboard-card full-width">
                    <h2>Health Trends</h2>
                    <div id="health-trends">
                        <div class="loading">Loading trend analysis...</div>
                    </div>
                </section>
            </div>
        </main>

        <footer class="dashboard-footer">
            <div class="dashboard-info">
                <span>Last Updated: <span id="last-updated">Never</span></span>
                <span>WebSocket: <span id="websocket-status">Disconnected</span></span>
            </div>
        </footer>
    </div>

    <script src="dashboard.js"></script>
</body>
</html>
        """.strip()

    def _generate_dashboard_js(self) -> str:
        """Generate dashboard JavaScript."""
        return f"""
class HealthDashboard {{
    constructor() {{
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.dashboardData = null;

        this.init();
    }}

    init() {{
        this.loadInitialData();
        this.setupWebSocket();
        this.setupEventListeners();

        // Refresh data every 30 seconds as fallback
        setInterval(() => this.loadInitialData(), 30000);
    }}

    async loadInitialData() {{
        try {{
            const response = await fetch('{self.dashboard_path}/api/dashboard');
            const result = await response.json();

            if (result.status === 'success') {{
                this.updateDashboard(result.data);
            }} else {{
                this.showError('Failed to load dashboard data: ' + result.message);
            }}
        }} catch (error) {{
            this.showError('Error loading dashboard data: ' + error.message);
        }}
    }}

    setupWebSocket() {{
        if (!window.WebSocket || !{str(self.enable_websocket).lower()}) {{
            document.getElementById('websocket-status').textContent = 'Not Available';
            return;
        }}

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${{protocol}}//${{window.location.host}}{self.dashboard_path}/ws`;

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {{
            console.log('WebSocket connected');
            document.getElementById('websocket-status').textContent = 'Connected';
            this.reconnectAttempts = 0;

            // Send ping to keep connection alive
            setInterval(() => {{
                if (this.websocket.readyState === WebSocket.OPEN) {{
                    this.websocket.send(JSON.stringify({{type: 'ping'}}));
                }}
            }}, 30000);
        }};

        this.websocket.onmessage = (event) => {{
            try {{
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            }} catch (error) {{
                console.error('Error parsing WebSocket message:', error);
            }}
        }};

        this.websocket.onclose = () => {{
            console.log('WebSocket disconnected');
            document.getElementById('websocket-status').textContent = 'Disconnected';
            this.attemptReconnect();
        }};

        this.websocket.onerror = (error) => {{
            console.error('WebSocket error:', error);
            document.getElementById('websocket-status').textContent = 'Error';
        }};
    }}

    handleWebSocketMessage(message) {{
        switch (message.type) {{
            case 'dashboard_update':
                this.updateDashboard(message.data);
                break;
            case 'pong':
                // Connection is alive
                break;
            case 'error':
                console.error('WebSocket error:', message.message);
                break;
            default:
                console.log('Unknown WebSocket message type:', message.type);
        }}
    }}

    attemptReconnect() {{
        if (this.reconnectAttempts < this.maxReconnectAttempts) {{
            this.reconnectAttempts++;
            const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff

            setTimeout(() => {{
                console.log(`Attempting WebSocket reconnection (${{this.reconnectAttempts}}/${{this.maxReconnectAttempts}})`);
                this.setupWebSocket();
            }}, delay);
        }}
    }}

    updateDashboard(data) {{
        this.dashboardData = data;

        // Update overall status
        this.updateOverallStatus(data.overall_status);

        // Update components
        this.updateSystemOverview(data);
        this.updateComponentHealth(data.component_health || {{}});
        this.updateActiveAlerts(data.active_alerts || {{}});
        this.updatePerformanceMetrics(data.visualization_data || {{}});
        this.updateHealthTrends(data.trend_analysis || {{}});

        // Update last updated time
        document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
    }}

    updateOverallStatus(status) {{
        const statusElement = document.getElementById('overall-status');
        const statusText = statusElement.querySelector('.status-text');

        statusText.textContent = status ? status.toUpperCase() : 'UNKNOWN';

        // Remove existing status classes
        statusElement.classList.remove('healthy', 'warning', 'critical', 'unknown');

        // Add appropriate status class
        if (status === 'healthy') {{
            statusElement.classList.add('healthy');
        }} else if (status === 'warning' || status === 'degraded') {{
            statusElement.classList.add('warning');
        }} else if (status === 'critical' || status === 'unhealthy') {{
            statusElement.classList.add('critical');
        }} else {{
            statusElement.classList.add('unknown');
        }}
    }}

    updateSystemOverview(data) {{
        const overviewElement = document.getElementById('system-overview');

        const metrics = data.base_health || {{}};
        const components = data.component_health || {{}};

        let html = '<div class="metrics-grid">';

        // System metrics
        if (metrics.components) {{
            const componentCount = Object.keys(metrics.components).length;
            const healthyComponents = Object.values(metrics.components).filter(
                comp => comp.status === 'healthy'
            ).length;

            html += `
                <div class="metric-item">
                    <div class="metric-label">Components</div>
                    <div class="metric-value">${{healthyComponents}}/${{componentCount}}</div>
                </div>
            `;
        }}

        // Performance overview
        if (data.performance_correlations) {{
            const correlationCount = Object.keys(data.performance_correlations).length;
            html += `
                <div class="metric-item">
                    <div class="metric-label">Correlations</div>
                    <div class="metric-value">${{correlationCount}}</div>
                </div>
            `;
        }}

        html += '</div>';

        overviewElement.innerHTML = html;
    }}

    updateComponentHealth(componentHealth) {{
        const healthElement = document.getElementById('component-health');

        let html = '<div class="component-grid">';

        for (const [componentName, health] of Object.entries(componentHealth)) {{
            const status = health.health_status || 'unknown';
            const responseTime = health.response_time_ms || 0;

            html += `
                <div class="component-item ${{status}}">
                    <div class="component-name">${{componentName.replace('_', ' ').toUpperCase()}}</div>
                    <div class="component-status">${{status.toUpperCase()}}</div>
                    <div class="component-metric">${{responseTime.toFixed(0)}}ms</div>
                </div>
            `;
        }}

        html += '</div>';

        healthElement.innerHTML = html;
    }}

    updateActiveAlerts(alerts) {{
        const alertsElement = document.getElementById('active-alerts');

        const recentAlerts = alerts.recent_alerts || [];

        if (recentAlerts.length === 0) {{
            alertsElement.innerHTML = '<div class="no-alerts">No active alerts</div>';
            return;
        }}

        let html = '<div class="alerts-list">';

        recentAlerts.slice(0, 5).forEach(alert => {{
            html += `
                <div class="alert-item ${{alert.severity}}">
                    <div class="alert-header">
                        <span class="alert-severity">${{alert.severity.toUpperCase()}}</span>
                        <span class="alert-component">${{alert.component}}</span>
                    </div>
                    <div class="alert-message">${{alert.message}}</div>
                    <div class="alert-time">${{new Date(alert.timestamp).toLocaleString()}}</div>
                </div>
            `;
        }});

        html += '</div>';

        alertsElement.innerHTML = html;
    }}

    updatePerformanceMetrics(visualizationData) {{
        const metricsElement = document.getElementById('performance-metrics');

        const componentStatus = visualizationData.component_status_chart || {{}};

        let html = '<div class="performance-grid">';

        for (const [component, data] of Object.entries(componentStatus)) {{
            const responseTime = data.response_time || 0;
            const errorRate = (data.error_rate || 0) * 100;

            html += `
                <div class="performance-item">
                    <div class="performance-label">${{component.replace('_', ' ')}}</div>
                    <div class="performance-metrics">
                        <div class="performance-metric">
                            <span class="metric-name">Response Time</span>
                            <span class="metric-value">${{responseTime.toFixed(0)}}ms</span>
                        </div>
                        <div class="performance-metric">
                            <span class="metric-name">Error Rate</span>
                            <span class="metric-value">${{errorRate.toFixed(1)}}%</span>
                        </div>
                    </div>
                </div>
            `;
        }}

        html += '</div>';

        metricsElement.innerHTML = html;
    }}

    updateHealthTrends(trendAnalysis) {{
        const trendsElement = document.getElementById('health-trends');

        if (Object.keys(trendAnalysis).length === 0) {{
            trendsElement.innerHTML = '<div class="no-trends">No trend data available</div>';
            return;
        }}

        let html = '<div class="trends-grid">';

        for (const [component, trend] of Object.entries(trendAnalysis)) {{
            const trendStatus = trend.trend || 'unknown';
            const confidence = ((trend.confidence || 0) * 100).toFixed(0);

            html += `
                <div class="trend-item">
                    <div class="trend-header">
                        <span class="trend-component">${{component.replace('_', ' ')}}</span>
                        <span class="trend-status ${{trendStatus}}">${{trendStatus.toUpperCase()}}</span>
                    </div>
                    <div class="trend-confidence">Confidence: ${{confidence}}%</div>
                    <div class="trend-indicators">
                        ${{(trend.key_indicators || []).slice(0, 2).map(indicator =>
                            `<div class="trend-indicator">${{indicator}}</div>`
                        ).join('')}}
                    </div>
                </div>
            `;
        }}

        html += '</div>';

        trendsElement.innerHTML = html;
    }}

    setupEventListeners() {{
        // Add event listeners for interactive elements
        document.addEventListener('click', (event) => {{
            if (event.target.classList.contains('alert-item')) {{
                // Handle alert clicks
                this.showAlertDetails(event.target);
            }}
        }});
    }}

    showAlertDetails(alertElement) {{
        // Implementation for showing alert details
        console.log('Alert clicked:', alertElement);
    }}

    showError(message) {{
        console.error('Dashboard error:', message);

        // Show error in UI
        const errorHtml = `<div class="error-message">${{message}}</div>`;
        document.querySelectorAll('.loading').forEach(element => {{
            element.innerHTML = errorHtml;
        }});
    }}
}}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {{
    new HealthDashboard();
}});
        """.strip()

    def _generate_dashboard_css(self) -> str:
        """Generate dashboard CSS."""
        return """
/* Health Dashboard Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f5f5f5;
    color: #333;
    line-height: 1.6;
}

.dashboard-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.dashboard-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
}

.status-indicator {
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
}

.status-indicator.healthy {
    background-color: #10b981;
}

.status-indicator.warning {
    background-color: #f59e0b;
}

.status-indicator.critical {
    background-color: #ef4444;
}

.status-indicator.unknown {
    background-color: #6b7280;
}

.dashboard-main {
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
}

.dashboard-card {
    background: white;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border: 1px solid #e5e7eb;
}

.dashboard-card.full-width {
    grid-column: 1 / -1;
}

.dashboard-card h2 {
    font-size: 1.25rem;
    margin-bottom: 1rem;
    color: #374151;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 0.5rem;
}

.loading {
    text-align: center;
    color: #6b7280;
    padding: 2rem;
}

.error-message {
    color: #ef4444;
    text-align: center;
    padding: 1rem;
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 4px;
}

/* System Overview */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
}

.metric-item {
    text-align: center;
    padding: 1rem;
    background-color: #f9fafb;
    border-radius: 6px;
}

.metric-label {
    font-size: 0.875rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #374151;
}

/* Component Health */
.component-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.component-item {
    padding: 1rem;
    border-radius: 6px;
    text-align: center;
    border-left: 4px solid #e5e7eb;
}

.component-item.healthy {
    background-color: #ecfdf5;
    border-left-color: #10b981;
}

.component-item.degraded,
.component-item.warning {
    background-color: #fffbeb;
    border-left-color: #f59e0b;
}

.component-item.unhealthy,
.component-item.critical {
    background-color: #fef2f2;
    border-left-color: #ef4444;
}

.component-item.unknown {
    background-color: #f9fafb;
    border-left-color: #6b7280;
}

.component-name {
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.component-status {
    font-size: 0.875rem;
    margin-bottom: 0.25rem;
}

.component-metric {
    font-size: 0.75rem;
    color: #6b7280;
}

/* Active Alerts */
.alerts-list {
    space-y: 0.75rem;
}

.alert-item {
    padding: 1rem;
    border-radius: 6px;
    border-left: 4px solid #e5e7eb;
    margin-bottom: 0.75rem;
    cursor: pointer;
    transition: background-color 0.2s;
}

.alert-item:hover {
    background-color: #f9fafb;
}

.alert-item.info {
    background-color: #eff6ff;
    border-left-color: #3b82f6;
}

.alert-item.warning {
    background-color: #fffbeb;
    border-left-color: #f59e0b;
}

.alert-item.critical,
.alert-item.emergency {
    background-color: #fef2f2;
    border-left-color: #ef4444;
}

.alert-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.alert-severity {
    font-weight: 600;
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 12px;
    background-color: rgba(0,0,0,0.1);
}

.alert-component {
    font-size: 0.875rem;
    color: #6b7280;
}

.alert-message {
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
}

.alert-time {
    font-size: 0.75rem;
    color: #6b7280;
}

.no-alerts {
    text-align: center;
    color: #10b981;
    padding: 2rem;
    font-weight: 500;
}

/* Performance Metrics */
.performance-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
}

.performance-item {
    padding: 1rem;
    background-color: #f9fafb;
    border-radius: 6px;
}

.performance-label {
    font-weight: 600;
    margin-bottom: 0.75rem;
    text-transform: capitalize;
}

.performance-metrics {
    display: flex;
    justify-content: space-between;
}

.performance-metric {
    text-align: center;
}

.metric-name {
    display: block;
    font-size: 0.75rem;
    color: #6b7280;
    margin-bottom: 0.25rem;
}

.metric-value {
    display: block;
    font-weight: 600;
    font-size: 1.125rem;
}

/* Health Trends */
.trends-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
}

.trend-item {
    padding: 1rem;
    background-color: #f9fafb;
    border-radius: 6px;
}

.trend-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.trend-component {
    font-weight: 600;
    text-transform: capitalize;
}

.trend-status {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 12px;
    font-weight: 600;
}

.trend-status.improving {
    background-color: #dcfce7;
    color: #166534;
}

.trend-status.stable {
    background-color: #e0f2fe;
    color: #0c4a6e;
}

.trend-status.degrading {
    background-color: #fef3c7;
    color: #92400e;
}

.trend-status.unstable,
.trend-status.critical_decline {
    background-color: #fee2e2;
    color: #991b1b;
}

.trend-confidence {
    font-size: 0.875rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

.trend-indicators {
    font-size: 0.75rem;
    color: #6b7280;
}

.trend-indicator {
    margin-bottom: 0.25rem;
}

.no-trends {
    text-align: center;
    color: #6b7280;
    padding: 2rem;
}

/* Dashboard Footer */
.dashboard-footer {
    background-color: white;
    border-top: 1px solid #e5e7eb;
    padding: 1rem 2rem;
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
}

.dashboard-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.875rem;
    color: #6b7280;
}

/* Responsive Design */
@media (max-width: 768px) {
    .dashboard-header {
        flex-direction: column;
        gap: 1rem;
        text-align: center;
    }

    .dashboard-main {
        padding: 1rem;
        margin-bottom: 80px;
    }

    .dashboard-grid {
        grid-template-columns: 1fr;
    }

    .metrics-grid,
    .component-grid,
    .performance-grid,
    .trends-grid {
        grid-template-columns: 1fr;
    }

    .performance-metrics {
        flex-direction: column;
        gap: 0.5rem;
    }

    .alert-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.25rem;
    }
}
        """.strip()


# Global health dashboard instance
_health_dashboard: HealthDashboard | None = None


async def get_health_dashboard(**kwargs) -> HealthDashboard:
    """Get or create global health dashboard instance."""
    global _health_dashboard

    if _health_dashboard is None:
        _health_dashboard = HealthDashboard(**kwargs)
        await _health_dashboard.initialize()

    return _health_dashboard


async def shutdown_health_dashboard():
    """Shutdown global health dashboard."""
    global _health_dashboard

    if _health_dashboard:
        await _health_dashboard.stop_server()
        _health_dashboard = None
