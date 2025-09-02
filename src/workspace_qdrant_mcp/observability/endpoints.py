
from ...observability import get_logger
logger = get_logger(__name__)
"""
Health and metrics endpoints for workspace-qdrant-mcp.

Provides HTTP endpoints for health checks, metrics exposition, and system diagnostics
compatible with monitoring systems, load balancers, and observability platforms.

Endpoints:
    GET /health - Basic health check for load balancers
    GET /health/detailed - Comprehensive health diagnostics  
    GET /metrics - Prometheus-format metrics exposition
    GET /metrics/json - JSON-format metrics for custom dashboards
    GET /diagnostics - Full system diagnostics and troubleshooting info

Integration:
    - Kubernetes liveness/readiness probes
    - Load balancer health checks
    - Prometheus metrics scraping
    - Grafana dashboard integration
    - Custom monitoring solutions

Example:
    ```python
    from fastapi import FastAPI
    from workspace_qdrant_mcp.observability.endpoints import add_observability_routes
    
    app = FastAPI()
    add_observability_routes(app)
    ```
"""

import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from .logger import get_logger
from .metrics import metrics_instance
from .health import health_checker_instance, HealthStatus

logger = get_logger(__name__)


async def health_check_basic() -> Dict[str, Any]:
    """Basic health check endpoint for load balancers."""
    try:
        health_status = await health_checker_instance.get_health_status()
        
        # Simple response for load balancer compatibility
        if health_status["status"] == "healthy":
            return {
                "status": "healthy",
                "timestamp": health_status["timestamp"],
                "message": health_status["message"]
            }
        elif health_status["status"] == "degraded":
            # Consider degraded as healthy for load balancer
            return {
                "status": "healthy", 
                "degraded": True,
                "timestamp": health_status["timestamp"],
                "message": health_status["message"]
            }
        else:
            # Return unhealthy status
            return {
                "status": "unhealthy",
                "timestamp": health_status["timestamp"],
                "message": health_status["message"]
            }
            
    except Exception as e:
        logger.error("Health check endpoint failed", error=str(e), exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "message": f"Health check failed: {e}"
        }


async def health_check_detailed() -> Dict[str, Any]:
    """Detailed health check with component information."""
    try:
        health_status = await health_checker_instance.get_health_status()
        
        # Add endpoint-specific information
        health_status["endpoint"] = "detailed"
        health_status["checks_performed"] = len(health_status.get("components", {}))
        
        return health_status
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e), exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "message": f"Detailed health check failed: {e}",
            "error": str(e),
            "components": {}
        }


async def metrics_prometheus() -> str:
    """Prometheus-format metrics endpoint."""
    try:
        # Update system metrics before export
        metrics_instance.update_system_metrics()
        
        # Export in Prometheus format
        prometheus_data = metrics_instance.export_prometheus_format()
        
        logger.debug("Metrics exported", format="prometheus", 
                    size_bytes=len(prometheus_data))
        
        return prometheus_data
        
    except Exception as e:
        logger.error("Prometheus metrics export failed", error=str(e), exc_info=True)
        # Return error metric in Prometheus format
        return f"""# Metrics export error
# TYPE metrics_export_errors_total counter
metrics_export_errors_total 1
# EOF
"""


async def metrics_json() -> Dict[str, Any]:
    """JSON-format metrics for custom dashboards."""
    try:
        # Update system metrics before export
        metrics_instance.update_system_metrics()
        
        # Get comprehensive metrics summary
        metrics_summary = metrics_instance.get_metrics_summary()
        
        # Add metadata
        metrics_summary["export_format"] = "json"
        metrics_summary["export_timestamp"] = time.time()
        
        logger.debug("Metrics exported", format="json", 
                    counters_count=len(metrics_summary.get("counters", {})),
                    gauges_count=len(metrics_summary.get("gauges", {})),
                    histograms_count=len(metrics_summary.get("histograms", {})))
        
        return metrics_summary
        
    except Exception as e:
        logger.error("JSON metrics export failed", error=str(e), exc_info=True)
        return {
            "error": str(e),
            "timestamp": time.time(),
            "export_format": "json",
            "counters": {},
            "gauges": {},
            "histograms": {}
        }


async def system_diagnostics() -> Dict[str, Any]:
    """Comprehensive system diagnostics endpoint."""
    try:
        # Get detailed diagnostics from health checker
        diagnostics = await health_checker_instance.get_detailed_diagnostics()
        
        # Add endpoint-specific metadata
        diagnostics["endpoint"] = "diagnostics"
        diagnostics["generated_at"] = time.time()
        
        logger.info("System diagnostics requested", 
                   health_status=diagnostics.get("health_status", {}).get("status"),
                   components_checked=len(diagnostics.get("health_status", {}).get("components", {})))
        
        return diagnostics
        
    except Exception as e:
        logger.error("System diagnostics failed", error=str(e), exc_info=True)
        return {
            "error": str(e),
            "timestamp": time.time(),
            "endpoint": "diagnostics",
            "health_status": {
                "status": "unhealthy",
                "message": f"Diagnostics failed: {e}",
                "components": {}
            }
        }


def add_observability_routes(app: FastAPI, prefix: str = "") -> None:
    """Add observability routes to a FastAPI application.
    
    Args:
        app: FastAPI application instance
        prefix: URL prefix for routes (optional)
    
    Example:
        ```python
        app = FastAPI()
        add_observability_routes(app, prefix="/api/v1")
        ```
    """
    
    @app.get(f"{prefix}/health")
    async def health_endpoint(request: Request) -> JSONResponse:
        """Basic health check endpoint for load balancers.
        
        Returns:
            200: Service is healthy or degraded (operational)
            503: Service is unhealthy (not operational)
        """
        start_time = time.perf_counter()
        
        try:
            health_data = await health_check_basic()
            
            # Record metrics
            metrics_instance.increment_counter("health_endpoint_requests_total", 
                                             status=health_data["status"])
            
            # Determine HTTP status code
            if health_data["status"] in ["healthy"]:
                status_code = 200
            else:
                status_code = 503
            
            response_time = time.perf_counter() - start_time
            metrics_instance.record_histogram("health_endpoint_duration_seconds", response_time)
            
            return JSONResponse(content=health_data, status_code=status_code)
            
        except Exception as e:
            logger.error("Health endpoint failed", error=str(e), exc_info=True)
            metrics_instance.increment_counter("health_endpoint_requests_total", status="error")
            
            return JSONResponse(
                content={
                    "status": "unhealthy",
                    "timestamp": time.time(),
                    "message": "Internal health check error"
                },
                status_code=503
            )
    
    @app.get(f"{prefix}/health/detailed")
    async def health_detailed_endpoint(request: Request) -> JSONResponse:
        """Detailed health check with component diagnostics.
        
        Returns comprehensive health information including per-component
        status, diagnostic details, and troubleshooting information.
        """
        start_time = time.perf_counter()
        
        try:
            health_data = await health_check_detailed()
            
            # Record metrics
            metrics_instance.increment_counter("health_detailed_requests_total", 
                                             status=health_data["status"])
            
            response_time = time.perf_counter() - start_time
            metrics_instance.record_histogram("health_detailed_duration_seconds", response_time)
            
            return JSONResponse(content=health_data, status_code=200)
            
        except Exception as e:
            logger.error("Detailed health endpoint failed", error=str(e), exc_info=True)
            metrics_instance.increment_counter("health_detailed_requests_total", status="error")
            
            return JSONResponse(
                content={
                    "status": "unhealthy", 
                    "timestamp": time.time(),
                    "message": "Internal detailed health check error",
                    "error": str(e)
                },
                status_code=500
            )
    
    @app.get(f"{prefix}/metrics")
    async def metrics_endpoint(request: Request) -> PlainTextResponse:
        """Prometheus-format metrics endpoint for scraping.
        
        Returns metrics in Prometheus exposition format compatible with
        Prometheus server, Grafana, and other monitoring systems.
        """
        start_time = time.perf_counter()
        
        try:
            prometheus_data = await metrics_prometheus()
            
            # Record metrics (but avoid recursive metrics explosion)
            response_time = time.perf_counter() - start_time
            metrics_instance.record_histogram("metrics_endpoint_duration_seconds", response_time)
            metrics_instance.increment_counter("metrics_endpoint_requests_total", format="prometheus")
            
            return PlainTextResponse(
                content=prometheus_data,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
            
        except Exception as e:
            logger.error("Metrics endpoint failed", error=str(e), exc_info=True)
            metrics_instance.increment_counter("metrics_endpoint_requests_total", 
                                             format="prometheus", status="error")
            
            error_response = f"""# Metrics endpoint error
# TYPE metrics_endpoint_errors_total counter
metrics_endpoint_errors_total 1
"""
            return PlainTextResponse(content=error_response, status_code=500)
    
    @app.get(f"{prefix}/metrics/json")
    async def metrics_json_endpoint(request: Request) -> JSONResponse:
        """JSON-format metrics endpoint for custom dashboards.
        
        Returns metrics in structured JSON format suitable for custom
        monitoring dashboards and alerting systems.
        """
        start_time = time.perf_counter()
        
        try:
            json_data = await metrics_json()
            
            # Record metrics
            response_time = time.perf_counter() - start_time
            metrics_instance.record_histogram("metrics_endpoint_duration_seconds", response_time)
            metrics_instance.increment_counter("metrics_endpoint_requests_total", format="json")
            
            return JSONResponse(content=json_data)
            
        except Exception as e:
            logger.error("JSON metrics endpoint failed", error=str(e), exc_info=True)
            metrics_instance.increment_counter("metrics_endpoint_requests_total", 
                                             format="json", status="error")
            
            return JSONResponse(
                content={
                    "error": str(e),
                    "timestamp": time.time(),
                    "export_format": "json"
                },
                status_code=500
            )
    
    @app.get(f"{prefix}/diagnostics")
    async def diagnostics_endpoint(request: Request) -> JSONResponse:
        """System diagnostics endpoint for troubleshooting.
        
        Returns comprehensive system information including health status,
        metrics, configuration, and troubleshooting data.
        """
        start_time = time.perf_counter()
        
        try:
            diagnostics_data = await system_diagnostics()
            
            # Record metrics
            response_time = time.perf_counter() - start_time
            metrics_instance.record_histogram("diagnostics_endpoint_duration_seconds", response_time)
            metrics_instance.increment_counter("diagnostics_endpoint_requests_total")
            
            return JSONResponse(content=diagnostics_data)
            
        except Exception as e:
            logger.error("Diagnostics endpoint failed", error=str(e), exc_info=True)
            metrics_instance.increment_counter("diagnostics_endpoint_requests_total", status="error")
            
            return JSONResponse(
                content={
                    "error": str(e),
                    "timestamp": time.time(),
                    "endpoint": "diagnostics"
                },
                status_code=500
            )
    
    # Add startup logging
    logger.info("Observability endpoints registered",
               routes=[f"{prefix}/health", f"{prefix}/health/detailed", 
                      f"{prefix}/metrics", f"{prefix}/metrics/json",
                      f"{prefix}/diagnostics"])


def setup_observability_middleware(app: FastAPI) -> None:
    """Set up middleware for observability and monitoring.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        """Middleware for request/response observability."""
        start_time = time.perf_counter()
        method = request.method
        path = request.url.path
        
        # Generate request ID for tracing
        request_id = f"{int(time.time() * 1000000)}"
        
        logger.debug("Request started",
                    method=method,
                    path=path,
                    request_id=request_id,
                    client_ip=request.client.host if request.client else "unknown")
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.perf_counter() - start_time
            
            # Record metrics
            metrics_instance.increment_counter("http_requests_total",
                                             method=method,
                                             status_code=str(response.status_code))
            metrics_instance.record_histogram("http_request_duration_seconds",
                                            response_time,
                                            method=method,
                                            path=path)
            
            # Log response
            logger.info("Request completed",
                       method=method,
                       path=path,
                       status_code=response.status_code,
                       response_time_seconds=response_time,
                       request_id=request_id)
            
            return response
            
        except Exception as e:
            response_time = time.perf_counter() - start_time
            
            # Record error metrics
            metrics_instance.increment_counter("http_requests_total",
                                             method=method,
                                             status_code="500")
            metrics_instance.record_histogram("http_request_duration_seconds",
                                            response_time,
                                            method=method,
                                            path=path)
            
            logger.error("Request failed",
                        method=method,
                        path=path,
                        response_time_seconds=response_time,
                        request_id=request_id,
                        error=str(e),
                        exc_info=True)
            
            raise
    
    logger.info("Observability middleware configured", 
               features=["request_timing", "error_tracking", "metrics_collection"])