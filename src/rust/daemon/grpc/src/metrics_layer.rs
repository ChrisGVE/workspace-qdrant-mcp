//! Prometheus metrics tower layer for gRPC services.
//!
//! Wraps every unary/streaming RPC with a timer that records request count
//! (labels: service, method, status) and duration (labels: service, method).
//! Status classification is HTTP-level: transport failures and HTTP != 200
//! are tagged `error`; HTTP 200 responses are tagged `ok` because tonic
//! puts business errors in trailers that this layer does not parse. This
//! keeps the layer free of trailer inspection while still surfacing
//! infrastructure breakage in Prometheus.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use http::{Request, Response};
use tower::{Layer, Service};
use workspace_qdrant_core::METRICS;

/// Tower layer that installs [`MetricsService`] middleware.
#[derive(Clone, Default)]
pub struct MetricsLayer;

impl<S> Layer<S> for MetricsLayer {
    type Service = MetricsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MetricsService { inner }
    }
}

/// Middleware that records gRPC call metrics on every request.
#[derive(Clone)]
pub struct MetricsService<S> {
    inner: S,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for MetricsService<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
    ReqBody: Send + 'static,
    ResBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let (service, method) = parse_grpc_path(req.uri().path());
        let start = Instant::now();
        // Tower services must be cloned before moving into async block so the
        // original can keep accepting requests.
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        Box::pin(async move {
            let result = inner.call(req).await;
            let duration = start.elapsed();
            let ok = result
                .as_ref()
                .map(|resp| resp.status().is_success())
                .unwrap_or(false);
            METRICS.record_grpc_call(&service, &method, ok, duration);
            result
        })
    }
}

/// Parse a gRPC path of the form `/<package>.<Service>/<Method>` into
/// `(service, method)`. Returns `("unknown", "unknown")` for unparsable paths
/// and strips the leading package component so label values stay short.
pub fn parse_grpc_path(path: &str) -> (String, String) {
    let mut parts = path.trim_start_matches('/').split('/');
    let full_service = parts.next().unwrap_or("");
    let method = parts.next().unwrap_or("").to_string();

    if full_service.is_empty() || method.is_empty() {
        return ("unknown".to_string(), "unknown".to_string());
    }

    let service = full_service
        .rsplit('.')
        .next()
        .unwrap_or(full_service)
        .to_string();
    (service, method)
}

#[cfg(test)]
mod tests {
    use super::parse_grpc_path;

    #[test]
    fn parses_fully_qualified_path() {
        assert_eq!(
            parse_grpc_path("/workspace_daemon.SystemService/HealthCheck"),
            ("SystemService".to_string(), "HealthCheck".to_string())
        );
    }

    #[test]
    fn parses_path_without_package() {
        assert_eq!(
            parse_grpc_path("/SystemService/HealthCheck"),
            ("SystemService".to_string(), "HealthCheck".to_string())
        );
    }

    #[test]
    fn returns_unknown_for_garbage_paths() {
        assert_eq!(
            parse_grpc_path("/"),
            ("unknown".to_string(), "unknown".to_string())
        );
        assert_eq!(
            parse_grpc_path(""),
            ("unknown".to_string(), "unknown".to_string())
        );
    }
}
