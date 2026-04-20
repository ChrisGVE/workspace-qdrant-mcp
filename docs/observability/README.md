# Observability reference (issue-64)

This directory holds reference artifacts for the memexd telemetry surface
introduced in issue-64. They are examples — not part of the shipped
product — intended to be copied into an existing Grafana / Prometheus
stack.

## Files

- `memexd-telemetry-dashboard.json` — Grafana 10 dashboard covering the
  new metrics (watcher events, gRPC request rate and latency, queue
  throughput/latency, embedding / SQLite / Qdrant latencies). Import via
  `Grafana → Dashboards → Import → Upload JSON file`.
- `prometheus-scrape-example.yaml` — snippet to paste into a
  Prometheus `scrape_configs` section so the daemon's `/metrics`
  endpoint is harvested on a 15s interval.

## Turning on the metrics endpoint

Either set the config section:

```yaml
observability:
  telemetry:
    prometheus:
      enabled: true
      port: 9464
      bind: 0.0.0.0
```

or pass env vars for a quick test:

```bash
WQM_PROMETHEUS_ENABLED=true WQM_PROMETHEUS_PORT=9464 memexd --foreground
curl http://localhost:9464/metrics | head
```

The `--metrics-port <N>` CLI flag is preserved as a shortcut: when set,
it forces `enabled=true` and overrides `port`.

## Turning on OTLP traces

Set `observability.telemetry.otlp.enabled: true` and `otlp.endpoint: ...`
in the config, or use the standard OpenTelemetry env vars
(`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`,
`OTEL_TRACES_SAMPLER_ARG`). Metrics over OTLP are **not** emitted yet;
Prometheus remains the canonical metrics surface.
