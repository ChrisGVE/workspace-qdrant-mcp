# Workspace Qdrant MCP – Security Audit

## Scope & Approach

- Examined the full repository at `/Users/chris/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp` with a focus on security posture across the Python MCP server, shared core modules, Rust daemon(s), CLI tooling, and exposed services.
- Reviewed existing policies (`SECURITY.md`) against implementation realities, assessed network exposure (gRPC, HTTP, CLI service management), and traced data flow for places where security controls are expected (LLM access control, authenticated writes).
- Static analysis only; no code execution, fuzzing, or dependency scanning was performed in this sandboxed context.

## Architecture Security Highlights

- Intended design channels all write operations through a Rust daemon enforcing policy (LLM access control, metadata normalization, queueing).
- Python layers include multiple gRPC client implementations, with access-control checks tied to the daemon path.
- Hybrid transport (MCP stdio, optional HTTP server, gRPC daemon) increases the attack surface; most transports default to plaintext/insecure configuration.
- CLI tooling assumes trusted local execution; service installation scripts interact with system service managers without additional safeguards.

## High-Risk Findings

- **Bypass of LLM Access Control via Direct Qdrant Fallback**  
  The `store` tool falls back to direct Qdrant writes whenever the daemon is unavailable (`src/python/workspace_qdrant_mcp/server.py:482-519`). This path omits the `validate_llm_collection_access` checks present in the daemon client (`src/python/common/core/daemon_client.py:342-353`), allowing writes to protected collections (`memory`, `__*`, etc.) and bypassing experimental metadata validation. Any transient daemon outage (or deliberate disabling) leads to unrestricted writes.

- **Unauthenticated & Unencrypted gRPC Control Plane**  
  The daemon client constructs an insecure channel by default (`src/python/common/grpc/daemon_client.py:111-127`) and the server never enables TLS or any form of authentication (`rust-engine/src/grpc/server.rs:81-137`). Anyone with network access to port 50051 can invoke daemon RPCs (collection creation, ingestion, watcher control). Since CLI/service scripts encourage daemon deployment as a user-level service, a local adversary gains high-impact manipulation primitives.

- **Authentication-Free HTTP Hook Server**  
  `http_server.py` exposes FastAPI endpoints for session start/stop and hook processing without authentication or CSRF protection (`src/python/workspace_qdrant_mcp/http_server.py:31-153`). If started with `--host 0.0.0.0`, an attacker can spoof Claude session events, trigger ingest actions, or manipulate daemon state in current implementation (the server forwards status notifications to the daemon). No origin or token validation is performed.

- **gRPC Protocol Mismatch Forces Fallback**  
  The Rust daemon compiled from `rust-engine` registers only `SystemService` (`rust-engine/src/grpc/services/mod.rs:3-13`), while Python expects `DocumentService`/`CollectionService`. This mismatch means every write immediately hits the insecure fallback path above, making the “daemon required” guard ineffective.

## Medium-Risk Findings

- **Plaintext Metadata & API Keys in Logs**  
  Many modules log metadata and payloads at info level (e.g., `DaemonClient.ingest_text` logs collection names and document IDs at `src/python/common/core/daemon_client.py:352-359`). Without log scrubbing, sensitive file paths or user content can leak. Log destinations differ across components; guidance for safe log handling is absent.

- **Service Installation Assumes Trusted Binary Paths**  
  The CLI’s service manager allows building and installing the daemon into user-writable locations (`src/python/wqm_cli/cli/commands/service.py:157-214`). The executable path is discovered via heuristics (`src/python/wqm_cli/cli/commands/service.py:61-99`); if an attacker plants a malicious `memexd` binary in a higher-priority location before installation, it will be adopted as the service binary.

- **Hyphenated Collection Names Expose Sensitive Metadata**  
  When the fallback path writes directly to Qdrant, arbitrary payload metadata (including `file_path`, `domain`) is stored verbatim (`src/python/workspace_qdrant_mcp/server.py:492-504`). No redaction or filtering occurs, so ingestion of sensitive repositories can expose internal file system structures, access tokens embedded in paths, etc.

- **No Rate Limiting or Abuse Detection**  
  The HTTP server and gRPC services have no explicit rate limiting; the connection manager on the Rust server enforces counts per client (`rust-engine/src/grpc/server.rs:30-52`), but without authentication attackers can trivially cycle sources to avoid penalties. The FastAPI server ties directly into daemon notifications without trust boundaries.

- **Multiple gRPC Clients with Divergent Behaviour**  
  Two active daemon clients (`src/python/common/core/daemon_client.py` vs. `src/python/common/grpc/daemon_client.py`) maintain separate retry, timeout, and error-handling logic. Divergent defaults (e.g., circuit breaker vs. none) make it difficult to reason about security guarantees (such as fail-closed vs. fail-open) across components.

## Low-Risk / Defense-in-Depth Issues

- **Transport Configuration Defaults to Loopback Only**  
  `default_configuration.yaml` sets loopback hosts for MCP and daemon, but documentation encourages manual overrides. Without clear warnings, operators might bind to `0.0.0.0` for convenience, exposing unauthenticated services.

- **Daemon Health Responses Contain Synthetic Data**  
  `SystemService` returns hard-coded metrics (`rust-engine/src/grpc/services/system_service.rs:42-118`), preventing reliable detection of misbehaviour. Operators may miss security incidents (e.g., queue flooding) because health responses always read “healthy.”

- **Service Discovery Stubbed Out**  
  Automatic discovery is disabled (`src/python/common/core/daemon_client.py:154-185`), but the scaffolding leaves open questions about how endpoints will be shared. A future discovery mechanism must authenticate peer advertisements to avoid rogue daemon injection.

## Positive Practices

- `SECURITY.md` provides a responsible disclosure process and encourages secure configuration (API keys via env vars, TLS for Qdrant).
- LLM access control (`src/python/common/core/llm_access_control.py`) is robust when requests pass through the daemon path, preventing accidental manipulation of protected collections.
- Configuration loader supports environment-based overrides, enabling secret injection via standard tooling.
- Observability pipeline disables noisy logging in MCP stdio mode to prevent accidental leakage over the protocol.

## Recommendations

1. **Enforce Daemon-Only Writes**  
   Remove or heavily restrict direct Qdrant fallback. If a fallback is required for availability, gate it behind explicit configuration and maintain LLM access control checks.
2. **Secure gRPC and HTTP Transports**  
   Require TLS (with mutual authentication for gRPC when cross-host), or at minimum provide token-based auth. Bind HTTP hooks to loopback by default and require an auth token/header for all endpoints.
3. **Align Daemon Implementations**  
   Deploy a single Rust daemon that exposes the expected protocol so Python never silently degrades. Add integration tests to enforce protocol compatibility.
4. **Sanitize Logs and Payloads**  
   Avoid logging raw metadata/content or provide masking utilities. Document secure log handling practices in deployment guides.
5. **Harden Service Installation**  
   Validate binaries before installation (checksum/ signature) and fail if the resolved path already exists with unexpected ownership or permissions.
6. **Document Exposure Risks**  
   Update operator docs to highlight the risks of exposing the HTTP server or gRPC daemon beyond localhost, and provide guidance for reverse-proxy hardening.
7. **Plan for Authentication in Service Discovery**  
   If automated discovery is revived, integrate cryptographic verification of endpoints to prevent rogue daemon control.

## Closing Notes

The project has a thoughtful security narrative, but current implementation gaps (notably the daemon fallback and lack of transport security) leave key protections ineffective. Addressing the high-risk items should be prioritized before production deployment. Once remediated, further security investment can focus on observability, rate limiting, and long-term hardening.
