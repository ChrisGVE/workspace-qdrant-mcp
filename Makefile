# Makefile - Linux/WSL operations for the workspace-qdrant-mcp fork.
#
# Container-first workflow: every build happens INSIDE Docker. You do NOT need
# a local Rust/cargo toolchain, a local ONNX Runtime, or a host npm build —
# `docker/Dockerfile.memexd` builds the daemon (Rust + static ONNX) and the
# root `Dockerfile` compiles the TypeScript MCP server and the Rust node addon.
#
# Run this from inside the WSL distro (native ext4), e.g.:
#   wsl -d ubuntu-24.04
#   cd /home/alkmimm/respositorios/workspace-qdrant-mcp
#   make help
#
# The Windows/PowerShell flow lives in `Makefile.win` (use `make -f Makefile.win`).
# This file is the Linux/WSL counterpart and intentionally has NO cargo/npm host
# targets — wiring the Windows side to build via the container is a future step.
#
# Useful variables (override on the command line, e.g. `make redeploy LOG_TAIL=100`):
#   REPO              repo root (default: this Makefile's directory)
#   COMPOSE_ENV_FILE  compose env file (default: docker/.env)
#   COMPOSE_FILE      compose file    (default: docker-compose.yml)
#   MCP_HTTP_PORT     host port for the MCP HTTP/admin endpoint (default: 6335)
#   QDRANT_HTTP_PORT  host port for Qdrant REST (default: 6333)
#   LOG_TAIL          lines for stack-logs (default: 50)

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

REPO ?= $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
COMPOSE_FILE ?= $(REPO)/docker-compose.yml
COMPOSE_ENV_FILE ?= $(REPO)/docker/.env
MCP_HTTP_PORT ?= 6335
QDRANT_HTTP_PORT ?= 6333
MEMEXD_GRPC_PORT ?= 50051
LOG_TAIL ?= 50

# Single source of truth for every compose invocation.
COMPOSE := docker compose --env-file "$(COMPOSE_ENV_FILE)" -f "$(COMPOSE_FILE)"

MCP_HEALTH_URL ?= http://localhost:$(MCP_HTTP_PORT)/admin/api/health
MCP_INIT_URL ?= http://localhost:$(MCP_HTTP_PORT)/admin/init
MEMEXD_DB_VOLUME ?= workspace-qdrant-mcp_memexd_db

.PHONY: help check-env first-time redeploy \
	stack-up stack-down stack-restart stack-status stack-logs \
	build-images mcp-rebuild memexd-recreate \
	health-quick scan register-all watch reindex reindex-status hooks-install clean

help:
	@echo "============================================================"
	@echo "workspace-qdrant-mcp — Linux/WSL (container-first) targets"
	@echo "============================================================"
	@echo "Stack lifecycle (day-to-day):"
	@echo "  first-time       SETUP FROM SCRATCH: create db volume + build + up + hooks + status"
	@echo "  redeploy         AFTER CODE CHANGES / git pull: rebuild mcp+memexd images + recreate + status"
	@echo "  stack-up         start the docker stack (no rebuild)"
	@echo "  stack-down       stop the docker stack"
	@echo "  stack-restart    down + up"
	@echo "  stack-status     compose ps + ping admin/qdrant/daemon"
	@echo "  stack-logs       tail mcp + memexd logs (LOG_TAIL=$(LOG_TAIL))"
	@echo "------------------------------------------------------------"
	@echo "Build / recreate (all builds run INSIDE Docker — no local cargo/npm):"
	@echo "  build-images     docker compose build mcp memexd"
	@echo "  mcp-rebuild      rebuild + recreate ONLY the mcp container"
	@echo "  memexd-recreate  recreate memexd (picks up env changes from docker/.env)"
	@echo "------------------------------------------------------------"
	@echo "Observability / indexing:"
	@echo "  health-quick     curl the MCP /admin/api/health endpoint"
	@echo "  scan             list git repos discovered under WQM_DEV_ROOT (no register)"
	@echo "  register-all     register every discovered repo with the daemon (starts indexing)"
	@echo "  watch            poll indexing progress until all projects drain (or timeout)"
	@echo "  reindex          trigger a full reindex of the watched projects (admin API)"
	@echo "  reindex-status   per-project indexing progress (one-shot)"
	@echo "  hooks-install    install POSIX git hooks into .wqm-fork/git-hooks"
	@echo "  clean            remove the MCP dist build artifacts"
	@echo "------------------------------------------------------------"
	@echo "Watch root (daemon-observed projects) is set via WQM_DEV_ROOT in"
	@echo "$(COMPOSE_ENV_FILE). For WSL use a native ext4 path, e.g."
	@echo "  WQM_DEV_ROOT=/home/<user>/respositorios"
	@echo "============================================================"

check-env:
	@if [[ ! -f "$(COMPOSE_ENV_FILE)" ]]; then \
		echo "ERROR: $(COMPOSE_ENV_FILE) not found. Copy docker/.env.example to docker/.env first." >&2; \
		exit 1; \
	fi

# ── Stack lifecycle ──────────────────────────────────────────────────────────

first-time: check-env
	@echo "=== First-time setup (container-first) ==="
	@echo "Step 1/4: ensure the external SQLite volume exists"
	@docker volume create "$(MEMEXD_DB_VOLUME)" >/dev/null
	@echo "Step 2/4: build + start the whole stack"
	@cd "$(REPO)" && $(COMPOSE) up -d --build
	@echo "Step 3/4: install POSIX git hooks"
	@$(MAKE) -f "$(lastword $(MAKEFILE_LIST))" hooks-install
	@echo "Step 4/4: status"
	@sleep 8
	@$(MAKE) -f "$(lastword $(MAKEFILE_LIST))" stack-status
	@echo ""
	@echo "=== Done. Open http://localhost:$(MCP_HTTP_PORT)/admin/ ==="

redeploy: check-env
	@echo "=== Redeploy after code changes (build runs inside Docker) ==="
	@echo "Step 1/4: rebuild mcp + memexd images"
	@docker volume create "$(MEMEXD_DB_VOLUME)" >/dev/null
	@cd "$(REPO)" && $(COMPOSE) build mcp memexd
	@echo "Step 2/4: recreate mcp + memexd (env may have changed)"
	@cd "$(REPO)" && $(COMPOSE) up -d --force-recreate mcp memexd
	@echo "Step 3/4: reinstall git hooks (idempotent — lives in the repo, not the image)"
	@$(MAKE) -f "$(lastword $(MAKEFILE_LIST))" hooks-install
	@echo "Step 4/4: status"
	@sleep 6
	@$(MAKE) -f "$(lastword $(MAKEFILE_LIST))" stack-status
	@echo ""
	@echo "=== Redeploy complete ==="

stack-up: check-env
	@cd "$(REPO)" && $(COMPOSE) up -d
	@echo "Stack started. Run 'make stack-status' to verify."

stack-down: check-env
	@cd "$(REPO)" && $(COMPOSE) down

stack-restart: stack-down stack-up

stack-status: check-env
	@echo "=== docker compose ps ==="
	@cd "$(REPO)" && $(COMPOSE) ps
	@echo ""
	@echo "=== endpoints ==="
	@if curl -fsS -o /dev/null -m 3 "$(MCP_INIT_URL)"; then echo "/admin/init     [ok]"; else echo "/admin/init     [fail]"; fi
	@if curl -fsS -o /dev/null -m 3 "http://localhost:$(QDRANT_HTTP_PORT)/collections"; then echo "qdrant          [ok]"; else echo "qdrant          [fail]"; fi
	@if (exec 3<>/dev/tcp/localhost/$(MEMEXD_GRPC_PORT)) 2>/dev/null; then echo "memexd gRPC     [ok] localhost:$(MEMEXD_GRPC_PORT)"; else echo "memexd gRPC     [fail]"; fi

stack-logs: check-env
	@cd "$(REPO)" && $(COMPOSE) logs --tail $(LOG_TAIL) mcp memexd

# ── Build / recreate (everything builds inside Docker) ───────────────────────

build-images: check-env
	@cd "$(REPO)" && $(COMPOSE) build mcp memexd

mcp-rebuild: check-env
	@echo "Rebuilding MCP image (TypeScript compiled inside the container)..."
	@cd "$(REPO)" && $(COMPOSE) build mcp
	@cd "$(REPO)" && $(COMPOSE) up -d mcp

memexd-recreate: check-env
	@echo "Recreating memexd container (picks up env changes from docker/.env)..."
	@cd "$(REPO)" && $(COMPOSE) up -d --force-recreate memexd

# ── Observability / indexing ─────────────────────────────────────────────────
#
# Host-side admin calls rely on MCP_HTTP_TRUST_LOCALHOST=1 (the compose default),
# which skips the Bearer check for loopback peers — so no token is needed here.
# JSON is pretty-printed with python3 (jq is not assumed to be installed).

PP := python3 -m json.tool

health-quick: check-env
	@curl -fsS -m 5 "$(MCP_HEALTH_URL)" | $(PP) 2>/dev/null || curl -fsS -m 5 "$(MCP_HEALTH_URL)"

# Discover git repos under the daemon's devRoot (WQM_DEV_ROOT) without registering.
scan: check-env
	@python3 "$(REPO)/scripts/wqm_admin.py" scan "http://localhost:$(MCP_HTTP_PORT)"

# Register EVERY git repo discovered under devRoot with the daemon (idempotent —
# already-registered projects are refreshed). Indexing starts in the background.
register-all: check-env
	@python3 "$(REPO)/scripts/wqm_admin.py" register-all "http://localhost:$(MCP_HTTP_PORT)"

# Poll indexing progress until every project drains (0 pending, 100%) or timeout.
# Override cadence/limit: make watch WATCH_INTERVAL=15 WATCH_MAX=1800
WATCH_INTERVAL ?= 10
WATCH_MAX ?= 900
watch: check-env
	@python3 "$(REPO)/scripts/wqm_admin.py" watch "http://localhost:$(MCP_HTTP_PORT)" $(WATCH_INTERVAL) $(WATCH_MAX)

# Force-rebuild the computed indexes (FTS5, tags, sparse vectors, components,
# keywords) for EVERY watched project. Enumerates tenants from the admin
# snapshot, then POSTs /admin/api/projects/reindex per tenant. Pass TENANT=<id>
# to reindex a single project instead of all.
TENANT ?=
reindex: check-env
	@base="http://localhost:$(MCP_HTTP_PORT)"; \
	if [[ -n "$(TENANT)" ]]; then tenants="$(TENANT)"; else \
		tenants=$$(python3 "$(REPO)/scripts/wqm_admin.py" tenants "$$base"); \
	fi; \
	if [[ -z "$$tenants" ]]; then echo "No watched projects found — nothing to reindex."; exit 0; fi; \
	for t in $$tenants; do \
		echo "==> reindex tenant $$t"; \
		curl -fsS -m 60 -X POST -H "Content-Type: application/json" \
			-d "{\"tenantId\":\"$$t\"}" "$$base/admin/api/projects/reindex" | $(PP) 2>/dev/null || true; \
	done; \
	echo "Reindex requested for all watched projects. Watch 'make reindex-status' / 'make stack-logs'."

# Per-project indexing progress (pending / done / total / percent) from snapshot.
reindex-status: check-env
	@python3 "$(REPO)/scripts/wqm_admin.py" status "http://localhost:$(MCP_HTTP_PORT)"

hooks-install:
	@echo "Installing POSIX git hooks (sh + curl -> MCP HTTP)..."
	@sh "$(REPO)/scripts/git-hooks/install.sh" --repo "$(REPO)" --hooks-dir "$(REPO)/.wqm-fork/git-hooks" --force

clean:
	@rm -rf "$(REPO)/src/typescript/mcp-server/dist"
	@echo "Removed src/typescript/mcp-server/dist"
