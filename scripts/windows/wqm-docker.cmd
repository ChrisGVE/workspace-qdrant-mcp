@echo off
REM ---------------------------------------------------------------------------
REM wqm-docker.cmd - run the bundled wqm CLI inside the dockerized daemon
REM container instead of a locally-installed wqm.
REM
REM Why: the daemon container (wqm-memexd) owns the authoritative SQLite
REM (memexd_db volume) and the local gRPC endpoint, so `docker exec wqm-memexd
REM wqm ...` returns correct data. A host-installed wqm reads the wrong DB for
REM queue/stats (see docs / memory: "wqm CLI reads wrong DB by default").
REM
REM Enable: set WQM_PATH (or WQM_EXECUTABLE) to this file's absolute path.
REM Resolve-WqmPath (PowerShell) and resolveWqmPath (TypeScript) both honor
REM that env var first, so the registry / observe / incremental-check scripts
REM transparently route through the container with no other changes.
REM Override the container name with WQM_DOCKER_CONTAINER (default wqm-memexd).
REM ---------------------------------------------------------------------------
setlocal
if not defined WQM_DOCKER_CONTAINER set "WQM_DOCKER_CONTAINER=wqm-memexd"
docker exec "%WQM_DOCKER_CONTAINER%" wqm %*
