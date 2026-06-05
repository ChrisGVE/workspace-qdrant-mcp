# Runbook: profiling de memória do daemon `memexd`

Como investigar crescimento/leak de memória no `memexd` (container-first, WSL2).
Escrito após o incidente de 2026-06-04 (memexd ia a 74GB e enchia a VM do WSL).

## Quando usar

`docker stats wqm-memexd` mostra a memória crescendo sem parar (dezenas de GB),
o WSL trava (`Wsl/Service/0x8007274c`) ou o host congela. Antes de profilar,
**classifique** a natureza:

```bash
# 1) É heap (anon) ou page cache? (decisivo)
docker exec wqm-memexd sh -lc 'grep -E "^anon |^file " /sys/fs/cgroup/memory.stat'
#   anon >> file  -> heap/leak (siga este runbook)
#   file >> anon  -> page cache de I/O (não é leak; ver scope/exclusões/.wslconfig)

# 2) O que está aberto? (leitura de arquivo grande vs estruturas internas)
docker exec wqm-memexd sh -lc 'pid=$(for p in /proc/[0-9]*; do [ "$(cat $p/comm 2>/dev/null)" = memexd ] && echo ${p##*/} && break; done); ls -l /proc/$pid/fd | grep /home'
```

## Conter primeiro (o host vem antes do diagnóstico)

```bash
docker update --restart=no wqm-memexd && docker stop wqm-memexd
# Se a VM do WSL já travou:  (no Windows)  wsl --shutdown   # recupera a memória
# Proteção durável já aplicada: ~/.wslconfig  memory=80GB + [experimental] autoMemoryReclaim=gradual
```

`search`/`retrieve` continuam via Qdrant com o memexd parado.

## Profiling com heaptrack (precisa de símbolos)

O binário de produção é **stripped** (`[profile.release] strip = true`), então
heaptrack só dá endereços. É preciso um build temporário com símbolos.

### 1. Build com símbolos — edite o **repo MAIN** (não a worktree!)

O `docker compose build` usa o contexto do repo **main**
(`/home/alkmimm/respositorios/workspace-qdrant-mcp`). Editar a Cargo.toml de uma
worktree NÃO tem efeito no build. Em `src/rust/Cargo.toml`:

```toml
[profile.release]
lto = false        # ~3x mais rápido + stacks menos inlined (mais claras)
codegen-units = 16
panic = "unwind"
strip = false      # mantém símbolos
debug = 1          # line tables (DWARF) — heaptrack resolve nomes de função
```

O `Dockerfile.memexd` NÃO faz strip separado (depende do `strip=true` do Cargo),
então `strip=false` basta. Rebuild:

```bash
docker compose --env-file docker/.env -f docker-compose.yml build memexd
# verificar DWARF:
docker run --rm --entrypoint sh workspace-qdrant-mcp-memexd:local -c \
  'apt-get install -y -qq binutils >/dev/null 2>&1; readelf -S /usr/local/bin/memexd | grep -i debug_info'
```

### 2. Imagem heaptrack (`docker/Dockerfile.memexd` é a base)

```dockerfile
FROM workspace-qdrant-mcp-memexd:local
USER root
RUN apt-get update -qq && apt-get install -y -qq heaptrack \
 && rm -rf /var/lib/apt/lists/* && mkdir -p /out && chmod 0777 /out
USER memexd
ENTRYPOINT ["heaptrack", "-o", "/out/memexd-heaptrack", "/usr/local/bin/memexd"]
CMD ["--foreground", "--grpc-port", "50051", "--metrics-port", "9091"]
```

### 3. Reproduzir o cenário (memexd parado, edição segura do DB)

```bash
# limpar a fila e deixar 1 projeto ativo (repro mínimo)
docker run --rm -v workspace-qdrant-mcp_memexd_db:/db alpine sh -c \
 "apk add -q sqlite; sqlite3 /db/memexd.db \"DELETE FROM unified_queue;\"; \
  sqlite3 /db/memexd.db \"UPDATE watch_folders SET enabled=(path='/home/alkmimm/respositorios/workspace-qdrant-mcp');\""
```

### 4. Rodar sob heaptrack (rede/volumes/env do compose)

```bash
ROOT=/home/alkmimm/respositorios/workspace-qdrant-mcp
docker run -d --name memexd-ht --network workspace-qdrant-mcp_workspace-network \
  --env-file docker/.env -e QDRANT_URL=http://qdrant:6334 \
  -v "$ROOT/.fastembed_cache:/home/memexd/.workspace-qdrant/models" \
  -v workspace-qdrant-mcp_memexd_db:/var/lib/memexd \
  -v "$ROOT/state/memexd/config.yaml:/etc/wqm/config.yaml" \
  -v "$ROOT:$ROOT" -v "$ROOT/state/memexd:/home/memexd/.workspace-qdrant" \
  -v /tmp/htout:/out  memexd-ht:local
# deixe processar; PARE GRACIOSO (SIGTERM) p/ heaptrack finalizar o trace:
docker stop -t 40 memexd-ht
```

### 5. Analisar

```bash
docker run --rm -v /tmp/htout:/out --entrypoint heaptrack_print memexd-ht:local \
  /out/memexd-heaptrack.gz 2>&1 | sed -n '/PEAK MEMORY CONSUMERS/,/temporary/p'
# --print-leaks p/ alocações nunca liberadas.
```

### 6. Limpar (SEMPRE)

```bash
# reverter src/rust/Cargo.toml para strip=true / lto=true / codegen-units=1 (sem debug)
git checkout -- src/rust/Cargo.toml
docker image rm memexd-ht:local; rm -rf /tmp/htout
# rebuild normal quando for religar o memexd corrigido
```

## ⚠️ Caveat crítico do heaptrack (lição de 2026-06-04)

**heaptrack instrumenta CADA alocação e desacelera o daemon 10-50×.** Isso
**estrangula o throughput de embedding (ONNX)** — exatamente o que dispara o leak
deste daemon. No incidente, sob heaptrack a memória ficou flat em 1.5GB e o leak
**não reproduziu**; sem heaptrack, ia a 8GB em 15s.

**Para leaks dirigidos por throughput (embedding/ONNX), prefira métodos leves:**

- **jemalloc heap profiling** (desacelera pouco): build com jemalloc + LD_PRELOAD
  `MALLOC_CONF=prof:true,prof_prefix:/out/jeprof`, depois `jeprof --show_bytes`.
- **Correlação RSS × operação** (sem profiler): rodar o memexd normal, com
  auto-stop, e amostrar `memory.stat anon` vs a contagem de logs
  `FastEmbedProvider chunk embedded`. Se a RSS sobe em lockstep com os embeds,
  o leak é o caminho de embedding (suspeito nº1: **arena de memória do ONNX
  Runtime cresce por inferência e não devolve** — fix no `ort` SessionBuilder:
  `with_memory_pattern(false)` / desabilitar arena de CPU, ou reciclar o
  embedder periodicamente).

## Auto-stop seguro (não use `bc` com "GiB" — parsing quebra)

Leia o cgroup em **bytes** e compare numericamente:

```bash
cur=$(docker exec wqm-memexd sh -lc 'cat /sys/fs/cgroup/memory.current')
[ "${cur:-0}" -gt $((10*1024*1024*1024)) ] && docker kill wqm-memexd
```
