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

## Profiling com jemalloc (MÉTODO PREFERIDO — localizou o leak em 2026-06-04)

jemalloc com `MALLOC_CONF=prof` amostra alocações com **overhead baixo** (não
estrangula o throughput de embedding como o heaptrack — ver caveat abaixo), e o
`jeprof` resolve os nomes contra o binário com símbolos. Foi o que pinou o leak
do `split_chunk_with_overlap` (loop não-terminante no chunker) num único run.

**Pré-requisito:** binário com DWARF+`.symtab` — faça o build com símbolos
(seção "Build com símbolos" abaixo, idêntica à do heaptrack). Sem símbolos o
jeprof só mostra endereços.

### 1. Imagem jemalloc (sobre a base com símbolos)

```dockerfile
# tmp/Dockerfile.jemalloc — mantém o entrypoint da base; o profiling entra via env.
FROM workspace-qdrant-mcp-memexd:local
USER root
RUN apt-get update -qq \
 && apt-get install -y -qq libjemalloc2 libjemalloc-dev binutils perl >/dev/null 2>&1 \
 && rm -rf /var/lib/apt/lists/* && mkdir -p /out && chmod 0777 /out
USER memexd
```
```bash
docker build -q -f tmp/Dockerfile.jemalloc -t memexd-jemalloc:local .
```

### 2. Override do compose (reusa rede/volumes/env-file da service base)

```yaml
# tmp/docker-compose.jemalloc.yml
services:
  memexd:
    image: memexd-jemalloc:local
    user: "root"          # /out (bind /tmp/jeout) é root:root; sem isso o uid memexd não escreve o dump
    environment:
      LD_PRELOAD: /usr/lib/x86_64-linux-gnu/libjemalloc.so.2
      MALLOC_CONF: "prof:true,prof_active:true,lg_prof_sample:19,lg_prof_interval:30,prof_prefix:/out/jeprof,prof_final:true"
    volumes:
      - /tmp/jeout:/out
    restart: "no"
```

`MALLOC_CONF` decodificado:
- `lg_prof_sample:19` → amostra 1 a cada 2^19 = **512KB** alocados (overhead baixo).
- `lg_prof_interval:30` → dump a cada 2^30 = **1GB** alocado (captura o crescimento ao vivo).
- `prof_final:true` → dump final no exit **limpo** (atexit → exige `docker stop` gracioso, não `kill`).

### 3. Rodar, reproduzir, parar gracioso

```bash
mkdir -p /tmp/jeout && chmod 0777 /tmp/jeout
# (limpe a fila / deixe 1 projeto ativo — ver "Reproduzir o cenário" na seção heaptrack)
docker compose --env-file docker/.env -f docker-compose.yml -f tmp/docker-compose.jemalloc.yml \
  up -d --force-recreate memexd
# monitore anon até o burst (ver "Auto-stop seguro"), depois PARE GRACIOSO p/ o dump final:
docker stop -t 50 wqm-memexd
ls -la /tmp/jeout/*.heap
```

Os dumps `jeprof.<pid>.<n>.iN.heap` são os de intervalo (1GB); `.f.heap` são finais
(exit limpo). Processos filhos de vida curta também geram `.f` — o do daemon é o
PID principal (o de maior série `iN`).

### 4. Analisar com jeprof

```bash
JE="docker run --rm -v /tmp/jeout:/out --user root --entrypoint sh memexd-jemalloc:local -c"

# top allocators do heap VIVO. ATENÇÃO: o default do jeprof é alloc-space CUMULATIVO
# (soma o que já foi liberado; pode passar o RSS real). Para o leak use --inuse_space.
# Se inuse == alloc, nada foi liberado -> leak genuíno (foi o caso).
$JE 'jeprof --inuse_space --show_bytes --text /usr/local/bin/memexd $(ls -t /out/*.heap | head -1) | head -20'

# Quem CHAMA o allocator suspeito (call path completo até a função do projeto):
$JE 'jeprof --inuse_space --show_bytes --text --focus=finish_grow /usr/local/bin/memexd /out/jeprof.7.23.i23.heap | head -30'
```

### 5. Limpar

```bash
docker image rm memexd-jemalloc:local; rm -rf /tmp/jeout
rm -f tmp/Dockerfile.jemalloc tmp/docker-compose.jemalloc.yml
# + reverter src/rust/Cargo.toml (símbolos) — ver seção heaptrack
```

> **Gotcha de background (WSL):** `wsl.exe ... &`/`nohup` MORRE quando a sessão
> `wsl.exe` do launcher fecha (cada chamada é uma sessão separada). Use o
> background rastreado da ferramenta (mantém o `wsl.exe` vivo até o script
> terminar), não `nohup` dentro de um `wsl.exe -lc`.

## Profiling com heaptrack (fallback — só p/ leaks NÃO dirigidos por throughput)

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

**Para leaks dirigidos por throughput, prefira métodos leves:**

- **jemalloc heap profiling** (desacelera pouco) — ver seção dedicada acima. Foi
  o método que pinou o leak.
- **Correlação RSS × operação** (sem profiler): rodar o memexd normal, com
  auto-stop, e amostrar `memory.stat anon` vs a contagem de logs
  `FastEmbedProvider chunk embedded`. Diagnóstico-chave em 2026-06-04: o burst de
  +8.8GB aconteceu com os embeds **congelados** (contador parado) → o leak NÃO
  era embedding/ONNX, e sim uma fase **pré-embedding** (o chunking). Não presuma
  ONNX só porque o leak acompanha o throughput de indexação.

## Resolução do incidente 2026-06-04

O leak **não era ONNX** (BFCArena ficou em ~2% do heap). O `jeprof --inuse_space`
mostrou 100% do heap vivo em:

```
48.8%  workspace_qdrant_core::tree_sitter::types::SemanticChunk::new   (Strings dos fragmentos)
36.6%  alloc::raw_vec::finish_grow                                     (o Vec que as segura)
```

Call path: `process_file_sync_inner → SemanticChunker::chunk_source →
handle_oversized_chunks → split_chunk_with_overlap`.

**Bug:** `split_chunk_with_overlap` (`tree_sitter/chunker/splitting.rs`) tinha um
**loop não-terminante**. `find_line_boundary` faz `rfind('\n')` e "grudava"
`actual_end` num newline antigo; quando havia um trecho longo **sem newline**
depois dele (> `target_size`), o próximo `start = actual_end - FRAGMENT_OVERLAP`
não avançava → repetia fragmentos de 500 chars infinitamente, alocando GBs até
estourar a RAM. Gatilho: qualquer arquivo com linha longa (minificado, JSON/data
numa linha, lockfile, base64) que virasse um chunk oversized; `max_ingest_file_bytes`
(10MB) não pega um minificado de 200KB.

**Fix:** estride garantido de `step_size` a partir do início da janela (não a
partir de `actual_end`), aceitando o corte em line-boundary só se ele cobrir um
stride inteiro — termina em ~`total_fragments` iterações, mantém o overlap, sem
gaps. + 2 testes de regressão em `chunker/tests.rs` (linha gigante sem `\n`;
tail longo após newline).

## Auto-stop seguro (não use `bc` com "GiB" — parsing quebra)

Leia o cgroup em **bytes** e compare numericamente:

```bash
cur=$(docker exec wqm-memexd sh -lc 'cat /sys/fs/cgroup/memory.current')
[ "${cur:-0}" -gt $((10*1024*1024*1024)) ] && docker kill wqm-memexd
```
