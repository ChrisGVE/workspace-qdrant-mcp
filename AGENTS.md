# AGENTS.md

Instruções para agentes trabalhando neste fork do `workspace-qdrant-mcp` e nos projetos indexados por ele.

## Objetivo do fork

Este fork existe para usar o `workspace-qdrant-mcp` como camada local de memória, busca, observabilidade e indexação incremental em Claude Desktop e Codex, especialmente em Windows.

Prioridades:

1. manter `main` como versão estável/funcional do fork (upstream rastreado em `upstream-sync`);
2. manter overlay Windows/Claude/Codex separado de correções upstreamáveis;
3. corrigir bugs que afetam uso real em projetos;
4. suportar vários projetos indexados e branches criadas por agentes;
5. aumentar confiabilidade de serviço, observabilidade e atualização incremental;
6. nunca commitar dados locais, bancos, segredos, logs ou artefatos gerados.

## Modelo de branches e regra sobre `main`

Modelo (GitFlow-lite, desde 2026-05-29):

- **`main` = versão estável/funcional do fork.** Linha principal. Recebe promoções de `dev` quando o trabalho está estável, e atualizações do upstream via `upstream-sync`. CI roda na `main`.
- **`dev` = branch de trabalho ativa.** Todo desenvolvimento acontece aqui (ou em feature branches a partir de `dev`). CI roda em `dev` (`ci.yml: branches: [main, dev]`).
- **`upstream-sync` = espelho limpo de `upstream/main`** (fetch-only; só fast-forward; nunca commitar nela).

**Trabalho de agente vai em `dev` (ou feature branch a partir de `dev`), nunca WIP direto na `main`.** Promover `dev → main` é decisão do humano (ou pedido explícito): quando estável, via PR `--base main` ou merge.

Proibido para agentes sem autorização explícita:

- commitar WIP direto na `main` (use `dev`);
- promover `dev → main` ou abrir PR `--base main` por conta própria;
- abrir PR para o `upstream` (`ChrisGVE`) — o push para o upstream está desabilitado de propósito;
- usar `git reset --hard`, `git clean -fd`, `stash`, rebase publicado ou force-push.

Sincronizar com o upstream (sempre seguro; `fetch` não envia nada):

```powershell
git fetch upstream
git switch upstream-sync; git merge --ff-only upstream/main   # atualiza o espelho
git switch dev; git merge upstream-sync                       # integra no trabalho
```

> Branches legadas do fork-kit (`fork/overlay`, `fork/fixes`, `personal/use-in-projects`) são mantidas como fallback, mas não são mais a linha ativa. Os targets de branch do `Makefile.win` / `branch-flow.ps1` ainda implementam o modelo antigo de 4 camadas e estão pendentes de migração — prefira os comandos git acima.

## Dois tipos de branch: não confundir

### 1. Branches do fork do MCP

Branches do fork `workspace-qdrant-mcp`:

```text
upstream/main → upstream-sync        (espelho fetch-only)
dev (trabalho, CI) → main (estável, CI)   [promover quando estável]
```

- `main`: versão estável/funcional do fork. Linha principal.
- `dev`: branch de trabalho ativa (com CI). Feature branches saem daqui.
- `upstream-sync`: espelho limpo de `upstream/main` (fetch-only).
- `fork/overlay`, `fork/fixes`, `personal/use-in-projects`: camadas legadas do fork-kit, mantidas como fallback (inativas).

Registry local recomendado: `.wqm-fork/fork-branches.json`.

### 2. Branches dos projetos indexados

Quando o usuário fala em múltiplas branches, normalmente significa branches criadas por agentes de IA dentro dos projetos indexados para fazer alterações antes de voltar à branch original.

Exemplo:

```text
C:\dev\meu-app              -> main
C:\dev\meu-app-agent-auth   -> agent/auth-retry-20260523
C:\dev\meu-app-agent-ui     -> agent/ui-fix-20260523
```

Registry local recomendado: `.wqm-fork/indexed-projects.json`.

## Política para branches criadas por agentes

Agentes podem criar branches de trabalho, mas **não fazem merge automático de volta**.

Ciclo seguro:

1. detectar projeto e branch base;
2. criar branch `agent/<slug>-<yyyymmdd-hhmm>` ou `fix/<slug>`;
3. preferir `git worktree` para branch paralela;
4. registrar a branch no `indexed-projects.json`;
5. rodar checks/incremental/observabilidade;
6. ao terminar, marcar como `ready_for_review`;
7. voltar para branch original quando for mesmo diretório;
8. aguardar humano decidir merge, PR, squash, cherry-pick ou descarte.

Proibido para agentes em projetos indexados:

- fazer merge para `main`, `master`, `develop` ou branch original sem pedido explícito;
- fazer PR automaticamente;
- deletar branch/worktree sem autorização explícita;
- trocar branch com working tree suja;
- alternar branch em diretório observado pelo daemon quando existe outro agente trabalhando nele; prefira worktree;
- reindexar tudo/destruir coleções sem pedido explícito.

## Comandos recomendados

Criar registry de projetos indexados:

```powershell
make -f Makefile.win index-init
```

Registrar projeto indexado:

```powershell
make -f Makefile.win index-project-add INDEX_PROJECT_NAME=meu-app INDEX_PROJECT=C:\dev\meu-app
```

Criar branch de agente em worktree paralelo:

```powershell
make -f Makefile.win index-agent-start `
  INDEX_PROJECT_NAME=meu-app `
  INDEX_BRANCH=agent/auth-retry-20260523 `
  INDEX_BASE_BRANCH=main `
  INDEX_USE_WORKTREE=true `
  INDEX_PURPOSE="corrigir retry de auth"
```

Marcar pronta para revisão, sem merge:

```powershell
make -f Makefile.win index-agent-finish INDEX_PROJECT_NAME=meu-app INDEX_BRANCH=agent/auth-retry-20260523
```

Observar todos os projetos/branches:

```powershell
make -f Makefile.win index-observe-all
```

Rodar checagem incremental:

```powershell
make -f Makefile.win index-incremental-check-all
```

## Ferramenta MCP `workspace_index`

Se instalada, a ferramenta MCP `workspace_index` permite que Claude/Codex consultem e, com autorização explícita, gerenciem os projetos indexados e branches de agentes.

Leitura segura:

- `list_projects`
- `project_status`
- `list_branches`
- `observe_project`
- `observe_all`
- `incremental_check`
- `agent_branch_status`

Mutação controlada:

- `add_project`
- `start_agent_branch`
- `finish_agent_branch`
- `abandon_agent_branch`
- `register_wqm`
- `repair_incremental`

Ações mutáveis exigem os dois sinais:

```powershell
$env:WQM_INDEX_MANAGER_ALLOW_MUTATION = "1"
```

E, na chamada MCP:

```json
{ "allowMutation": true }
```

Sem os dois, a mutação é recusada.

## Confiabilidade e observabilidade

Antes de declarar um projeto estável:

```powershell
make -f Makefile.win service-stabilize PROJECT=C:\dev\meu-projeto
make -f Makefile.win index-project-status INDEX_PROJECT_NAME=meu-projeto
make -f Makefile.win index-incremental-check INDEX_PROJECT_NAME=meu-projeto
make -f Makefile.win health PROJECT=C:\dev\meu-projeto
```

Observabilidade mínima por projeto/branch:

- Qdrant responde;
- daemon TCP responde;
- `wqm status health` responde;
- fila tem profundidade aceitável;
- branch atual e commit HEAD;
- dirty working tree;
- ahead/behind contra branch base;
- watch ativo;
- `project check` sem divergências críticas;
- último snapshot em `.wqm-fork/observability/`.

## Correção de tenant em `rules`

Comportamento obrigatório:

- `rules add/update/remove` com `scope="project"` usa `projectId` como tenant;
- `scope="project"` sem projeto registrado/resolvido retorna erro; nunca cai para `global`;
- `scope="global"` usa explicitamente tenant global;
- duplicate detection de regras project-scoped considera projeto atual + regras globais, não todos os projetos.

## Lições aprendidas até agora

- Para `list/search` sem `projectId`, o projeto precisa estar registrado no daemon em `watch_folders`; sem isso o detector não resolve `projectPath` e `projectId`.
- O padrão do fork é Docker: `make -f Makefile.win apply-config` gera clientes para `localhost:50051`; o FastEmbed continua disponível dentro do `memexd` containerizado, enquanto `apply-config-fastembed`/`55151` é só o caminho local opcional.
- Em Docker, `WQM_DEV_ROOT` precisa espelhar o caminho real do repositório no host; se o bind mount não bater com o path git, o daemon pode logar `Not a Git repository, defaulting to 'main'` e a resolução de branch fica degradada.
- `daemon_unavailable` no bootstrap pode ser só endpoint/profile errado; antes de culpar o daemon, confirme `50051` vs `55151` e qual stack está de pé.
- Para wrappers host-side do fluxo Docker (`status`, `project list`, `project status`), também é preciso apontar `WQM_DATABASE_PATH` para o `state/memexd/memexd.db` do bind mount; sem isso o CLI cai no `state.db` local e pode reclamar de banco ausente mesmo com os containers saudáveis.
- `wqm status health` pode sair como `degraded` por `Embedding Provider: degraded / probe pending` mesmo quando conexão, gRPC, fila e Qdrant estão OK; distinguir isso de falha real de conexão.
- `healthCheck()` do daemon em `src/typescript/mcp-server/src/clients/daemon-client/system-methods.ts` precisa de timeout explícito, ou o bootstrap pode ficar preso esperando a RPC.
- Em WSL, `npx tsc -p tsconfig.json` funcionou, mas `npm run build` pode falhar por IPC/temporários do `tsx` em caminho Windows; trate isso como diferença de ambiente, não como falha do TypeScript.
- Ao validar o MCP, não assumir que o handshake completo está resolvido só porque o transporte conectou; confirmar `initialize` e a resposta real das tools antes de declarar sucesso.
- Para testar o servidor compilado, garantir que `dist/proto/workspace_daemon.proto` acompanhe o JS gerado.
- libgit2 (via crate `git2`) ignora as envs `GIT_CONFIG_*` e só lê arquivos gitconfig; o `GIT_CONFIG_COUNT/KEY/VALUE: safe.directory = *` já presente no compose libera o git CLI mas não o daemon. Para o `memexd` abrir repos bind-mounted, o `Dockerfile.memexd` cria `/etc/gitconfig` com `[safe] directory = *`. Sem isso, `Repository::open` falha por ownership e `get_current_branch` cai para `"main"` emitindo WARN a cada chamada — em árvores grandes vira centenas de WARNs/min e mascara warns reais.
- O limite de RSS do `unified_queue_processor` (`DEFAULT_MAX_RSS_MB`) era 2048MB hardcoded; em hosts com FastEmbed + heap + mmap o processo fica colado nesse teto e entra em pause-loop (5s ativo / 10s pausado), derrubando o throughput em ~150×. Agora `WQM_MAX_RSS_MB` faz override via env (cacheada com `OnceLock`); compose default é 4096.
- Sem `MCP_HTTP_TRUST_LOCALHOST=1`, `/metrics`, `/readyz` e outras rotas do MCP HTTP retornam 401 mesmo do próprio host — a detecção em `auth-middleware.ts` cobre loopback e Docker bridge (172.16-31) quando o processo roda em contêiner. O compose consolidado seta `1` por padrão; trocar para `0` se a porta for exposta fora de localhost.
- `/proc/self/statm` (usado pelo `check_process_rss`) e `docker stats` divergem em ordem de grandeza — o primeiro é o RSS do processo, o segundo é `memory.current` do cgroup (inclui page cache e mmap files). Não tratar `docker stats` como autoridade para o que o guard interno vê.
- Build com `docker compose build` quando o builder ativo usa driver `docker-container` (ex.: `multiarch-builder`) produz manifest list em registro interno do buildkit, mas a tag `:local` no docker daemon continua apontando para a imagem antiga — `docker compose up` recria o contêiner com o binário antigo mesmo após build `DONE`. Confirmar o builder com `docker buildx ls` e usar o de driver `docker` (`desktop-linux` no Docker Desktop), ou passar `--load` explícito. Sintoma: `docker image inspect` mostra `Created` recente mas `docker history` não tem as layers novas, e `docker run --rm ... cat /etc/<arquivo-novo>` falha.
- `FtsBatchProcessor::process_batch` carrega `old_content + new_content + DiffResult` de TODAS as mudanças do batch em RAM (Vec) antes de abrir transação. Um único arquivo de 600k linhas (CSV dump, proto gerado, lockfile) entrando no mesmo batch que 49 arquivos pequenos coloca o processo em RSS de 11GB+ e trips o pause-loop do `unified_queue_processor` documentado acima. Dois env guards (search.db v8, default 0/off): `WQM_FTS5_SINGLE_MODE_THRESHOLD` força single-mode quando qualquer pending change > N linhas (protege os pequenos); `WQM_FTS5_HARD_CAP` pula `code_lines`/FTS5 inteiro para arquivos > N linhas e marca `file_metadata.fts5_skipped = 1` (a busca semântica continua). Sintoma do problema sem o cap: `INFO FTS5 flush complete: 1 files, 604616 inserted, ..., 105829ms (batch)` seguido de WARN de RSS. Achar culprits: `docker run --rm -v workspace-qdrant-mcp_memexd_db:/data keinos/sqlite3 sqlite3 /data/search.db 'SELECT file_path, size_bytes FROM file_metadata ORDER BY size_bytes DESC LIMIT 15'`.
- `global.wqmignore` é aplicado em **três** caminhos de enqueue, e eles precisam concordar: (1) file watcher via `patterns::global_ignore::is_globally_ignored()` em `watching_queue/file_watcher_ops.rs` (gates `should_filter_event`, `should_filter_debounced_event`, e o chokepoint `enqueue_file_operation`); (2) folder-scan via `is_ignored_by_matcher()` em `strategies/processing/folder/scan.rs` (checa dir antes de descer + arquivo); (3) reconciler via `WalkBuilder::add_ignore()` em `startup/reconciliation/ignore_sync.rs`. Antes de existir o módulo `global_ignore`, só o reconciler aplicava o arquivo — o watcher reenfileirava `state/qdrant` a cada rotação de segmento (feedback loop), e o folder-scan reindexava generated/ que só o global excluía. O matcher de `is_globally_ignored` é cacheado e recarregado por mtime (edição via admin UI vale no próximo evento, sem restart).
- **Anchor diverge entre os caminhos**: `is_globally_ignored` ancora em `/` (`GitignoreBuilder::new("/")` + `matched_path_or_any_parents`); o reconciler usa `WalkBuilder::add_ignore(global.wqmignore)` que ancora no **dir-pai do ignore-file** (`/var/lib/memexd/`). Padrões com prefixo `**/` e globs de extensão (a forma usada em todo o `global.wqmignore`) casam idêntico nos dois. Reservar padrões root-anchored (`/foo`) para `.wqmignore` de projeto.
- **Re-inclusão (`!`) precisa negar o DIRETÓRIO, não só o conteúdo**: para re-incluir arquivos sob um dir excluído (ex.: `!**/.../testlink/cfg/**` sob `**/.../testlink/**`), negue também o dir nu (`!**/.../testlink/cfg/`). Um walk que poda diretórios descarta o dir ainda-ignorado antes de descer, então a negação só-conteúdo nunca dispara — sintoma observado: `cfg/`+`custom/` com 14 arquivos a 0 indexados apesar do `!.../cfg/**`. `matched_path_or_any_parents` re-inclui o *arquivo* mas não o *dir nu*; cobertura em `patterns/global_ignore.rs` tests (`testlink_cfg_*`).

## Antes de alterar código

1. Rode `git status --short --branch`.
2. Para trabalho no fork, confirme que está em `dev` (ou feature branch a partir de `dev`), não na `main` — a `main` recebe só promoções estáveis.
3. Confirme se está em branch do fork ou branch de projeto indexado.
4. Para projeto indexado, registre `project`, `path`, `branch`, `baseBranch` e `returnBranch`.
5. Não faça merge automático de volta à branch original.
6. Rode validações compatíveis.

## Dados locais e segurança

Não commitar:

- `.env`, tokens, API keys;
- bancos SQLite;
- storage Qdrant;
- `.wqm-fork/*.json` de runtime;
- logs, snapshots, reports, metrics;
- `node_modules`, `dist`, `target`;
- configs geradas com paths pessoais.

Use `.ignore` para reduzir ruído no indexador.
