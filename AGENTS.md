# AGENTS.md

Instruções para agentes trabalhando neste fork do `workspace-qdrant-mcp` e nos projetos indexados por ele.

## Objetivo do fork

Este fork existe para usar o `workspace-qdrant-mcp` como camada local de memória, busca, observabilidade e indexação incremental em Claude Desktop e Codex, especialmente em Windows.

Prioridades:

1. manter `main` alinhada ao upstream original;
2. manter overlay Windows/Claude/Codex separado de correções upstreamáveis;
3. corrigir bugs que afetam uso real em projetos;
4. suportar vários projetos indexados e branches criadas por agentes;
5. aumentar confiabilidade de serviço, observabilidade e atualização incremental;
6. nunca commitar dados locais, bancos, segredos, logs ou artefatos gerados.

## Regra absoluta sobre `main`

**Nunca faça merge, commit, push de trabalho, nem PR para `main`.**

A `main` deste fork é somente espelho do upstream original. Ela só pode receber sincronização mecânica de `upstream/main`, preferencialmente por humano:

```powershell
git fetch upstream
git checkout main
git merge --ff-only upstream/main
```

Proibido para agentes:

- `git merge <branch>` estando em `main`;
- `git commit` estando em `main`;
- `git push origin main` depois de qualquer trabalho local;
- `gh pr create --base main` no fork;
- abrir/sugerir/preparar PR para `main` do fork;
- abrir PR upstream sem pedido explícito do usuário;
- resolver conflitos colocando overlay/correções diretamente em `main`;
- usar `git reset --hard`, `git clean -fd`, `stash`, rebase publicado ou force-push sem autorização explícita.

Correções upstreamáveis nascem em `fix/*` a partir da `main`. Depois de validadas, podem ser promovidas para `fork/fixes` e `personal/use-in-projects`.

## Dois tipos de branch: não confundir

### 1. Branches do fork do MCP

Cadeia do fork `workspace-qdrant-mcp`:

```text
upstream/main -> main -> fork/overlay -> fork/fixes -> personal/use-in-projects
```

- `main`: espelho limpo do upstream.
- `fork/overlay`: Makefile, scripts Windows, docs, templates, `AGENTS.md`, `.ignore` e patches operacionais.
- `fork/fixes`: overlay + correções usadas no fork.
- `personal/use-in-projects`: branch diária para instalar/usar.

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

## Antes de alterar código

1. Rode `git status --short --branch`.
2. Confirme que não está na `main`.
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
