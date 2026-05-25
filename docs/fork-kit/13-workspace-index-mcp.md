# 13 — Ferramenta MCP `workspace_index`

A ferramenta MCP `workspace_index` permite que Claude/Codex consultem e, com autorização explícita, gerenciem projetos indexados e branches criadas por agentes.

## Instalação

Aplique o patch no fork, de preferência em `fork/fixes`:

```powershell
git checkout fork/fixes
make -f Makefile.win workspace-index-check
make -f Makefile.win workspace-index-apply
make -f Makefile.win typecheck-ts
make -f Makefile.win test-ts
git add src/typescript/mcp-server/src
git commit -m "feat(mcp): add workspace index management tool"
```

## Ações de leitura

- `list_projects`
- `project_status`
- `list_branches`
- `agent_branch_status`
- `observe_project`
- `observe_all`
- `incremental_check`
- `incremental_check_all`

## Ações mutáveis

- `init`
- `add_project`
- `start_agent_branch`
- `finish_agent_branch`
- `abandon_agent_branch`
- `register_wqm`
- `register_all_wqm`
- `cleanup_orphans`

Mutação exige:

```powershell
$env:WQM_INDEX_MANAGER_ALLOW_MUTATION = "1"
```

E `allowMutation: true` na chamada MCP.

## Exemplo MCP: listar projetos

```json
{ "action": "list_projects" }
```

## Exemplo MCP: criar branch de agente

```json
{
  "action": "start_agent_branch",
  "projectName": "meu-app",
  "branchName": "agent/auth-retry-20260523",
  "baseBranch": "main",
  "useWorktree": true,
  "purpose": "corrigir retry de auth",
  "allowMutation": true
}
```

## Garantia importante

`workspace_index` **não faz merge para a branch original**. `finish_agent_branch` apenas marca a branch como `ready_for_review`.

Quando `start_agent_branch` cria uma branch nova, o fluxo tenta registrar automaticamente o caminho no daemon para que a branch fique pesquisável sem um passo manual extra.

O mesmo registry também pode ser alimentado por hooks locais instalados com `make -f Makefile.win index-hooks-install`. Isso fecha a brecha das criações manuais com `git checkout -b` e `git worktree add`, que passam a aparecer sem depender de helper específico.

Para higienização periódica, use `cleanup_orphans`. A ação remove do registry as entradas cujas branches ou worktrees já não existem mais, sem apagar nada do disco. Ela é mutável, então continua sujeita ao `allowMutation: true` e ao opt-in do ambiente. Isso permite rodar a limpeza de forma assíncrona depois que um worktree foi removido.
