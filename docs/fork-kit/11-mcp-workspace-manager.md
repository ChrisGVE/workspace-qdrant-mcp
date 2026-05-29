# 11 — Gestão via MCP: `fork_workspace`

Este kit inclui um patch opcional para adicionar uma ferramenta MCP local chamada `fork_workspace`.

A ferramenta foi desenhada para permitir que Claude/Codex consultem e coordenem vários projetos/branches sem tocar em `main` por acidente.

## Instalação do patch MCP

Aplique na branch de trabalho `dev`, não na `main`:

```powershell
git checkout dev
make -f Makefile.win mcp-manager-check
make -f Makefile.win mcp-manager-apply
make -f Makefile.win typecheck-ts
make -f Makefile.win test-ts
git add src/typescript/mcp-server/src
git commit -m "feat(mcp): add safe fork workspace manager tool"
```

## Segurança por padrão

Por padrão, a ferramenta é somente leitura. Ações que alteram estado exigem duas coisas:

1. variável de ambiente:

```powershell
$env:WQM_WORKSPACE_MANAGER_ALLOW_MUTATION = "1"
```

2. argumento da chamada MCP:

```json
{
  "allowMutation": true
}
```

Sem os dois sinais, a ferramenta recusa qualquer mutação.

## Ações disponíveis

Leitura:

- `registry:list`
- `project:status`
- `projects:status`

Mutáveis, protegidas:

- `registry:add`
- `registry:remove`
- `project:sync-chain`
- `projects:sync-chain`
- `project:start-fix`
- `project:promote-fix`

## Garantias

A ferramenta MCP:

- não faz PR;
- não faz merge para `main`;
- não cria branch chamada `main`/`master`;
- não atualiza a `main` local;
- usa `upstream/main` como base quando precisa criar branches limpas;
- exige working tree limpa antes de ações mutáveis;
- retorna JSON com status e erros para observabilidade.

## Config recomendada para Claude/Codex

Inclua a variável somente quando quiser permitir mutações pelo agente:

```json
{
  "WQM_WORKSPACE_REGISTRY_PATH": "C:\\dev\\workspace-qdrant-mcp\\.wqm-fork\\projects.json",
  "WQM_WORKSPACE_MANAGER_ALLOW_MUTATION": "0"
}
```

Para operação normal, mantenha `WQM_WORKSPACE_MANAGER_ALLOW_MUTATION=0` e deixe o agente apenas consultar status. Quando quiser uma ação, peça explicitamente e mude temporariamente para `1`.
