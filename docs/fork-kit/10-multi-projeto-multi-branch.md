# 10 — Gestão multi-projeto e multi-branch

Este fork pode operar vários clones/projetos ao mesmo tempo. Cada projeto pode ter a cadeia:

```text
upstream/main -> main -> fork/overlay -> fork/fixes -> personal/use-in-projects
```

Além disso, cada correção isolada deve viver em uma branch própria:

```text
fix/rules-tenant-scope
fix/daemon-reconnect
fix/incremental-watchers
local/meu-ajuste
```

## Registry local

O registry fica em:

```text
.wqm-fork/projects.json
```

Ele é local e não deve ser commitado. Existe um exemplo versionado em:

```text
templates/fork-kit/projects.example.json
```

## Comandos principais

Inicializar registry:

```powershell
make -f Makefile.win workspace-init
```

Adicionar o projeto atual:

```powershell
make -f Makefile.win workspace-add WORKSPACE_NAME=workspace-qdrant WORKSPACE_PROJECT=C:\dev\workspace-qdrant-mcp
```

Listar projetos:

```powershell
make -f Makefile.win workspace-list
```

Ver status de todos:

```powershell
make -f Makefile.win workspace-status-all
```

Sincronizar um projeto pela cadeia pessoal:

```powershell
make -f Makefile.win workspace-sync WORKSPACE_NAME=workspace-qdrant PUSH=true
```

Sincronizar todos:

```powershell
make -f Makefile.win workspace-sync-all PUSH=true
```

Criar uma branch de correção limpa sem mexer na `main` local:

```powershell
make -f Makefile.win workspace-fix-start WORKSPACE_NAME=workspace-qdrant WORKSPACE_FIX_BRANCH=fix/minha-correcao
```

Promover a correção para `fork/fixes` e `personal/use-in-projects`:

```powershell
make -f Makefile.win workspace-fix-promote WORKSPACE_NAME=workspace-qdrant WORKSPACE_FIX_BRANCH=fix/minha-correcao PUSH=true
```

## Segurança

O fluxo multi-projeto usa `upstream/main` como base para overlay/fixes/use e evita fazer checkout/merge em `main` durante ações automatizadas. A `main` continua sendo espelho limpo do upstream e só deve ser sincronizada manualmente por humano.

Regras:

- não operar com working tree suja;
- não criar branch chamada `main`, `master`, `fork/overlay`, `fork/fixes` ou `personal/use-in-projects` como branch de correção;
- branches de correção devem começar com `fix/`, `chore/`, `refactor/`, `docs/`, `test/` ou `local/`;
- ações mutáveis exigem `-Mutate true`, usado pelos targets do Makefile somente em comandos explícitos.
