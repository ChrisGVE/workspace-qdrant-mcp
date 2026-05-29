# 10 — Gestão multi-projeto e multi-branch

> Modelo de branches por projeto (desde 2026-05-29): `main` (estável) ← `dev` (trabalho, com CI) ← `upstream-sync` (espelho fetch-only de `upstream/main`). Veja `AGENTS.md` › "Modelo de branches e regra sobre `main`". Os targets `workspace-*` operam nesse modelo (versão multi-projeto).

Este fork pode operar vários clones/projetos ao mesmo tempo. Cada projeto pode ter a cadeia:

```text
upstream/main --(ff)--> upstream-sync --(merge)--> dev --(promote)--> main
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

Integrar a correção na linha de trabalho (`dev`):

```powershell
make -f Makefile.win workspace-fix-promote WORKSPACE_NAME=workspace-qdrant WORKSPACE_FIX_BRANCH=fix/minha-correcao PUSH=true
```

## Segurança

O fluxo multi-projeto faz fast-forward de `upstream-sync` a partir de `upstream/main` e mergeia em `dev`; ações automatizadas nunca fazem checkout/merge na `main`. A `main` é a versão estável e só recebe promoções de `dev` (decisão humana). O `upstream` é fetch-only (push desabilitado).

Regras:

- não operar com working tree suja;
- não criar branch chamada `main`, `dev`, `upstream-sync`, `master` (ou as legadas `fork/overlay`/`fork/fixes`/`personal/use-in-projects`) como branch de correção;
- branches de correção devem começar com `fix/`, `chore/`, `refactor/`, `docs/`, `test/` ou `local/`;
- ações mutáveis exigem `-Mutate true`, usado pelos targets do Makefile somente em comandos explícitos.
