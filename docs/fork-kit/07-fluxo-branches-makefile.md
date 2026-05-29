# Fluxo de branches no Makefile

Modelo GitFlow-lite (desde 2026-05-29). Veja também `AGENTS.md` › "Modelo de branches e regra sobre `main`".

```text
upstream/main --(ff)--> upstream-sync --(merge)--> dev --(promote quando estável)--> main
```

## Papel de cada branch

- `main`: versão estável/funcional do fork. Linha principal; só recebe promoções estáveis. CI roda na `main`.
- `dev`: linha de trabalho ativa (CI roda em `dev`). Feature branches `fix/*` saem daqui.
- `upstream-sync`: espelho limpo de `upstream/main` (fetch-only; só fast-forward; nunca commitar).

> As branches legadas `fork/overlay`, `fork/fixes` e `personal/use-in-projects` foram mantidas como fallback, mas não fazem mais parte do fluxo ativo.

## Targets principais

```powershell
make -f Makefile.win branch-help      # explica o modelo
make -f Makefile.win branch-status    # branches + log resumido
make -f Makefile.win sync-upstream    # ff upstream-sync a partir de upstream/main
make -f Makefile.win sync-dev         # sync-upstream + merge de upstream-sync em dev
make -f Makefile.win promote          # promove dev -> main (só quando estável)
```

## Atualizar o trabalho com o upstream

`fetch` nunca envia nada; o push para `upstream` está desabilitado de propósito.

```powershell
make -f Makefile.win sync-dev PUSH=true
```

Equivale a:

```text
upstream/main --(ff)--> upstream-sync
upstream-sync --(merge)--> dev
```

Por padrão, `SYNC_STRATEGY=merge`. Para rebase:

```powershell
make -f Makefile.win sync-dev SYNC_STRATEGY=rebase PUSH=true
```

Para uso pessoal, `merge` costuma ser mais seguro porque não reescreve histórico publicado.

## Criar uma correção (feature branch)

```powershell
make -f Makefile.win fix-start FIX_BRANCH=fix/rules-tenant-scope
```

A branch é criada a partir de `dev`. Depois aplique o patch, valide e commite:

```powershell
git apply .\fix-rules-tenant-scope.patch
make -f Makefile.win typecheck-ts
make -f Makefile.win test-ts
git add src\typescript\mcp-server\src\tools
git commit -m "fix(mcp/rules): scope rule mutations by tenant"
git push -u origin fix/rules-tenant-scope
```

## Integrar a correção na linha de trabalho

```powershell
make -f Makefile.win fix-promote FIX_BRANCH=fix/rules-tenant-scope PUSH=true
```

Faz merge de `fix/*` em `dev`. A partir daí o trabalho segue em `dev` até a próxima promoção para `main`.

## Promover para a main (versão estável)

Só quando `dev` estiver estável (CI verde):

```powershell
make -f Makefile.win promote PUSH=true
# ou via PR: gh pr create --base main --head dev
```

## Criar tag de uma versão estável

```powershell
make -f Makefile.win tag-main TAG=fork-claude-codex-ready-v0.1.0 PUSH=true
```

Use tags sempre que uma versão da `main` estiver validada.

## Regra sobre a main

A `main` é a versão estável: recebe promoções de `dev` (decisão humana/explícita), nunca WIP direto. O `upstream` é fetch-only — o push para o upstream está desabilitado de propósito; contribuições de volta vão por `origin` + PR cross-fork.

## Tenant fix integrado ao fluxo

```powershell
make -f Makefile.win fix-start FIX_BRANCH=fix/rules-tenant-scope
make -f Makefile.win tenant-check
make -f Makefile.win tenant-apply
make -f Makefile.win tenant-validate
make -f Makefile.win typecheck-ts
make -f Makefile.win test-ts
git add src/typescript/mcp-server/src/tools
git commit -m "fix(mcp/rules): scope rule mutations by tenant"
make -f Makefile.win fix-promote FIX_BRANCH=fix/rules-tenant-scope PUSH=true
```
