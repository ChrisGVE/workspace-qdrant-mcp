# Tenant fix e proteção da main

Este fork carrega o patch `patches/fixes/fix-rules-tenant-scope.patch` e um hardening complementar em `patches/fixes/harden-rules-add-scope-error.patch`.

## Problema

Labels de regras não são globalmente únicas. Um projeto A e um projeto B podem ter uma regra com o mesmo `label`, mas conteúdo e escopo diferentes. O comportamento seguro é sempre rotear `add`, `update`, `remove` e duplicate detection pelo tenant correto.

## Correção esperada

- `scope = "project"` usa o `projectId` resolvido como tenant.
- `scope = "project"` sem projeto resolvido retorna erro claro.
- `scope = "global"` usa explicitamente o tenant global.
- duplicate detection project-scoped consulta o projeto atual + regras globais.
- duplicate detection não executa busca ampla se o tenant de projeto não for resolvido.

## Aplicação

```powershell
make -f Makefile.win fix-start FIX_BRANCH=fix/rules-tenant-scope
make -f Makefile.win tenant-check
make -f Makefile.win tenant-apply
make -f Makefile.win tenant-validate
make -f Makefile.win typecheck-ts
make -f Makefile.win test-ts
```

Depois:

```powershell
git add src/typescript/mcp-server/src/tools
git commit -m "fix(mcp/rules): scope rule mutations by tenant"
make -f Makefile.win fix-promote FIX_BRANCH=fix/rules-tenant-scope PUSH=true
```

## Branch de trabalho

Trabalho de correção vai em `dev` (ou feature branch `fix/*` a partir de `dev`), nunca WIP direto na `main`. A `main` é a versão estável e só recebe promoções de `dev` (`make -f Makefile.win promote`), decisão humana/explícita. O `upstream` é fetch-only — o push para o upstream está desabilitado de propósito.

## Validação semântica rápida

```powershell
git grep "buildDuplicateScopeFilter" -- src/typescript/mcp-server/src/tools
git grep "scopeError" -- src/typescript/mcp-server/src/tools/rules.ts
git grep "const tenantId = resolvedProjectId ?? TENANT_GLOBAL" -- src/typescript/mcp-server/src/tools
```

Se todos os marcadores aparecerem, o core tenant fix + hardening estão presentes.
