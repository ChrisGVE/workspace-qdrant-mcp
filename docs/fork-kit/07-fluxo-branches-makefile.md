# Fluxo de branches no Makefile

Este fork usa uma cadeia de branches para separar upstream, overlay operacional, correções e uso diário.

```text
upstream/main -> main -> fork/overlay -> fork/fixes -> personal/use-in-projects
```

## Papel de cada branch

- `main`: espelho limpo do upstream original. Não recebe commits pessoais.
- `fork/overlay`: camada operacional do fork, com `Makefile.win`, scripts Windows, docs, templates, `AGENTS.md` e `.ignore`.
- `fork/fixes`: overlay + correções que você quer usar localmente.
- `personal/use-in-projects`: branch do dia a dia, usada nos seus projetos.

Correções que podem virar PR upstream devem nascer da `main`, em branches `fix/*`. Depois elas são promovidas para `fork/fixes` e para `personal/use-in-projects`.

## Targets principais

```powershell
make -f Makefile.win branch-help
make -f Makefile.win branch-status
make -f Makefile.win branch-init
make -f Makefile.win sync-chain
```

`branch-init` cria/sincroniza a cadeia inteira. `sync-chain` atualiza a cadeia inteira depois que a `main` recebe novidades do upstream.

## Atualizar tudo após puxar upstream

```powershell
make -f Makefile.win sync-chain PUSH=true
```

Equivale a:

```text
upstream/main -> main
main -> fork/overlay
fork/overlay -> fork/fixes
fork/fixes -> personal/use-in-projects
```

Por padrão, `SYNC_STRATEGY=merge`. Para usar rebase:

```powershell
make -f Makefile.win sync-chain SYNC_STRATEGY=rebase PUSH=true
```

Para uso pessoal, `merge` costuma ser mais seguro porque não reescreve histórico publicado.

## Criar correção limpa para PR upstream

```powershell
make -f Makefile.win fix-start FIX_BRANCH=fix/rules-tenant-scope
```

Depois aplique o patch, valide e commite:

```powershell
git apply .\fix-rules-tenant-scope.patch
make -f Makefile.win typecheck-ts
make -f Makefile.win test-ts
git add src\typescript\mcp-server\src\tools
git commit -m "fix(mcp/rules): scope rule mutations by tenant"
git push -u origin fix/rules-tenant-scope
```

## Promover uma correção para sua branch diária

```powershell
make -f Makefile.win fix-promote FIX_BRANCH=fix/rules-tenant-scope PUSH=true
```

Isso incorpora a correção em:

```text
fork/fixes -> personal/use-in-projects
```

## Criar tag da versão usada nos projetos

```powershell
make -f Makefile.win tag-use TAG=fork-claude-codex-ready-v0.1.0 PUSH=true
```

Use tags sempre que uma versão estiver validada em Claude Desktop/Codex.

## Regra para agentes: nunca mexer na main

Agentes não devem commitar, fazer merge ou abrir PR para `main`. A `main` serve apenas como espelho do upstream. A única operação permitida é sincronização mecânica com `upstream/main` via `sync-main`/`sync-chain`.

O target `pr` do Makefile foi bloqueado para evitar PR acidental com base `main`.

## Tenant fix integrado ao fluxo

Para aplicar o patch de tenant no fluxo correto:

```powershell
make -f Makefile.win fix-start FIX_BRANCH=fix/rules-tenant-scope
make -f Makefile.win tenant-check
make -f Makefile.win tenant-apply
make -f Makefile.win tenant-validate
make -f Makefile.win typecheck-ts
make -f Makefile.win test-ts
git add src	ypescript\mcp-server\src	ools
git commit -m "fix(mcp/rules): scope rule mutations by tenant"
make -f Makefile.win fix-promote FIX_BRANCH=fix/rules-tenant-scope PUSH=true
```
