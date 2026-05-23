# 12 — Projetos indexados, branches de agentes e worktrees

Este documento corrige a distinção entre:

1. **branches do fork do MCP**: `main`, `fork/overlay`, `fork/fixes`, `personal/use-in-projects`;
2. **branches dos projetos indexados**: branches que Claude/Codex/agentes criam para trabalhar antes de voltar à branch original.

## Registry correto

Projetos indexados ficam em:

```text
.wqm-fork/indexed-projects.json
```

Esse arquivo é local e não deve ser commitado. O exemplo versionado fica em:

```text
templates/fork-kit/indexed-projects.example.json
```

## Modelo mental

```text
meu-app
  main                         branch base/original
  agent/auth-retry-20260523    branch criada por agente
  agent/ui-fix-20260523        outra branch criada por agente
```

Para branches paralelas, prefira `git worktree`:

```text
C:\dev\meu-app                 main
C:\dev\meu-app-agent-auth      agent/auth-retry-20260523
C:\dev\meu-app-agent-ui        agent/ui-fix-20260523
```

## Fluxo seguro para agentes

1. `index-agent-start`: cria branch e registra no registry.
2. agente altera código na branch.
3. `index-incremental-check`: valida indexação incremental.
4. `index-observe`: captura status, fila, daemon, Qdrant e Git.
5. `index-agent-finish`: marca `ready_for_review`, sem merge.
6. humano decide merge/cherry-pick/PR/descarte.

## Comandos

Inicializar registry:

```powershell
make -f Makefile.win index-init
```

Adicionar projeto:

```powershell
make -f Makefile.win index-project-add INDEX_PROJECT_NAME=meu-app INDEX_PROJECT=C:\dev\meu-app
```

Criar branch de agente em worktree:

```powershell
make -f Makefile.win index-agent-start `
  INDEX_PROJECT_NAME=meu-app `
  INDEX_BRANCH=agent/auth-retry-20260523 `
  INDEX_BASE_BRANCH=main `
  INDEX_USE_WORKTREE=true `
  INDEX_PURPOSE="corrigir retry de auth"
```

Marcar pronta para revisão:

```powershell
make -f Makefile.win index-agent-finish INDEX_PROJECT_NAME=meu-app INDEX_BRANCH=agent/auth-retry-20260523
```

Abandonar sem deletar branch/worktree:

```powershell
make -f Makefile.win index-agent-abandon INDEX_PROJECT_NAME=meu-app INDEX_BRANCH=agent/auth-retry-20260523
```

Observabilidade:

```powershell
make -f Makefile.win index-observe-all
make -f Makefile.win index-incremental-check-all
```

## Políticas

- Nenhuma ação faz merge automático para a branch original.
- Nenhuma ação cria PR automaticamente.
- Remover worktree exige pedido explícito e parâmetro próprio.
- Branches de agentes devem ser tratadas como unidades temporárias de trabalho.
- Se o projeto estiver com working tree suja, o script bloqueia troca/criação sem worktree.
