# 04 - Fluxo Git, branch, push e PR

## Remotes

Use:

```powershell
git remote -v
```

Esperado:

```text
origin   https://github.com/SEU_USUARIO/workspace-qdrant-mcp.git
upstream https://github.com/ChrisGVE/workspace-qdrant-mcp.git
```

Se faltar upstream:

```powershell
git remote add upstream https://github.com/ChrisGVE/workspace-qdrant-mcp.git
```

## Sincronizar com upstream

```powershell
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git push origin main
```

## Criar branch de trabalho

```powershell
make -f Makefile.win fork-branch BRANCH=personal/windows-hardening
```

ou manualmente:

```powershell
git fetch upstream
git checkout -B personal/windows-hardening upstream/main
```

## Commits pequenos

Prefira:

```text
chore(windows): add fork operations kit
docs(codex): document mcp config for workspace-qdrant
test(rules): cover add rule uuid generation
fix(rules): reject project scope without project id
fix(daemon): reconnect grpc client after transient disconnect
```

## Publicar branch

```powershell
make -f Makefile.win push BRANCH=personal/windows-hardening
```

## Abrir PR

Com GitHub CLI:

```powershell
make -f Makefile.win pr BRANCH=personal/windows-hardening
```

Manual:

1. Abra seu fork no GitHub.
2. Clique em Compare & pull request.
3. Base: `ChrisGVE/workspace-qdrant-mcp:main`.
4. Head: `SEU_USUARIO:personal/windows-hardening`.

## Política de fork

Mantenha três tipos de branch:

- `personal/*`: coisas específicas suas; não precisam ir upstream.
- `fix/*`: correções genéricas; envie upstream.
- `docs/*` ou `chore/*`: docs, scripts, tooling; envie upstream se forem neutras.

## Rebase ou merge?

Para fork pessoal:

```powershell
git merge --ff-only upstream/main
```

Para branches de PR curtas, rebase é aceitável:

```powershell
git fetch upstream
git rebase upstream/main
```

Evite rebase em branch que outras pessoas já usam.
