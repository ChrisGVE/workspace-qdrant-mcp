# 01 - Fork e setup no Windows

## Pré-requisitos

Instale ou valide:

- Git.
- GitHub CLI (`gh`), opcional, mas recomendado.
- Node.js 18+; Node 22 LTS é uma boa escolha.
- npm.
- Rust via rustup, se for compilar `wqm`/`memexd`.
- Visual Studio Build Tools com workload C++.
- Docker Desktop para Qdrant.
- GNU Make para usar `Makefile.win`.
- PowerShell 7 é recomendado, embora os scripts evitem recursos muito novos quando possível.

## Criar fork

Opção com GitHub CLI:

```powershell
gh repo fork ChrisGVE/workspace-qdrant-mcp --clone=false
git clone https://github.com/SEU_USUARIO/workspace-qdrant-mcp.git C:\dev\workspace-qdrant-mcp
cd C:\dev\workspace-qdrant-mcp
git remote add upstream https://github.com/ChrisGVE/workspace-qdrant-mcp.git
git fetch upstream
```

Opção manual:

1. Abra `https://github.com/ChrisGVE/workspace-qdrant-mcp`.
2. Clique em Fork.
3. Clone o fork.
4. Adicione o remote upstream.

## Aplicar o overlay deste pacote

A partir da pasta onde você extraiu o zip:

```powershell
pwsh .\apply-overlay.ps1 -RepoDir C:\dev\workspace-qdrant-mcp
```

Depois, dentro do fork:

```powershell
cd C:\dev\workspace-qdrant-mcp
git checkout -b personal/windows-hardening
git add Makefile.win scripts/windows docs/fork-kit templates/fork-kit
git commit -m "chore(windows): add fork operations kit"
```

## Rodar doctor

```powershell
make -f Makefile.win doctor
```

O doctor valida:

- ferramentas no PATH;
- estrutura do repo;
- `dist/index.js`;
- Qdrant;
- porta do daemon;
- arquivos de configuração Claude/Codex.

## Subir Qdrant

```powershell
make -f Makefile.win qdrant-up
```

Se você quiser garantir o fluxo gRPC usado pelo daemon local, use:

```powershell
make -f Makefile.win qdrant-up-grpc
```

Isto cria/inicia um container Docker:

```text
wqm-qdrant
```

com volume:

```text
qdrant_storage
```

## Compilar MCP TypeScript

```powershell
make -f Makefile.win build-ts
```

## Padrão do fork: Docker

Se você quer usar tudo containerizado, este é o caminho padrão do fork:

```powershell
make -f Makefile.win compose-up
make -f Makefile.win apply-config
```

Isso sobe o stack Docker e gera as configs de Claude Desktop e Codex: o Claude Desktop continua apontando para o daemon em `http://localhost:50051`, e o Codex usa o MCP HTTP containerizado em `http://localhost:6335/mcp`.

## Daemon e CLI

Se você já instalou binários pré-compilados:

```powershell
wqm --version
wqm service start
wqm status health
```

Ou:

```powershell
make -f Makefile.win start-daemon
```

Se você quiser o fluxo local opcional com FastEmbed e gRPC:

```powershell
make -f Makefile.win service-stabilize-fastembed
```

Esse caminho não é o padrão do fork; use-o só quando você realmente quiser o daemon fora do Docker. Nesse modo, o `apply-config-fastembed` volta a gerar o perfil stdio local para o Codex.

Compilar Rust localmente pode exigir configuração de ONNX Runtime estático. Para começar rápido, prefira binários pré-compilados e use source build só quando for mexer no daemon/CLI.

## Smoke test

```powershell
make -f Makefile.win smoke PROJECT=C:\dev\meu-projeto
```

O smoke test tenta:

1. Qdrant `/collections`.
2. Porta do daemon.
3. `wqm status health`.
4. Build TypeScript se necessário.
5. Registro do projeto.
6. `rules add/list/remove`.
7. Busca global simples.
