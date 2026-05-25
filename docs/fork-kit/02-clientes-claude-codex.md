# 02 - Configuração Claude Desktop e Codex

## Estratégia recomendada

Use o stack Docker como backend padrão do fork. O cliente continua em STDIO para Claude Desktop e Codex, mas o daemon e a indexação ficam containerizados, com o caminho padrão em `localhost:50051`.

Esse caminho também suporta FastEmbed, mas dentro do `memexd` do Docker; o endpoint continua em `localhost:50051`.

Motivos:

- Menos dependência de um daemon local no host.
- Uma configuração única para Claude Desktop e Codex.
- O caminho padrão já sai pronto com `make -f Makefile.win apply-config`.
- O fluxo FastEmbed local fica disponível só como alternativa explícita.

## Gerar configs

Para apenas gerar arquivos em `generated/`:

```powershell
make -f Makefile.win config
```

Para aplicar diretamente em Claude Desktop e Codex:

```powershell
make -f Makefile.win apply-config
```

Se você realmente quiser o fluxo local FastEmbed + gRPC, use:

```powershell
make -f Makefile.win apply-config-fastembed
```

Por padrão, os destinos são:

```text
%APPDATA%\Claude\claude_desktop_config.json
%USERPROFILE%\.codex\config.toml
```

## Claude Desktop

Exemplo:

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "node",
      "args": [
        "C:/dev/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js"
      ],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "WQM_DAEMON_ENDPOINT": "localhost:50051"
      }
    }
  }
}
```

Esse é o perfil padrão do fork para o stack Docker. Só troque para `http://localhost:55151` se você estiver usando o fluxo local FastEmbed.

Depois de alterar, reinicie o Claude Desktop.

## Codex

Codex usa `config.toml`. Exemplo:

```toml
[mcp_servers.workspace-qdrant]
command = "node"
args = ["C:/dev/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js"]
startup_timeout_sec = 20
tool_timeout_sec = 120
required = true
enabled_tools = ["search", "retrieve", "grep", "list", "store", "rules"]

[mcp_servers.workspace-qdrant.env]
QDRANT_URL = "http://localhost:6333"
WQM_DAEMON_ENDPOINT = "localhost:50051"
```

No fluxo local FastEmbed, use `WQM_DAEMON_ENDPOINT = "http://localhost:55151"`. No padrão do fork, mantenha `localhost:50051`.

Use `/mcp` no TUI do Codex para ver servidores ativos.

## Tool allowlist inicial

Recomendado:

```text
search,retrieve,grep,list,store,rules
```

Só habilite `embedding` depois de validar:

- daemon saudável;
- latência aceitável;
- uso de CPU sob controle;
- sem duplicação de ferramentas no cliente.

## Instruções para agentes

Copie o conteúdo de:

```text
templates/fork-kit/AGENTS_WORKSPACE_QDRANT.md
templates/fork-kit/CLAUDE_WORKSPACE_QDRANT.md
```

para os arquivos de instrução dos seus projetos, ajustando o tom conforme necessário.
