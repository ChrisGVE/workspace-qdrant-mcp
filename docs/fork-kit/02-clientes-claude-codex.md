# 02 - Configuração Claude Desktop e Codex

## Estratégia recomendada

Use STDIO como transporte padrão no Windows local. Evite HTTP até ter necessidade real.

Motivos:

- Menos superfície de rede.
- Não precisa token HTTP.
- Funciona bem com clientes MCP locais.
- Isola o processo por cliente.

## Gerar configs

Para apenas gerar arquivos em `generated/`:

```powershell
make -f Makefile.win config
```

Para aplicar diretamente em Claude Desktop e Codex:

```powershell
make -f Makefile.win apply-config
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
