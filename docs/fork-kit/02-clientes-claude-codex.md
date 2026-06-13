# 02 - Configuração Claude Desktop e Codex

## Estratégia recomendada

Use o stack Docker como backend padrão do fork. O daemon, a indexação e o MCP ficam containerizados; o Codex fala com o MCP HTTP do stack em `http://localhost:6335/mcp`. O Claude Desktop pode continuar no perfil STDIO local enquanto você ainda não migrou esse cliente.

Esse caminho também suporta FastEmbed, mas dentro do `memexd` do Docker. O fluxo local FastEmbed continua disponível como alternativa explícita.

Motivos:

- Menos dependência de um daemon local no host.
- O Codex não depende de um processo MCP local; ele fala direto com o container HTTP.
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
        "WQM_DAEMON_ENDPOINT": "http://localhost:50051"
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
[projects.'\\wsl.localhost\Ubuntu-24.04\home\alkmimm\respositorios\workspace-qdrant-mcp']
trust_level = "trusted"

[mcp_servers.workspace-qdrant]
url = "http://localhost:6335/mcp"
bearer_token_env_var = "MCP_HTTP_TOKEN"
startup_timeout_sec = 20
tool_timeout_sec = 120
required = true
enabled_tools = ["search", "retrieve", "grep", "list", "store", "rules", "workspace_index", "graph"]
```

No Windows/WSL, o `apply-config` também registra o projeto como trusted nas
formas de caminho que o Codex pode receber (`\\wsl.localhost\...`, `\\?\UNC\...`
e `/home/...`). Sem isso, o MCP pode aparecer configurado, mas o Codex pode
ignorar camadas locais do projeto por não reconhecer o caminho como confiável.

Use o mesmo valor de `MCP_HTTP_TOKEN` que você colocou em `docker/.env` para o stack Docker.
No Windows, deixe `MCP_HTTP_TOKEN` disponível no ambiente antes de abrir o Codex, ou inicie o Codex a partir de um shell onde esse env var já esteja exportado.
Se quiser evitar export manual, use `make -f Makefile.win codex-open` ou `.\scripts\windows\start-codex.ps1`; ambos leem `docker\.env` e exportam `MCP_HTTP_TOKEN` antes de abrir o Codex.

No fluxo local FastEmbed, `apply-config-fastembed` gera o perfil stdio tradicional e usa `WQM_DAEMON_ENDPOINT = "http://localhost:55151"`.

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
