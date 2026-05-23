# 00 - Visão geral do fork

## Objetivo

Criar um fork mantível do `workspace-qdrant-mcp` para uso local com Claude Desktop, Codex e projetos de desenvolvimento, com foco em:

1. Operação previsível no Windows.
2. Configuração MCP reproduzível.
3. Validação das correções críticas já feitas no upstream.
4. Pequenas melhorias próprias, preferencialmente contribuíveis como PR.
5. Uso seguro por projeto, sem misturar contexto global e contexto de projeto.

## Decisão

Faça o fork, mas não transforme o fork em um produto paralelo. A melhor estratégia é manter o upstream perto e usar branches pequenas:

- `personal/windows-hardening`
- `fix/<area>-<bug>`
- `docs/<tema>`
- `chore/<tema>`

## Estado técnico observado

O projeto usa três peças principais:

- `memexd`: daemon Rust para file watching, embeddings e fila.
- `wqm`: CLI Rust para serviço, projeto, regras, filas e administração.
- MCP Server TypeScript: expõe ferramentas para clientes MCP.

O MCP TypeScript expõe ferramentas como `search`, `retrieve`, `rules`, `store`, `grep`, `list` e `embedding`. Para uso inicial em Codex/Claude, recomendo expor apenas:

```text
search,retrieve,grep,list,store,rules
```

Mantenha `embedding` desabilitado no começo para reduzir superfície e ruído operacional.

## O que este pacote adiciona

Este pacote é um overlay operacional:

- Makefile Windows.
- Scripts PowerShell.
- Documentação de setup e operação.
- Templates Claude/Codex.
- Smoke tests para validar Qdrant, daemon, regras e registro de projeto.

Ele não altera lógica de produção do MCP por padrão. A primeira branch recomendada é de operação/documentação, não de refatoração.
