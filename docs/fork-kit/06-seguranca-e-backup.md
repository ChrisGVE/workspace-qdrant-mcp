# 06 - Segurança, privacidade e backup

## Transporte MCP

Use STDIO por padrão.

Evite HTTP até precisar integrar outra máquina ou um cliente que não consiga iniciar processo local.

Se usar HTTP:

- defina `MCP_HTTP_TOKEN` forte;
- restrinja bind a localhost quando possível;
- coloque atrás de reverse proxy com TLS quando expor rede;
- não use token curto.

## Dados sensíveis

Antes de registrar um projeto:

1. revise `.gitignore`;
2. crie `.wqmignore`;
3. remova secrets do workspace;
4. não indexe `.env`, dumps e bancos locais.

## Qdrant local

O Qdrant do script usa volume Docker:

```text
qdrant_storage
```

Backup simples:

```powershell
docker run --rm -v qdrant_storage:/data -v ${PWD}:/backup alpine tar czf /backup/qdrant_storage_backup.tgz -C /data .
```

Restore:

```powershell
docker volume create qdrant_storage
docker run --rm -v qdrant_storage:/data -v ${PWD}:/backup alpine sh -c "cd /data && tar xzf /backup/qdrant_storage_backup.tgz"
```

## Estado local do wqm

Também faça backup de:

```text
%USERPROFILE%\.config\workspace-qdrant
%APPDATA%\Claude\claude_desktop_config.json
%USERPROFILE%\.codex\config.toml
```

Os caminhos exatos podem variar por implementação e config; valide com `wqm status health` e com o arquivo `config.yaml` do workspace-qdrant.

## Regras globais

Regras globais afetam múltiplos projetos. Use poucas e objetivas.

Exemplos bons:

```text
prefer-uv: Em projetos Python, prefira uv quando já houver uv.lock.
no-secrets: Nunca imprimir ou armazenar secrets; redigir valores sensíveis.
```

Evite regras globais muito específicas de um projeto.
