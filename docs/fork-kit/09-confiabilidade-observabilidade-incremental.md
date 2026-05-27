# Confiabilidade, observabilidade e atualização incremental

Este fork deve ser operado como um serviço local: Qdrant, `memexd`, `wqm` e o MCP TypeScript precisam estar saudáveis ao mesmo tempo. Esta camada adiciona comandos de observação e reparo não destrutivo para reduzir falhas silenciosas.

## Objetivos

- Detectar rapidamente quando Qdrant está saudável, mas daemon/gRPC/MCP não está.
- Registrar snapshots em `.wqm-fork/logs/` antes de qualquer reparo.
- Validar se um projeto está registrado, com watcher ativo e índice incremental consistente.
- Evitar rebuild/delete/admin destrutivo como primeira reação.

## Comandos principais

```powershell
make -f Makefile.win service-stabilize PROJECT=C:\dev\meu-projeto
make -f Makefile.win health PROJECT=C:\dev\meu-projeto
make -f Makefile.win incremental-check PROJECT=C:\dev\meu-projeto
make -f Makefile.win observe PROJECT=C:\dev\meu-projeto OBS_INTERVAL=30
```

## Snapshot de saúde

`health` executa `scripts/windows/service-observe.ps1 -Once` e grava uma linha JSONL em:

```text
.wqm-fork/logs/service-observe-YYYYMMDD.jsonl
```

O snapshot inclui:

- branch e status Git;
- disponibilidade HTTP do Qdrant (`/collections`);
- disponibilidade TCP do daemon gRPC;
- `wqm status health`;
- `wqm queue stats`;
- `wqm project status <PROJECT>`;
- `wqm project watch list --json`;
- processos `node` e `memexd` encontrados.

## Monitor contínuo

```powershell
make -f Makefile.win observe PROJECT=C:\dev\meu-projeto OBS_INTERVAL=15
```

Use durante sessões longas ou quando o daemon parece perder conexão. Interrompa com `Ctrl+C`.

## Checagem incremental

```powershell
make -f Makefile.win incremental-check PROJECT=C:\dev\meu-projeto
```

Coleta:

- `wqm project status <PROJECT>`;
- `wqm project check <PROJECT> --json`;
- `wqm project watch list --json`;
- `wqm queue stats`.

O resultado completo fica em:

```text
.wqm-fork/logs/incremental-check-YYYYMMDD-HHMMSS.json
```

## Reparo não destrutivo

```powershell
make -f Makefile.win incremental-repair PROJECT=C:\dev\meu-projeto
```

O reparo tenta apenas ações conservadoras:

1. `wqm project register <PROJECT> --yes`;
2. `wqm project watch resume`;
3. `wqm project check <PROJECT> --json`;
4. `wqm queue stats`.

Não faz rebuild, delete, truncate, prune ou alteração destrutiva de coleções.

## Estabilização do serviço

```powershell
make -f Makefile.win service-stabilize PROJECT=C:\dev\meu-projeto
```

Sequência:

1. snapshot inicial;
2. tenta subir Qdrant via script existente;
3. tenta iniciar daemon via script existente;
4. snapshot final.

Use este comando antes de culpar o MCP server quando houver erro de `Client not connected`, `Daemon unavailable` ou timeouts intermitentes.

## Ordem recomendada em incidentes

1. `health`
2. `incremental-check`
3. `service-stabilize`
4. `incremental-repair`
5. `smoke-full`
6. só então considerar rebuild/admin, e apenas com aprovação explícita.

## Sinais de alerta

- Qdrant `/collections` OK, mas daemon TCP falha: problema provável em `memexd`/gRPC.
- Daemon TCP OK, mas `wqm status health` falha: problema provável em CLI/config/env.
- `project status` falha, mas `project list` funciona: projeto pode não estar registrado ou path divergente.
- `project check` falha com watcher ativo: investigar fila, ignore rules e branch.
- queue cresce sem baixar: olhar logs do daemon e recursos de embedding.

## Política para agentes

Agentes devem coletar logs antes de alterar estado. Reparos automáticos devem ser não destrutivos. Rebuilds e deletes exigem pedido explícito do usuário.
