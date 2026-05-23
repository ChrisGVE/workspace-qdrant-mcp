# 03 - Correções urgentes e validação

## Ponto importante

Alguns bugs que pareciam urgentes já aparecem corrigidos/fechados no upstream atual. Portanto, a prioridade do fork não é reimplementar tudo: é validar, criar testes locais e só corrigir regressões.

## Matriz de validação

| Área | Estado esperado no upstream atual | Como validar no fork | Ação se falhar |
|---|---|---|---|
| `rules add` com erro de UUID | Deve estar corrigido usando UUID em add | `make -f Makefile.win smoke` | Inspecionar `src/typescript/mcp-server/src/tools/rules-mutation-helpers.ts`, especialmente `persistAddRule` e `document_id` |
| Regra project caindo para global | Deve retornar erro quando projeto não registrado | chamar `rules add scope=project` fora de projeto registrado | Garantir guard em `resolveProjectScopeId` |
| `wqm project register` não aparece em list | Deve estar corrigido | `wqm project register . --yes` e `wqm project list` | Verificar persistência no SQLite e watch folders |
| Daemon "Client not connected" | Deve estar corrigido ou mitigado | `wqm status health`, smoke e teste com múltiplos clientes | Adicionar reconnect/backoff/diagnóstico |
| Startup grande bloqueando gRPC | Deve estar melhorado | testar em repo grande e medir readiness | Priorizar batch + background reconciliation |
| `.gitignore`/`.wqmignore` cascata | Deve estar corrigido | criar `.wqmignore` na raiz e subpastas; validar que não indexa pasta ignorada | Corrigir matcher para acumular regras pai |
| Config Windows/Claude/Codex | Este pacote adiciona | `make -f Makefile.win config` | Corrigir scripts/templates |
| Observabilidade operacional | Ainda vale melhorar | `wqm status health`, logs, métricas | Criar `wqm doctor` upstreamável |

## Primeiro PR recomendado no seu fork

Uma branch não invasiva:

```powershell
git checkout -b chore/windows-fork-kit
git add Makefile.win scripts/windows docs/fork-kit templates/fork-kit
git commit -m "chore(windows): add fork operations kit"
git push -u origin chore/windows-fork-kit
```

Este commit já é útil para você e potencialmente PR upstream, porque melhora documentação/operabilidade sem mexer em core.

## Segundo PR recomendado

Adicionar testes de regressão para os bugs já fechados:

1. `rules add` deve usar UUID em criação.
2. `scope=project` sem projeto registrado deve falhar, não cair para global.
3. `project register` deve persistir e aparecer em `project list`.
4. `.wqmignore` de raiz deve ser aplicado em child scans.
5. daemon deve ficar pronto antes de reconciliação pesada.

## Terceiro PR recomendado

Melhorar `doctor`/diagnóstico upstream:

- detectar Qdrant saudável;
- detectar daemon gRPC;
- detectar múltiplos MCP Node presos;
- validar versão de `wqm`, `memexd`, MCP package;
- sugerir correção objetiva.

## Critério de pronto para usar em projetos reais

Antes de confiar em um projeto importante, confirme:

```powershell
make -f Makefile.win doctor
make -f Makefile.win smoke PROJECT=C:\dev\projeto-real
wqm project check C:\dev\projeto-real --verbose
wqm search project "termo conhecido do projeto"
```

Se a busca recuperar arquivos corretos e o daemon continuar saudável, habilite no Claude/Codex.
