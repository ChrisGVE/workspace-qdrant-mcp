# Plano de evolucao de embedding e rerank

**Data:** 2026-05-25
**Status:** Proposto
**Decisao atual:** manter o embedding atual em `fastembed` / `AllMiniLM-L6-v2` / 384 dimensoes por enquanto.

---

## Contexto

O fork usa o `workspace-qdrant-mcp` como camada local de memoria, busca,
observabilidade e indexacao incremental. A busca atual ja combina:

1. embedding denso local com FastEmbed;
2. vetor esparso BM25;
3. busca hibrida no Qdrant;
4. fusao por RRF no MCP server.

Como o projeto ainda esta no comeco, mudar dimensao e reindexar colecoes nao e
um bloqueio relevante. Mesmo assim, a decisao operacional para agora e manter o
modelo atual e preparar os proximos passos de forma incremental.

O objetivo deste plano e deixar pronto um caminho claro para evoluir qualidade de
busca sem misturar tres assuntos diferentes:

1. **qualidade do embedding**: trocar para um modelo maior/melhor;
2. **qualidade do ranking**: adicionar rerank de segunda fase;
3. **performance operacional**: usar GPU quando ingestao ou rerank ficarem caros.

---

## Principios

1. **Nao trocar modelo sem medir baseline.** Antes de mudar dimensao, registrar
   latencia, throughput e exemplos de qualidade com o estado atual.
2. **Rerank vem depois da recuperacao ampla.** A busca hibrida deve buscar mais
   candidatos do que o usuario pediu; o reranker reordena apenas o topo.
3. **GPU e aceleracao, nao garantia de qualidade.** GPU ajuda throughput e
   latencia de inferencia, mas a qualidade vem de modelo, dados, filtros e rerank.
4. **Mudancas devem ser reversiveis.** Cada fase precisa ter flag/config para
   voltar ao comportamento anterior.
5. **Evitar reindexacao acidental.** Toda troca de dimensao deve ser explicita e
   acompanhada de plano de rebuild/reembed.
6. **Dar opcoes sem expor complexidade demais.** O usuario deve poder escolher
   perfis prontos, mas tambem deve existir um modo `custom` para ambientes mais
   avancados.

---

## Modelo configuravel

A evolucao deve ser desenhada como uma matriz de opcoes configuraveis. O default
continua simples e local, mas usuarios podem escolher outro perfil quando
quiserem priorizar qualidade, custo, latencia, privacidade ou GPU.

### Perfis de alto nivel

```yaml
search_quality_profile: stable-local
```

Perfis sugeridos:

| Perfil | Objetivo | Embedding | Rerank | GPU |
|--------|----------|-----------|--------|-----|
| `stable-local` | estabilidade e privacidade local | FastEmbed 384d | desligado | nao |
| `local-quality` | melhor ranking sem servico remoto | FastEmbed 384d | local top-k | opcional |
| `remote-quality` | qualidade maior com endpoint externo | OpenAI-compatible 768d+ | remoto top-k | opcional |
| `multilingual` | corpus misto PT/EN/outros | modelo multilingue | multilingue | opcional |
| `gpu-service` | throughput alto | endpoint GPU | endpoint GPU | sim |
| `custom` | usuario controla tudo | configuravel | configuravel | configuravel |

O perfil nao deve esconder a configuracao final. Ele deve preencher defaults, e
o usuario ainda pode sobrescrever campos especificos.

### Exemplo: perfil estavel atual

```yaml
search_quality_profile: stable-local

embedding:
  provider: fastembed
  model: AllMiniLM-L6-v2
  output_dim: 384

search:
  candidate_multiplier: 2
  rerank:
    enabled: false
  diversity:
    enabled: false
```

### Exemplo: perfil com rerank local

```yaml
search_quality_profile: local-quality

embedding:
  provider: fastembed
  model: AllMiniLM-L6-v2
  output_dim: 384

search:
  candidate_multiplier: 4
  max_candidates_for_rerank: 50
  rerank:
    enabled: true
    provider: fastembed
    model: BAAI/bge-reranker-base
    top_k: 50
    timeout_ms: 1500
    fallback_to_rrf: true
```

### Exemplo: perfil remoto de qualidade

```yaml
search_quality_profile: remote-quality

embedding:
  provider: openai_compatible
  base_url: http://wqm-embeddings:8080
  model: BAAI/bge-base-en-v1.5
  output_dim: 768
  remote_batch_size: 128

search:
  candidate_multiplier: 4
  max_candidates_for_rerank: 50
  rerank:
    enabled: true
    provider: openai_compatible
    base_url: http://wqm-reranker:8080
    model: BAAI/bge-reranker-base
    top_k: 50
    timeout_ms: 1500
    fallback_to_rrf: true
```

### Exemplo: perfil multilingue

```yaml
search_quality_profile: multilingual

embedding:
  provider: openai_compatible
  base_url: http://wqm-embeddings:8080
  model: intfloat/multilingual-e5-base
  output_dim: 768

search:
  rerank:
    enabled: true
    provider: openai_compatible
    base_url: http://wqm-reranker:8080
    model: jinaai/jina-reranker-v2-base-multilingual
    top_k: 50
    fallback_to_rrf: true
```

### Exemplo: perfil GPU

```yaml
search_quality_profile: gpu-service

embedding:
  provider: openai_compatible
  base_url: http://wqm-gpu-embeddings:8080
  model: BAAI/bge-base-en-v1.5
  output_dim: 768
  remote_batch_size: 256

search:
  rerank:
    enabled: true
    provider: openai_compatible
    base_url: http://wqm-gpu-reranker:8080
    model: BAAI/bge-reranker-base
    top_k: 100
    timeout_ms: 2500
    fallback_to_rrf: true
```

### Exemplo: modo custom

```yaml
search_quality_profile: custom

embedding:
  provider: openai_compatible
  base_url: http://localhost:8080
  model: custom/code-embedding-model
  output_dim: 1024
  remote_batch_size: 64

search:
  candidate_multiplier: 6
  max_candidates_for_rerank: 100
  rerank:
    enabled: true
    provider: openai_compatible
    base_url: http://localhost:8081
    model: custom/code-reranker
    top_k: 100
    rrf_weight: 0.2
    rerank_weight: 0.8
    timeout_ms: 3000
    fallback_to_rrf: true
  diversity:
    enabled: true
    max_per_file: 3
    score_tier_threshold: 0.03
```

### Superficie para o usuario

O usuario nao deveria precisar editar todos os campos na mao para experimentar.
O produto pode oferecer comandos de alto nivel para listar, explicar e ativar
perfis.

Exemplo de CLI:

```powershell
wqm search profile list
wqm search profile explain stable-local
wqm search profile explain gpu-service
wqm search profile use local-quality --scope project
wqm search profile use stable-local --scope global
```

Exemplo de saida esperada:

```text
stable-local
  embedding: fastembed / AllMiniLM-L6-v2 / 384d
  rerank: disabled
  gpu: disabled
  best for: local privacy, low complexity, stable default

local-quality
  embedding: fastembed / AllMiniLM-L6-v2 / 384d
  rerank: fastembed / BAAI/bge-reranker-base / top 50
  gpu: optional
  best for: better final ranking without remote services
```

Exemplo de override por projeto:

```powershell
wqm search profile use multilingual --scope project --project C:\dev\meu-app
```

Exemplo de override em uma busca especifica:

```typescript
search({
  query: "como o watcher registra branches de agentes",
  mode: "hybrid",
  limit: 10,
  searchProfile: "local-quality",
  rerankTopK: 50
})
```

### Precedencia de configuracao

Para evitar surpresa, a precedencia deve ser explicita:

```text
request override
  > project profile
  > global user profile
  > daemon config
  > built-in default stable-local
```

### Validacao de configuracao

Validacoes minimas:

1. se `embedding.output_dim` muda, exigir plano de reembed;
2. se `rerank.enabled = true`, exigir `provider`, `model`, `top_k` e timeout;
3. se `provider = openai_compatible`, exigir `base_url` e endpoint saudavel;
4. se `gpu.mode != disabled`, exigir fallback ou falhar de forma explicita;
5. se `candidate_multiplier` ou `top_k` passam do limite, recusar config;
6. se perfil e override entram em conflito, mostrar a configuracao efetiva.

Exemplo:

```powershell
wqm search profile validate --profile remote-quality
wqm search profile effective --project C:\dev\meu-app
```

---

## Estado atual recomendado

Manter:

```yaml
embedding:
  provider: fastembed
  model: AllMiniLM-L6-v2
  output_dim: 384
```

Manter tambem a busca hibrida como default:

```typescript
search({
  query: "queue processing and retry logic",
  mode: "hybrid",
  limit: 10,
  scope: "project"
})
```

O comportamento esperado e:

1. gerar embedding denso da query;
2. gerar sparse vector BM25;
3. buscar candidatos densos e esparsos;
4. aplicar RRF;
5. retornar `limit` resultados.

---

## Fase 1: baseline de qualidade e performance

Antes de mexer no modelo, criar um conjunto pequeno e repetivel de consultas.
Esse conjunto deve cobrir perguntas reais de agentes usando os projetos
indexados.

### Consultas exemplo

```text
Como o daemon detecta mudanca de branch?
Onde a fila de ingestao aplica retry e backoff?
Como o tenant de rules project-scoped e resolvido?
Como o MCP server decide o projectId quando search nao recebe projectId?
Onde o Qdrant recebe upsert de chunks?
Como o watcher evita reindexar tudo?
Como a branch de agente e registrada no indexed-projects.json?
```

### Casos esperados

Para cada consulta, registrar manualmente:

```yaml
- query: "Como o daemon detecta mudanca de branch?"
  expected_files:
    - "src/rust/daemon/core/src/config/git*"
    - "src/rust/daemon/core/src/git*"
    - "docs/specs/19-branch-worktree-audit.md"
  expected_behavior:
    - "resultados devem mencionar branch atual"
    - "resultados devem separar projeto, branch e worktree"
```

### Metricas minimas

Medir:

1. latencia p50/p95 da busca;
2. top-1/top-3/top-10 com resultado util;
3. quantidade de resultados duplicados do mesmo arquivo;
4. casos em que BM25 salva a busca semantica;
5. casos em que semantica salva a busca literal.

### Exemplo de tabela de avaliacao

```markdown
| Query | Top-1 util | Top-3 util | Duplicacao | Latencia ms | Observacao |
|-------|------------|------------|------------|-------------|------------|
| branch detection | sim | sim | baixa | 85 | bom baseline |
| rules tenant | nao | sim | media | 92 | precisa rerank |
```

---

## Fase 2: overfetch controlado

Antes de adicionar reranker, ajustar a recuperacao para buscar mais candidatos do
que o usuario pediu.

Hoje o pipeline ja multiplica o `limit` por uma margem antes de buscar em cada
colecao. Para rerank, tornar isso explicito como configuracao.

### Configuracao proposta

```yaml
search:
  candidate_multiplier: 4
  max_candidates_for_rerank: 50
```

Esses campos devem aceitar override por perfil e tambem por chamada, quando a
tool expuser esse nivel de controle. Um usuario investigando uma refatoracao
grande pode pedir mais candidatos; um usuario fazendo lookup rapido pode manter
valores menores.

Exemplo de override por chamada:

```typescript
search({
  query: "how branch worktrees are registered",
  mode: "hybrid",
  limit: 10,
  candidateMultiplier: 6,
  maxCandidatesForRerank: 80
})
```

Exemplo de limites seguros:

```yaml
search:
  candidate_multiplier:
    default: 4
    min: 1
    max: 8
  max_candidates_for_rerank:
    default: 50
    min: 10
    max: 200
```

### Exemplo

Se o usuario pede:

```typescript
search({
  query: "tenant rules project scope",
  mode: "hybrid",
  limit: 10
})
```

O pipeline pode recuperar:

```text
dense candidates: 40
sparse candidates: 40
after RRF: up to 80 unique candidates
rerank input cap: 50
final output: 10
```

Esse passo sozinho ja ajuda porque da mais material para uma futura segunda
fase, sem mudar modelo ou dimensao.

---

## Fase 3: rerank de segunda fase

Adicionar uma etapa opcional depois do RRF e antes do corte final por `limit`.
Essa etapa deve ser pluggable: o mesmo contrato precisa aceitar rerank local,
rerank remoto, rerank desativado e futuras implementacoes especificas para codigo.

Fluxo desejado:

```text
query
  -> dense + sparse search
  -> RRF fusion
  -> candidate normalization
  -> rerank top K
  -> optional diversity pass
  -> final top N
```

### Formato do documento para rerank

O reranker nao deve receber apenas o chunk bruto. Para codigo, metadados ajudam
muito.

Exemplo de texto enviado ao reranker:

```text
path: src/typescript/mcp-server/src/tools/search-helpers.ts
title: search helper pipeline
component: mcp-server
branch: fork/fixes

content:
Resolve project context, generate embeddings, search all collections,
apply RRF fusion, expand context, and build final SearchResponse.
```

### Interface proposta

```typescript
interface RerankCandidate {
  id: string;
  collection: string;
  score: number;
  content: string;
  metadata: Record<string, unknown>;
}

interface RerankResult {
  id: string;
  rerankScore: number;
}

interface RerankProvider {
  rerank(query: string, candidates: RerankCandidate[]): Promise<RerankResult[]>;
}
```

### Providers de rerank

| Provider | Quando usar | Observacoes |
|----------|-------------|-------------|
| `disabled` | perfil estavel ou debug | retorna RRF puro |
| `fastembed` | privacidade local | bom primeiro passo, pode ser CPU-bound |
| `openai_compatible` | servico local/remoto | melhor para TEI, vLLM, LM Studio ou gateway proprio |
| `external_command` | experimentos locais | util para prototipos, deve ter timeout forte |
| `custom` | integracoes futuras | reservado para provider registrado por plugin/config |

Exemplo com rerank desligado explicitamente:

```yaml
search:
  rerank:
    enabled: false
    provider: disabled
```

Exemplo com provider local:

```yaml
search:
  rerank:
    enabled: true
    provider: fastembed
    model: BAAI/bge-reranker-base
    top_k: 30
    timeout_ms: 1500
    fallback_to_rrf: true
```

Exemplo com provider remoto:

```yaml
search:
  rerank:
    enabled: true
    provider: openai_compatible
    base_url: http://wqm-reranker:8080
    model: BAAI/bge-reranker-base
    top_k: 50
    timeout_ms: 1500
    fallback_to_rrf: true
```

Exemplo experimental via comando externo:

```yaml
search:
  rerank:
    enabled: true
    provider: external_command
    command: wqm-rerank-local
    args:
      - "--model"
      - "my-code-reranker"
    top_k: 20
    timeout_ms: 1000
    fallback_to_rrf: true
```

### Combinacao de score

Nao substituir completamente o score original no primeiro momento. Combinar RRF
e rerank para evitar regressao brusca:

```text
final_score = (0.25 * normalized_rrf_score) + (0.75 * normalized_rerank_score)
```

Depois de medir, ajustar pesos:

```yaml
search:
  rerank:
    enabled: true
    top_k: 50
    rrf_weight: 0.25
    rerank_weight: 0.75
```

### Fallback

Se o reranker falhar:

1. registrar `status = "uncertain"`;
2. incluir `status_reason`;
3. retornar os resultados RRF originais;
4. nao falhar a chamada inteira de `search`.

Exemplo:

```json
{
  "status": "uncertain",
  "status_reason": "rerank provider unavailable; returned RRF results"
}
```

---

## Fase 4: diversidade pos-rerank

Depois do rerank, aplicar uma regra leve para evitar concentrar todos os
resultados no mesmo arquivo, biblioteca ou tenant quando scores forem parecidos.

### Regra inicial

```yaml
search:
  diversity:
    enabled: true
    max_per_file: 3
    score_tier_threshold: 0.03
```

Tambem vale oferecer presets, porque diversidade demais pode atrapalhar quando o
usuario quer um arquivo especifico.

```yaml
search:
  diversity:
    preset: balanced
```

Presets sugeridos:

| Preset | Comportamento |
|--------|---------------|
| `off` | nao altera a ordem apos rerank/RRF |
| `light` | evita repeticao extrema, quase nao mexe no ranking |
| `balanced` | boa opcao default para exploracao |
| `strong` | privilegia cobertura de fontes diferentes |
| `custom` | usa `max_per_file`, `max_per_source` e `score_tier_threshold` |

### Exemplo

Entrada:

```text
1. src/tools/search.ts              score 0.91
2. src/tools/search.ts              score 0.90
3. src/tools/search.ts              score 0.89
4. src/tools/search-helpers.ts      score 0.88
5. docs/specs/08-api-reference.md   score 0.87
```

Saida desejada:

```text
1. src/tools/search.ts
2. src/tools/search-helpers.ts
3. docs/specs/08-api-reference.md
4. src/tools/search.ts
5. src/tools/search.ts
```

A diversidade nao deve derrubar um resultado muito melhor. Ela so deve mexer em
resultados dentro de uma mesma faixa de score.

---

## Fase 5: troca de modelo de embedding

Como o projeto esta no comeco, mudar dimensao e reindexar e aceitavel. Ainda
assim, essa mudanca deve ser feita depois da baseline e preferencialmente depois
do rerank, porque rerank tende a melhorar qualidade com menor impacto estrutural.

### Candidatos de modelo

Essa fase deve aceitar diferentes familias de modelo, nao apenas uma lista fixa.
O importante e validar dimensao, registrar o modelo ativo e exigir reembed quando
a dimensao mudar.

### Providers de embedding

| Provider | Quando usar | Exemplo |
|----------|-------------|---------|
| `fastembed` | default local e simples | `AllMiniLM-L6-v2` |
| `openai_compatible` | endpoint local/remoto | TEI, LM Studio, gateway proprio |
| `openai` | API OpenAI direta, se for separado no futuro | `text-embedding-3-small` |
| `custom` | provider futuro registrado por plugin/config | modelos internos |

Para corpus majoritariamente em ingles:

```yaml
embedding:
  provider: openai_compatible
  model: BAAI/bge-base-en-v1.5
  output_dim: 768
```

Para corpus misto com portugues, ingles e documentacao variada:

```yaml
embedding:
  provider: openai_compatible
  model: intfloat/multilingual-e5-base
  output_dim: 768
```

Para qualidade maior com custo maior:

```yaml
embedding:
  provider: openai_compatible
  model: BAAI/bge-large-en-v1.5
  output_dim: 1024
```

### Impacto esperado

| Mudanca | Beneficio | Custo |
|---------|-----------|-------|
| 384d -> 768d | melhor recall semantico | mais memoria e reindexacao |
| 384d -> 1024d | melhor qualidade em alguns dominios | maior latencia e storage |
| rerank top 50 | melhor precisao final | inferencia extra por query |
| GPU | maior throughput | mais complexidade operacional |

### Checklist de troca

1. parar ingestao automatica;
2. atualizar config de embedding e dimensao;
3. recriar ou reembedar colecoes;
4. rodar baseline novamente;
5. comparar tabela antes/depois;
6. manter rollback documentado.

Exemplo de operacao:

```powershell
make -f Makefile.win service-stabilize
wqm admin reembed --confirm
wqm status health
```

---

## Fase 6: GPU como aceleracao opcional

GPU passa a valer a pena quando pelo menos uma destas condicoes for verdadeira:

1. reindexacao inicial demora o suficiente para atrapalhar o fluxo diario;
2. bibliotecas grandes sao adicionadas com frequencia;
3. rerank top 50 aumenta latencia perceptivelmente;
4. o modelo escolhido e grande demais para CPU confortavel;
5. varios agentes consultam o MCP ao mesmo tempo.

### Caminho preferencial

Usar um servico externo OpenAI-compatible para embedding/rerank, mantendo o
daemon e o MCP simples.

Exemplo conceitual:

```yaml
embedding:
  provider: openai_compatible
  base_url: http://wqm-embeddings:8080
  model: BAAI/bge-base-en-v1.5
  remote_batch_size: 128
  output_dim: 768
  api_key_env_var: WQM_LOCAL_EMBEDDING_API_KEY
```

Vantagens:

1. troca de backend sem reescrever o pipeline;
2. GPU isolada em container proprio;
3. possibilidade de subir embedding e rerank separadamente;
4. rollback simples para FastEmbed local.

### Modos de GPU configuraveis

```yaml
gpu:
  mode: disabled
```

Modos sugeridos:

| Modo | Descricao |
|------|-----------|
| `disabled` | nenhum servico GPU |
| `embedding-only` | GPU apenas para ingestao/reembed/query embedding |
| `rerank-only` | GPU apenas para segunda fase de busca |
| `embedding-and-rerank` | GPU para os dois caminhos |
| `auto` | seleciona GPU quando endpoint estiver saudavel, senao volta para CPU |

Exemplo de GPU apenas para rerank:

```yaml
gpu:
  mode: rerank-only

embedding:
  provider: fastembed
  model: AllMiniLM-L6-v2
  output_dim: 384

search:
  rerank:
    enabled: true
    provider: openai_compatible
    base_url: http://wqm-gpu-reranker:8080
    model: BAAI/bge-reranker-base
    top_k: 50
    fallback_to_rrf: true
```

Exemplo de modo automatico:

```yaml
gpu:
  mode: auto
  health_probe_secs: 30
  fallback_profile: stable-local
```

### Caminho alternativo

Compilar FastEmbed com suporte CUDA/CuDNN. Esse caminho tende a ser mais
sensivel ao ambiente local e deve ficar para quando o container remoto nao
atender.

---

## Fase 7: observabilidade especifica de busca

Adicionar metricas separadas para entender onde a busca melhora ou degrada.

### Metricas propostas

```text
search_dense_ms
search_sparse_ms
search_rrf_ms
search_rerank_ms
search_total_ms
search_candidates_before_rerank
search_candidates_after_dedup
search_rerank_fallback_total
search_results_distinct_files
```

### Evento de busca enriquecido

```json
{
  "query": "tenant rules project scope",
  "mode": "hybrid",
  "limit": 10,
  "dense_ms": 21,
  "sparse_ms": 9,
  "rrf_ms": 1,
  "rerank_ms": 42,
  "candidate_count": 50,
  "final_count": 10,
  "distinct_files": 6,
  "rerank_enabled": true
}
```

---

## Criterios de decisao

### Manter 384d sem rerank

Adequado se:

1. top-3 util fica acima de 80% nas consultas reais;
2. agentes conseguem achar arquivos corretos sem muitas tentativas;
3. latencia atual e mais importante que precisao;
4. corpus ainda e pequeno.

### Adicionar rerank

Adequado se:

1. resultado correto aparece no top-10, mas nao no top-3;
2. ha muitos chunks parecidos competindo;
3. queries conceituais retornam arquivos certos em ordem ruim;
4. usuarios reclamam de "quase achou".

### Trocar embedding

Adequado se:

1. resultado correto nao aparece nem no top-20;
2. sinonimos e queries em linguagem natural falham;
3. corpus tem varios idiomas;
4. documentacao e codigo usam vocabularios muito diferentes.

### Usar GPU

Adequado se:

1. ingestao/reembed e gargalo real;
2. rerank torna busca lenta;
3. modelo maior e necessario;
4. ha concorrencia de muitos agentes.

---

## Sequencia recomendada

1. **Agora:** manter `AllMiniLM-L6-v2` 384d e criar baseline.
2. **Proximo passo tecnico:** introduzir perfis configuraveis e validacao de config.
3. **Depois:** adicionar overfetch configuravel e metricas de candidatos.
4. **Depois:** implementar rerank opcional top 50 com fallback para RRF.
5. **Depois:** aplicar diversidade leve por arquivo/origem.
6. **Depois:** testar 768d em branch separada com reindexacao limpa.
7. **Por ultimo:** mover embedding/rerank para GPU se metricas justificarem.

---

## Exemplos de configuracao futura completa

### Stable local

```yaml
search_quality_profile: stable-local

embedding:
  provider: fastembed
  model: AllMiniLM-L6-v2
  output_dim: 384

search:
  candidate_multiplier: 2
  max_candidates_for_rerank: 20
  rerank:
    enabled: false
    provider: disabled
  diversity:
    enabled: false
```

### Local quality

```yaml
search_quality_profile: local-quality

embedding:
  provider: fastembed
  model: AllMiniLM-L6-v2
  output_dim: 384

search:
  candidate_multiplier: 4
  max_candidates_for_rerank: 50
  rerank:
    enabled: true
    provider: fastembed
    model: BAAI/bge-reranker-base
    top_k: 50
    rrf_weight: 0.25
    rerank_weight: 0.75
    timeout_ms: 1500
    fallback_to_rrf: true
  diversity:
    enabled: true
    preset: balanced
```

### Remote quality

```yaml
search_quality_profile: remote-quality

embedding:
  provider: openai_compatible
  base_url: http://wqm-embeddings:8080
  model: BAAI/bge-base-en-v1.5
  output_dim: 768
  remote_batch_size: 128
  health_probe_cache_secs: 60

search:
  candidate_multiplier: 4
  max_candidates_for_rerank: 50
  rerank:
    enabled: true
    provider: openai_compatible
    base_url: http://wqm-reranker:8080
    model: BAAI/bge-reranker-base
    top_k: 50
    rrf_weight: 0.25
    rerank_weight: 0.75
    timeout_ms: 1500
    fallback_to_rrf: true
  diversity:
    enabled: true
    preset: balanced
```

### GPU service

```yaml
search_quality_profile: gpu-service

gpu:
  mode: embedding-and-rerank
  fallback_profile: stable-local

embedding:
  provider: openai_compatible
  base_url: http://wqm-gpu-embeddings:8080
  model: BAAI/bge-base-en-v1.5
  output_dim: 768
  remote_batch_size: 256
  health_probe_cache_secs: 60

search:
  candidate_multiplier: 4
  max_candidates_for_rerank: 100
  rerank:
    enabled: true
    provider: openai_compatible
    base_url: http://wqm-gpu-reranker:8080
    model: BAAI/bge-reranker-base
    top_k: 100
    rrf_weight: 0.25
    rerank_weight: 0.75
    timeout_ms: 2500
    fallback_to_rrf: true
  diversity:
    enabled: true
    preset: balanced
```

Essas configuracoes nao devem ser aplicadas agora. Elas representam alvos de
desenho para quando a baseline indicar que o ganho compensa a complexidade e
para que o produto possa oferecer opcoes ao usuario sem exigir edicao manual de
todos os detalhes.

---

## Resultado esperado

Com esse plano, o projeto fica com uma trajetoria clara:

1. preservar estabilidade agora;
2. medir o comportamento real;
3. melhorar ranking sem reindexar;
4. trocar dimensao quando fizer sentido;
5. usar GPU como aceleracao depois que o gargalo estiver comprovado.

O ponto mais importante: nao tratar embedding, rerank e GPU como uma unica
decisao. Cada um resolve um problema diferente, e medir separadamente evita
otimizacao prematura.
