# 05 - Operação em projetos

## Antes de registrar um projeto

Crie ou revise `.wqmignore` na raiz do projeto.

Sugestão inicial:

```gitignore
node_modules/
dist/
build/
target/
.venv/
venv/
.env/
.cache/
tmp/
coverage/
.next/
.turbo/
data/raw/
datasets/
*.log
*.sqlite
*.db
```

Ajuste para não indexar:

- secrets;
- dumps;
- dados grandes;
- vendors;
- artefatos gerados;
- bases locais.

## Registrar projeto

```powershell
make -f Makefile.win register PROJECT=C:\dev\meu-projeto
```

ou:

```powershell
wqm project register C:\dev\meu-projeto --yes
wqm project status C:\dev\meu-projeto
wqm project check C:\dev\meu-projeto --verbose
```

## Validar busca

Dentro do projeto:

```powershell
wqm search project "nome de uma função conhecida" -n 5
wqm project list
wqm project check --verbose
```

## Claude/Codex

Depois de aplicar config:

```powershell
make -f Makefile.win apply-config REPO=C:\dev\workspace-qdrant-mcp
```

Esse comando gera o padrão Docker do fork: Claude Desktop continua apontando para o daemon em `http://localhost:50051`, e o Codex usa o MCP HTTP do stack em `http://localhost:6335/mcp`. Se você realmente precisar do fluxo local FastEmbed, use `apply-config-fastembed`.

No prompt do agente, use uma instrução parecida com:

```text
Antes de explorar arquivos manualmente, use workspace-qdrant search/grep/list quando a pergunta depender da estrutura ou histórico deste projeto.
```

## Uso de coleções

- `projects`: código e docs dos projetos registrados.
- `libraries`: documentação externa e referências.
- `scratchpad`: notas de investigação.
- `rules`: preferências persistentes.

## Rotina semanal

```powershell
cd C:\dev\workspace-qdrant-mcp
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git push origin main
make -f Makefile.win doctor
```

Em cada projeto ativo:

```powershell
wqm project status C:\dev\projeto
wqm project check C:\dev\projeto --verbose
```

## Quando pausar watchers

Durante refactors massivos, renames em lote ou geração de arquivos:

```powershell
wqm project watch pause
# faça o refactor
wqm project watch resume
wqm project check --verbose
```
