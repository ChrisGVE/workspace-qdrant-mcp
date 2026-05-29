#Requires -Version 5.1
<#
.SYNOPSIS
    Sobe o stack Docker do workspace-qdrant-mcp.

.DESCRIPTION
    Aguarda o Docker Desktop ficar disponível e executa docker compose up -d.
    Usa docker/\.env como env file (o .env real nao fica na raiz por estar no .gitignore).

    Use tambem via Makefile: make up | make build | make down

.PARAMETER Build
    Rebuilda as imagens locais (memexd) antes de subir. Use após atualizar o branch.

.PARAMETER Timeout
    Segundos máximos aguardando Docker ficar pronto (default: 120).
#>
param(
    [switch]$Build,
    [int]$Timeout = 120
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"
$EnvFile     = Join-Path $ProjectRoot "docker\.env"

if (-not (Test-Path $EnvFile)) {
    Write-Error "Arquivo de ambiente nao encontrado: $EnvFile`nCopie docker\.env.example para docker\.env e preencha os valores."
    exit 1
}

# Aguarda Docker Desktop estar pronto
Write-Host "Aguardando Docker Desktop..." -ForegroundColor Cyan
$elapsed = 0
while ($elapsed -lt $Timeout) {
    $info = docker info 2>$null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 5
    $elapsed += 5
    Write-Host "  ${elapsed}s / ${Timeout}s..."
}

if ($elapsed -ge $Timeout) {
    Write-Error "Docker nao ficou disponivel em ${Timeout}s. Verifique se o Docker Desktop esta rodando."
    exit 1
}

Write-Host "Docker pronto." -ForegroundColor Green

$composeArgs = @(
    "compose",
    "-f", $ComposeFile,
    "--env-file", $EnvFile,
    "up", "-d"
)

if ($Build) {
    $composeArgs += "--build"
    Write-Host "Modo --build ativado (rebuilding imagens locais)..." -ForegroundColor Yellow
}

Write-Host "Subindo stack workspace-qdrant-mcp..." -ForegroundColor Cyan
& docker @composeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "docker compose up falhou com codigo $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Stack no ar:" -ForegroundColor Green
docker compose -f $ComposeFile --env-file $EnvFile ps
