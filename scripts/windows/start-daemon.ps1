[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

if (Get-Command wqm -ErrorAction SilentlyContinue) {
  Write-Host "Iniciando serviço via wqm..."
  wqm service start
  wqm status health
  exit 0
}

$memexd = Get-Command memexd -ErrorAction SilentlyContinue
if ($memexd) {
  Write-Host "Iniciando memexd do PATH em nova janela..."
  Start-Process -FilePath $memexd.Source
  Start-Sleep -Seconds 2
  exit 0
}

$localMemexd = Join-Path $RepoDir "src\rust\target\release\memexd.exe"
if (Test-Path $localMemexd) {
  Write-Host "Iniciando memexd local em nova janela..."
  Start-Process -FilePath $localMemexd
  Start-Sleep -Seconds 2
  exit 0
}

throw "Não encontrei wqm nem memexd. Instale os binários ou rode build-rust."
