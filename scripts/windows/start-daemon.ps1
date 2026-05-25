[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

function Resolve-MemexdPath {
  param([string]$BaseDir)

  $candidates = @(
    (Join-Path $BaseDir "src\rust\target\debug\memexd.exe"),
    (Join-Path $BaseDir "src\rust\target\release\memexd.exe")
  )

  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }

  $memexd = Get-Command memexd -ErrorAction SilentlyContinue
  if ($memexd) {
    return $memexd.Source
  }

  return $null
}

if (Get-Command wqm -ErrorAction SilentlyContinue) {
  Write-Host "Iniciando serviço via wqm..."
  wqm service start
  wqm status health
  exit 0
}

$memexd = Resolve-MemexdPath -BaseDir $RepoDir
if ($memexd) {
  Write-Host "Iniciando memexd local em nova janela..."
  Start-Process -FilePath $memexd
  Start-Sleep -Seconds 2
  exit 0
}

throw "Não encontrei wqm nem memexd. Instale os binários ou rode build-rust."
