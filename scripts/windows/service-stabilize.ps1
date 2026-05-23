[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$ProjectDir = (Get-Location).Path,
  [string]$QdrantUrl = "http://localhost:6333",
  [string]$DaemonEndpoint = "localhost:50051",
  [string]$LogDir = ".wqm-fork/logs"
)

$ErrorActionPreference = "Continue"

function Invoke-Step {
  param([string]$Name, [scriptblock]$Body)
  Write-Host "==> $Name"
  try {
    & $Body
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { Write-Host "Aviso: $Name retornou codigo $LASTEXITCODE" -ForegroundColor Yellow }
  } catch {
    Write-Host "Aviso: $Name falhou: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

Push-Location $RepoDir
try {
  Invoke-Step "snapshot inicial" { .\scripts\windows\service-observe.ps1 -RepoDir $RepoDir -ProjectDir $ProjectDir -QdrantUrl $QdrantUrl -DaemonEndpoint $DaemonEndpoint -LogDir $LogDir -Once }
  Invoke-Step "subir Qdrant se necessario" { .\scripts\windows\start-qdrant.ps1 -QdrantUrl $QdrantUrl }
  Invoke-Step "iniciar daemon se necessario" { .\scripts\windows\start-daemon.ps1 -RepoDir $RepoDir }
  Start-Sleep -Seconds 3
  Invoke-Step "snapshot final" { .\scripts\windows\service-observe.ps1 -RepoDir $RepoDir -ProjectDir $ProjectDir -QdrantUrl $QdrantUrl -DaemonEndpoint $DaemonEndpoint -LogDir $LogDir -Once }
  Write-Host "service-stabilize concluido. Veja logs em $LogDir"
} finally {
  Pop-Location
}
