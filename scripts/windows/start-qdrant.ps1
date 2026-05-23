[CmdletBinding()]
param(
  [string]$ContainerName = "wqm-qdrant",
  [string]$VolumeName = "qdrant_storage",
  [string]$QdrantImage = "qdrant/qdrant",
  [string]$QdrantUrl = "http://localhost:6333"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "docker não encontrado no PATH"
}

$exists = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }
if ($exists) {
  $running = docker ps --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }
  if ($running) {
    Write-Host "Container $ContainerName já está rodando."
  } else {
    docker start $ContainerName | Out-Null
    Write-Host "Container $ContainerName iniciado."
  }
} else {
  docker volume create $VolumeName | Out-Null
  docker run -d --name $ContainerName -p 6333:6333 -v "${VolumeName}:/qdrant/storage" $QdrantImage | Out-Null
  Write-Host "Container $ContainerName criado e iniciado."
}

Write-Host "Aguardando Qdrant responder..."
for ($i = 0; $i -lt 30; $i++) {
  try {
    $resp = Invoke-WebRequest -Uri "$QdrantUrl/collections" -UseBasicParsing -TimeoutSec 2
    if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
      Write-Host "Qdrant OK: $QdrantUrl"
      exit 0
    }
  } catch {
    Start-Sleep -Seconds 1
  }
}
throw "Qdrant não respondeu em $QdrantUrl"
