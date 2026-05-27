[CmdletBinding()]
param(
  [string]$ContainerName = "wqm-qdrant",
  [string]$VolumeName = "qdrant_storage",
  [string]$QdrantImage = "qdrant/qdrant",
  [string]$QdrantUrl = "http://localhost:6333",
  [int]$HttpPort = 6333,
  [int]$GrpcPort = 6334,
  [switch]$EnsureGrpc
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "docker não encontrado no PATH"
}

function Test-ContainerGrpcPort {
  param(
    [string]$Name,
    [int]$Port
  )

  try {
    $bindings = docker port $Name 2>$null
    return $null -ne ($bindings | Where-Object { $_ -match "^\s*$Port/tcp\b" })
  } catch {
    return $false
  }
}

$needsRecreate = $false
$exists = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }
if ($exists) {
  $running = docker ps --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }
  if ($EnsureGrpc -and -not (Test-ContainerGrpcPort -Name $ContainerName -Port $GrpcPort)) {
    Write-Host "Container $ContainerName existe sem a porta gRPC $GrpcPort; recriando."
    docker rm -f $ContainerName | Out-Null
    $needsRecreate = $true
  } elseif ($running) {
    Write-Host "Container $ContainerName já está rodando."
  } else {
    docker start $ContainerName | Out-Null
    Write-Host "Container $ContainerName iniciado."
  }
}

if (-not $exists -or $needsRecreate) {
  docker volume create $VolumeName | Out-Null
  docker run -d --name $ContainerName -p "${HttpPort}:6333" -p "${GrpcPort}:6334" -v "${VolumeName}:/qdrant/storage" $QdrantImage | Out-Null
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
