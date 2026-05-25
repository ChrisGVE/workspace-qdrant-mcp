[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$CacheDir = ".fastembed_cache",
  [string]$ConfigPath = ".memexd-fastembed.yaml",
  [string]$QdrantUrl = "http://127.0.0.1:6334",
  [int]$GrpcPort = 55151,
  [int]$ControlPort = 7798,
  [int]$MetricsPort = 9091,
  [string]$LogLevel = "info",
  [string]$LogDir = ".wqm-fork/logs"
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

  throw "Não encontrei memexd. Compile com build-rust ou instale os binários."
}

function Resolve-LocalPath {
  param(
    [string]$BaseDir,
    [string]$Value
  )

  if ([System.IO.Path]::IsPathRooted($Value)) {
    return $Value
  }

  return (Join-Path $BaseDir $Value)
}

function Escape-CmdValue {
  param([string]$Value)
  return $Value.Replace('"', '""')
}

function Test-TcpPort {
  param(
    [string]$HostName,
    [int]$Port,
    [int]$TimeoutMs = 1200
  )

  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect($HostName, $Port, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
    if (-not $ok) {
      $client.Close()
      return $false
    }
    $client.EndConnect($iar)
    $client.Close()
    return $true
  } catch {
    return $false
  }
}

function Test-HealthyDaemon {
  param(
    [string]$QdrantUrl,
    [int]$GrpcPort
  )

  $wqm = Get-Command wqm -ErrorAction SilentlyContinue
  if (-not $wqm) {
    return $null
  }

  $previousDaemonAddr = $env:WQM_DAEMON_ADDR
  $previousQdrantUrl = $env:WQM_QDRANT_URL
  $previousQdrantLegacyUrl = $env:QDRANT_URL
  try {
    $env:WQM_DAEMON_ADDR = "http://127.0.0.1:$GrpcPort"
    $env:WQM_QDRANT_URL = $QdrantUrl
    $env:QDRANT_URL = $QdrantUrl

    $probeOutput = & wqm status health 2>&1 | Out-String
    $healthy = ($LASTEXITCODE -eq 0)

    return [pscustomobject]@{
      healthy = $healthy
      output = $probeOutput.Trim()
    }
  } finally {
    if ($null -eq $previousDaemonAddr) {
      Remove-Item Env:WQM_DAEMON_ADDR -ErrorAction SilentlyContinue
    } else {
      $env:WQM_DAEMON_ADDR = $previousDaemonAddr
    }

    if ($null -eq $previousQdrantUrl) {
      Remove-Item Env:WQM_QDRANT_URL -ErrorAction SilentlyContinue
    } else {
      $env:WQM_QDRANT_URL = $previousQdrantUrl
    }

    if ($null -eq $previousQdrantLegacyUrl) {
      Remove-Item Env:QDRANT_URL -ErrorAction SilentlyContinue
    } else {
      $env:QDRANT_URL = $previousQdrantLegacyUrl
    }
  }
}

$ResolvedCacheDir = Resolve-LocalPath -BaseDir $RepoDir -Value $CacheDir
New-Item -ItemType Directory -Force -Path $ResolvedCacheDir | Out-Null
$ResolvedLogDir = Resolve-LocalPath -BaseDir $RepoDir -Value $LogDir
New-Item -ItemType Directory -Force -Path $ResolvedLogDir | Out-Null
$ResolvedConfigPath = Resolve-LocalPath -BaseDir $RepoDir -Value $ConfigPath
$HasConfig = Test-Path $ResolvedConfigPath

if (Test-TcpPort -HostName "127.0.0.1" -Port $GrpcPort -TimeoutMs 800) {
  $daemonHealth = Test-HealthyDaemon -QdrantUrl $QdrantUrl -GrpcPort $GrpcPort
  if ($null -ne $daemonHealth -and $daemonHealth.healthy) {
    Write-Host "memexd já responde em 127.0.0.1:$GrpcPort."
    exit 0
  }

  Write-Warning "A porta 127.0.0.1:$GrpcPort já está ocupada, mas o health check do daemon falhou."
  Write-Warning "Vou tentar iniciar memexd mesmo assim para evitar aceitar um processo incompatível."
  if ($daemonHealth -and $daemonHealth.output) {
    Write-Host $daemonHealth.output
  }
}

$Memexd = Resolve-MemexdPath -BaseDir $RepoDir
$LaunchCommandParts = @(
  'set "HF_HOME=' + (Escape-CmdValue $ResolvedCacheDir) + '"',
  'set "WQM_EMBEDDING_PROVIDER=fastembed"',
  'set "WQM_EMBEDDING_MODEL_CACHE_DIR=' + (Escape-CmdValue $ResolvedCacheDir) + '"',
  'set "QDRANT_URL=' + (Escape-CmdValue $QdrantUrl) + '"',
  'set "WORKSPACE_QDRANT_QDRANT__URL=' + (Escape-CmdValue $QdrantUrl) + '"',
  'set "WORKSPACE_QDRANT_QDRANT__TRANSPORT=grpc"'
)
if ($HasConfig) {
  $LaunchCommandParts += 'call "' + (Escape-CmdValue $Memexd) + '" -c "' + (Escape-CmdValue $ResolvedConfigPath) + '" --grpc-port ' + $GrpcPort + ' --control-port ' + $ControlPort + ' --metrics-port ' + $MetricsPort + ' --foreground --log-level ' + $LogLevel
} else {
  $LaunchCommandParts += 'call "' + (Escape-CmdValue $Memexd) + '" --grpc-port ' + $GrpcPort + ' --control-port ' + $ControlPort + ' --metrics-port ' + $MetricsPort + ' --foreground --log-level ' + $LogLevel
}
$LaunchCommand = $LaunchCommandParts -join " && "

if ($HasConfig) {
  Write-Host "Usando config FastEmbed: $ResolvedConfigPath"
}
Write-Host "Iniciando memexd FastEmbed em gRPC na porta $GrpcPort..."
$StdOutLog = Join-Path $ResolvedLogDir "memexd-fastembed-start.out.log"
$StdErrLog = Join-Path $ResolvedLogDir "memexd-fastembed-start.err.log"
$Process = Start-Process -FilePath cmd.exe -ArgumentList @('/c', $LaunchCommand) -WorkingDirectory $RepoDir -PassThru -RedirectStandardOutput $StdOutLog -RedirectStandardError $StdErrLog

Write-Host "Aguardando memexd responder em 127.0.0.1:$GrpcPort..."
for ($i = 0; $i -lt 30; $i++) {
  if (Test-TcpPort -HostName "127.0.0.1" -Port $GrpcPort -TimeoutMs 1000) {
    Write-Host "memexd OK: 127.0.0.1:$GrpcPort"
    exit 0
  }
  if ($Process.HasExited) {
    $stdout = if (Test-Path $StdOutLog) { Get-Content -Raw $StdOutLog } else { "" }
    $stderr = if (Test-Path $StdErrLog) { Get-Content -Raw $StdErrLog } else { "" }
    throw "memexd encerrou antes de abrir a porta $GrpcPort. stdout: $stdout stderr: $stderr"
  }
  Start-Sleep -Seconds 1
}

throw "memexd não respondeu em 127.0.0.1:$GrpcPort. Veja $StdOutLog e $StdErrLog"
