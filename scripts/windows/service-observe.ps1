[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$ProjectDir = (Get-Location).Path,
  [string]$QdrantUrl = "http://localhost:6333",
  [string]$DaemonEndpoint = "localhost:50051",
  [string]$LogDir = ".wqm-fork/logs",
  [int]$IntervalSeconds = 30,
  [switch]$Once
)

$ErrorActionPreference = "Continue"

function Resolve-LogDir {
  param([string]$Base, [string]$PathValue)
  if ([System.IO.Path]::IsPathRooted($PathValue)) { return $PathValue }
  return (Join-Path $Base $PathValue)
}

function Invoke-Captured {
  param(
    [string]$File,
    [string[]]$Args = @(),
    [string]$WorkingDirectory = $RepoDir,
    [int]$TimeoutSeconds = 15
  )
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $outFile = [System.IO.Path]::GetTempFileName()
  $errFile = [System.IO.Path]::GetTempFileName()
  try {
    $p = Start-Process -FilePath $File -ArgumentList $Args -WorkingDirectory $WorkingDirectory -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    if (-not $p.WaitForExit($TimeoutSeconds * 1000)) {
      try { $p.Kill() } catch {}
      return @{ ok = $false; exitCode = -1; stdout = ""; stderr = "timeout after ${TimeoutSeconds}s"; durationMs = $sw.ElapsedMilliseconds }
    }
    return @{
      ok = ($p.ExitCode -eq 0)
      exitCode = $p.ExitCode
      stdout = (Get-Content -Raw -Path $outFile -ErrorAction SilentlyContinue)
      stderr = (Get-Content -Raw -Path $errFile -ErrorAction SilentlyContinue)
      durationMs = $sw.ElapsedMilliseconds
    }
  } catch {
    return @{ ok = $false; exitCode = -1; stdout = ""; stderr = $_.Exception.Message; durationMs = $sw.ElapsedMilliseconds }
  } finally {
    Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue
  }
}

function Test-TcpEndpoint {
  param([string]$Endpoint)
  $clean = $Endpoint -replace '^https?://',''
  $parts = $clean.Split(':')
  $host = $parts[0]
  $port = if ($parts.Length -gt 1) { [int]$parts[1] } else { 50051 }
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $iar = $client.BeginConnect($host, $port, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(2000, $false)
    if ($ok) { $client.EndConnect($iar) }
    return @{ ok = [bool]$ok; host = $host; port = $port }
  } catch {
    return @{ ok = $false; host = $host; port = $port; error = $_.Exception.Message }
  } finally {
    try { $client.Close() } catch {}
  }
}

function Test-Qdrant {
  param([string]$Url)
  try {
    $resp = Invoke-WebRequest -Uri ($Url.TrimEnd('/') + '/collections') -UseBasicParsing -TimeoutSec 5
    return @{ ok = ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300); statusCode = $resp.StatusCode; bodyPrefix = $resp.Content.Substring(0, [Math]::Min(300, $resp.Content.Length)) }
  } catch {
    return @{ ok = $false; error = $_.Exception.Message }
  }
}

function Get-GitInfo {
  if (-not (Test-Path (Join-Path $RepoDir '.git'))) { return @{ ok = $false; error = 'not a git repo' } }
  $branch = Invoke-Captured git @('branch','--show-current') $RepoDir 5
  $status = Invoke-Captured git @('status','--short','--branch') $RepoDir 5
  $head = Invoke-Captured git @('log','-1','--oneline') $RepoDir 5
  return @{ ok = $branch.ok; branch = $branch.stdout.Trim(); status = $status.stdout.Trim(); head = $head.stdout.Trim() }
}

function New-Snapshot {
  $wqmHealth = Invoke-Captured wqm @('status','health') $RepoDir 20
  $queueStats = Invoke-Captured wqm @('queue','stats') $RepoDir 20
  $projectStatus = Invoke-Captured wqm @('project','status', $ProjectDir) $ProjectDir 20
  $projectList = Invoke-Captured wqm @('project','list') $RepoDir 20
  $watchList = Invoke-Captured wqm @('project','watch','list','--json') $ProjectDir 20

  $nodeProc = Get-Process -Name node -ErrorAction SilentlyContinue | Select-Object -First 10 Id,ProcessName,CPU,StartTime,Path
  $memexProc = Get-Process -Name memexd -ErrorAction SilentlyContinue | Select-Object -First 10 Id,ProcessName,CPU,StartTime,Path

  return [ordered]@{
    timestamp = (Get-Date).ToString('o')
    repoDir = $RepoDir
    projectDir = $ProjectDir
    git = Get-GitInfo
    qdrant = Test-Qdrant $QdrantUrl
    daemonTcp = Test-TcpEndpoint $DaemonEndpoint
    wqm = @{ health = $wqmHealth; queueStats = $queueStats; projectStatus = $projectStatus; projectList = $projectList; watchList = $watchList }
    processes = @{ node = $nodeProc; memexd = $memexProc }
  }
}

$ResolvedLogDir = Resolve-LogDir -Base $RepoDir -PathValue $LogDir
New-Item -ItemType Directory -Force -Path $ResolvedLogDir | Out-Null
$logFile = Join-Path $ResolvedLogDir ("service-observe-" + (Get-Date -Format 'yyyyMMdd') + ".jsonl")

while ($true) {
  $snap = New-Snapshot
  $json = $snap | ConvertTo-Json -Depth 12 -Compress
  Add-Content -Path $logFile -Value $json

  $q = if ($snap.qdrant.ok) { 'ok' } else { 'fail' }
  $d = if ($snap.daemonTcp.ok) { 'ok' } else { 'fail' }
  $h = if ($snap.wqm.health.ok) { 'ok' } else { 'fail' }
  Write-Host ("[{0}] qdrant={1} daemonTcp={2} wqmHealth={3} log={4}" -f (Get-Date -Format 'HH:mm:ss'), $q, $d, $h, $logFile)

  if ($Once) { break }
  Start-Sleep -Seconds $IntervalSeconds
}
