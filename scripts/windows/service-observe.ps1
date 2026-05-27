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

function Resolve-CommandOnPath([string]$CommandName) {
  if (-not $CommandName) { return $null }

  if ([System.IO.Path]::IsPathRooted($CommandName)) {
    if (Test-Path -LiteralPath $CommandName) {
      return [System.IO.Path]::GetFullPath($CommandName)
    }
    return $null
  }

  $searchNames = @($CommandName)
  if (-not [System.IO.Path]::HasExtension($CommandName)) {
    $searchNames += @(
      "$CommandName.exe",
      "$CommandName.cmd",
      "$CommandName.bat"
    )
  }

  if ($env:PATH) {
    foreach ($entry in @($env:PATH -split ';' | Where-Object { $_ })) {
      foreach ($name in $searchNames) {
        $candidate = Join-Path $entry $name
        if (Test-Path -LiteralPath $candidate) {
          return [System.IO.Path]::GetFullPath($candidate)
        }
      }
    }
  }

  $cmd = Get-Command $CommandName -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($cmd) {
    foreach ($property in @('Source', 'Path', 'Definition')) {
      $value = $cmd.$property
      if ($value) {
        if ([System.IO.Path]::IsPathRooted($value)) {
          return [System.IO.Path]::GetFullPath($value)
        }
        $resolved = Resolve-Path -LiteralPath $value -ErrorAction SilentlyContinue
        if ($resolved) {
          return [System.IO.Path]::GetFullPath($resolved.Path)
        }
        return $value
      }
    }
  }

  return $null
}

function Resolve-WqmPath {
  param([string]$BaseDir)

  foreach ($envName in @('WQM_PATH', 'WQM_EXECUTABLE')) {
    $configured = [Environment]::GetEnvironmentVariable($envName)
    if ($configured) {
      $resolvedConfigured = Resolve-CommandOnPath $configured
      if ($resolvedConfigured) {
        return $resolvedConfigured
      }
    }
  }

  $candidates = @(
    (Join-Path $BaseDir "src\rust\target\debug\wqm.exe"),
    (Join-Path $BaseDir "src\rust\target\release\wqm.exe")
  )

  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }

  foreach ($name in @('wqm','wqm.exe','wqm.cmd','wqm.bat')) {
    $resolved = Resolve-CommandOnPath $name
    if ($resolved) {
      return $resolved
    }
  }

  return $null
}

function Truncate-Text {
  param(
    [AllowNull()][string]$Value,
    [int]$MaxChars = 2000
  )

  if ([string]::IsNullOrEmpty($Value)) {
    return $Value
  }

  if ($Value.Length -le $MaxChars) {
    return $Value
  }

  return $Value.Substring(0, $MaxChars) + "...<truncated>"
}

function Resolve-HttpEndpoint {
  param([string]$Endpoint)

  if ($Endpoint -match '^https?://') {
    return $Endpoint
  }

  return "http://$Endpoint"
}

function Resolve-DockerDatabasePath {
  param([string]$BaseDir)

  $candidates = @(
    (Join-Path $BaseDir "state\memexd\memexd.db"),
    (Join-Path $BaseDir "state\memexd\state.db"),
    (Join-Path $BaseDir "state\state.db")
  )

  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }

  return (Join-Path $BaseDir "state\memexd\memexd.db")
}

function Trace-Observe {
  param([string]$Message)
  if ($env:WQM_OBSERVE_DEBUG) {
    Write-Host $Message
  }
}

function Invoke-Captured {
  param(
    [string]$File,
    [string[]]$CommandArgs = @(),
    [string]$WorkingDirectory = $RepoDir,
    [int]$TimeoutSeconds = 15
  )
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $outFile = [System.IO.Path]::GetTempFileName()
  $errFile = [System.IO.Path]::GetTempFileName()
  $previousConsoleEncoding = [Console]::OutputEncoding
  $previousOutputEncoding = $OutputEncoding
  $utf8Encoding = [System.Text.UTF8Encoding]::new($false)
  try {
    [Console]::OutputEncoding = $utf8Encoding
    $OutputEncoding = $utf8Encoding
    $sanitizedArgs = @($CommandArgs | Where-Object { $_ -ne $null -and $_ -ne '' })
    $resolvedFile = $File
    if ($resolvedFile -and -not [System.IO.Path]::IsPathRooted($resolvedFile)) {
      $resolvedCandidate = Resolve-CommandOnPath $resolvedFile
      if ($resolvedCandidate) {
        $resolvedFile = $resolvedCandidate
      }
    }
    Push-Location $WorkingDirectory
    try {
      if ($resolvedFile -and $resolvedFile -match '\.ps1$') {
        $scriptArgs = @('-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $resolvedFile) + $sanitizedArgs
        & powershell.exe @scriptArgs 1> $outFile 2> $errFile
      } elseif ($sanitizedArgs.Count -gt 0) {
        & $resolvedFile @sanitizedArgs 1> $outFile 2> $errFile
      } else {
        & $resolvedFile 1> $outFile 2> $errFile
      }
      $exitCode = $LASTEXITCODE
    } finally {
      Pop-Location
    }

    if ($exitCode -eq $null) {
      $exitCode = -1
    }
    return @{
      ok = ($exitCode -eq 0)
      exitCode = $exitCode
      stdout = (Truncate-Text (Get-Content -Raw -Path $outFile -ErrorAction SilentlyContinue))
      stderr = (Truncate-Text (Get-Content -Raw -Path $errFile -ErrorAction SilentlyContinue))
      durationMs = $sw.ElapsedMilliseconds
    }
  } catch {
    return @{ ok = $false; exitCode = -1; stdout = ""; stderr = $_.Exception.Message; durationMs = $sw.ElapsedMilliseconds }
  } finally {
    [Console]::OutputEncoding = $previousConsoleEncoding
    $OutputEncoding = $previousOutputEncoding
    Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue
  }
}

function Invoke-OptionalWatchCapture {
  param(
    [string]$File,
    [string[]]$CommandArgs = @(),
    [string]$WorkingDirectory = $RepoDir,
    [int]$TimeoutSeconds = 15
  )

  $result = Invoke-Captured $File $CommandArgs $WorkingDirectory $TimeoutSeconds
  if ($result.ok) {
    return $result
  }

  $combined = @($result.stdout, $result.stderr) -join "`n"
  if ($combined -match "unrecognized subcommand 'watch'") {
    return @{
      ok = $true
      skipped = $true
      available = $false
      reason = 'watch subcommand unavailable'
      exitCode = 0
      stdout = $result.stdout
      stderr = $result.stderr
      durationMs = $result.durationMs
    }
  }

  return $result
}

function Test-TcpEndpoint {
  param([string]$Endpoint)
  $clean = $Endpoint -replace '^https?://',''
  $parts = $clean.Split(':')
  $targetHost = $parts[0]
  $targetPort = if ($parts.Length -gt 1) { [int]$parts[1] } else { 50051 }
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $iar = $client.BeginConnect($targetHost, $targetPort, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(2000, $false)
    if ($ok) { $client.EndConnect($iar) }
    return @{ ok = [bool]$ok; host = $targetHost; port = $targetPort }
  } catch {
    return @{ ok = $false; host = $targetHost; port = $targetPort; error = $_.Exception.Message }
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
  $statusLines = @($status.stdout -split "`r?`n" | Where-Object { $_ })
  $dirtyLines = @($statusLines | Where-Object { $_ -notmatch '^## ' })
  $statusPreview = @($statusLines | Select-Object -First 20)
  return @{
    ok = ($branch.ok -and $status.ok -and $head.ok)
    branch = $branch.stdout.Trim()
    head = $head.stdout.Trim()
    dirtyCount = $dirtyLines.Count
    statusPreview = ($statusPreview -join "`n")
  }
}

function Get-HookInfo {
  param([string]$BaseDir)

  $hookScript = Join-Path $BaseDir 'scripts\windows\indexed-projects-hooks.ps1'
  $report = [ordered]@{
    available = (Test-Path -LiteralPath $hookScript)
    installed = $false
    healthy = $false
    scriptPath = $hookScript
  }

  if (-not $report.available) {
    $report.reason = 'hook script missing'
    return $report
  }

  $result = Invoke-Captured $hookScript @('-Action', 'status', '-RepoDir', $BaseDir) $BaseDir 15
  $report.lastCheck = $result

  if (-not $result.ok) {
    $report.reason = $(if ($result.stderr) { $result.stderr } else { 'hook status command failed' })
    return $report
  }

  try {
    $status = $result.stdout | ConvertFrom-Json -ErrorAction Stop
    $hookCount = 0
    if ($null -ne $status.hookCount) {
      $hookCount = [int]$status.hookCount
    }
    if ($null -ne $status.installed) {
      $installed = [bool]$status.installed
    } else {
      $installed = [bool]$status.hooksPathMatches -and [bool]$status.runnerExists -and ($hookCount -ge 5)
    }
    if ($null -ne $status.healthy) {
      $healthy = [bool]$status.healthy
    } else {
      $healthy = $installed
    }
    $report.status = $status
    $report.installed = $installed
    $report.healthy = $healthy
    if (-not $installed) {
      $report.reason = $(if ($status.reason) { $status.reason } else { 'hooks not fully installed' })
    }
  } catch {
    $report.reason = $_.Exception.Message
  }

  return $report
}

function New-Snapshot {
  Trace-Observe "snapshot: resolve wqm"
  $resolvedWqmPath = Resolve-WqmPath -BaseDir $RepoDir
  if (-not $resolvedWqmPath) { $resolvedWqmPath = "wqm" }
  Trace-Observe "snapshot: git"
  $git = Get-GitInfo
  Trace-Observe "snapshot: hooks"
  $hooks = Get-HookInfo -BaseDir $RepoDir
  Trace-Observe "snapshot: qdrant"
  $qdrant = Test-Qdrant $QdrantUrl
  Trace-Observe "snapshot: daemon"
  $daemonTcp = Test-TcpEndpoint $DaemonEndpoint
  Trace-Observe "snapshot: wqm"
  $previousWqmDaemonAddr = $env:WQM_DAEMON_ADDR
  $previousWqmQdrantUrl = $env:WQM_QDRANT_URL
  $previousQdrantUrl = $env:QDRANT_URL
  $previousWqmDatabasePath = $env:WQM_DATABASE_PATH
  try {
    $env:WQM_DAEMON_ADDR = (Resolve-HttpEndpoint -Endpoint $DaemonEndpoint)
    $env:WQM_QDRANT_URL = $QdrantUrl
    $env:QDRANT_URL = $QdrantUrl
    $env:WQM_DATABASE_PATH = (Resolve-DockerDatabasePath -BaseDir $RepoDir)

    $wqmHealth = Invoke-Captured $resolvedWqmPath @('status','health') $RepoDir 20
    $queueStats = Invoke-Captured $resolvedWqmPath @('queue','stats') $RepoDir 20
    $projectStatus = Invoke-Captured $resolvedWqmPath @('project','status', $ProjectDir) $ProjectDir 20
    $projectList = Invoke-Captured $resolvedWqmPath @('project','list') $RepoDir 20
    $watchList = Invoke-OptionalWatchCapture $resolvedWqmPath @('watch','list','--json') $ProjectDir 20
  } finally {
    if ($null -eq $previousWqmDaemonAddr) {
      Remove-Item Env:WQM_DAEMON_ADDR -ErrorAction SilentlyContinue
    } else {
      $env:WQM_DAEMON_ADDR = $previousWqmDaemonAddr
    }

    if ($null -eq $previousWqmQdrantUrl) {
      Remove-Item Env:WQM_QDRANT_URL -ErrorAction SilentlyContinue
    } else {
      $env:WQM_QDRANT_URL = $previousWqmQdrantUrl
    }

    if ($null -eq $previousQdrantUrl) {
      Remove-Item Env:QDRANT_URL -ErrorAction SilentlyContinue
    } else {
      $env:QDRANT_URL = $previousQdrantUrl
    }

    if ($null -eq $previousWqmDatabasePath) {
      Remove-Item Env:WQM_DATABASE_PATH -ErrorAction SilentlyContinue
    } else {
      $env:WQM_DATABASE_PATH = $previousWqmDatabasePath
    }
  }
  Trace-Observe "snapshot: processes"

  $nodeProc = @(Get-Process -Name node -ErrorAction SilentlyContinue | Select-Object -First 10 | ForEach-Object {
    [ordered]@{
      Id = $_.Id
      ProcessName = $_.ProcessName
      CPU = $_.CPU
      StartTime = $(if ($_.StartTime) { $_.StartTime.ToString('o') } else { $null })
      Path = $_.Path
    }
  })
  $memexProc = @(Get-Process -Name memexd -ErrorAction SilentlyContinue | Select-Object -First 10 | ForEach-Object {
    [ordered]@{
      Id = $_.Id
      ProcessName = $_.ProcessName
      CPU = $_.CPU
      StartTime = $(if ($_.StartTime) { $_.StartTime.ToString('o') } else { $null })
      Path = $_.Path
    }
  })

  return [ordered]@{
    timestamp = (Get-Date).ToString('o')
    repoDir = $RepoDir
    projectDir = $ProjectDir
    wqmPath = $resolvedWqmPath
    git = $git
    qdrant = $qdrant
    daemonTcp = $daemonTcp
    hooks = $hooks
    wqm = @{ health = $wqmHealth; queueStats = $queueStats; projectStatus = $projectStatus; projectList = $projectList; watchList = $watchList }
    processes = @{ node = $nodeProc; memexd = $memexProc }
  }
}

$ResolvedLogDir = Resolve-LogDir -Base $RepoDir -PathValue $LogDir
New-Item -ItemType Directory -Force -Path $ResolvedLogDir | Out-Null
$logFile = Join-Path $ResolvedLogDir ("service-observe-" + (Get-Date -Format 'yyyyMMdd') + ".jsonl")

while ($true) {
  Trace-Observe "loop: snapshot start"
  $snap = New-Snapshot
  Trace-Observe "loop: snapshot done"
  Trace-Observe "loop: json"
  $json = $snap | ConvertTo-Json -Depth 12 -Compress
  Trace-Observe "loop: add-content"
  Add-Content -Path $logFile -Value $json
  Trace-Observe "loop: wrote"

  $q = if ($snap.qdrant.ok) { 'ok' } else { 'fail' }
  $d = if ($snap.daemonTcp.ok) { 'ok' } else { 'fail' }
  $h = if ($snap.wqm.health.ok) { 'ok' } else { 'fail' }
  $hooks = if ($snap.hooks.healthy) { 'ok' } elseif ($snap.hooks.available) { 'warn' } else { 'off' }
  Write-Host ("[{0}] qdrant={1} daemonTcp={2} wqmHealth={3} hooks={4} log={5}" -f (Get-Date -Format 'HH:mm:ss'), $q, $d, $h, $hooks, $logFile)

  if ($Once) { break }
  Start-Sleep -Seconds $IntervalSeconds
}
