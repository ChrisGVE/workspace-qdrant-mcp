[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$ProjectDir = (Get-Location).Path,
  [string]$LogDir = ".wqm-fork/logs",
  [switch]$Repair
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

function Invoke-Captured {
  param(
    [string]$File,
    [string[]]$CommandArgs = @(),
    [string]$WorkingDirectory = $ProjectDir,
    [int]$TimeoutSeconds = 60
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
      if ($sanitizedArgs.Count -gt 0) {
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
    return @{ ok = ($exitCode -eq 0); exitCode = $exitCode; stdout = (Truncate-Text (Get-Content -Raw -Path $outFile -ErrorAction SilentlyContinue)); stderr = (Truncate-Text (Get-Content -Raw -Path $errFile -ErrorAction SilentlyContinue)); durationMs = $sw.ElapsedMilliseconds }
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
    [string]$WorkingDirectory = $ProjectDir,
    [int]$TimeoutSeconds = 60
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

if (-not (Test-Path $ProjectDir)) { throw "ProjectDir nao existe: $ProjectDir" }
$WqmPath = Resolve-WqmPath -BaseDir $RepoDir
if (-not $WqmPath) { $WqmPath = "wqm" }
$ResolvedLogDir = Resolve-LogDir -Base $RepoDir -PathValue $LogDir
New-Item -ItemType Directory -Force -Path $ResolvedLogDir | Out-Null
$logFile = Join-Path $ResolvedLogDir ("incremental-check-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".json")

$before = [ordered]@{
  timestamp = (Get-Date).ToString('o')
  repoDir = $RepoDir
  projectDir = $ProjectDir
  repairRequested = [bool]$Repair
  projectStatus = Invoke-Captured $WqmPath @('project','status', $ProjectDir) $ProjectDir 30
  projectCheck = Invoke-Captured $WqmPath @('project','check', $ProjectDir, '--json') $ProjectDir 120
  watchList = Invoke-OptionalWatchCapture $WqmPath @('watch','list','--json') $ProjectDir 60
  queueStats = Invoke-Captured $WqmPath @('queue','stats') $ProjectDir 60
}

$repairResult = $null
if ($Repair) {
  Write-Host "Rodando reparos nao destrutivos: register --yes, watch resume, project check."
  $repairResult = [ordered]@{
    register = Invoke-Captured $WqmPath @('project','register', $ProjectDir, '--yes') $ProjectDir 120
    watchResume = Invoke-OptionalWatchCapture $WqmPath @('watch','resume') $ProjectDir 60
    projectCheckAfter = Invoke-Captured $WqmPath @('project','check', $ProjectDir, '--json') $ProjectDir 120
    queueStatsAfter = Invoke-Captured $WqmPath @('queue','stats') $ProjectDir 60
  }
}

$result = [ordered]@{ before = $before; repair = $repairResult }
$result | ConvertTo-Json -Depth 12 | Set-Content -Path $logFile -Encoding UTF8

$checkOk = $before.projectCheck.ok
$statusOk = $before.projectStatus.ok
$watchOk = $before.watchList.ok
$watchState = if ($before.watchList.skipped) { 'skip' } elseif ($watchOk) { 'ok' } else { 'fail' }
Write-Host "incremental-check status=$statusOk check=$checkOk watchList=$watchState log=$logFile"

if (-not $statusOk -or -not $checkOk) {
  Write-Host "Aviso: verificacao incremental encontrou falhas. Use incremental-repair para reparos nao destrutivos antes de rebuild/delete." -ForegroundColor Yellow
  exit 2
}
