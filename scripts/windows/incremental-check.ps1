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

function Invoke-Captured {
  param(
    [string]$File,
    [string[]]$Args = @(),
    [string]$WorkingDirectory = $ProjectDir,
    [int]$TimeoutSeconds = 60
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
    return @{ ok = ($p.ExitCode -eq 0); exitCode = $p.ExitCode; stdout = (Get-Content -Raw $outFile -ErrorAction SilentlyContinue); stderr = (Get-Content -Raw $errFile -ErrorAction SilentlyContinue); durationMs = $sw.ElapsedMilliseconds }
  } catch {
    return @{ ok = $false; exitCode = -1; stdout = ""; stderr = $_.Exception.Message; durationMs = $sw.ElapsedMilliseconds }
  } finally {
    Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue
  }
}

if (-not (Test-Path $ProjectDir)) { throw "ProjectDir nao existe: $ProjectDir" }
$ResolvedLogDir = Resolve-LogDir -Base $RepoDir -PathValue $LogDir
New-Item -ItemType Directory -Force -Path $ResolvedLogDir | Out-Null
$logFile = Join-Path $ResolvedLogDir ("incremental-check-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".json")

$before = [ordered]@{
  timestamp = (Get-Date).ToString('o')
  repoDir = $RepoDir
  projectDir = $ProjectDir
  repairRequested = [bool]$Repair
  projectStatus = Invoke-Captured wqm @('project','status', $ProjectDir) $ProjectDir 30
  projectCheck = Invoke-Captured wqm @('project','check', $ProjectDir, '--json') $ProjectDir 120
  watchList = Invoke-Captured wqm @('project','watch','list','--json') $ProjectDir 60
  queueStats = Invoke-Captured wqm @('queue','stats') $ProjectDir 60
}

$repairResult = $null
if ($Repair) {
  Write-Host "Rodando reparos nao destrutivos: register --yes, watch resume, project check."
  $repairResult = [ordered]@{
    register = Invoke-Captured wqm @('project','register', $ProjectDir, '--yes') $ProjectDir 120
    watchResume = Invoke-Captured wqm @('project','watch','resume') $ProjectDir 60
    projectCheckAfter = Invoke-Captured wqm @('project','check', $ProjectDir, '--json') $ProjectDir 120
    queueStatsAfter = Invoke-Captured wqm @('queue','stats') $ProjectDir 60
  }
}

$result = [ordered]@{ before = $before; repair = $repairResult }
$result | ConvertTo-Json -Depth 12 | Set-Content -Path $logFile -Encoding UTF8

$checkOk = $before.projectCheck.ok
$statusOk = $before.projectStatus.ok
$watchOk = $before.watchList.ok
Write-Host "incremental-check status=$statusOk check=$checkOk watchList=$watchOk log=$logFile"

if (-not $statusOk -or -not $checkOk) {
  Write-Host "Aviso: verificacao incremental encontrou falhas. Use incremental-repair para reparos nao destrutivos antes de rebuild/delete." -ForegroundColor Yellow
  exit 2
}
