[CmdletBinding()]
param(
  [string]$ProjectDir = (Get-Location).Path,
  [string]$Name = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command wqm -ErrorAction SilentlyContinue)) {
  throw "wqm não encontrado no PATH"
}
if (-not (Test-Path $ProjectDir)) {
  throw "ProjectDir não existe: $ProjectDir"
}

function Convert-ToWqmPath([string]$PathValue) {
  if (-not $PathValue) { return $PathValue }

  $absolute = [System.IO.Path]::GetFullPath($PathValue)
  $normalized = $absolute -replace '\\', '/'

  if ($normalized -match '^/mnt/[a-z]/') {
    return $normalized
  }

  if ($normalized -match '^([A-Za-z]):/(.*)$') {
    $drive = $Matches[1].ToLower()
    $rest = $Matches[2].TrimStart('/')
    return "/mnt/$drive/$rest"
  }

  return $normalized
}

$WqmProjectDir = Convert-ToWqmPath $ProjectDir
$args = @("project", "register", $WqmProjectDir, "--yes")
if ($Name) {
  $args += @("--name", $Name)
}

Write-Host "Registrando projeto: $ProjectDir"
& wqm @args

Write-Host ""
Write-Host "Status:"
wqm project status $WqmProjectDir

Write-Host ""
Write-Host "Lista:"
wqm project list
