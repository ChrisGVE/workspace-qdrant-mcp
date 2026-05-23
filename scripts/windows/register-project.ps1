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

$args = @("project", "register", $ProjectDir, "--yes")
if ($Name) {
  $args += @("--name", $Name)
}

Write-Host "Registrando projeto: $ProjectDir"
& wqm @args

Write-Host ""
Write-Host "Status:"
wqm project status $ProjectDir

Write-Host ""
Write-Host "Lista:"
wqm project list
