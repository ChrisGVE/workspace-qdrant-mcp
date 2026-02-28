# PowerShell Test Script - Regression test for .ps1 language detection
# Expected: file_type=code, language=powershell

param(
    [string]$Name = "World",
    [switch]$Verbose
)

function Get-Greeting {
    param([string]$Target)
    return "Hello, $Target!"
}

$greeting = Get-Greeting -Target $Name

if ($Verbose) {
    Write-Host "Verbose mode enabled"
    Write-Host "Generated greeting: $greeting"
}

Write-Output $greeting

# Pipeline example
Get-ChildItem -Path . -Filter "*.txt" |
    Where-Object { $_.Length -gt 1KB } |
    Sort-Object -Property LastWriteTime -Descending |
    Select-Object -First 5 -Property Name, Length, LastWriteTime
