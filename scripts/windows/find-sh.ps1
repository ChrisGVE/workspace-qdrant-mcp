# Locate sh.exe from Git for Windows, in a way that survives GNU make's
# fragile quoting when invoking `powershell.exe -Command "<inline one-liner>"`.
#
# Probe order:
#   1. PATH (`Get-Command sh`)
#   2. Standard Git install dirs (`Program Files\Git\usr\bin\sh.exe`, x86 sibling,
#      LOCALAPPDATA fallback for per-user installs)
#   3. Derive from `Get-Command git` -> two Split-Path levels up + `usr\bin\sh.exe`
#
# Writes the absolute path to stdout on success; writes a friendly message to
# stderr and exits 1 on failure.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

if ($cmd = Get-Command sh -ErrorAction SilentlyContinue) {
    Write-Output $cmd.Source
    exit 0
}

$candidates = @(
    "$env:ProgramFiles\Git\usr\bin\sh.exe",
    "${env:ProgramFiles(x86)}\Git\usr\bin\sh.exe",
    "$env:LOCALAPPDATA\Programs\Git\usr\bin\sh.exe"
)

foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
        Write-Output $candidate
        exit 0
    }
}

if ($g = Get-Command git -ErrorAction SilentlyContinue) {
    # git.exe usually lives at <GitRoot>\cmd\git.exe or <GitRoot>\bin\git.exe.
    # Two Split-Path -Parent levels gets us to <GitRoot>; append usr\bin\sh.exe.
    $gitRoot = Split-Path (Split-Path $g.Source -Parent) -Parent
    $candidate = Join-Path $gitRoot 'usr\bin\sh.exe'
    if (Test-Path -LiteralPath $candidate) {
        Write-Output $candidate
        exit 0
    }
}

[Console]::Error.WriteLine("ERROR: 'sh' not on PATH and not found in any known Git\usr\bin location.")
[Console]::Error.WriteLine("Install Git Bash (https://gitforwindows.org) or add C:\Program Files\Git\usr\bin to PATH.")
exit 1
