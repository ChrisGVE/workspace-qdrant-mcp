# Verify Windows binaries have only allowed DLL dependencies.
# Uses dumpbin /dependents to check for external DLL dependencies.
# Fails with exit code 1 if unexpected DLLs are found.
param(
    [Parameter(Mandatory=$true, Position=0, ValueFromRemainingArguments=$true)]
    [string[]]$Binaries
)

$ErrorActionPreference = "Stop"

# Windows system DLLs that are always present
$AllowedDlls = @(
    "KERNEL32.dll",
    "ADVAPI32.dll",
    "WS2_32.dll",
    "USERENV.dll",
    "ntdll.dll",
    "bcrypt.dll",
    "OLEAUT32.dll",
    "ole32.dll",
    "SHELL32.dll",
    "USER32.dll",
    "GDI32.dll",
    "CRYPT32.dll",
    "SECUR32.dll",
    "RPCRT4.dll",
    "IPHLPAPI.dll",
    "PSAPI.dll",
    "DBGHELP.dll",
    "SYNCHRONIZATION.dll",
    # MSVC runtime (statically linked preferred, but allowed if present)
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll",
    "MSVCP140.dll",
    "ucrtbase.dll",
    "api-ms-win-*"
)

function Test-Binary {
    param([string]$BinaryPath)

    if (-not (Test-Path $BinaryPath)) {
        Write-Host "ERROR: Binary not found: $BinaryPath"
        return $false
    }

    Write-Host "Checking dependencies for: $BinaryPath"

    # Try dumpbin first (Visual Studio), fall back to objdump (MinGW)
    $deps = $null
    try {
        $dumpbinOutput = & dumpbin /dependents $BinaryPath 2>$null
        if ($LASTEXITCODE -eq 0) {
            # Parse dumpbin output - dependencies are listed as plain DLL names
            $deps = $dumpbinOutput | Where-Object { $_ -match '^\s+\S+\.dll$' } | ForEach-Object { $_.Trim() }
        }
    } catch {
        # dumpbin not available
    }

    if ($null -eq $deps) {
        try {
            $objdumpOutput = & objdump -p $BinaryPath 2>$null
            if ($LASTEXITCODE -eq 0) {
                $deps = $objdumpOutput | Where-Object { $_ -match 'DLL Name:' } | ForEach-Object {
                    ($_ -split 'DLL Name:\s*')[1].Trim()
                }
            }
        } catch {
            # objdump not available either
        }
    }

    if ($null -eq $deps) {
        Write-Host "  WARN: Neither dumpbin nor objdump available, skipping dependency check"
        return $true
    }

    $violations = 0
    foreach ($dep in $deps) {
        $allowed = $false
        foreach ($pattern in $AllowedDlls) {
            if ($pattern.Contains("*")) {
                if ($dep -like $pattern) {
                    $allowed = $true
                    break
                }
            } else {
                if ($dep -ieq $pattern) {
                    $allowed = $true
                    break
                }
            }
        }

        if ($allowed) {
            Write-Host "  OK: $dep"
        } else {
            Write-Host "  VIOLATION: Unexpected dependency: $dep"
            $violations++
        }
    }

    if ($violations -gt 0) {
        Write-Host "FAILED: $violations unexpected dependencies found in $BinaryPath"
        return $false
    }

    Write-Host "PASSED: All dependencies are allowed for $BinaryPath"
    return $true
}

$exitCode = 0
foreach ($binary in $Binaries) {
    if (-not (Test-Binary $binary)) {
        $exitCode = 1
    }
    Write-Host ""
}

exit $exitCode
