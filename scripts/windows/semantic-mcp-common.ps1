function Get-EnvFileValue {
  param(
    [string]$RepoDir,
    [string]$Name
  )

  $candidateFiles = @(
    (Join-Path $RepoDir "docker\.env"),
    (Join-Path $RepoDir ".env")
  )

  foreach ($file in $candidateFiles) {
    if (-not (Test-Path -LiteralPath $file)) {
      continue
    }

    foreach ($line in Get-Content -LiteralPath $file) {
      $trimmed = $line.Trim()
      if (-not $trimmed -or $trimmed.StartsWith("#")) {
        continue
      }

      $pattern = "^{0}=(.*)$" -f [regex]::Escape($Name)
      $match = [regex]::Match($trimmed, $pattern)
      if ($match.Success) {
        return $match.Groups[1].Value.Trim().Trim('"')
      }
    }
  }

  return $null
}

function Resolve-SettingValue {
  param(
    [string]$RepoDir,
    [string]$Name,
    [string]$DefaultValue
  )

  $envValue = [Environment]::GetEnvironmentVariable($Name)
  if ($envValue) {
    return $envValue
  }

  $fileValue = Get-EnvFileValue -RepoDir $RepoDir -Name $Name
  if ($fileValue) {
    return $fileValue
  }

  return $DefaultValue
}

function Resolve-McpToken {
  param(
    [string]$RepoDir,
    [string]$Token
  )

  if ($Token) {
    return $Token.Trim()
  }

  $resolved = Resolve-SettingValue -RepoDir $RepoDir -Name "MCP_HTTP_TOKEN" -DefaultValue ""
  if ($resolved) {
    return $resolved.Trim()
  }

  throw "MCP_HTTP_TOKEN is required. Set it in the environment or in docker\.env."
}

function Resolve-McpUrl {
  param(
    [string]$RepoDir,
    [string]$McpUrl
  )

  if ($McpUrl) {
    return $McpUrl
  }

  $port = Resolve-SettingValue -RepoDir $RepoDir -Name "MCP_HTTP_PORT" -DefaultValue "6335"
  return "http://127.0.0.1:${port}/mcp"
}

function Wait-McpHealth {
  param(
    [string]$McpUrl,
    [int]$TimeoutSeconds = 120,
    [int]$PollSeconds = 2
  )

  $healthUrl = $McpUrl -replace '/mcp$', '/healthz'
  $elapsed = 0

  while ($true) {
    try {
      $code = (Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 5).StatusCode
      if ($code -eq 200) {
        return
      }
    } catch {
      # Keep polling until timeout.
    }

    if ($elapsed -ge $TimeoutSeconds) {
      throw "MCP health check did not return 200 after $TimeoutSeconds seconds: $healthUrl"
    }

    Start-Sleep -Seconds $PollSeconds
    $elapsed += $PollSeconds
  }
}

function ConvertFrom-McpRawResponse {
  param(
    [string]$RawResponse
  )

  $lines = @($RawResponse -split "`r?`n")
  foreach ($line in $lines) {
    $trimmed = $line.Trim()
    if ($trimmed.StartsWith("data:")) {
      $candidate = $trimmed.Substring(5).Trim()
      if ($candidate.StartsWith("{") -and $candidate.EndsWith("}")) {
        return $candidate | ConvertFrom-Json
      }
    }
  }

  foreach ($line in $lines) {
    $trimmed = $line.Trim()
    if ($trimmed.StartsWith("{") -and $trimmed.EndsWith("}")) {
      return $trimmed | ConvertFrom-Json
    }
  }

  $start = $RawResponse.IndexOf("{")
  $end = $RawResponse.LastIndexOf("}")
  if ($start -ge 0 -and $end -gt $start) {
    return $RawResponse.Substring($start, $end - $start + 1) | ConvertFrom-Json
  }

  throw "Unable to parse MCP response body as JSON."
}

function Invoke-McpRequest {
  param(
    [string]$McpUrl,
    [string]$Token,
    [object]$Request,
    [int]$TimeoutSeconds = 120
  )

  $body = $Request | ConvertTo-Json -Depth 32 -Compress
  $curlArgs = @(
    '-sk',
    '--max-time',
    "$TimeoutSeconds",
    '-H',
    "Authorization: Bearer $Token",
    '-H',
    'Content-Type: application/json',
    '-H',
    'Accept: application/json, text/event-stream',
    '-d',
    $body,
    $McpUrl
  )

  $raw = & curl.exe @curlArgs
  if ($LASTEXITCODE -ne 0) {
    throw "curl.exe failed with exit code ${LASTEXITCODE} while calling ${McpUrl}"
  }

  $rawText = @($raw) -join "`n"
  return ConvertFrom-McpRawResponse -RawResponse $rawText
}

function Invoke-McpToolCall {
  param(
    [string]$McpUrl,
    [string]$Token,
    [string]$ToolName,
    [hashtable]$Arguments,
    [int]$RequestId = 1,
    [int]$TimeoutSeconds = 120
  )

  $request = [ordered]@{
    jsonrpc = '2.0'
    id = $RequestId
    method = 'tools/call'
    params = [ordered]@{
      name = $ToolName
      arguments = $Arguments
    }
  }

  $response = Invoke-McpRequest -McpUrl $McpUrl -Token $Token -Request $request -TimeoutSeconds $TimeoutSeconds
  if ($response.error) {
    $message = $response.error.message
    if (-not $message) {
      $message = 'unknown MCP JSON-RPC error'
    }
    throw "MCP request failed for ${ToolName}: ${message}"
  }

  if (-not $response.result) {
    throw "MCP response for $ToolName did not include a result payload."
  }

  return $response.result
}

function Get-PlainText {
  param(
    [string]$Text
  )

  if (-not $Text) {
    return ''
  }

  return [regex]::Replace($Text, "`e\[[0-9;]*m", '')
}

function Get-ProjectIdFromPath {
  param(
    [string]$ProjectPath
  )

  $absolutePath = [System.IO.Path]::GetFullPath($ProjectPath)
  $sha256 = [System.Security.Cryptography.SHA256]::Create()
  try {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($absolutePath)
    $hash = $sha256.ComputeHash($bytes)
    $hex = ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
    return 'local_{0}' -f $hex.Substring(0, 12)
  } finally {
    $sha256.Dispose()
  }
}

function Get-McpToolText {
  param(
    [object]$ToolResult
  )

  if (-not $ToolResult.content -or $ToolResult.content.Count -eq 0) {
    return ''
  }

  return [string]$ToolResult.content[0].text
}

function Get-McpToolJsonText {
  param(
    [object]$ToolResult
  )

  $text = Get-McpToolText -ToolResult $ToolResult
  if (-not $text) {
    throw "MCP tool result did not contain any text."
  }

  if ($text.TrimStart().StartsWith('{')) {
    return $text | ConvertFrom-Json
  }

  return $text
}
