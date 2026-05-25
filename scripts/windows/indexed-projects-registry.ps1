[CmdletBinding()]
param(
  [string]$RegistryPath = ".wqm-fork\indexed-projects.json",
  [string]$RepoDir = (Get-Location).Path,
  [ValidateSet("init","list-projects","add-project","remove-project","project-status","status-all","list-branches","add-branch","start-agent-branch","finish-agent-branch","abandon-agent-branch","agent-branch-status","observe-project","observe-all","incremental-check","incremental-check-all","register-wqm","register-all-wqm","sync-current-branch","cleanup-orphans")]
  [string]$Action = "list-projects",
  [string]$ProjectName = "",
  [string]$ProjectDir = "",
  [string]$BranchName = "",
  [string]$BaseBranch = "main",
  [string]$ReturnBranch = "",
  [string]$WorktreePath = "",
  [string]$WorktreeRoot = "",
  [string]$UseWorktree = "true",
  [string]$Purpose = "agent change",
  [string]$CreatedBy = "ai-agent",
  [string]$QdrantUrl = "http://localhost:6333",
  [string]$DaemonEndpoint = "localhost:50051",
  [string]$LogDir = ".wqm-fork\logs",
  [string]$Mutate = "false",
  [string]$Json = "false",
  [string]$RemoveWorktree = "false"
)

$ErrorActionPreference = "Stop"

function Test-TrueValue([string]$Value) { return $Value -match '^(1|true|yes|y)$' }
function UtcNow { return (Get-Date).ToUniversalTime().ToString("o") }
function Assert-MutationAllowed { if (-not (Test-TrueValue $Mutate)) { throw "Acao mutavel bloqueada. Reexecute com -Mutate true apos confirmacao humana explicita." } }
function Convert-ToAbs([string]$PathValue) {
  if (-not $PathValue) { return $PathValue }
  $resolved = Resolve-Path -LiteralPath $PathValue -ErrorAction SilentlyContinue
  if ($resolved) { return [System.IO.Path]::GetFullPath($resolved.Path) }
  return [System.IO.Path]::GetFullPath($PathValue)
}
function Convert-ToWqmPath([string]$PathValue) {
  if (-not $PathValue) { return $PathValue }

  $absolute = Convert-ToAbs $PathValue
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
function Resolve-AbsolutePath([string]$BaseDir, [string]$PathValue) {
  if (-not $PathValue) { return $PathValue }

  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $PathValue))
}
function Safe-BranchSlug([string]$Name) { return ($Name -replace '[^A-Za-z0-9._-]+','-').Trim('-') }
function Ensure-Dir([string]$PathValue) { if (-not (Test-Path -LiteralPath $PathValue)) { New-Item -ItemType Directory -Path $PathValue -Force | Out-Null } }
function Resolve-GitPath {
  $cmd = Get-Command git -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($cmd) {
    if ($cmd.Source) { return $cmd.Source }
    if ($cmd.Path) { return $cmd.Path }
  }
  return "git"
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

  $pathEntries = @()
  if ($env:PATH) {
    $pathEntries = @($env:PATH -split ';' | Where-Object { $_ })
  }

  foreach ($entry in $pathEntries) {
    foreach ($name in $searchNames) {
      $candidate = Join-Path $entry $name
      if (Test-Path -LiteralPath $candidate) {
        return [System.IO.Path]::GetFullPath($candidate)
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
function New-Registry { [ordered]@{ schemaVersion = 2; kind = "indexed-projects"; updatedAt = UtcNow; projects = @() } }
function Read-Registry {
  if (-not (Test-Path -LiteralPath $RegistryPath)) { return New-Registry }
  $raw = Get-Content -LiteralPath $RegistryPath -Raw
  if (-not $raw.Trim()) { return New-Registry }
  return $raw | ConvertFrom-Json
}
function Write-Registry($Registry) {
  $Registry.updatedAt = UtcNow
  $parent = Split-Path -Parent $RegistryPath
  if ($parent) { Ensure-Dir $parent }
  $encoding = [System.Text.UTF8Encoding]::new($false)
  [System.IO.File]::WriteAllText($RegistryPath, ($Registry | ConvertTo-Json -Depth 20), $encoding)
}
function To-Array($MaybeArray) { if ($null -eq $MaybeArray) { return @() }; return @($MaybeArray) }
function Normalize-Project($Project) {
  $branches = To-Array $Project.branches
  [pscustomobject]@{
    name = $Project.name
    root = (Convert-ToAbs $Project.root)
    projectId = $Project.projectId
    qdrantUrl = $(if ($Project.qdrantUrl) { $Project.qdrantUrl } else { $QdrantUrl })
    daemonEndpoint = $(if ($Project.daemonEndpoint) { $Project.daemonEndpoint } else { $DaemonEndpoint })
    defaultBranch = $(if ($Project.defaultBranch) { $Project.defaultBranch } else { "main" })
    tenantStrategy = $(if ($Project.tenantStrategy) { $Project.tenantStrategy } else { "project" })
    enabled = $(if ($null -ne $Project.enabled) { [bool]$Project.enabled } else { $true })
    updatedAt = $(if ($Project.updatedAt) { $Project.updatedAt } else { $null })
    branches = $branches
  }
}
function Get-Projects($Registry) { To-Array $Registry.projects | ForEach-Object { Normalize-Project $_ } }
function Find-Project($Registry) {
  $projects = @(Get-Projects $Registry)
  if ($ProjectDir) {
    $resolvedRoot = Resolve-ProjectRoot $ProjectDir
    $m = @($projects | Where-Object { (Convert-ToAbs $_.root) -eq $resolvedRoot })
    if ($m.Count -eq 1) { return $m[0] }
    if ($m.Count -gt 1) { throw "Projeto ambiguo: $ProjectName $ProjectDir" }
  }

  if ($ProjectName) { $m = @($projects | Where-Object { $_.name -eq $ProjectName }) }
  elseif ($ProjectDir) { $m = @() }
  else { throw "Informe -ProjectName ou -ProjectDir." }
  if ($m.Count -eq 0) { throw "Projeto indexado nao encontrado: $ProjectName $ProjectDir" }
  if ($m.Count -gt 1) { throw "Projeto ambiguo: $ProjectName $ProjectDir" }
  return $m[0]
}
function Find-ProjectByRoot($Registry, [string]$RootPath) {
  if (-not $RootPath) { return $null }
  $abs = Resolve-ProjectRoot $RootPath
  return @(Get-Projects $Registry | Where-Object { (Convert-ToAbs $_.root) -eq $abs }) | Select-Object -First 1
}
function Invoke-Git([string]$Repo, [string[]]$CommandArgs) {
  & $GitPath -C $Repo @CommandArgs
  if ($LASTEXITCODE -ne 0) { throw "git -C $Repo $($CommandArgs -join ' ') falhou com codigo $LASTEXITCODE" }
}
function Invoke-GitText([string]$Repo, [string[]]$CommandArgs) {
  $out = & $GitPath -C $Repo @CommandArgs 2>$null
  [pscustomobject]@{ code = $LASTEXITCODE; ok = ($LASTEXITCODE -eq 0); output = (@($out) -join "`n") }
}
function Assert-Clean([string]$Repo) {
  $s = & $GitPath -C $Repo status --porcelain
  if ($s) { throw "Working tree suja em $Repo. Commit/stash/limpe manualmente antes." }
}
function Get-Head([string]$Repo) {
  $result = Invoke-GitText $Repo @('rev-parse','HEAD')
  if ($result.ok) {
    $headLines = @($result.output -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    foreach ($candidate in $headLines) {
      if ($candidate -match '^[0-9a-fA-F]{40,}$') {
        return $candidate
      }
      if ($candidate -match '([0-9a-fA-F]{40,})') {
        return $Matches[1]
      }
    }
  }

  $gitDir = Get-GitDir $Repo
  if (-not $gitDir) { return $null }

  $headFile = Join-Path $gitDir 'HEAD'
  if (-not (Test-Path -LiteralPath $headFile)) { return $null }

  $head = (Get-Content -LiteralPath $headFile -Raw).Trim()
  if (-not $head) { return $null }

  if ($head -match '^ref:\s*(.+)$') {
    $refName = $Matches[1].Trim()
    $commonDir = Get-GitCommonDir $Repo
    if (-not $commonDir) { return $null }

    $refPath = Resolve-AbsolutePath $commonDir $refName
    if (Test-Path -LiteralPath $refPath) {
      return (Get-Content -LiteralPath $refPath -Raw).Trim()
    }

    $packedRefs = Join-Path $commonDir 'packed-refs'
    if (Test-Path -LiteralPath $packedRefs) {
      foreach ($line in (Get-Content -LiteralPath $packedRefs -ErrorAction SilentlyContinue)) {
        if ($line -and -not $line.StartsWith('#') -and $line -notmatch '^\^' -and $line -match "^[0-9a-fA-F]{40,}\s+$([regex]::Escape($refName))$") {
          return ($line -split '\s+')[0]
        }
      }
    }

    return $null
  }

  if ($head -match '^[0-9a-fA-F]{40,}$') { return $head }
  return $null
}
function Get-CurrentBranch([string]$Repo) {
  $gitDir = Get-GitDir $Repo
  if (-not $gitDir) { return $null }

  $headFile = Join-Path $gitDir 'HEAD'
  if (-not (Test-Path -LiteralPath $headFile)) { return $null }

  $head = (Get-Content -LiteralPath $headFile -Raw).Trim()
  if ($head -match '^ref:\s*refs/heads/(.+)$') {
    return $Matches[1].Trim()
  }
  return $null
}
function Branch-Exists([string]$Repo, [string]$Branch) {
  $commonDir = Get-GitCommonDir $Repo
  if (-not $commonDir) { return $false }

  $refPath = Join-Path $commonDir (Join-Path 'refs\heads' $Branch)
  if (Test-Path -LiteralPath $refPath) { return $true }

  $packedRefs = Join-Path $commonDir 'packed-refs'
  if (-not (Test-Path -LiteralPath $packedRefs)) { return $false }

  $refName = "refs/heads/$Branch"
  foreach ($line in (Get-Content -LiteralPath $packedRefs -ErrorAction SilentlyContinue)) {
    if ($line -and -not $line.StartsWith('#') -and $line -notmatch '^\^' -and $line -match "^[0-9a-fA-F]{40,}\s+$([regex]::Escape($refName))$") {
      return $true
    }
  }

  return $false
}
function Get-PreferredDefaultBranch([string]$Repo, [string]$Fallback) {
  foreach ($candidate in @('main','master','develop')) {
    if (Branch-Exists $Repo $candidate) {
      return $candidate
    }
  }

  if ($Fallback) {
    return $Fallback
  }

  return 'main'
}
function Find-Branch($Project, [string]$Name) { @(To-Array $Project.branches | Where-Object { $_.name -eq $Name }) | Select-Object -First 1 }
function Upsert-Project($Registry, $ProjectObject) {
  $items = @()
  foreach ($p in To-Array $Registry.projects) {
    if ($p.name -ne $ProjectObject.name) {
      $items += $p
    }
  }
  $items += [pscustomobject]$ProjectObject
  $Registry.projects = @($items)
}
function Upsert-Branch($Registry, $Project, $BranchObject) {
  $stored = @(To-Array $Registry.projects | Where-Object { $_.name -eq $Project.name }) | Select-Object -First 1
  if (-not $stored) { throw "Projeto nao encontrado para atualizar branch: $($Project.name)" }
  $list = @()
  foreach ($b in To-Array $stored.branches) {
    if ($b.name -ne $BranchObject.name) {
      $list += $b
    }
  }
  $list += [pscustomobject]$BranchObject
  $stored.branches = @($list)
  $stored.updatedAt = UtcNow
  Upsert-Project $Registry $stored
}
function Strip-AnsiEscapeSequences([string]$Text) {
  if (-not $Text) { return $Text }
  return ($Text -replace "`e\[[0-9;?]*[ -/]*[@-~]", '')
}
function Get-ProjectIdFromRegisterOutput([string]$Stdout, [string]$Stderr) {
  foreach ($chunk in @($Stdout, $Stderr)) {
    if (-not $chunk) { continue }

    $plain = Strip-AnsiEscapeSequences $chunk
    foreach ($line in ($plain -split "`r?`n")) {
      if ($line -match '^\s*(?:Existing\s+)?Project I[Dd]:\s*(?<id>.+?)\s*$') {
        return $Matches['id'].Trim()
      }
    }
  }

  return $null
}
function Get-GitDir([string]$Repo) {
  $repoAbs = Convert-ToAbs $Repo
  $gitMarker = Join-Path $repoAbs '.git'
  if (-not (Test-Path -LiteralPath $gitMarker)) { return $null }

  $item = Get-Item -LiteralPath $gitMarker -Force
  if ($item.PSIsContainer) {
    return [System.IO.Path]::GetFullPath($gitMarker)
  }

  $raw = Get-Content -LiteralPath $gitMarker -Raw
  if (-not $raw) { return $null }
  if ($raw -notmatch '^gitdir:\s*(.+)$') { return $null }
  return Resolve-AbsolutePath $repoAbs $Matches[1].Trim()
}
function Get-GitCommonDir([string]$Repo) {
  $result = Invoke-GitText $Repo @('rev-parse','--git-common-dir')
  if (-not $result.ok) { return $null }
  $commonDir = $result.output.Trim()
  if (-not $commonDir) { return $null }
  return Resolve-AbsolutePath $Repo $commonDir
}
function Get-ProjectRootFromRepo([string]$Repo) {
  $repoAbs = Convert-ToAbs $Repo
  $gitMarker = Join-Path $repoAbs '.git'
  if (Test-Path -LiteralPath $gitMarker) {
    $gitItem = Get-Item -LiteralPath $gitMarker -Force
    if ($gitItem.PSIsContainer) {
      return $repoAbs
    }
  }

  $commonDir = Get-GitCommonDir $Repo
  if (-not $commonDir) { return $null }
  return (Resolve-Path -LiteralPath (Split-Path -Parent $commonDir)).Path
}
function Resolve-ProjectRoot([string]$PathValue) {
  if (-not $PathValue) { return $null }

  $resolved = Get-ProjectRootFromRepo $PathValue
  if ($resolved) {
    return $resolved
  }

  return Convert-ToAbs $PathValue
}
function Get-SyncBranchSnapshot([string]$Repo, $ExistingBranch, [string]$ProjectDefaultBranch, [bool]$UseWorktree) {
  $currentBranch = Get-CurrentBranch $Repo
  if (-not $currentBranch) {
    # Detached HEAD has no branch to track, so leave the registry untouched.
    return $null
  }

  $head = Get-Head $Repo
  $projectRoot = Get-ProjectRootFromRepo $Repo
  if (-not $projectRoot) {
    $projectRoot = Convert-ToAbs $Repo
  }

  $branchName = $currentBranch
  $existing = $ExistingBranch
  $createdAt = if ($existing.createdAt) { $existing.createdAt } else { UtcNow }
  $baseBranch = if ($existing.baseBranch) { $existing.baseBranch } elseif ($ProjectDefaultBranch) { $ProjectDefaultBranch } else { 'main' }
  $returnBranch = if ($existing.returnBranch) { $existing.returnBranch } else { $baseBranch }
  $kind = if ($existing.kind) { $existing.kind } else { $(if ($UseWorktree) { 'manual-worktree' } else { 'manual' }) }
  $createdBy = if ($existing.createdBy) { $existing.createdBy } else { 'git-hook' }
  $purpose = if ($existing.purpose) { $existing.purpose } else { 'tracked by git hook' }
  $path = Convert-ToAbs $Repo

  return [pscustomobject]@{
    projectRoot = $projectRoot
    path = $path
    branchName = $branchName
    head = $head
    baseBranch = $baseBranch
    returnBranch = $returnBranch
    kind = $kind
    createdBy = $createdBy
    createdAt = $createdAt
    purpose = $purpose
    useWorktree = $UseWorktree
  }
}
function Sync-CurrentBranch($Registry, [string]$Repo) {
  $repoPath = Convert-ToAbs $Repo
  $projectRoot = Get-ProjectRootFromRepo $repoPath
  if (-not $projectRoot) {
    return [pscustomobject]@{ success = $true; action = 'sync-current-branch'; skipped = $true; reason = 'project-root-unresolved' }
  }

  $project = Find-ProjectByRoot $Registry $projectRoot
  $createdProject = $false
  if (-not $project) {
    $defaultBranch = Get-PreferredDefaultBranch $repoPath (Get-CurrentBranch $repoPath)
    $project = [ordered]@{
      name = $(if ($ProjectName) { $ProjectName } else { (Split-Path -Leaf $projectRoot) })
      root = $projectRoot
      projectId = $null
      qdrantUrl = $QdrantUrl
      daemonEndpoint = $DaemonEndpoint
      defaultBranch = $defaultBranch
      tenantStrategy = 'project'
      enabled = $true
      createdAt = UtcNow
      updatedAt = UtcNow
      branches = @()
    }
    Upsert-Project $Registry $project
    $createdProject = $true
  }

  $useWorktree = (Convert-ToAbs $repoPath) -ne (Convert-ToAbs $projectRoot)
  $existingBranch = Find-Branch $project (Get-CurrentBranch $repoPath)
  $sync = Get-SyncBranchSnapshot $repoPath $existingBranch $project.defaultBranch $useWorktree
  if (-not $sync) {
    return [pscustomobject]@{ success = $true; action = 'sync-current-branch'; skipped = $true; reason = 'detached-head' }
  }

  $registerResult = Invoke-WqmProjectRegister -PathValue $sync.path -ProjectName $project.name -Cwd $sync.path
  $branch = [ordered]@{
    name = $sync.branchName
    kind = $sync.kind
    path = $sync.path
    baseBranch = $sync.baseBranch
    returnBranch = $sync.returnBranch
    status = 'active'
    createdBy = $sync.createdBy
    createdAt = $sync.createdAt
    lastSeenAt = UtcNow
    baseCommit = $(if ($existingBranch.baseCommit) { $existingBranch.baseCommit } else { $null })
    headCommit = $sync.head
    lastIndexedCommit = $sync.head
    watchEnabled = $true
    indexed = $registerResult.ok
    purpose = $sync.purpose
    useWorktree = $sync.useWorktree
  }
  Upsert-Branch $Registry $project $branch

  $storedProject = @(To-Array $Registry.projects | Where-Object { $_.name -eq $project.name }) | Select-Object -First 1
  if ($storedProject) {
    $storedProject.projectId = $(if ($registerResult.ok -and $registerResult.project_id) { $registerResult.project_id } else { $storedProject.projectId })
    $storedProject.updatedAt = UtcNow
    Upsert-Project $Registry $storedProject
  }
  Write-Registry $Registry

  return [pscustomobject]@{
    success = $true
    action = 'sync-current-branch'
    createdProject = $createdProject
    project = $project.name
    root = $projectRoot
    branch = $branch
    register = $registerResult
  }
}
function Test-IndexedBranchHealth($Branch) {
  $path = Convert-ToAbs $Branch.path
  $pathExists = Test-Path -LiteralPath $path
  $gitRepo = $false
  $branchExists = $false
  $reasons = New-Object System.Collections.Generic.List[string]

  if (-not $pathExists) {
    $reasons.Add('path_missing')
  } else {
    $gitRepo = Test-Path -LiteralPath (Join-Path $path '.git')
    if (-not $gitRepo) {
      $reasons.Add('git_repo_missing')
    } elseif (-not (Branch-Exists $path $Branch.name)) {
      $reasons.Add('branch_missing')
    } else {
      $branchExists = $true
    }
  }

  [pscustomobject]@{
    path = $path
    pathExists = $pathExists
    gitRepo = $gitRepo
    branchExists = $branchExists
    stale = ($reasons.Count -gt 0)
    reason = ($reasons -join ',')
    branchName = $Branch.name
    kind = $Branch.kind
    status = $Branch.status
    useWorktree = $Branch.useWorktree
  }
}
function Cleanup-OrphanedIndex($Registry, [bool]$Mutate) {
  $removedBranches = New-Object System.Collections.Generic.List[object]
  $removedProjects = New-Object System.Collections.Generic.List[object]
  $keptProjects = New-Object System.Collections.Generic.List[object]

  foreach ($project in Get-Projects $Registry) {
    $keptBranches = New-Object System.Collections.Generic.List[object]

    foreach ($branch in To-Array $project.branches) {
      $health = Test-IndexedBranchHealth $branch
      if ($health.stale) {
        $removedBranches.Add([pscustomobject]@{
          project = $project.name
          branch = $branch.name
          path = $health.path
          kind = $health.kind
          status = $health.status
          reason = $health.reason
        })
        continue
      }

      $keptBranches.Add([pscustomobject]$branch)
    }

    if ($keptBranches.Count -eq 0) {
      $removedProjects.Add([pscustomobject]@{
        project = $project.name
        root = $project.root
      })
      continue
    }

    if ($Mutate) {
      $project.branches = @($keptBranches.ToArray())
      $project.updatedAt = UtcNow
    }
    $keptProjects.Add([pscustomobject]$project)
  }

  if ($Mutate) {
    $Registry.projects = @($keptProjects.ToArray())
    Write-Registry $Registry
  }

  return [pscustomobject]@{
    success = $true
    action = 'cleanup-orphans'
    mutated = $Mutate
    removedBranchCount = $removedBranches.Count
    removedProjectCount = $removedProjects.Count
    removedBranches = @($removedBranches.ToArray())
    removedProjects = @($removedProjects.ToArray())
    keptProjectCount = $keptProjects.Count
  }
}
function Invoke-WqmProjectRegister([string]$PathValue, [string]$ProjectName, [string]$Cwd) {
  $projectWqmPath = Convert-ToWqmPath $PathValue
  $commandArgs = @('project', 'register', $projectWqmPath, '--yes')
  if ($ProjectName) { $commandArgs += @('--name', $ProjectName) }
  $result = Invoke-Captured $WqmPath $commandArgs $Cwd 60
  $projectId = Get-ProjectIdFromRegisterOutput $result.stdout $result.stderr
  if (-not $result.ok) {
    return [pscustomobject]@{
      ok = $false
      path = $PathValue
      wqmPath = $projectWqmPath
      projectName = $ProjectName
      exitCode = $result.exitCode
      stdout = $result.stdout
      stderr = $result.stderr
      project_id = $projectId
    }
  }
  [pscustomobject]@{
    ok = $true
    path = $PathValue
    wqmPath = $projectWqmPath
    projectName = $ProjectName
    exitCode = $result.exitCode
    stdout = $result.stdout
    stderr = $result.stderr
    project_id = $projectId
  }
}
function Test-TcpEndpoint([string]$Endpoint) {
  $clean = $Endpoint -replace '^https?://',''
  $parts = $clean.Split(':')
  $host = $parts[0]
  $port = if ($parts.Length -gt 1) { [int]$parts[1] } else { 50051 }
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $iar = $client.BeginConnect($host, $port, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(2000, $false)
    if ($ok) { $client.EndConnect($iar) }
    [pscustomobject]@{ ok = [bool]$ok; host = $host; port = $port }
  } catch { [pscustomobject]@{ ok = $false; host = $host; port = $port; error = $_.Exception.Message } }
  finally { try { $client.Close() } catch {} }
}
function Test-Qdrant([string]$Url) {
  try {
    $uri = $Url.TrimEnd('/') + '/collections'
    $resp = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 5
    [pscustomobject]@{ ok = ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300); statusCode = $resp.StatusCode; endpoint = $uri }
  } catch { [pscustomobject]@{ ok = $false; endpoint = ($Url.TrimEnd('/') + '/collections'); error = $_.Exception.Message } }
}
function Invoke-Captured([string]$File, [string[]]$CommandArgs, [string]$Cwd, [int]$TimeoutSeconds = 20) {
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
    Push-Location $Cwd
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
    [pscustomobject]@{ ok=($exitCode -eq 0); exitCode=$exitCode; file=$resolvedFile; stdout=(Get-Content -Raw $outFile -ErrorAction SilentlyContinue); stderr=(Get-Content -Raw $errFile -ErrorAction SilentlyContinue) }
  } catch {
    [pscustomobject]@{ ok=$false; exitCode=-1; file=$resolvedFile; stderr=$_.Exception.Message; stdout="" }
  } finally {
    [Console]::OutputEncoding = $previousConsoleEncoding
    $OutputEncoding = $previousOutputEncoding
    Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue
  }
}

function Invoke-OptionalWatchCapture([string]$File, [string[]]$CommandArgs, [string]$Cwd, [int]$TimeoutSeconds = 20) {
  $result = Invoke-Captured $File $CommandArgs $Cwd $TimeoutSeconds
  if ($result.ok) {
    return $result
  }

  $combined = @($result.stdout, $result.stderr) -join "`n"
  if ($combined -match "unrecognized subcommand 'watch'") {
    return [pscustomobject]@{
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
function Get-GitSnapshot([string]$Repo, [string]$Base = "") {
  $branch = Get-CurrentBranch $Repo
  $head = Get-Head $Repo
  $status = Invoke-GitText $Repo @('status','--short','--branch')
  $statusLines = @()
  if ($status.output) { $statusLines = @($status.output -split "`n" | Where-Object { $_ }) }
  $dirty = @($statusLines | Where-Object { $_ -notmatch '^## ' })
  $statusPreview = @($statusLines | Select-Object -First 20)
  $ahead = $null; $behind = $null
  if ($Base) {
    $ab = Invoke-GitText $Repo @('rev-list','--left-right','--count',"$Base...HEAD")
    if ($ab.ok) {
      $parts = $ab.output.Trim() -split '\s+'
      if ($parts.Count -ge 2) { $behind = [int]$parts[0]; $ahead = [int]$parts[1] }
    }
  }
  [pscustomobject]@{ ok=($status.ok); currentBranch=$branch; head=$head; clean=($dirty.Count -eq 0); dirtyCount=$dirty.Count; statusPreview=($statusPreview -join "`n"); ahead=$ahead; behind=$behind; base=$Base }
}
function New-Observation($Project) {
  $branches = New-Object System.Collections.Generic.List[object]
  foreach ($b in To-Array $Project.branches) {
    $path = if ($b.path) { Convert-ToAbs $b.path } else { $Project.root }
    $git = if (Test-Path -LiteralPath $path) { Get-GitSnapshot $path $b.baseBranch } else { [pscustomobject]@{ ok=$false; error='path missing' } }
    $branches.Add([pscustomobject]@{ name=$b.name; kind=$b.kind; status=$b.status; path=$path; baseBranch=$b.baseBranch; returnBranch=$b.returnBranch; git=$git; lastIndexedCommit=$b.lastIndexedCommit; watchEnabled=$b.watchEnabled })
  }
  [ordered]@{
    timestamp = UtcNow
    project = $Project.name
    root = $Project.root
    qdrant = Test-Qdrant $Project.qdrantUrl
    daemonTcp = Test-TcpEndpoint $Project.daemonEndpoint
    wqmHealth = Invoke-Captured $WqmPath @('status','health') $Project.root 20
    queue = Invoke-Captured $WqmPath @('queue','stats') $Project.root 20
    branches = @($branches)
  }
}
function Save-Observation($Obs) {
  Ensure-Dir $LogDir
  $obsDir = Join-Path (Split-Path -Parent $RegistryPath) 'observability'
  Ensure-Dir $obsDir
  $safe = Safe-BranchSlug $Obs.project
  $file = Join-Path $obsDir ("$safe-latest.json")
  $Obs | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $file -Encoding UTF8
  $line = ($Obs | ConvertTo-Json -Depth 20 -Compress)
  $log = Join-Path $LogDir ("indexed-projects-" + (Get-Date -Format 'yyyyMMdd') + ".jsonl")
  $line | Add-Content -LiteralPath $log -Encoding UTF8
  return $file
}
function Output-Result($Value) {
  if (Test-TrueValue $Json) { $Value | ConvertTo-Json -Depth 20 }
  else { $Value | ConvertTo-Json -Depth 20 }
}

$WqmPath = Resolve-WqmPath -BaseDir $RepoDir
if (-not $WqmPath) { $WqmPath = "wqm" }
$GitPath = Resolve-GitPath

try {
  switch ($Action) {
    'init' {
      Assert-MutationAllowed
      if (-not (Test-Path -LiteralPath $RegistryPath)) { Write-Registry (New-Registry) }
      Output-Result ([ordered]@{ success=$true; action=$Action; registry=$RegistryPath })
    }
    'add-project' {
      Assert-MutationAllowed
      if (-not $ProjectName) { throw "ProjectName obrigatorio" }
      if (-not $ProjectDir) { throw "ProjectDir obrigatorio" }
      $root = Convert-ToAbs $ProjectDir
      if (-not (Test-Path -LiteralPath $root)) { throw "ProjectDir nao existe: $root" }
      if (-not (Test-Path -LiteralPath (Join-Path $root '.git'))) { throw "ProjectDir nao parece ser repo git: $root" }
      $registry = Read-Registry
      $currentBranch = Get-CurrentBranch $root
      $head = Get-Head $root
      $registerResult = Invoke-WqmProjectRegister -PathValue $root -ProjectName $ProjectName -Cwd $root
      if (-not $registerResult.ok) {
        Write-Warning "Auto-registration falhou para $root (codigo $($registerResult.exitCode)). O registro local foi mantido; rode index-register-wqm depois se quiser reintentar."
      }
      $project = [ordered]@{ name=$ProjectName; root=$root; projectId=$null; qdrantUrl=$QdrantUrl; daemonEndpoint=$DaemonEndpoint; defaultBranch=$BaseBranch; tenantStrategy='project'; enabled=$true; createdAt=UtcNow; updatedAt=UtcNow; branches=@([ordered]@{ name=$currentBranch; kind='primary'; path=$root; baseBranch=$BaseBranch; returnBranch=$currentBranch; status='active'; createdBy='human'; createdAt=UtcNow; lastSeenAt=UtcNow; baseCommit=$null; headCommit=$head; lastIndexedCommit=$(if ($registerResult.ok) { $head } else { $null }); watchEnabled=$true; indexed=$registerResult.ok; purpose='primary working tree' }) }
      Upsert-Project $registry $project
      Write-Registry $registry
      Output-Result ([ordered]@{ success=$true; action=$Action; project=$project; register=$registerResult })
    }
    'list-projects' {
      $registry = Read-Registry
      Output-Result ([ordered]@{ success=$true; registry=$RegistryPath; projects=@(Get-Projects $registry | Select-Object name,root,defaultBranch,tenantStrategy,enabled) })
    }
    'project-status' {
      $registry = Read-Registry; $project = Find-Project $registry
      Output-Result ([ordered]@{ success=$true; project=$project.name; root=$project.root; branches=@($project.branches); observation=(New-Observation $project) })
    }
    'status-all' {
      $registry = Read-Registry; $items = @()
      foreach ($p in Get-Projects $registry | Where-Object { $_.enabled }) { $items += [pscustomobject](New-Observation $p) }
      Output-Result ([ordered]@{ success=$true; count=$items.Count; projects=$items })
    }
    'list-branches' {
      $registry = Read-Registry; $project = Find-Project $registry
      Output-Result ([ordered]@{ success=$true; project=$project.name; branches=@($project.branches) })
    }
    'start-agent-branch' {
      Assert-MutationAllowed
      if (-not $BranchName) { throw "BranchName obrigatorio" }
      $createdProject = $false
      $registry = Read-Registry
      try {
        $project = Find-Project $registry
      } catch {
        $lookupError = $_.Exception.Message
        if ($lookupError -notmatch 'Projeto indexado nao encontrado|Informe -ProjectName ou -ProjectDir') {
          throw
        }

        $bootstrapSource = if ($ProjectDir) { $ProjectDir } else { $RepoDir }
        if (-not $bootstrapSource) { throw $lookupError }

        $bootstrap = Sync-CurrentBranch $registry $bootstrapSource
        if (-not $bootstrap -or $bootstrap.skipped) {
          throw "Nao foi possivel indexar automaticamente o projeto a partir de ${bootstrapSource}: $($bootstrap.reason)"
        }
        $createdProject = [bool]$bootstrap.createdProject

        $registry = Read-Registry
        $bootstrapRoot = Get-ProjectRootFromRepo $bootstrapSource
        if (-not $bootstrapRoot) {
          $bootstrapRoot = Convert-ToAbs $bootstrapSource
        }

        $project = Find-ProjectByRoot $registry $bootstrapRoot
        if (-not $project) {
          throw "Projeto indexado nao encontrado apos auto-indexacao: $bootstrapSource"
        }
      }
      $repo = $project.root
      $return = if ($ReturnBranch) { $ReturnBranch } else { Get-CurrentBranch $repo }
      $baseCommit = $null
      if (Test-TrueValue $UseWorktree) {
        $slug = Safe-BranchSlug $BranchName
        if (-not $WorktreePath) {
          $parent = if ($WorktreeRoot) { Convert-ToAbs $WorktreeRoot } else { Split-Path -Parent $project.root }
          $WorktreePath = Join-Path $parent ((Split-Path -Leaf $project.root) + '-' + $slug)
        }
        $wt = Convert-ToAbs $WorktreePath
        if (Test-Path -LiteralPath $wt) { throw "WorktreePath ja existe: $wt" }
        $baseCommit = (Invoke-GitText $repo @('rev-parse', $BaseBranch)).output.Trim()
        if (Branch-Exists $repo $BranchName) { Invoke-Git $repo @('worktree','add', $wt, $BranchName) }
        else { Invoke-Git $repo @('worktree','add','-b', $BranchName, $wt, $BaseBranch) }
        $branchPath = $wt
      } else {
        Assert-Clean $repo
        Invoke-Git $repo @('checkout', $BaseBranch)
        $baseCommit = Get-Head $repo
        if (Branch-Exists $repo $BranchName) { Invoke-Git $repo @('checkout', $BranchName) }
        else { Invoke-Git $repo @('checkout','-b', $BranchName) }
        $branchPath = $repo
      }
      $head = Get-Head $branchPath
      $registerResult = Invoke-WqmProjectRegister -PathValue $branchPath -ProjectName $project.name -Cwd $branchPath
      if (-not $registerResult.ok) {
        Write-Warning "Auto-registration falhou para $branchPath (codigo $($registerResult.exitCode)). O branch foi registrado localmente; rode index-register-wqm depois se quiser reintentar."
      }
      $branch = [ordered]@{ name=$BranchName; kind='agent'; path=(Convert-ToAbs $branchPath); baseBranch=$BaseBranch; returnBranch=$return; status='active'; createdBy=$CreatedBy; createdAt=UtcNow; lastSeenAt=UtcNow; baseCommit=$baseCommit; headCommit=$head; lastIndexedCommit=$(if ($registerResult.ok) { $head } else { $null }); watchEnabled=$true; indexed=$registerResult.ok; purpose=$Purpose; useWorktree=(Test-TrueValue $UseWorktree) }
      Upsert-Branch $registry $project $branch
      Write-Registry $registry
      Output-Result ([ordered]@{ success=$true; action=$Action; createdProject=$createdProject; project=$project.name; branch=$branch; register=$registerResult; message='Branch de agente criada/registrada. Nao houve merge para branch original.' })
    }
    'finish-agent-branch' {
      Assert-MutationAllowed
      if (-not $BranchName) { throw "BranchName obrigatorio" }
      $registry = Read-Registry; $project = Find-Project $registry; $branch = Find-Branch $project $BranchName
      if (-not $branch) { throw "Branch nao registrada: $BranchName" }
      $path = Convert-ToAbs $branch.path
      $branch.headCommit = Get-Head $path
      $branch.lastSeenAt = UtcNow
      $branch.status = 'ready_for_review'
      $branch.note = 'Pronta para revisao humana. Merge nao executado.'
      Upsert-Branch $registry $project $branch
      Write-Registry $registry
      Output-Result ([ordered]@{ success=$true; action=$Action; project=$project.name; branch=$branch; message='Marcada como ready_for_review sem merge.' })
    }
    'abandon-agent-branch' {
      Assert-MutationAllowed
      if (-not $BranchName) { throw "BranchName obrigatorio" }
      $registry = Read-Registry; $project = Find-Project $registry; $branch = Find-Branch $project $BranchName
      if (-not $branch) { throw "Branch nao registrada: $BranchName" }
      $branch.status = 'abandoned'; $branch.lastSeenAt = UtcNow; $branch.note = 'Abandonada no registry. Worktree/branch nao deletados automaticamente.'
      if ((Test-TrueValue $RemoveWorktree) -and $branch.useWorktree) { Invoke-Git $project.root @('worktree','remove', (Convert-ToAbs $branch.path)) ; $branch.note = 'Abandonada e worktree removida por solicitacao explicita.' }
      Upsert-Branch $registry $project $branch; Write-Registry $registry
      Output-Result ([ordered]@{ success=$true; action=$Action; project=$project.name; branch=$branch })
    }
    'agent-branch-status' {
      $registry = Read-Registry; $project = Find-Project $registry; $branch = Find-Branch $project $BranchName
      if (-not $branch) { throw "Branch nao registrada: $BranchName" }
      Output-Result ([ordered]@{ success=$true; project=$project.name; branch=$branch; git=(Get-GitSnapshot (Convert-ToAbs $branch.path) $branch.baseBranch) })
    }
    'observe-project' {
      $registry = Read-Registry; $project = Find-Project $registry; $obs = New-Observation $project; $file = Save-Observation $obs
      Output-Result ([ordered]@{ success=$true; action=$Action; observation=$obs; savedTo=$file })
    }
    'observe-all' {
      $registry = Read-Registry; $items=@(); foreach ($p in Get-Projects $registry | Where-Object { $_.enabled }) { $obs = New-Observation $p; Save-Observation $obs | Out-Null; $items += [pscustomobject]$obs }
      Output-Result ([ordered]@{ success=$true; action=$Action; count=$items.Count; observations=$items })
    }
    'incremental-check' {
      $registry = Read-Registry; $project = Find-Project $registry; $results=@()
      foreach ($b in To-Array $project.branches) { $path = Convert-ToAbs $b.path; $projectWqmPath = Convert-ToWqmPath $path; $results += [pscustomobject]@{ branch=$b.name; path=$path; projectStatus=(Invoke-Captured $WqmPath @('project','status',$projectWqmPath) $path 20); projectCheck=(Invoke-Captured $WqmPath @('project','check',$projectWqmPath,'--json') $path 60); watchList=(Invoke-OptionalWatchCapture $WqmPath @('watch','list','--json') $path 20); queue=(Invoke-Captured $WqmPath @('queue','stats') $path 20) } }
      Output-Result ([ordered]@{ success=$true; action=$Action; project=$project.name; results=$results })
    }
    'incremental-check-all' {
      $registry = Read-Registry; $all=@(); foreach ($p in Get-Projects $registry | Where-Object { $_.enabled }) { $ProjectName=$p.name; $project=$p; $results=@(); foreach ($b in To-Array $project.branches) { $path=Convert-ToAbs $b.path; $projectWqmPath = Convert-ToWqmPath $path; $results += [pscustomobject]@{ project=$project.name; branch=$b.name; path=$path; projectCheck=(Invoke-Captured $WqmPath @('project','check',$projectWqmPath,'--json') $path 60); queue=(Invoke-Captured $WqmPath @('queue','stats') $path 20) } }; $all += $results }
      Output-Result ([ordered]@{ success=$true; action=$Action; results=$all })
    }
    'register-wqm' {
      Assert-MutationAllowed
      $registry = Read-Registry; $project = Find-Project $registry; $results=@(); $changed=$false
      foreach ($b in To-Array $project.branches) {
        $path = Convert-ToAbs $b.path
        $register = Invoke-WqmProjectRegister -PathValue $path -ProjectName $project.name -Cwd $path
        if ($register.ok) {
          $b.lastIndexedCommit = Get-Head $path
          $b.indexed = $true
          $changed = $true
        } else {
          $b.indexed = $false
          $changed = $true
        }
        $results += [pscustomobject]@{ branch=$b.name; path=$path; register=$register }
      }
      if ($changed) { Upsert-Project $registry $project; Write-Registry $registry }
      Output-Result ([ordered]@{ success=$true; action=$Action; project=$project.name; results=$results })
    }
    'register-all-wqm' {
      Assert-MutationAllowed
      $registry = Read-Registry; $all=@(); $changed=$false
      foreach ($p in Get-Projects $registry | Where-Object { $_.enabled }) {
        foreach ($b in To-Array $p.branches) {
          $path = Convert-ToAbs $b.path
          $register = Invoke-WqmProjectRegister -PathValue $path -ProjectName $p.name -Cwd $path
          if ($register.ok) {
            $b.lastIndexedCommit = Get-Head $path
            $b.indexed = $true
            $changed = $true
          } else {
            $b.indexed = $false
            $changed = $true
          }
          $all += [pscustomobject]@{ project=$p.name; branch=$b.name; path=$path; register=$register }
        }
        Upsert-Project $registry $p
      }
      if ($changed) { Write-Registry $registry }
      Output-Result ([ordered]@{ success=$true; action=$Action; results=$all })
    }
    'sync-current-branch' {
      Assert-MutationAllowed
      $registry = Read-Registry
      $result = Sync-CurrentBranch $registry $RepoDir
      if ($result -and $result.skipped) {
        Output-Result $result
      } else {
        Output-Result $result
      }
    }
    'cleanup-orphans' {
      $mutateCleanup = Test-TrueValue $Mutate
      if ($mutateCleanup) { Assert-MutationAllowed }
      $registry = Read-Registry
      $report = Cleanup-OrphanedIndex $registry $mutateCleanup
      Output-Result $report
    }
    default { throw "Acao nao implementada: $Action" }
  }
} catch {
  $err = [ordered]@{
    success = $false
    action = $Action
    error = $_.Exception.Message
    stack = $_.ScriptStackTrace
    position = $_.InvocationInfo.PositionMessage
    registry = $RegistryPath
  }
  $err | ConvertTo-Json -Depth 12
  exit 1
}
