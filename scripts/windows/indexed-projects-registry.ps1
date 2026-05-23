[CmdletBinding()]
param(
  [string]$RegistryPath = ".wqm-fork\indexed-projects.json",
  [ValidateSet("init","list-projects","add-project","remove-project","project-status","status-all","list-branches","add-branch","start-agent-branch","finish-agent-branch","abandon-agent-branch","agent-branch-status","observe-project","observe-all","incremental-check","incremental-check-all","register-wqm","register-all-wqm")]
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
function Safe-BranchSlug([string]$Name) { return ($Name -replace '[^A-Za-z0-9._-]+','-').Trim('-') }
function Ensure-Dir([string]$PathValue) { if (-not (Test-Path -LiteralPath $PathValue)) { New-Item -ItemType Directory -Path $PathValue -Force | Out-Null } }
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
  $Registry | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $RegistryPath -Encoding UTF8
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
    branches = $branches
  }
}
function Get-Projects($Registry) { To-Array $Registry.projects | ForEach-Object { Normalize-Project $_ } }
function Find-Project($Registry) {
  $projects = @(Get-Projects $Registry)
  if ($ProjectName) { $m = @($projects | Where-Object { $_.name -eq $ProjectName }) }
  elseif ($ProjectDir) { $abs = Convert-ToAbs $ProjectDir; $m = @($projects | Where-Object { (Convert-ToAbs $_.root) -eq $abs }) }
  else { throw "Informe -ProjectName ou -ProjectDir." }
  if ($m.Count -eq 0) { throw "Projeto indexado nao encontrado: $ProjectName $ProjectDir" }
  if ($m.Count -gt 1) { throw "Projeto ambiguo: $ProjectName $ProjectDir" }
  return $m[0]
}
function Invoke-Git([string]$Repo, [string[]]$Args) {
  & git -C $Repo @Args
  if ($LASTEXITCODE -ne 0) { throw "git -C $Repo $($Args -join ' ') falhou com codigo $LASTEXITCODE" }
}
function Invoke-GitText([string]$Repo, [string[]]$Args) {
  $out = & git -C $Repo @Args 2>&1
  [pscustomobject]@{ code = $LASTEXITCODE; ok = ($LASTEXITCODE -eq 0); output = (@($out) -join "`n") }
}
function Assert-Clean([string]$Repo) {
  $s = & git -C $Repo status --porcelain
  if ($s) { throw "Working tree suja em $Repo. Commit/stash/limpe manualmente antes." }
}
function Get-Head([string]$Repo) {
  $r = Invoke-GitText $Repo @('rev-parse','HEAD')
  if ($r.ok) { return $r.output.Trim() }
  return $null
}
function Get-CurrentBranch([string]$Repo) {
  $r = Invoke-GitText $Repo @('branch','--show-current')
  if ($r.ok) { return $r.output.Trim() }
  return $null
}
function Branch-Exists([string]$Repo, [string]$Branch) {
  & git -C $Repo show-ref --verify --quiet "refs/heads/$Branch"
  return $LASTEXITCODE -eq 0
}
function Find-Branch($Project, [string]$Name) { @(To-Array $Project.branches | Where-Object { $_.name -eq $Name }) | Select-Object -First 1 }
function Upsert-Project($Registry, $ProjectObject) {
  $items = New-Object System.Collections.Generic.List[object]
  foreach ($p in To-Array $Registry.projects) { if ($p.name -ne $ProjectObject.name) { $items.Add($p) } }
  $items.Add([pscustomobject]$ProjectObject)
  $Registry.projects = @($items)
}
function Upsert-Branch($Registry, $Project, $BranchObject) {
  $stored = @(To-Array $Registry.projects | Where-Object { $_.name -eq $Project.name }) | Select-Object -First 1
  if (-not $stored) { throw "Projeto nao encontrado para atualizar branch: $($Project.name)" }
  $list = New-Object System.Collections.Generic.List[object]
  foreach ($b in To-Array $stored.branches) { if ($b.name -ne $BranchObject.name) { $list.Add($b) } }
  $list.Add([pscustomobject]$BranchObject)
  $stored.branches = @($list)
  $stored.updatedAt = UtcNow
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
function Invoke-Captured([string]$File, [string[]]$Args, [string]$Cwd, [int]$TimeoutSeconds = 20) {
  $outFile = [System.IO.Path]::GetTempFileName(); $errFile = [System.IO.Path]::GetTempFileName()
  try {
    $p = Start-Process -FilePath $File -ArgumentList $Args -WorkingDirectory $Cwd -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    if (-not $p.WaitForExit($TimeoutSeconds * 1000)) { try { $p.Kill() } catch {}; return [pscustomobject]@{ ok=$false; exitCode=-1; stderr="timeout ${TimeoutSeconds}s"; stdout="" } }
    [pscustomobject]@{ ok=($p.ExitCode -eq 0); exitCode=$p.ExitCode; stdout=(Get-Content -Raw $outFile -ErrorAction SilentlyContinue); stderr=(Get-Content -Raw $errFile -ErrorAction SilentlyContinue) }
  } catch { [pscustomobject]@{ ok=$false; exitCode=-1; stderr=$_.Exception.Message; stdout="" } }
  finally { Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue }
}
function Get-GitSnapshot([string]$Repo, [string]$Base = "") {
  $branch = Get-CurrentBranch $Repo
  $head = Get-Head $Repo
  $status = Invoke-GitText $Repo @('status','--short','--branch')
  $dirty = @()
  if ($status.output) { $dirty = @($status.output -split "`n" | Where-Object { $_ -and $_ -notmatch '^## ' }) }
  $ahead = $null; $behind = $null
  if ($Base) {
    $ab = Invoke-GitText $Repo @('rev-list','--left-right','--count',"$Base...HEAD")
    if ($ab.ok) {
      $parts = $ab.output.Trim() -split '\s+'
      if ($parts.Count -ge 2) { $behind = [int]$parts[0]; $ahead = [int]$parts[1] }
    }
  }
  [pscustomobject]@{ ok=$status.ok; currentBranch=$branch; head=$head; clean=($dirty.Count -eq 0); dirtyCount=$dirty.Count; status=$status.output; ahead=$ahead; behind=$behind; base=$Base }
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
    wqmHealth = Invoke-Captured wqm @('status','health') $Project.root 20
    queue = Invoke-Captured wqm @('queue','stats') $Project.root 20
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
      $project = [ordered]@{ name=$ProjectName; root=$root; projectId=$null; qdrantUrl=$QdrantUrl; daemonEndpoint=$DaemonEndpoint; defaultBranch=$BaseBranch; tenantStrategy='project'; enabled=$true; createdAt=UtcNow; updatedAt=UtcNow; branches=@([ordered]@{ name=$currentBranch; kind='primary'; path=$root; baseBranch=$BaseBranch; returnBranch=$currentBranch; status='active'; createdBy='human'; createdAt=UtcNow; lastSeenAt=UtcNow; baseCommit=$null; headCommit=$head; lastIndexedCommit=$null; watchEnabled=$true; indexed=$true; purpose='primary working tree' }) }
      Upsert-Project $registry $project
      Write-Registry $registry
      Output-Result ([ordered]@{ success=$true; action=$Action; project=$project })
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
      $registry = Read-Registry; $project = Find-Project $registry
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
      $branch = [ordered]@{ name=$BranchName; kind='agent'; path=(Convert-ToAbs $branchPath); baseBranch=$BaseBranch; returnBranch=$return; status='active'; createdBy=$CreatedBy; createdAt=UtcNow; lastSeenAt=UtcNow; baseCommit=$baseCommit; headCommit=$head; lastIndexedCommit=$null; watchEnabled=$true; indexed=$true; purpose=$Purpose; useWorktree=(Test-TrueValue $UseWorktree) }
      Upsert-Branch $registry $project $branch
      Write-Registry $registry
      Output-Result ([ordered]@{ success=$true; action=$Action; project=$project.name; branch=$branch; message='Branch de agente criada/registrada. Nao houve merge para branch original.' })
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
      foreach ($b in To-Array $project.branches) { $path = Convert-ToAbs $b.path; $results += [pscustomobject]@{ branch=$b.name; path=$path; projectStatus=(Invoke-Captured wqm @('project','status',$path) $path 20); projectCheck=(Invoke-Captured wqm @('project','check',$path,'--json') $path 60); watchList=(Invoke-Captured wqm @('project','watch','list','--json') $path 20); queue=(Invoke-Captured wqm @('queue','stats') $path 20) } }
      Output-Result ([ordered]@{ success=$true; action=$Action; project=$project.name; results=$results })
    }
    'incremental-check-all' {
      $registry = Read-Registry; $all=@(); foreach ($p in Get-Projects $registry | Where-Object { $_.enabled }) { $ProjectName=$p.name; $project=$p; $results=@(); foreach ($b in To-Array $project.branches) { $path=Convert-ToAbs $b.path; $results += [pscustomobject]@{ project=$project.name; branch=$b.name; path=$path; projectCheck=(Invoke-Captured wqm @('project','check',$path,'--json') $path 60); queue=(Invoke-Captured wqm @('queue','stats') $path 20) } }; $all += $results }
      Output-Result ([ordered]@{ success=$true; action=$Action; results=$all })
    }
    'register-wqm' {
      Assert-MutationAllowed
      $registry = Read-Registry; $project = Find-Project $registry; $results=@()
      foreach ($b in To-Array $project.branches) { $path=Convert-ToAbs $b.path; $results += [pscustomobject]@{ branch=$b.name; path=$path; register=(Invoke-Captured wqm @('project','register',$path,'--yes') $path 60) } }
      Output-Result ([ordered]@{ success=$true; action=$Action; project=$project.name; results=$results })
    }
    'register-all-wqm' {
      Assert-MutationAllowed
      $registry = Read-Registry; $all=@(); foreach ($p in Get-Projects $registry | Where-Object { $_.enabled }) { foreach ($b in To-Array $p.branches) { $path=Convert-ToAbs $b.path; $all += [pscustomobject]@{ project=$p.name; branch=$b.name; path=$path; register=(Invoke-Captured wqm @('project','register',$path,'--yes') $path 60) } } }
      Output-Result ([ordered]@{ success=$true; action=$Action; results=$all })
    }
    default { throw "Acao nao implementada: $Action" }
  }
} catch {
  $err = [ordered]@{ success=$false; action=$Action; error=$_.Exception.Message; registry=$RegistryPath }
  $err | ConvertTo-Json -Depth 12
  exit 1
}
