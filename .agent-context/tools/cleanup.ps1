<#
.SYNOPSIS
  Remove old files from .agent-context/logs. Dry run unless -Apply is given.

.DESCRIPTION
  Touches nothing outside .agent-context/logs and never deletes .gitkeep.
  Without -Apply it only lists what would be removed.

.EXAMPLE
  pwsh .agent-context/tools/cleanup.ps1               # list files older than 14 days
  pwsh .agent-context/tools/cleanup.ps1 -Days 3 -Apply
#>
[CmdletBinding()]
param(
    [ValidateRange(0, 3650)][int]$Days = 14,
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'

$logs = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\logs'))
if (-not (Test-Path -LiteralPath $logs -PathType Container)) {
    Write-Output "no logs directory at $logs"
    exit 0
}
$prefix = $logs.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
$cutoff = (Get-Date).AddDays(-$Days)

$candidates = @(Get-ChildItem -LiteralPath $logs -File -Recurse -Force |
    Where-Object { $_.Name -ne '.gitkeep' -and $_.LastWriteTime -lt $cutoff })

$removed = 0
foreach ($file in $candidates) {
    # Guard against links or odd paths resolving outside logs/.
    if (-not $file.FullName.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Warning "skipped, outside logs/: $($file.FullName)"
        continue
    }
    if ($Apply) {
        Remove-Item -LiteralPath $file.FullName -Force
        $removed++
        Write-Output "deleted $($file.FullName)"
    } else {
        Write-Output "would delete $($file.FullName)"
    }
}

if ($Apply) {
    Write-Output "deleted $removed file(s) older than $Days day(s)"
} else {
    Write-Output "dry run: $($candidates.Count) file(s) older than $Days day(s); pass -Apply to delete"
}
