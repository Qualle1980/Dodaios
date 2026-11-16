<#
.SYNOPSIS
  Claude Code PreToolUse hook: ask before a destructive shell command runs.

.DESCRIPTION
  Reads the hook payload (JSON) from stdin. For Bash and PowerShell tool calls
  whose command matches a destructive pattern it prints

    {"hookSpecificOutput":{"hookEventName":"PreToolUse",
     "permissionDecision":"ask","permissionDecisionReason":"..."}}

  and exits 0. Anything else produces no output and exits 0, so the normal
  permission flow decides. A payload it cannot parse is allowed through with a
  warning on stderr: the guard must never wedge every tool call.

  The hook registration lives in user or local settings (.claude/* is not
  versioned); see .agent-context/CURRENT.md.
#>
$ErrorActionPreference = 'Stop'
# The caller reads UTF-8; Windows PowerShell defaults to the OEM codepage.
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
[Console]::InputEncoding = $utf8

$patterns = [ordered]@{
    'rm -r / rm -rf'            = '(^|[\s;&|(])rm\s+(-[A-Za-z]*[rR][A-Za-z]*|--recursive)\b'
    'Remove-Item -Recurse'      = '(?i)(Remove-Item|\b(rd|rmdir|del|erase|ri)\b)[^\r\n]*\s-(Recurse|r)\b'
    'cmd rd/del /s'             = '(?i)\b(rd|rmdir|del|erase)\b[^\r\n]*\s/s\b'
    'git reset --hard'          = '\bgit\b[^\r\n]*\breset\b[^\r\n]*--hard\b'
    'git push --force'          = '\bgit\b[^\r\n]*\bpush\b[^\r\n]*(\s--force(-with-lease)?\b|\s-[A-Za-z]*f\b|\s\+\S)'
    'git clean -f'              = '\bgit\b[^\r\n]*\bclean\b[^\r\n]*\s-[A-Za-z]*f'
    'git checkout/restore .'    = '\bgit\b[^\r\n]*\b(checkout|restore)\b[^\r\n]*\s(--\s+)?\.(\s|$)'
    'git branch -D'             = '\bgit\b[^\r\n]*\bbranch\b[^\r\n]*\s(-D|--delete\s+--force|-d\s+-f)\b'
    'git stash drop/clear'      = '\bgit\b[^\r\n]*\bstash\s+(drop|clear)\b'
    'git worktree remove -f'    = '\bgit\b[^\r\n]*\bworktree\s+(remove|prune)\b[^\r\n]*(--force|-f)\b'
    'git update-ref -d'         = '\bgit\b[^\r\n]*\bupdate-ref\s+-d\b'
    'format / diskpart'         = '(?i)(^|[\s;&|])(format(\.com)?\s+[A-Za-z]:|diskpart|Format-Volume|Clear-Disk)\b'
    'SQL DROP/TRUNCATE'         = '(?i)\b(DROP\s+(TABLE|DATABASE|SCHEMA)|TRUNCATE\s+TABLE)\b'
}

try {
    $raw = [Console]::In.ReadToEnd()
    $payload = $raw | ConvertFrom-Json
} catch {
    [Console]::Error.WriteLine("guard-destructive: unreadable hook payload, allowing: $($_.Exception.Message)")
    exit 0
}

if ($payload.tool_name -notin @('Bash', 'PowerShell')) { exit 0 }
$command = [string]$payload.tool_input.command
if ([string]::IsNullOrWhiteSpace($command)) { exit 0 }

$hits = @()
foreach ($name in $patterns.Keys) {
    # Case-sensitive on purpose: `git branch -D` is destructive, `-d` is not.
    if ($command -cmatch $patterns[$name]) { $hits += $name }
}
if ($hits.Count -eq 0) { exit 0 }

$result = @{
    hookSpecificOutput = @{
        hookEventName            = 'PreToolUse'
        permissionDecision       = 'ask'
        permissionDecisionReason = "Destructive command ($($hits -join ', ')). Confirm before it runs."
    }
}
[Console]::Out.Write(($result | ConvertTo-Json -Depth 4 -Compress))
exit 0
