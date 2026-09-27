# Daily job discovery for Vouch. Run by a Windows scheduled task; safe to run by hand.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\daily-discover.ps1 [extra vouch args]
# Starts Docker Desktop if needed, waits for the database, runs `vouch discover`, and writes
# everything to logs\discover-<date>.log.

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force (Join-Path $root "logs") | Out-Null
$log = Join-Path $root ("logs\discover-{0:yyyy-MM-dd}.log" -f (Get-Date))

function Write-Log($message) {
    "{0:HH:mm:ss} {1}" -f (Get-Date), $message | Out-File -Append -Encoding utf8 $log
}

function Invoke-Logged($exe, [string[]]$arguments) {
    # cmd /c merges stderr into stdout without PowerShell 5.1 turning it into error records.
    $line = ($arguments | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    cmd /c "$exe $line 2>&1" | Out-File -Append -Encoding utf8 $log
    return $LASTEXITCODE
}

Write-Log "Starting job discovery"

# 1. Docker Desktop (the database runs in it).
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $desktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $desktop) {
        Write-Log "Docker isn't running; starting Docker Desktop"
        Start-Process $desktop
    }
    $deadline = (Get-Date).AddMinutes(3)
    do {
        Start-Sleep -Seconds 5
        docker info *> $null
    } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: Docker didn't start within 3 minutes; skipping today's run"
        exit 1
    }
}

# 2. The database container, then wait until Postgres accepts connections.
Invoke-Logged "docker" @("compose", "up", "-d") | Out-Null
$deadline = (Get-Date).AddMinutes(1)
do {
    Start-Sleep -Seconds 2
    docker compose exec -T db pg_isready -U jobmatcher *> $null
} while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)

# 3. Discovery.
$code = Invoke-Logged "python" (@("-m", "uv", "run", "vouch", "discover") + $args)
Write-Log ("Finished with exit code {0}" -f $code)
exit $code
