param(
  [switch]$Restart
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeRoot = if (Test-Path -LiteralPath "A:\") { "A:\lenta_web_runtime" } else { Join-Path $Root "runtime" }
$PipelineRoot = if (Test-Path -LiteralPath "A:\lenta_pipeline\handoff_v19_speed_full_pipeline_20260518_040946") {
  "A:\lenta_pipeline\handoff_v19_speed_full_pipeline_20260518_040946"
} else {
  Join-Path (Split-Path -Parent $Root) "pipeline"
}
$CatalogPath = if (Test-Path -LiteralPath "A:\lenta_data\db_hack.csv") {
  "A:\lenta_data\db_hack.csv"
} else {
  Join-Path (Split-Path -Parent $Root) "artifacts\db_hack.csv"
}
$CheckpointPath = if (Test-Path -LiteralPath "A:\lenta_pipeline\handoff_v19_speed_full_pipeline_20260518_040946\models\rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth") {
  "A:\lenta_pipeline\handoff_v19_speed_full_pipeline_20260518_040946\models\rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth"
} else {
  Join-Path (Split-Path -Parent $Root) "artifacts\models\rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth"
}
$ToolsRoot = Join-Path $RuntimeRoot "tools"
$LogsRoot = Join-Path $RuntimeRoot "logs"
$Cloudflared = Join-Path $ToolsRoot "cloudflared.exe"
$ServerOut = Join-Path $LogsRoot "server_stdout.log"
$ServerErr = Join-Path $LogsRoot "server_stderr.log"
$TunnelOut = Join-Path $LogsRoot "cloudflared_stdout.log"
$TunnelErr = Join-Path $LogsRoot "cloudflared_stderr.log"
$PublicUrlFile = Join-Path $Root "public_url.txt"

function Stop-ByName {
  param([string]$Name)
  Get-Process -Name $Name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

if ($Restart) {
  $listener = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($listener) {
    Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
  }
  Stop-ByName "cloudflared"
  Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
    Where-Object { $_.CommandLine -like "*keep_awake.ps1*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 1
}

New-Item -ItemType Directory -Force -Path $ToolsRoot, $LogsRoot | Out-Null
if (!(Test-Path -LiteralPath $Cloudflared)) {
  Invoke-WebRequest `
    -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
    -OutFile $Cloudflared `
    -UseBasicParsing
}

$server = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (!$server) {
  $env:LENTA_PIPELINE_ROOT = $PipelineRoot
  $env:LENTA_WEB_RUNTIME_ROOT = $RuntimeRoot
  $env:LENTA_CATALOG_PATH = $CatalogPath
  $env:LENTA_DETECTOR_CHECKPOINT = $CheckpointPath
  Start-Process `
    -FilePath python `
    -ArgumentList @("server.py", "--host", "127.0.0.1", "--port", "5173") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $ServerOut `
    -RedirectStandardError $ServerErr `
    -WindowStyle Hidden | Out-Null
  Start-Sleep -Seconds 2
}

$CurrentPid = $PID
$awake = Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
  Where-Object { $_.ProcessId -ne $CurrentPid -and $_.CommandLine -like "*keep_awake.ps1*" } |
  Select-Object -First 1
if (!$awake) {
  Start-Process `
    -FilePath powershell.exe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "keep_awake.ps1")) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden | Out-Null
}

$tunnel = Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Select-Object -First 1
if (!$tunnel) {
  Remove-Item -LiteralPath $TunnelOut, $TunnelErr -ErrorAction SilentlyContinue
  Start-Process `
    -FilePath $Cloudflared `
    -ArgumentList @("tunnel", "--url", "http://127.0.0.1:5173", "--no-autoupdate") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $TunnelOut `
    -RedirectStandardError $TunnelErr `
    -WindowStyle Hidden | Out-Null
}

$url = ""
for ($i = 0; $i -lt 90; $i++) {
  $url = Select-String -Path $TunnelErr, $TunnelOut -Pattern "https://(?!api\.)[-a-z0-9]+\.trycloudflare\.com" -AllMatches -ErrorAction SilentlyContinue |
    ForEach-Object { $_.Matches.Value } |
    Select-Object -First 1
  if ($url) {
    break
  }
  Start-Sleep -Seconds 1
}

if (!$url) {
  throw "Cloudflare Tunnel URL was not found. Check $TunnelErr"
}

$url | Set-Content -LiteralPath $PublicUrlFile -Encoding UTF8
Write-Host "Public URL: $url"
Write-Host "Health: $url/api/health"
Write-Host "Local: http://127.0.0.1:5173"
