param(
  [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = if ($env:LENTA_WEB_RUNTIME_ROOT) { $env:LENTA_WEB_RUNTIME_ROOT } else { Join-Path $RepoRoot "runtime" }
$PipelineRoot = if ($env:LENTA_PIPELINE_ROOT) { $env:LENTA_PIPELINE_ROOT } else { Join-Path $RepoRoot "pipeline" }
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
  $ConfigPath = if ($env:LENTA_RETRAIN_CONFIG) { $env:LENTA_RETRAIN_CONFIG } else { Join-Path $RuntimeRoot "retrain_config.json" }
}

$defaults = [ordered]@{
  enabled = $false
  schedule = "weekly"
  uncertain_root = if ($env:LENTA_UNCERTAIN_ROOT) { $env:LENTA_UNCERTAIN_ROOT } else { Join-Path $RuntimeRoot "uncertain_predictions" }
  pipeline_root = $PipelineRoot
  training_script = Join-Path $PipelineRoot "run_train_rfdetr_all_data.ps1"
  logs_root = Join-Path $RuntimeRoot "logs\retrain"
  epochs = 8
  note = "Auto retraining is intentionally disabled until enabled manually."
}

function Save-Config {
  param([object]$Config)
  $configDir = Split-Path -Parent $ConfigPath
  if (![string]::IsNullOrWhiteSpace($configDir)) {
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null
  }
  $Config | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8
}

if (!(Test-Path -LiteralPath $ConfigPath)) {
  Save-Config ([pscustomobject]$defaults)
}

$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$merged = [ordered]@{}
foreach ($key in $defaults.Keys) {
  $merged[$key] = $defaults[$key]
}
foreach ($property in $config.PSObject.Properties) {
  $merged[$property.Name] = $property.Value
}
$config = [pscustomobject]$merged
Save-Config $config

if (-not $config.enabled) {
  Write-Host "Auto retraining is disabled."
  exit 0
}

if (!(Test-Path -LiteralPath $config.uncertain_root)) {
  Write-Host "No uncertain predictions folder yet:"
  Write-Host $config.uncertain_root
  exit 0
}
if (!(Test-Path -LiteralPath $config.training_script)) {
  throw "Training script is missing: $($config.training_script)"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path $config.logs_root | Out-Null
$logPath = Join-Path $config.logs_root "weekly_retrain_$timestamp.log"

Start-Transcript -Path $logPath -Force | Out-Null
try {
  Write-Host "Weekly retraining started at $(Get-Date -Format o)"
  Write-Host "Pipeline: $($config.pipeline_root)"
  Write-Host "Uncertain predictions: $($config.uncertain_root)"
  Write-Host "Training script: $($config.training_script)"
  & powershell -NoProfile -ExecutionPolicy Bypass -File $config.training_script -Epochs ([int]$config.epochs)
  if ($LASTEXITCODE -ne 0) {
    throw "Retraining failed with exit code $LASTEXITCODE"
  }
  Write-Host "Weekly retraining finished at $(Get-Date -Format o)"
} finally {
  Stop-Transcript | Out-Null
}
