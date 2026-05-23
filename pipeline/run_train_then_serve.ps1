param(
  [string]$Python = "",
  [string]$DataRoot = "",
  [string]$DatasetDir = "",
  [string]$RunDir = "",
  [string]$ModelOutputPath = "",
  [int]$Epochs = 8,
  [int]$BatchSize = 4,
  [int]$GradAccumSteps = 4,
  [int]$Resolution = 512,
  [int]$NumWorkers = 0,
  [string]$Device = "cuda",
  [string]$PretrainWeights = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $Root

if ([string]::IsNullOrWhiteSpace($Python)) {
  $Python = if (![string]::IsNullOrWhiteSpace($env:LENTA_PIPELINE_PYTHON)) {
    $env:LENTA_PIPELINE_PYTHON
  } elseif (![string]::IsNullOrWhiteSpace($env:PIPELINE_PYTHON)) {
    $env:PIPELINE_PYTHON
  } else {
    "python"
  }
}

$RuntimeRoot = if (![string]::IsNullOrWhiteSpace($env:RUNTIME_DIR)) {
  $env:RUNTIME_DIR
} elseif (![string]::IsNullOrWhiteSpace($env:LENTA_WEB_RUNTIME_ROOT)) {
  $env:LENTA_WEB_RUNTIME_ROOT
} else {
  Join-Path $ProjectRoot "runtime"
}
$TrainingRoot = Join-Path $RuntimeRoot "training"

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
  $DataRoot = if (![string]::IsNullOrWhiteSpace($env:TRAINING_DATA_DIR)) {
    $env:TRAINING_DATA_DIR
  } else {
    Join-Path $ProjectRoot "artifacts/training_data"
  }
}
if ([string]::IsNullOrWhiteSpace($DatasetDir)) {
  $DatasetDir = Join-Path $TrainingRoot "datasets/rfdetr_price_tag_all_annotated_tiled1280"
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $TrainingRoot "runs/rfdetr_from_scratch_e${Epochs}"
}
if ([string]::IsNullOrWhiteSpace($ModelOutputPath)) {
  $ModelOutputPath = Join-Path $TrainingRoot "models/rfdetr_from_scratch_e${Epochs}_checkpoint_best_total.pth"
}

& (Join-Path $Root "run_train_rfdetr.ps1") `
  -Python $Python `
  -DataRoot $DataRoot `
  -DatasetDir $DatasetDir `
  -RunDir $RunDir `
  -ModelOutputPath $ModelOutputPath `
  -Epochs $Epochs `
  -BatchSize $BatchSize `
  -GradAccumSteps $GradAccumSteps `
  -Resolution $Resolution `
  -NumWorkers $NumWorkers `
  -Device $Device `
  -PretrainWeights $PretrainWeights
if ($LASTEXITCODE -ne 0) {
  throw "Training process failed with exit code $LASTEXITCODE."
}
if (!(Test-Path -LiteralPath $ModelOutputPath)) {
  throw "TRAINING_CHECKPOINT_NOT_FOUND: expected output was not created: $ModelOutputPath"
}

$env:MODEL_PATH = $ModelOutputPath
$env:INFERENCE_DEVICE = $Device
$AppHost = if (![string]::IsNullOrWhiteSpace($env:APP_HOST)) { $env:APP_HOST } else { "0.0.0.0" }
$AppPort = if (![string]::IsNullOrWhiteSpace($env:APP_PORT)) { $env:APP_PORT } else { "8000" }

Write-Host "Training complete. Production artifact checkpoint was not changed."
Write-Host "Starting Shelf Vision with trained checkpoint: $ModelOutputPath"
& $Python (Join-Path $ProjectRoot "web/server.py") --host $AppHost --port $AppPort
if ($LASTEXITCODE -ne 0) {
  throw "Web server exited with code $LASTEXITCODE."
}
