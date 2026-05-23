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
  [string]$PretrainWeights = "",
  [switch]$SkipDatasetBuild,
  [switch]$BuildOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $Root
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if ([string]::IsNullOrWhiteSpace($Python)) {
  if (![string]::IsNullOrWhiteSpace($env:LENTA_PIPELINE_PYTHON)) {
    $Python = $env:LENTA_PIPELINE_PYTHON
  } elseif (![string]::IsNullOrWhiteSpace($env:PIPELINE_PYTHON)) {
    $Python = $env:PIPELINE_PYTHON
  } else {
    $Python = "python"
  }
}
$command = Get-Command $Python -ErrorAction SilentlyContinue
if ($null -ne $command -and ![string]::IsNullOrWhiteSpace($command.Source)) {
  $Python = $command.Source
}
$ArtifactsRoot = if (![string]::IsNullOrWhiteSpace($env:ARTIFACTS_DIR)) {
  $env:ARTIFACTS_DIR
} elseif (![string]::IsNullOrWhiteSpace($env:LENTA_ARTIFACTS_ROOT)) {
  $env:LENTA_ARTIFACTS_ROOT
} else {
  Join-Path $ProjectRoot "artifacts"
}
$RuntimeRoot = if (![string]::IsNullOrWhiteSpace($env:RUNTIME_DIR)) {
  $env:RUNTIME_DIR
} elseif (![string]::IsNullOrWhiteSpace($env:LENTA_WEB_RUNTIME_ROOT)) {
  $env:LENTA_WEB_RUNTIME_ROOT
} else {
  Join-Path $ProjectRoot "runtime"
}
$TrainingRuntimeRoot = Join-Path $RuntimeRoot "training"

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
  $DataRoot = if (![string]::IsNullOrWhiteSpace($env:TRAINING_DATA_DIR)) {
    $env:TRAINING_DATA_DIR
  } else {
    Join-Path $ArtifactsRoot "training_data"
  }
}
if ([string]::IsNullOrWhiteSpace($DatasetDir)) {
  $DatasetDir = if (![string]::IsNullOrWhiteSpace($env:TRAINING_DATASET_DIR)) {
    $env:TRAINING_DATASET_DIR
  } else {
    Join-Path $TrainingRuntimeRoot "datasets/rfdetr_price_tag_all_annotated_tiled1280"
  }
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = if (![string]::IsNullOrWhiteSpace($env:TRAINING_RUN_DIR)) {
    $env:TRAINING_RUN_DIR
  } else {
    Join-Path $TrainingRuntimeRoot "runs/rfdetr_small_price_tag_all_annotated_tiled1280_e${Epochs}"
  }
}
if ([string]::IsNullOrWhiteSpace($ModelOutputPath)) {
  $ModelOutputPath = if (![string]::IsNullOrWhiteSpace($env:TRAINING_MODEL_OUTPUT_PATH)) {
    $env:TRAINING_MODEL_OUTPUT_PATH
  } else {
    Join-Path $TrainingRuntimeRoot "models/rfdetr_small_price_tag_all_annotated_tiled1280_e${Epochs}_checkpoint_best_total.pth"
  }
}
if ([string]::IsNullOrWhiteSpace($PretrainWeights)) {
  if (![string]::IsNullOrWhiteSpace($env:RFDETR_PRETRAIN_WEIGHTS)) {
    $PretrainWeights = $env:RFDETR_PRETRAIN_WEIGHTS
  } else {
    $DefaultPretrainWeights = Join-Path $ProjectRoot "rf-detr-small.pth"
    if (Test-Path -LiteralPath $DefaultPretrainWeights) {
      $PretrainWeights = $DefaultPretrainWeights
    }
  }
}

$ArtifactsModelsRoot = [System.IO.Path]::GetFullPath((Join-Path $ArtifactsRoot "models")).TrimEnd(
  [System.IO.Path]::DirectorySeparatorChar,
  [System.IO.Path]::AltDirectorySeparatorChar
)
$ResolvedModelOutputPath = [System.IO.Path]::GetFullPath($ModelOutputPath)
$ArtifactsModelsPrefix = $ArtifactsModelsRoot + [System.IO.Path]::DirectorySeparatorChar
if (
  $ResolvedModelOutputPath.Equals($ArtifactsModelsRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
  $ResolvedModelOutputPath.StartsWith($ArtifactsModelsPrefix, [System.StringComparison]::OrdinalIgnoreCase)
) {
  throw "TRAINING_OUTPUT_CONFLICT: refuse to write trained weights into artifacts/models. Use runtime/training/models and promote a verified checkpoint manually."
}
$ModelOutputPath = $ResolvedModelOutputPath

function Assert-LastExitCode {
  param([string]$Step)
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed with exit code $LASTEXITCODE"
  }
}

if (!(Test-Path -LiteralPath $DataRoot)) {
  throw "DataRoot is missing: $DataRoot"
}

Write-Host "Training data root: $DataRoot"
Write-Host "Only folders with matching <folder>.mp4 and <folder>.csv annotation files are included."
Write-Host "Training checkpoint output: $ModelOutputPath"
if (![string]::IsNullOrWhiteSpace($PretrainWeights)) {
  Write-Host "RF-DETR pretrain weights: $PretrainWeights"
}

if (!$SkipDatasetBuild) {
  Write-Host "1/2 build RF-DETR tiled dataset from all annotated videos"
  & $Python scripts/build_rfdetr_tiled_dataset.py `
    --data-root $DataRoot `
    --output-dir $DatasetDir `
    --all-train `
    --mirror-train-to-valid `
    --tile-size 1280 `
    --tile-overlap 320 `
    --valid-fraction 0 `
    --min-visible-fraction 0.2 `
    --seed 42 `
    --overwrite
  Assert-LastExitCode "build RF-DETR all-data tiled dataset"
}

if ($BuildOnly) {
  Write-Host "Dataset ready:"
  Write-Host $DatasetDir
  exit 0
}

Write-Host "2/2 train RF-DETR small on all annotated videos"
$TrainArgs = @(
  "scripts/train_rfdetr_price_tag.py",
  "--dataset-dir", $DatasetDir,
  "--output-dir", $RunDir,
  "--model-size", "small",
  "--epochs", $Epochs,
  "--batch-size", $BatchSize,
  "--grad-accum-steps", $GradAccumSteps,
  "--resolution", $Resolution,
  "--num-workers", $NumWorkers,
  "--device", $Device
)
if (![string]::IsNullOrWhiteSpace($PretrainWeights)) {
  $TrainArgs += @("--pretrain-weights", $PretrainWeights)
}
& $Python @TrainArgs
Assert-LastExitCode "train RF-DETR small on all annotated data"

$Checkpoint = Join-Path $RunDir "checkpoint_best_total.pth"
if (!(Test-Path -LiteralPath $Checkpoint)) {
  throw "Training finished, but checkpoint was not found: $Checkpoint"
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ModelOutputPath) | Out-Null
Copy-Item -LiteralPath $Checkpoint -Destination $ModelOutputPath -Force

Write-Host "Done:"
Write-Host $ModelOutputPath
