param(
  [string]$Python = "A:\rfdetr_envs\lenta-rfdetr-gpu\Scripts\python.exe",
  [string]$DataRoot = "",
  [string]$DatasetDir = "",
  [string]$RunDir = "",
  [string]$ModelOutputPath = "",
  [int]$Epochs = 8,
  [int]$BatchSize = 4,
  [int]$GradAccumSteps = 4,
  [int]$Resolution = 512,
  [int]$NumWorkers = 0,
  [switch]$SkipDatasetBuild,
  [switch]$BuildOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$HoldoutVideo = "26_12-20"
$HoldoutSafe = $HoldoutVideo -replace '[^A-Za-z0-9]+', '_'

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
  $DataDirName = "$([char]0x0414)$([char]0x0430)$([char]0x043d)$([char]0x043d)$([char]0x044b)$([char]0x0435)"
  $ParentData = Join-Path (Split-Path -Parent $Root) $DataDirName
  if (Test-Path -LiteralPath $ParentData) {
    $DataRoot = $ParentData
  } else {
    $DataRoot = Join-Path $Root "data"
  }
}
if ([string]::IsNullOrWhiteSpace($DatasetDir)) {
  $DatasetDir = Join-Path $Root "datasets\rfdetr_price_tag_except_${HoldoutSafe}_tiled1280"
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $Root "runs\rfdetr_small_price_tag_except_${HoldoutSafe}_tiled1280_e${Epochs}"
}
if ([string]::IsNullOrWhiteSpace($ModelOutputPath)) {
  $ModelOutputPath = Join-Path $Root "models\rfdetr_small_price_tag_except_${HoldoutSafe}_tiled1280_e${Epochs}_checkpoint_best_total.pth"
}

function Assert-LastExitCode {
  param([string]$Step)
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed with exit code $LASTEXITCODE"
  }
}

if (!(Test-Path -LiteralPath (Join-Path $DataRoot $HoldoutVideo))) {
  throw "Holdout data folder is missing: $(Join-Path $DataRoot $HoldoutVideo)"
}
if (!(Test-Path -LiteralPath (Join-Path $DataRoot "43_15"))) {
  throw "43_15 folder is missing from DataRoot, but it must now be included in training: $(Join-Path $DataRoot '43_15')"
}

if (!$SkipDatasetBuild) {
  Write-Host "1/2 build RF-DETR tiled dataset with holdout $HoldoutVideo"
  & $Python scripts\build_rfdetr_tiled_dataset.py `
    --data-root $DataRoot `
    --output-dir $DatasetDir `
    --holdout-video $HoldoutVideo `
    --tile-size 1280 `
    --tile-overlap 320 `
    --valid-fraction 0.15 `
    --min-visible-fraction 0.2 `
    --seed 42 `
    --overwrite
  Assert-LastExitCode "build RF-DETR tiled dataset"
}

if ($BuildOnly) {
  Write-Host "Dataset ready:"
  Write-Host $DatasetDir
  exit 0
}

Write-Host "2/2 train RF-DETR small, holdout $HoldoutVideo"
& $Python scripts\train_rfdetr_price_tag.py `
  --dataset-dir $DatasetDir `
  --output-dir $RunDir `
  --model-size small `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --grad-accum-steps $GradAccumSteps `
  --resolution $Resolution `
  --num-workers $NumWorkers `
  --device cuda `
  --run-test
Assert-LastExitCode "train RF-DETR small"

$Checkpoint = Join-Path $RunDir "checkpoint_best_total.pth"
if (!(Test-Path -LiteralPath $Checkpoint)) {
  throw "Training finished, but checkpoint was not found: $Checkpoint"
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ModelOutputPath) | Out-Null
Copy-Item -LiteralPath $Checkpoint -Destination $ModelOutputPath -Force

Write-Host "Done:"
Write-Host $ModelOutputPath
