param(
  [string]$Python = "A:\rfdetr_envs\lenta-rfdetr-gpu\Scripts\python.exe",
  [int]$NumericJobs = 8,
  [string]$VideoName = "26_12-20",
  [string]$DataRoot = "",
  [string]$Checkpoint = "",
  [string]$OutRoot = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VideoSafeName = $VideoName -replace '[^A-Za-z0-9]+', '_'
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
  $DataDirName = "$([char]0x0414)$([char]0x0430)$([char]0x043d)$([char]0x043d)$([char]0x044b)$([char]0x0435)"
  $ParentData = Join-Path (Split-Path -Parent $Root) $DataDirName
  if (Test-Path -LiteralPath $ParentData) {
    $DataRoot = $ParentData
  } else {
    $DataRoot = Join-Path $Root "data"
  }
}
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
  $OutRoot = Join-Path $Root "repro_outputs\quality_$VideoSafeName"
}
if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
  $Checkpoint = Join-Path $Root "models\rfdetr_small_price_tag_except_26_12_20_tiled1280_e8_checkpoint_best_total.pth"
}
$VideoPath = Join-Path $DataRoot "$VideoName\$VideoName.mp4"

if (!(Test-Path -LiteralPath $VideoPath)) {
  throw "Video file is missing: $VideoPath"
}
if (!(Test-Path -LiteralPath $Checkpoint)) {
  throw "Detector checkpoint is missing: $Checkpoint. Run .\run_train_rfdetr_except_26_12_20.ps1 first, or pass -Checkpoint explicitly."
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

function Assert-LastExitCode {
  param([string]$Step)
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed with exit code $LASTEXITCODE"
  }
}

Write-Host "1/4 detection + tracking"
& $Python scripts\track_price_tag_crops.py `
  --video-name $VideoName `
  --video-path $VideoPath `
  --checkpoint $Checkpoint `
  --output "$OutRoot\tracking" `
  --variant small `
  --sampling-mode stops `
  --adaptive-stop-percentile 45 `
  --threshold 0.05 `
  --selection-mode track `
  --max-track-candidates 24 `
  --contact-sheet-page-size 80 `
  --top-k 3 `
  --min-tag-likeness 0.30 `
  --min-structure-score 0.45 `
  --min-red-fraction 0.035 `
  --min-white-fraction 0.12 `
  --min-color-coverage 0.25 `
  --min-center-price-fraction 0.06 `
  --min-price-center-score 0.45 `
  --max-yellow-fraction 0.14 `
  --max-dark-fraction 0.28 `
  --max-non-tag-fraction 0.48 `
  --min-qr-like-score 0.30 `
  --min-export-crop-score 0.80 `
  --qr-like-weight 0.12 `
  --reject-no-valid-refinement `
  --disable-overlays `
  --skip-contact-sheets
Assert-LastExitCode "detection + tracking"

Write-Host "2/4 OCR zones"
& $Python scripts\export_ocr_zones.py `
  --tracks-csv "$OutRoot\tracking\tracks_top_crops.csv" `
  --output "$OutRoot\ocr_zones_core_fixed" `
  --top-k 3 `
  --full-width 960 `
  --zone-pad 0.015 `
  --preferred-crop rectified `
  --skip-contact-sheets `
  --zones product_name,price_default_wide,price_card_number,discount_amount,qr,barcode
Assert-LastExitCode "OCR zones"

Write-Host "3/4 numeric OCR"
& $Python scripts\run_zone_ocr_baseline.py `
  --zones-manifest "$OutRoot\ocr_zones_core_fixed\ocr_zones_manifest.csv" `
  --output "$OutRoot\ocr_numeric_quality_core_fast_fixed" `
  --top-k 3 `
  --engines rapidocr,easyocr `
  --engine-plan lenta_fast `
  --decoders zxing,opencv_qr `
  --zones price_default_wide,price_card_number,discount_amount,qr,barcode `
  --image-variants enhanced,tight_enhanced `
  --decoder-variants tight,tight_enhanced,tight_binary,enhanced,binary,raw `
  --fast-no-decoders `
  --jobs $NumericJobs `
  --parallel-engine-mode shared `
  --gpu
Assert-LastExitCode "numeric OCR"

Write-Host "4/4 product_name OCR"
& $Python scripts\run_product_name_line_paddle.py `
  --zones-manifest "$OutRoot\ocr_zones_core_fixed\ocr_zones_manifest.csv" `
  --base-submission-csv "$OutRoot\ocr_numeric_quality_core_fast_fixed\ocr_aggregated_submission.csv" `
  --base-debug-csv "$OutRoot\ocr_numeric_quality_core_fast_fixed\ocr_aggregated_debug.csv" `
  --output "$OutRoot\ocr_final_quality_core_fast_fixed" `
  --top-k 3 `
  --image-variants enhanced `
  --max-lines 0
Assert-LastExitCode "product_name OCR"

Write-Host "Done:"
Write-Host "$OutRoot\ocr_final_quality_core_fast_fixed\ocr_aggregated_submission_product_lines.csv"
