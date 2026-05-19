param(
  [string]$Python = "A:\rfdetr_envs\lenta-rfdetr-gpu\Scripts\python.exe",
  [string]$TesseractExe = "A:\tesseract_env\Library\bin\tesseract.exe",
  [string]$TessdataDir = "A:\tesseract_env\Library\share\tessdata",
  [int]$NumericJobs = 8,
  [int]$TesseractJobs = 8,
  [string]$CatalogPath = "",
  [string]$OutputRunName = "",
  [string]$VideoName = "26_12-20",
  [string]$DataRoot = "",
  [string]$DetectorCheckpoint = "",
  [switch]$RebuildBase,
  [switch]$SkipVisualReport
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
if ([string]::IsNullOrWhiteSpace($DetectorCheckpoint)) {
  $DetectorCheckpoint = Join-Path $Root "models\rfdetr_small_price_tag_except_26_12_20_tiled1280_e8_checkpoint_best_total.pth"
}
if ([string]::IsNullOrWhiteSpace($OutputRunName)) {
  $OutputRunName = "quality_${VideoSafeName}_tesseract_catalog_recovery_v19_db_hack"
}

$BaseRun = Join-Path $Root "repro_outputs\quality_$VideoSafeName"
$TesseractRun = Join-Path $Root "repro_outputs\quality_${VideoSafeName}_tesseract"
$CropQualityRun = Join-Path $Root "repro_outputs\quality_${VideoSafeName}_tesseract_crop_quality_weighted_w005"
$OutputRun = Join-Path $Root "repro_outputs\$OutputRunName"
$GtCsv = Join-Path $DataRoot "$VideoName\$VideoName.csv"
if ([string]::IsNullOrWhiteSpace($CatalogPath)) {
  $ParentCatalog = Join-Path (Split-Path -Parent $Root) "db_hack.csv"
  $LocalCatalog = Join-Path $Root "data\db_hack.csv"
  if (Test-Path -LiteralPath $ParentCatalog) {
    $CatalogPath = $ParentCatalog
  } elseif (Test-Path -LiteralPath $LocalCatalog) {
    $CatalogPath = $LocalCatalog
  } else {
    $CatalogPath = Join-Path $Root "data\lenta_all_products_prices_weight.json"
  }
}
$CatalogJson = $CatalogPath

if (!(Test-Path -LiteralPath $GtCsv)) {
  throw "GT CSV is missing: $GtCsv"
}
if (!(Test-Path -LiteralPath $DetectorCheckpoint)) {
  throw "Detector checkpoint is missing: $DetectorCheckpoint. Run .\run_train_rfdetr_except_26_12_20.ps1 first, or pass -DetectorCheckpoint explicitly."
}

function Copy-ZonesForRun {
  param(
    [string]$FromRun,
    [string]$ToRun
  )
  $src = Join-Path $FromRun "ocr_zones_core_fixed"
  if (!(Test-Path -LiteralPath $src)) {
    throw "Missing OCR zones folder: $src"
  }
  New-Item -ItemType Directory -Force -Path $ToRun | Out-Null
  Copy-Item -LiteralPath $src -Destination $ToRun -Recurse -Force
}

function Assert-LastExitCode {
  param([string]$Step)
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed with exit code $LASTEXITCODE"
  }
}

if ($RebuildBase) {
  Write-Host "1/6 full base detection + numeric OCR"
  & (Join-Path $Root "run_quality_pipeline_43_15.ps1") `
    -Python $Python `
    -NumericJobs $NumericJobs `
    -VideoName $VideoName `
    -DataRoot $DataRoot `
    -Checkpoint $DetectorCheckpoint `
    -OutRoot $BaseRun
  Assert-LastExitCode "full base detection + numeric OCR"

  Write-Host "2/6 Tesseract product_name OCR"
  New-Item -ItemType Directory -Force -Path (Join-Path $TesseractRun "ocr_final_quality_core_fast_fixed") | Out-Null
  & $Python scripts\run_product_name_tesseract.py `
    --zones-manifest "$BaseRun\ocr_zones_core_fixed\ocr_zones_manifest.csv" `
    --base-submission-csv "$BaseRun\ocr_numeric_quality_core_fast_fixed\ocr_aggregated_submission.csv" `
    --base-debug-csv "$BaseRun\ocr_numeric_quality_core_fast_fixed\ocr_aggregated_debug.csv" `
    --output "$TesseractRun\ocr_final_quality_core_fast_fixed" `
    --tesseract-exe $TesseractExe `
    --tessdata-dir $TessdataDir `
    --language "rus+eng" `
    --top-k 3 `
    --image-variants "enhanced,raw" `
    --preprocess-modes "none,up2,binary_up2" `
    --psms "6,11" `
    --line-variants "enhanced" `
    --line-preprocess-mode "up2" `
    --line-psms "7" `
    --max-lines 5 `
    --accept-policy all `
    --jobs $TesseractJobs `
    --tesseract-omp-thread-limit 1
  Assert-LastExitCode "Tesseract product_name OCR"
  Copy-ZonesForRun -FromRun $BaseRun -ToRun $TesseractRun

  Write-Host "3/6 crop-quality reselect"
  New-Item -ItemType Directory -Force -Path (Join-Path $CropQualityRun "ocr_final_quality_core_fast_fixed") | Out-Null
  & $Python scripts\reselect_product_name_crop_quality.py `
    --candidate-csv "$TesseractRun\ocr_final_quality_core_fast_fixed\product_name_line_candidates.csv" `
    --zones-manifest "$BaseRun\ocr_zones_core_fixed\ocr_zones_manifest.csv" `
    --base-submission-csv "$TesseractRun\ocr_final_quality_core_fast_fixed\ocr_aggregated_submission_product_lines.csv" `
    --base-debug-csv "$TesseractRun\ocr_final_quality_core_fast_fixed\ocr_aggregated_debug_product_lines.csv" `
    --output "$CropQualityRun\ocr_final_quality_core_fast_fixed" `
    --policy weighted `
    --weight 0.05 `
    --top-k 3 `
    --image-variants "enhanced,raw"
  Assert-LastExitCode "crop-quality reselect"
  Copy-ZonesForRun -FromRun $BaseRun -ToRun $CropQualityRun

  Write-Host "4/6 evaluate crop-quality baseline"
  & $Python scripts\evaluate_43_15_report.py `
    --run-dir $CropQualityRun `
    --gt-csv $GtCsv `
    --output-dir "$CropQualityRun\evaluation"
  Assert-LastExitCode "evaluate crop-quality baseline"
} else {
  Write-Host "1/4 using existing crop-quality baseline"
  if (!(Test-Path -LiteralPath "$CropQualityRun\ocr_final_quality_core_fast_fixed\ocr_aggregated_submission_product_lines.csv")) {
    throw "Baseline run is missing. Re-run with -RebuildBase or restore $CropQualityRun"
  }
}

Write-Host "catalog recovery v12 vote decoder + v19 hybrid order source + db_hack exact catalog"
& $Python scripts\apply_catalog_product_name_recovery.py `
  --source-run-dir $CropQualityRun `
  --output-run-dir $OutputRun `
  --catalog-json $CatalogJson `
  --accept-score 0.65 `
  --accept-margin 0.035 `
  --strong-score 1.01 `
  --preselect 900 `
  --candidate-pool-size 12 `
  --preserve-ocr-order `
  --hybrid-order-source-pool `
  --catalog-decoder-mode vote `
  --vote-top-k 3 `
  --vote-min-score 0.50 `
  --vote-min-support 2 `
  --vote-accept-score 0.20 `
  --vote-accept-margin 0.05 `
  --vote-explain-ratio 0.50 `
  --disable-glass-packaging-tail
Assert-LastExitCode "catalog recovery"

Write-Host "fill sample-schema auxiliary and QR fields"
& $Python scripts\fill_submission_aux_fields.py `
  --run-dir $OutputRun `
  --infer-price2-qr `
  --overwrite-derived-qr `
  --restore-price-default-cents `
  --price-consistency-postprocess `
  --price-sanity-postprocess `
  --extract-special-symbols `
  --barcode-from-catalog `
  --catalog-barcode-csv $CatalogJson `
  --overwrite-barcode-from-catalog `
  --no-id-sku-from-catalog
Assert-LastExitCode "fill sample-schema auxiliary and QR fields"

Write-Host "deduplicate final submission rows"
& $Python scripts\deduplicate_submission_rows.py `
  --run-dir $OutputRun `
  --tracking-csv "$BaseRun\tracking\best_per_track.csv"
Assert-LastExitCode "deduplicate final submission rows"

Write-Host "evaluate v12"
& $Python scripts\evaluate_43_15_report.py `
  --run-dir $OutputRun `
  --gt-csv $GtCsv `
  --output-dir "$OutputRun\evaluation"
Assert-LastExitCode "evaluate v12"

Write-Host "evaluate all sample submission fields"
& $Python scripts\evaluate_full_submission_fields.py `
  --run-dir $OutputRun `
  --gt-csv $GtCsv `
  --output-dir "$OutputRun\evaluation_full"
Assert-LastExitCode "evaluate all sample submission fields"

if (!$SkipVisualReport) {
  Write-Host "manual visual report"
  & $Python scripts\make_catalog_recovery_visual_report.py `
    --recovery-csv "$OutputRun\ocr_final_quality_core_fast_fixed\product_name_catalog_recovery.csv" `
    --baseline-matched-csv "$CropQualityRun\evaluation\matched_predictions.csv" `
    --catalog-matched-csv "$OutputRun\evaluation\matched_predictions.csv" `
    --zones-manifest "$BaseRun\ocr_zones_core_fixed\ocr_zones_manifest.csv" `
    --output-dir "$OutputRun\manual_review"
  Assert-LastExitCode "manual visual report"
}

Write-Host "Done:"
Write-Host "$OutputRun\evaluation\metrics.json"
Write-Host "$OutputRun\evaluation_full\full_metrics.json"
Write-Host "$OutputRun\evaluation_full\full_submission_review.html"
Write-Host "$OutputRun\manual_review\catalog_recovery_visual_review.html"
