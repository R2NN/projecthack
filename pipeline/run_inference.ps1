param(
  [Parameter(Mandatory = $true)]
  [string]$VideoPath,
  [string]$VideoId = "",
  [string]$RunRoot = "",
  [string]$Python = "",
  [string]$TesseractExe = "",
  [string]$TessdataDir = "",
  [string]$DetectorCheckpoint = "",
  [string]$CatalogPath = "",
  [string]$SpecialSymbolTemplateDir = "",
  [string]$Device = "",
  [int]$NumericJobs = 8,
  [int]$TesseractJobs = 8
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $Root
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Import-DotEnv {
  param([string]$PathValue)
  if (!(Test-Path -LiteralPath $PathValue)) {
    return
  }
  foreach ($line in Get-Content -LiteralPath $PathValue) {
    $trimmed = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#") -or !$trimmed.Contains("=")) {
      continue
    }
    $parts = $trimmed.Split("=", 2)
    $key = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if (![string]::IsNullOrWhiteSpace($key) -and [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($key))) {
      [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
  }
}

$EnvFile = if (![string]::IsNullOrWhiteSpace($env:ENV_FILE)) { $env:ENV_FILE } else { Join-Path $ProjectRoot ".env" }
Import-DotEnv $EnvFile
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

function Resolve-RequiredPath {
  param(
    [string]$PathValue,
    [string]$Label
  )
  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    throw "$Label is empty"
  }
  $resolved = Resolve-Path -LiteralPath $PathValue -ErrorAction SilentlyContinue
  if ($null -eq $resolved) {
    throw "$Label is missing: $PathValue"
  }
  return $resolved.Path
}

function Resolve-ExecutablePath {
  param(
    [string]$PathValue,
    [string]$Label
  )
  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    throw "$Label is empty"
  }
  $resolved = Resolve-Path -LiteralPath $PathValue -ErrorAction SilentlyContinue
  if ($null -ne $resolved) {
    return $resolved.Path
  }
  $command = Get-Command $PathValue -ErrorAction SilentlyContinue
  if ($null -ne $command -and ![string]::IsNullOrWhiteSpace($command.Source)) {
    return $command.Source
  }
  throw "$Label is missing: $PathValue"
}

function Join-PathParts {
  param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Parts
  )
  if ($Parts.Count -eq 0) {
    return ""
  }
  $current = $Parts[0]
  for ($index = 1; $index -lt $Parts.Count; $index++) {
    $current = Join-Path $current $Parts[$index]
  }
  return $current
}

function Resolve-InferenceDevice {
  param(
    [string]$Requested,
    [string]$PythonPath
  )
  $normalized = if ([string]::IsNullOrWhiteSpace($Requested)) { "auto" } else { $Requested.Trim().ToLowerInvariant() }
  if ($normalized -notin @("auto", "cpu", "cuda")) {
    throw "Unsupported Device: $Requested. Use auto, cpu, or cuda."
  }
  if ($normalized -eq "cpu") {
    return "cpu"
  }
  $probe = & $PythonPath -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
  $available = if ($LASTEXITCODE -eq 0 -and ![string]::IsNullOrWhiteSpace($probe)) { $probe.Trim() } else { "cpu" }
  if ($normalized -eq "cuda" -and $available -ne "cuda") {
    throw "GPU_NOT_AVAILABLE: CUDA device was requested but torch.cuda.is_available() is false."
  }
  return $available
}

function Save-Timings {
  $script:Timings | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $script:TimingsJson -Encoding UTF8
  $script:Timings | Export-Csv -LiteralPath $script:TimingsCsv -NoTypeInformation -Encoding UTF8
}

function Invoke-TimedStep {
  param(
    [string]$Name,
    [scriptblock]$Body
  )
  Write-Host "START $Name"
  $started = Get-Date
  $watch = [System.Diagnostics.Stopwatch]::StartNew()
  $status = "ok"
  $exitCode = 0
  $message = ""
  try {
    & $Body
    if ($LASTEXITCODE -ne 0) {
      $exitCode = [int]$LASTEXITCODE
      throw "$Name failed with exit code $exitCode"
    }
  } catch {
    $status = "failed"
    $message = $_.Exception.Message
    if ($exitCode -eq 0 -and $LASTEXITCODE -ne 0) {
      $exitCode = [int]$LASTEXITCODE
    }
    throw
  } finally {
    $watch.Stop()
    $finished = Get-Date
    $row = [pscustomobject]@{
      step = $Name
      status = $status
      seconds = [math]::Round($watch.Elapsed.TotalSeconds, 3)
      started_at = $started.ToString("o")
      finished_at = $finished.ToString("o")
      exit_code = $exitCode
      message = $message
    }
    [void]$script:Timings.Add($row)
    Save-Timings
    Write-Host ("END {0}: {1}s status={2}" -f $Name, $row.seconds, $status)
  }
}

function Copy-MinimalZones {
  param(
    [string]$FromRun,
    [string]$ToRun
  )
  $src = Join-Path $FromRun "ocr_zones_core_fixed"
  $dst = Join-Path $ToRun "ocr_zones_core_fixed"
  $zonesManifest = Join-Path $src "ocr_zones_manifest.csv"
  $ocrManifest = Join-Path $src "ocr_manifest.csv"
  if (!(Test-Path -LiteralPath $zonesManifest)) {
    throw "Missing zones manifest: $zonesManifest"
  }
  if (Test-Path -LiteralPath $dst) {
    Remove-Item -LiteralPath $dst -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  Copy-Item -LiteralPath $zonesManifest -Destination (Join-Path $dst "ocr_zones_manifest.csv") -Force
  if (Test-Path -LiteralPath $ocrManifest) {
    Copy-Item -LiteralPath $ocrManifest -Destination (Join-Path $dst "ocr_manifest.csv") -Force
  }
}

$VideoPath = Resolve-RequiredPath $VideoPath "VideoPath"
if ([string]::IsNullOrWhiteSpace($Python)) {
  if (![string]::IsNullOrWhiteSpace($env:LENTA_PIPELINE_PYTHON)) {
    $Python = $env:LENTA_PIPELINE_PYTHON
  } elseif (![string]::IsNullOrWhiteSpace($env:PIPELINE_PYTHON)) {
    $Python = $env:PIPELINE_PYTHON
  } else {
    $Python = "python"
  }
}
if ([string]::IsNullOrWhiteSpace($TesseractExe)) {
  if (![string]::IsNullOrWhiteSpace($env:LENTA_TESSERACT_EXE)) {
    $TesseractExe = $env:LENTA_TESSERACT_EXE
  } elseif (![string]::IsNullOrWhiteSpace($env:TESSERACT_EXE)) {
    $TesseractExe = $env:TESSERACT_EXE
  } else {
    $TesseractExe = "tesseract"
  }
}
if ([string]::IsNullOrWhiteSpace($TessdataDir)) {
  if (![string]::IsNullOrWhiteSpace($env:LENTA_TESSDATA_DIR)) {
    $TessdataDir = $env:LENTA_TESSDATA_DIR
  } elseif (![string]::IsNullOrWhiteSpace($env:TESSDATA_DIR)) {
    $TessdataDir = $env:TESSDATA_DIR
  }
}
$Python = Resolve-ExecutablePath $Python "Python"
$TesseractExe = Resolve-ExecutablePath $TesseractExe "Tesseract"
$PythonDir = Split-Path -Parent $Python
if (![string]::IsNullOrWhiteSpace($PythonDir)) {
  $env:PATH = "$PythonDir$([System.IO.Path]::PathSeparator)$env:PATH"
  $env:PYTHONEXECUTABLE = $Python
}
if ([string]::IsNullOrWhiteSpace($Device)) {
  if (![string]::IsNullOrWhiteSpace($env:LENTA_INFERENCE_DEVICE)) {
    $Device = $env:LENTA_INFERENCE_DEVICE
  } elseif (![string]::IsNullOrWhiteSpace($env:INFERENCE_DEVICE)) {
    $Device = $env:INFERENCE_DEVICE
  } else {
    $Device = "auto"
  }
}
$ResolvedDevice = Resolve-InferenceDevice -Requested $Device -PythonPath $Python
$UseGpu = $ResolvedDevice -eq "cuda"
$env:LENTA_RESOLVED_DEVICE = $ResolvedDevice
if ([string]::IsNullOrWhiteSpace($DetectorCheckpoint)) {
  $DetectorCheckpoint = Join-PathParts $ArtifactsRoot "models" "rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth"
}
$DetectorCheckpoint = Resolve-RequiredPath $DetectorCheckpoint "DetectorCheckpoint"
if ([string]::IsNullOrWhiteSpace($CatalogPath)) {
  $CatalogPath = Join-PathParts $ArtifactsRoot "data" "db_hack.csv"
}
$CatalogPath = Resolve-RequiredPath $CatalogPath "CatalogPath"
if ([string]::IsNullOrWhiteSpace($SpecialSymbolTemplateDir)) {
  $SpecialSymbolTemplateDir = Join-PathParts $ArtifactsRoot "special_symbol_templates" "full_tags"
}
$SpecialSymbolTemplateDir = Resolve-RequiredPath $SpecialSymbolTemplateDir "SpecialSymbolTemplateDir"
$SampleCsv = Resolve-RequiredPath (Join-PathParts $ArtifactsRoot "data" "sample.csv") "SampleCsv"
if ([string]::IsNullOrWhiteSpace($TessdataDir)) {
  throw "TessdataDir is empty. Set TESSDATA_DIR/LENTA_TESSDATA_DIR or pass -TessdataDir."
}
$TessdataDir = Resolve-RequiredPath $TessdataDir "TessdataDir"

$videoFile = Get-Item -LiteralPath $VideoPath
if ([string]::IsNullOrWhiteSpace($VideoId)) {
  $VideoId = "unlabeled_$($videoFile.BaseName)"
}
$VideoSafeName = $VideoId -replace '[^A-Za-z0-9]+', '_'
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $RunRoot = Join-PathParts $RuntimeRoot "runs" "inference_${VideoSafeName}_$stamp"
}
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$RunRoot = (Resolve-Path -LiteralPath $RunRoot).Path

$BaseRun = Join-Path $RunRoot "base"
$TesseractRun = Join-Path $RunRoot "tesseract"
$CropQualityRun = Join-Path $RunRoot "tesseract_crop_quality_weighted_w005"
$OutputRun = Join-Path $RunRoot "catalog_recovery_v19_db_hack"
$FinalCsv = Join-Path $RunRoot "final_submission.csv"
$BaseTracking = Join-Path $BaseRun "tracking"
$BaseZones = Join-Path $BaseRun "ocr_zones_core_fixed"
$BaseNumeric = Join-Path $BaseRun "ocr_numeric_quality_core_fast_fixed"
$TesseractOutput = Join-Path $TesseractRun "ocr_final_quality_core_fast_fixed"
$CropQualityOutput = Join-Path $CropQualityRun "ocr_final_quality_core_fast_fixed"
$OutputFinal = Join-Path $OutputRun "ocr_final_quality_core_fast_fixed"
$CatalogCache = Join-PathParts $RuntimeRoot "cache" "lenta_catalog_product_name_index_v3.pkl"
$QrDebugDir = Join-Path $RunRoot "qr_priority"
$TrackScript = Join-PathParts $Root "scripts" "track_price_tag_crops.py"
$ExportZonesScript = Join-PathParts $Root "scripts" "export_ocr_zones.py"
$NumericOcrScript = Join-PathParts $Root "scripts" "run_zone_ocr_baseline.py"
$TesseractScript = Join-PathParts $Root "scripts" "run_product_name_tesseract.py"
$CropQualityScript = Join-PathParts $Root "scripts" "reselect_product_name_crop_quality.py"
$CatalogRecoveryScript = Join-PathParts $Root "scripts" "apply_catalog_product_name_recovery.py"
$FillAuxScript = Join-PathParts $Root "scripts" "fill_submission_aux_fields.py"
$DeduplicateScript = Join-PathParts $Root "scripts" "deduplicate_submission_rows.py"
$QrPriorityScript = Join-PathParts $Root "scripts" "apply_fulltag_qr_priority.py"
$script:Timings = New-Object System.Collections.Generic.List[object]
$script:TimingsJson = Join-Path $RunRoot "timings.json"
$script:TimingsCsv = Join-Path $RunRoot "timings.csv"

$runInfo = [pscustomobject]@{
  video_path = $VideoPath
  video_id = $VideoId
  run_root = $RunRoot
  detector_checkpoint = $DetectorCheckpoint
  catalog_path = $CatalogPath
  artifacts_root = $ArtifactsRoot
  runtime_root = $RuntimeRoot
  sample_csv = $SampleCsv
  special_symbol_template_dir = $SpecialSymbolTemplateDir
  python = $Python
  tesseract_exe = $TesseractExe
  tessdata_dir = $TessdataDir
  requested_device = $Device
  resolved_device = $ResolvedDevice
  numeric_jobs = $NumericJobs
  tesseract_jobs = $TesseractJobs
  created_at = (Get-Date).ToString("o")
}
$runInfo | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunRoot "run_info.json") -Encoding UTF8
Save-Timings

Invoke-TimedStep "01_detection_tracking" {
  & $Python $TrackScript `
    --video-name $VideoId `
    --video-path $VideoPath `
    --checkpoint $DetectorCheckpoint `
    --output $BaseTracking `
    --variant small `
    --device $ResolvedDevice `
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
}

Invoke-TimedStep "02_export_ocr_zones" {
  & $Python $ExportZonesScript `
    --tracks-csv (Join-Path $BaseTracking "tracks_top_crops.csv") `
    --output $BaseZones `
    --top-k 3 `
    --full-width 960 `
    --zone-pad 0.015 `
    --preferred-crop rectified `
    --skip-contact-sheets `
    --zones product_name,price_default_wide,price_card_number,discount_amount,qr,barcode
}

Invoke-TimedStep "03_numeric_ocr" {
  $numericArgs = @(
    $NumericOcrScript,
    "--zones-manifest", (Join-Path $BaseZones "ocr_zones_manifest.csv"),
    "--output", $BaseNumeric,
    "--top-k", "3",
    "--engines", "rapidocr,easyocr",
    "--engine-plan", "lenta_fast",
    "--decoders", "zxing,opencv_qr",
    "--zones", "price_default_wide,price_card_number,discount_amount,qr,barcode",
    "--image-variants", "enhanced,tight_enhanced",
    "--decoder-variants", "tight,tight_enhanced,tight_binary,enhanced,binary,raw",
    "--fast-no-decoders",
    "--jobs", "$NumericJobs",
    "--parallel-engine-mode", "shared"
  )
  if ($UseGpu) {
    $numericArgs += "--gpu"
  }
  & $Python @numericArgs
}

Invoke-TimedStep "04_tesseract_product_name" {
  New-Item -ItemType Directory -Force -Path $TesseractOutput | Out-Null
  & $Python $TesseractScript `
    --zones-manifest (Join-Path $BaseZones "ocr_zones_manifest.csv") `
    --base-submission-csv (Join-Path $BaseNumeric "ocr_aggregated_submission.csv") `
    --base-debug-csv (Join-Path $BaseNumeric "ocr_aggregated_debug.csv") `
    --output $TesseractOutput `
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
  Copy-MinimalZones -FromRun $BaseRun -ToRun $TesseractRun
}

Invoke-TimedStep "05_crop_quality_reselect" {
  New-Item -ItemType Directory -Force -Path $CropQualityOutput | Out-Null
  & $Python $CropQualityScript `
    --candidate-csv (Join-Path $TesseractOutput "product_name_line_candidates.csv") `
    --zones-manifest (Join-Path $BaseZones "ocr_zones_manifest.csv") `
    --base-submission-csv (Join-Path $TesseractOutput "ocr_aggregated_submission_product_lines.csv") `
    --base-debug-csv (Join-Path $TesseractOutput "ocr_aggregated_debug_product_lines.csv") `
    --output $CropQualityOutput `
    --policy weighted `
    --weight 0.05 `
    --top-k 3 `
    --image-variants "enhanced,raw"
  Copy-MinimalZones -FromRun $BaseRun -ToRun $CropQualityRun
}

Invoke-TimedStep "06_catalog_recovery_db_hack" {
  & $Python $CatalogRecoveryScript `
    --source-run-dir $CropQualityRun `
    --output-run-dir $OutputRun `
    --catalog-json $CatalogPath `
    --catalog-cache $CatalogCache `
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
}

Invoke-TimedStep "07_fill_aux_barcode" {
  & $Python $FillAuxScript `
    --run-dir $OutputRun `
    --sample-csv $SampleCsv `
    --infer-price2-qr `
    --overwrite-derived-qr `
    --restore-price-default-cents `
    --price-consistency-postprocess `
    --price-sanity-postprocess `
    --extract-special-symbols `
    --special-symbol-template-dir $SpecialSymbolTemplateDir `
    --barcode-from-catalog `
    --catalog-barcode-csv $CatalogPath `
    --overwrite-barcode-from-catalog `
    --no-template-net-defaults `
    --no-id-sku-from-catalog
}

Invoke-TimedStep "08_deduplicate_rows" {
  & $Python $DeduplicateScript `
    --run-dir $OutputRun `
    --tracking-csv (Join-Path $BaseTracking "best_per_track.csv")
}

Invoke-TimedStep "09_export_final_submission_with_qr_priority" {
  $sourceCsv = Join-Path $OutputFinal "ocr_aggregated_submission_product_lines.csv"
  if (!(Test-Path -LiteralPath $sourceCsv)) {
    throw "Final submission source is missing: $sourceCsv"
  }
  $preQrCsv = Join-Path $RunRoot "final_submission_before_qr.csv"
  Copy-Item -LiteralPath $sourceCsv -Destination $preQrCsv -Force
  & $Python $QrPriorityScript `
    --submission-csv $preQrCsv `
    --zones-manifest (Join-Path $BaseZones "ocr_zones_manifest.csv") `
    --output-csv $FinalCsv `
    --debug-dir $QrDebugDir `
    --workers 6
  $rowCount = (Import-Csv -LiteralPath $FinalCsv).Count
  [pscustomobject]@{
    final_submission_csv = $FinalCsv
    final_submission_before_qr_csv = $preQrCsv
    qr_summary_json = (Join-Path $QrDebugDir "summary.json")
    output_run = $OutputRun
    rows = $rowCount
    completed_at = (Get-Date).ToString("o")
  } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunRoot "inference_summary.json") -Encoding UTF8
}

Write-Host "Done:"
Write-Host $FinalCsv
Write-Host $script:TimingsJson
Write-Host $script:TimingsCsv
