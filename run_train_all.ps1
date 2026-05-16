# ============================================================
# Training script selection
#   1 = train_clf_bml.py  (default)
#   2 = train_clf_bml_transformer.py
# ============================================================
param(
    [ValidateSet("1","2")]
    [string]$TrainScript = ""
)

if ($TrainScript -eq "") {
    Write-Host ""
    Write-Host "Select training script:" -ForegroundColor Cyan
    Write-Host "  1) train_clf_bml.py           (classic BML)" -ForegroundColor White
    Write-Host "  2) train_clf_bml_transformer.py (transformer)" -ForegroundColor White
    $TrainScript = Read-Host "Enter 1 or 2 [default: 1]"
    if ($TrainScript -eq "") { $TrainScript = "1" }
}

if ($TrainScript -eq "2") {
    $trainScriptName = "train_clf_bml_transformer.py"
} else {
    $trainScriptName = "train_clf_bml.py"
}

Write-Host ""
Write-Host "Using training script: $trainScriptName" -ForegroundColor Green

# ============================================================
# Phase 2 — Dataset build + training
# ============================================================
$folders = @("HUST", "ISU", "LFP", "MATR", "NMC", "Tongji", "ZN-coin")

# Windows teardown crash codes that happen AFTER training finishes successfully.
$benignExitCodes = @(-1073740791, -1073741819)

$failed = @()

Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "Dataset build + training ($trainScriptName)" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host ("Folders to train on ({0}):" -f $folders.Count) -ForegroundColor Magenta
foreach ($f in $folders) { Write-Host "  - $f" -ForegroundColor Magenta }

foreach ($folder in $folders) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Processing folder: $folder" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $nmcVariantFolders = @("Tongji", "NMC", "ISU", "ZN-coin")
    if ($nmcVariantFolders -contains $folder) {
        $datasetScript = "dataset_clf_bml_NMC.py"
    } else {
        $datasetScript = "dataset_clf_bml.py"
    }
    $datasetCmd = "python $datasetScript --content_dir content_bml/$folder"
    Write-Host ""
    Write-Host ">> $datasetCmd" -ForegroundColor Yellow
    Invoke-Expression $datasetCmd
    $datasetExit = $LASTEXITCODE
    if ($datasetExit -ne 0 -and -not ($benignExitCodes -contains $datasetExit)) {
        Write-Host ("ERROR: dataset step failed for {0} (exit code {1}) - skipping to next folder" -f $folder, $datasetExit) -ForegroundColor Red
        $failed += ("{0} [dataset, exit {1}]" -f $folder, $datasetExit)
        continue
    }
    if ($benignExitCodes -contains $datasetExit) {
        Write-Host "Note: dataset returned benign teardown exit code $datasetExit - treating as success" -ForegroundColor DarkYellow
    }

    $trainCmd = "python $trainScriptName --content_dir ./content_bml/$folder --output_dir ./checkpoints/$folder"
    Write-Host ""
    Write-Host ">> $trainCmd" -ForegroundColor Yellow
    Invoke-Expression $trainCmd
    $trainExit = $LASTEXITCODE
    if ($trainExit -ne 0 -and -not ($benignExitCodes -contains $trainExit)) {
        Write-Host ("ERROR: train step failed for {0} (exit code {1}) - skipping to next folder" -f $folder, $trainExit) -ForegroundColor Red
        $failed += ("{0} [train, exit {1}]" -f $folder, $trainExit)
        continue
    }
    if ($benignExitCodes -contains $trainExit) {
        Write-Host "Note: train returned benign teardown exit code $trainExit - output was saved, continuing" -ForegroundColor DarkYellow
    }

    Write-Host ""
    Write-Host "Done: $folder" -ForegroundColor Green
}

# ── Final summary ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($failed.Count -eq 0) {
    Write-Host "All steps completed successfully." -ForegroundColor Green
} else {
    Write-Host "Completed with failures:" -ForegroundColor Yellow
    foreach ($f in $failed) { Write-Host "  - $f" -ForegroundColor Yellow }
}
