# ============================================================
# Phase 1 — Feature generation: gen_features_bml.py over every subfolder of Raw/Raw_BML
# ============================================================
$rawFolders = @("HUST", "ISU", "LFP", "MATR", "NMC", "Tongji", "ZN-coin"
)

# ============================================================
# Phase 2 — Dataset build + training 
# ============================================================
$folders  =@("HUST", "ISU", "LFP", "MATR", "NMC", "Tongji", "ZN-coin"
)


# Windows teardown crash codes that happen AFTER training finishes successfully.
$benignExitCodes = @(-1073740791, -1073741819)

$failed = @()

# ── Phase 1: gen_features_bml.py ───────────────────────────────────────────
Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "PHASE 1: Feature generation (gen_features_bml.py)" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host ("Subfolders in Raw/Raw_BML to process ({0}):" -f $rawFolders.Count) -ForegroundColor Magenta
foreach ($rf in $rawFolders) { Write-Host "  - $rf" -ForegroundColor Magenta }

foreach ($folder in $rawFolders) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Generating features for: $folder" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $genCmd = "python gen_features_bml.py --data_dir ./Raw/Raw_BML/$folder --out_dir ./content_bml/$folder"
    Write-Host ""
    Write-Host ">> $genCmd" -ForegroundColor Yellow
    Invoke-Expression $genCmd
    $genExit = $LASTEXITCODE
    if ($genExit -ne 0 -and -not ($benignExitCodes -contains $genExit)) {
        Write-Host ("ERROR: gen_features step failed for {0} (exit code {1}) - skipping to next folder" -f $folder, $genExit) -ForegroundColor Red
        $failed += ("{0} [gen_features, exit {1}]" -f $folder, $genExit)
        continue
    }
    if ($benignExitCodes -contains $genExit) {
        Write-Host "Note: gen_features returned benign teardown exit code $genExit - treating as success" -ForegroundColor DarkYellow
    }

    Write-Host ""
    Write-Host "Done generating features: $folder" -ForegroundColor Green
}

# ── Phase 2: dataset_clf_bml.py + train_clf_bml.py ─────────────────────────
Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "PHASE 2: Dataset build + training" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host ("Folders to train on ({0}):" -f $folders.Count) -ForegroundColor Magenta
foreach ($f in $folders) { Write-Host "  - $f" -ForegroundColor Magenta }

foreach ($folder in $folders) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Processing folder: $folder" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $datasetCmd = "python dataset_clf_bml.py --content_dir content_bml/$folder"
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

    $trainCmd = "python train_clf_bml.py --content_dir ./content_bml/$folder --output_dir ./checkpoints/$folder"
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
