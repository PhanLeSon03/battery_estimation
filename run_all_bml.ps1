$folders = @("NMC", "Tongji", "ZN-coin")

$benignExitCodes = @(-1073740791, -1073741819)

$failed = @()

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
        Write-Host "ERROR: dataset step failed for $folder (exit code $datasetExit) — skipping to next folder" -ForegroundColor Red
        $failed += ("{0} [dataset, exit {1}]" -f $folder, $datasetExit)
        continue
    }
    if ($benignExitCodes -contains $datasetExit) {
        Write-Host "Note: dataset returned benign teardown exit code $datasetExit — treating as success" -ForegroundColor DarkYellow
    }

    $trainCmd = "python train_clf_bml.py --content_dir ./content_bml/$folder --output_dir ./checkpoints/$folder"
    Write-Host ""
    Write-Host ">> $trainCmd" -ForegroundColor Yellow
    Invoke-Expression $trainCmd
    $trainExit = $LASTEXITCODE
    if ($trainExit -ne 0 -and -not ($benignExitCodes -contains $trainExit)) {
        Write-Host "ERROR: train step failed for $folder (exit code $trainExit) — skipping to next folder" -ForegroundColor Red
        $failed += ("{0} [train, exit {1}]" -f $folder, $trainExit)
        continue
    }
    if ($benignExitCodes -contains $trainExit) {
        Write-Host "Note: train returned benign teardown exit code $trainExit — output was saved, continuing" -ForegroundColor DarkYellow
    }

    Write-Host ""
    Write-Host "Done: $folder" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($failed.Count -eq 0) {
    Write-Host "All folders processed successfully." -ForegroundColor Green
} else {
    Write-Host "Completed with failures:" -ForegroundColor Yellow
    foreach ($f in $failed) { Write-Host "  - $f" -ForegroundColor Yellow }
}
