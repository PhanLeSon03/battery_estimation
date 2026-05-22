# ============================================================
# run_model_adjust.ps1
# Grid search: cnn_dim x gru_dim x gru_layers  (4x4x3 = 48 combos)
# Each combo trained 3 times  ->  144 total runs, sequential
#
# Usage:
#   .\run_model_adjust.ps1
# ============================================================

$cnnDims         = @(8, 16, 32, 64)
$gruDims         = @(8, 16, 32, 64)
$gruLayers       = @(1, 2, 3)
$nRuns           = 3
$contentDir      = "./content_bml/LFP"
$baseOut         = "./checkpoints_clf_bml/Model_adjust"
$benignExitCodes = @(-1073740791, -1073741819)

$jobs = [System.Collections.Generic.List[PSCustomObject]]::new()
foreach ($cnn in $cnnDims) {
    foreach ($grud in $gruDims) {
        foreach ($grul in $gruLayers) {
            for ($r = 1; $r -le $nRuns; $r++) {
                $jobs.Add([PSCustomObject]@{
                    cnn_dim    = $cnn
                    gru_dim    = $grud
                    gru_layers = $grul
                    run        = $r
                    output_dir = "$baseOut/CNN_GRUd_GRUl_${cnn}_${grud}_${grul}_run${r}"
                })
            }
        }
    }
}

Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "Model Grid Search  --  $($jobs.Count) runs total" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "Total jobs  : $($jobs.Count)  (4 cnn x 4 grud x 3 grul x 3 runs)" -ForegroundColor White
Write-Host "Content dir : $contentDir"                -ForegroundColor White
Write-Host "Output base : $baseOut"                   -ForegroundColor White
Write-Host ""

$done    = 0
$nFailed = 0

foreach ($job in $jobs) {
    $done++
    Write-Host ""
    Write-Host "[$done / $($jobs.Count)]  cnn=$($job.cnn_dim)  grud=$($job.gru_dim)  grul=$($job.gru_layers)  run=$($job.run)" -ForegroundColor Cyan

    $cmd = "python train_clf_bml.py" +
           " --content_dir `"$contentDir`"" +
           " --output_dir `"$($job.output_dir)`"" +
           " --cnn_dim $($job.cnn_dim)" +
           " --gru_dim $($job.gru_dim)" +
           " --gru_layers $($job.gru_layers)"

    Write-Host ">> $cmd" -ForegroundColor Yellow
    Invoke-Expression $cmd
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0 -and -not ($benignExitCodes -contains $exitCode)) {
        Write-Host "ERROR: job failed (exit $exitCode) — continuing to next job" -ForegroundColor Red
        $nFailed++
    } elseif ($benignExitCodes -contains $exitCode) {
        Write-Host "Note: benign teardown exit $exitCode — treating as success" -ForegroundColor DarkYellow
    } else {
        Write-Host "Done." -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "All jobs finished." -ForegroundColor Green
Write-Host "  Completed : $($jobs.Count - $nFailed) / $($jobs.Count)" -ForegroundColor Green
if ($nFailed -gt 0) {
    Write-Host "  Failed    : $nFailed" -ForegroundColor Yellow
}
Write-Host "============================================" -ForegroundColor Green
