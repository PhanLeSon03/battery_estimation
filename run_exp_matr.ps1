# ============================================================
# run_exp_matr.ps1
# Grid search: N_EARLY x N_RANDOM  (5x5 = 25 combos)
# Each combo trained 5 times  ->  125 total runs, sequential
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_exp_matr.ps1
# ============================================================

$earlyValues     = @(2, 4, 6, 8, 10)
$randomValues    = @(2, 4, 6, 8, 10)
$nRuns           = 5
$contentDir      = "./content_bml/MATR"
$baseOut         = "./checkpoints_clf_bml/EXP_MATR"
$benignExitCodes = @(-1073740791, -1073741819)

# ── Build job list ────────────────────────────────────────────────────────────
$jobs = [System.Collections.Generic.List[PSCustomObject]]::new()
foreach ($early in $earlyValues) {
    foreach ($random in $randomValues) {
        for ($r = 1; $r -le $nRuns; $r++) {
            $jobs.Add([PSCustomObject]@{
                n_early  = $early
                n_random = $random
                run      = $r
                output_dir = "$baseOut/E_R_${early}_${random}_run${r}"
            })
        }
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "EXP_MATR Grid Search  --  $($jobs.Count) runs total" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "Total jobs  : $($jobs.Count)  (5 N_EARLY x 5 N_RANDOM x 5 runs)" -ForegroundColor White
Write-Host "Content dir : $contentDir"  -ForegroundColor White
Write-Host "Output base : $baseOut"     -ForegroundColor White
Write-Host ""

$done    = 0
$nFailed = 0

foreach ($job in $jobs) {
    $done++
    Write-Host ""
    Write-Host "[$done / $($jobs.Count)]  N_EARLY=$($job.n_early)  N_RANDOM=$($job.n_random)  run=$($job.run)" -ForegroundColor Cyan

    # Set env vars so dataset_clf_bml.py picks them up
    $env:N_EARLY  = $job.n_early
    $env:N_RANDOM = $job.n_random

    $cmd = "python train_clf_bml.py" +
           " --content_dir `"$contentDir`"" +
           " --output_dir `"$($job.output_dir)`""

    Write-Host ">> $cmd" -ForegroundColor Yellow
    Invoke-Expression $cmd
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0 -and -not ($benignExitCodes -contains $exitCode)) {
        Write-Host "ERROR: job failed (exit $exitCode) -- continuing to next job" -ForegroundColor Red
        $nFailed++
    } elseif ($benignExitCodes -contains $exitCode) {
        Write-Host "Note: benign teardown exit $exitCode -- treating as success" -ForegroundColor DarkYellow
    } else {
        Write-Host "Done." -ForegroundColor Green
    }
}

# Clear env vars when finished
$env:N_EARLY  = $null
$env:N_RANDOM = $null

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "All jobs finished." -ForegroundColor Green
Write-Host "  Completed : $($jobs.Count - $nFailed) / $($jobs.Count)" -ForegroundColor Green
if ($nFailed -gt 0) {
    Write-Host "  Failed    : $nFailed" -ForegroundColor Yellow
}
Write-Host "============================================" -ForegroundColor Green
