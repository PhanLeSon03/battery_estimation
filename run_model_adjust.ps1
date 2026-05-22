# ============================================================
# run_model_adjust.ps1
# Grid search: cnn_dim x gru_dim x gru_layers  (4x4x3 = 48 combos)
# Each combo trained 3 times  ->  144 total runs
# Distributed across 10 parallel visible PowerShell windows (chunked)
#
# Usage:
#   .\run_model_adjust.ps1
#   .\run_model_adjust.ps1 -WorkerID 0
#   .\run_model_adjust.ps1 -WorkerID 1
#   ... up to -WorkerID 9
# ============================================================
param(
    [int]$WorkerID     = -1,
    [int]$TotalWorkers = 10
)

# ── Shared job definitions (same in every instance) ──────────────────────────
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

# ── Worker mode (each spawned terminal enters here) ───────────────────────────
if ($WorkerID -ge 0) {
    $chunkSize = [Math]::Ceiling($jobs.Count / $TotalWorkers)
    $start     = $WorkerID * $chunkSize
    $end       = [Math]::Min($start + $chunkSize, $jobs.Count) - 1

    if ($start -ge $jobs.Count) {
        Write-Host "Worker $($WorkerID + 1): no jobs assigned." -ForegroundColor DarkGray
        exit 0
    }

    $chunk = $jobs[$start..$end]

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Magenta
    Write-Host "Worker $($WorkerID + 1) / $TotalWorkers  --  $($chunk.Count) jobs" -ForegroundColor Magenta
    Write-Host "Jobs $($start + 1) to $($end + 1) of $($jobs.Count)" -ForegroundColor Magenta
    Write-Host "============================================" -ForegroundColor Magenta

    $done    = 0
    $nFailed = 0
    foreach ($job in $chunk) {
        $done++
        Write-Host ""
        Write-Host "[$done / $($chunk.Count)]  cnn=$($job.cnn_dim)  grud=$($job.gru_dim)  grul=$($job.gru_layers)  run=$($job.run)" -ForegroundColor Cyan

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
    Write-Host "Worker $($WorkerID + 1) finished." -ForegroundColor Green
    Write-Host "  Completed : $($chunk.Count - $nFailed) / $($chunk.Count)" -ForegroundColor Green
    if ($nFailed -gt 0) {
        Write-Host "  Failed    : $nFailed" -ForegroundColor Yellow
    }
    Write-Host "============================================" -ForegroundColor Green
    exit 0
}

# ── Launcher mode (runs when you execute the script normally) ─────────────────
$scriptPath = $MyInvocation.MyCommand.Path
if (-not $scriptPath) {
    Write-Host "ERROR: Run this script via -File, not interactively." -ForegroundColor Red
    Write-Host "Example:  powershell -ExecutionPolicy Bypass -File .\run_model_adjust.ps1" -ForegroundColor Yellow
    exit 1
}
$workDir   = Split-Path $scriptPath
$chunkSize = [Math]::Ceiling($jobs.Count / $TotalWorkers)

Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "Model Grid Search" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "Total jobs  : $($jobs.Count)  (4 cnn x 4 grud x 3 grul x 3 runs)" -ForegroundColor White
Write-Host "Workers     : $TotalWorkers"              -ForegroundColor White
Write-Host "Chunk size  : ~$chunkSize jobs per worker" -ForegroundColor White
Write-Host "Content dir : $contentDir"                -ForegroundColor White
Write-Host "Output base : $baseOut"                   -ForegroundColor White
Write-Host ""

for ($w = 0; $w -lt $TotalWorkers; $w++) {
    $workerArgs = "-NoExit -ExecutionPolicy Bypass -File `"$scriptPath`" -WorkerID $w -TotalWorkers $TotalWorkers"
    Start-Process powershell -ArgumentList $workerArgs -WorkingDirectory $workDir
    Write-Host "Launched worker $($w + 1) / $TotalWorkers" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "All $TotalWorkers workers launched." -ForegroundColor Green
