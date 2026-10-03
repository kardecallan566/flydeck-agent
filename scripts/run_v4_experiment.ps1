# v4 opt-in research: matched Fly/no-Fly experiments on the ORIGINAL history.
# Exactly the same GitHub branch and frozen v3 checkpoints. NO real trading.
[CmdletBinding()]
param(
    [int[]] $Seeds = @(42, 143, 244),
    [ValidateSet(100, 200, 300)] [int] $Population = 100,
    [ValidateRange(1, 16)] [int] $Threads = 4,
    [string] $History = "data/cache/BNBUSDT_5m.csv",
    [string] $Circuit = "data/malecns/motion_visual.json",
    [string] $FlyCache = "data/cache/fly_shared_features.npz",
    [string] $OutputRoot = "data/evolution/v4-balanced",
    [switch] $SkipTests,
    [switch] $RebuildFlyCache
)
$ErrorActionPreference = "Stop"
$expectedBranch = "feat/evolution-v2-survival-control-recent"
$actualBranch = (& git branch --show-current)
if ($LASTEXITCODE -ne 0 -or $actualBranch.Trim() -ne $expectedBranch) {
    throw "Refusing experiment outside $expectedBranch. Current branch: $actualBranch"
}
foreach ($item in @($History, $Circuit)) {
    if (!(Test-Path -LiteralPath $item -PathType Leaf)) {
        throw "Required local file not found: $item"
    }
}
foreach ($item in @(
    "data/evolution/controlled-v3/with_fly/population_checkpoint.json",
    "data/evolution/controlled-v3/without_fly/population_checkpoint.json"
)) {
    if (!(Test-Path -LiteralPath $item -PathType Leaf)) {
        throw "Frozen v3 checkpoint missing (never overwrite these): $item"
    }
}
if (!(Test-Path -LiteralPath $FlyCache -PathType Leaf) -and !$RebuildFlyCache) {
    throw "Missing shared historical MaleCNS cache. Provide -RebuildFlyCache explicitly if necessary."
}
if ($Seeds.Count -eq 0 -or ($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) {
    throw "Provide one or more UNIQUE seed integers."
}
$env:OPENBLAS_NUM_THREADS = "$Threads"
$env:OMP_NUM_THREADS = "$Threads"
if (!$SkipTests) {
    & python -m pytest tests/test_v4_research.py tests/test_evolution.py tests/test_forward_eval.py
    if ($LASTEXITCODE -ne 0) { throw "Tests failed. No training was started." }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$first = $true
foreach ($seed in $Seeds) {
    $out = Join-Path $OutputRoot ("seed-{0}" -f $seed)
    if (Test-Path -LiteralPath $out) {
        throw "Refusing to overwrite existing experiment: $out. Change -OutputRoot or -Seeds."
    }
    New-Item -ItemType Directory -Path $out | Out-Null
    $plan = [ordered]@{
        research_only = $true
        frozen_v3_unchanged = $true
        experimental_v4 = $true
        branch = $expectedBranch
        seed = $seed
        population = $Population
        history = $History
        circuit = $Circuit
        male_cns_shared_cache = $FlyCache
        class_balance_alpha = 0.75
        class_weight_cap = 1.5
        bias_l2_multiplier = 4.0
        matching_ablation = @("with_fly", "without_fly")
        output = $out
        hypothesis = "online resolved-only class weighting and stronger intercept shrinkage"
        never_select_on = @("historical_audit_already_examined", "repaired_300_candles_already_examined")
        no_actual_pancakeswap_pnl = $true
    }
    $plan | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $out "experiment_plan.json")
    $cliArgs = @(
        "-m", "flydeck.evolution_cli",
        "--data", $History,
        "--circuit", $Circuit,
        "--fly-cache", $FlyCache,
        "--compare-no-fly",
        "--population", "$Population",
        "--seed", "$seed",
        "--class-balance-alpha", "0.75",
        "--class-weight-cap", "1.5",
        "--bias-l2-multiplier", "4.0",
        "--output", $out
    )
    if ($RebuildFlyCache -and $first) {
        $cliArgs += "--rebuild-fly-cache"
    }
    Write-Host ("v4 matched experiment seed={0} agents={1} output={2}" -f $seed, $Population, $out)
    & python @cliArgs 2>&1 | Tee-Object -FilePath (Join-Path $out "training.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Seed $seed failed. Preserve $out/training.log; no model should be promoted."
    }
    foreach ($required in @(
        "with_fly/population_checkpoint.json",
        "without_fly/population_checkpoint.json",
        "ablation_report.json"
    )) {
        if (!(Test-Path -LiteralPath (Join-Path $out $required))) {
            throw "Missing required result for seed $seed : $required"
        }
    }
    $first = $false
}
Write-Host "Completed the requested number of SEQUENTIAL development experiments."
Write-Host "This script is NOT a 7-day continuous retraining daemon and does not place trades."
Write-Host "Evaluate matched populations on a FUTURE independent sealed window."
