[CmdletBinding()]
param(
    [string]$InputDir = "..\data\inference",
    [string]$OutputDir = "result\patient_clustered\beta-0.01_alpha-0.20_source-cap-2000",
    [string]$Paper2Dir = "..\Paper2.0\Clinical_Triage__New_Version_",
    [string]$PythonExe = "python",
    [int]$Splits = 500,
    [int]$Bootstraps = 1000,
    [switch]$RenderOnly,
    [switch]$NoInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $workflowTemp = Join-Path $projectRoot "result\.cache\patient_paper2_temp"
    New-Item -ItemType Directory -Force -Path $workflowTemp | Out-Null
    $env:TEMP = $workflowTemp
    $env:TMP = $workflowTemp
    $pythonCommand = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw "Python executable not found: $PythonExe" }
    if (-not $RenderOnly -and -not (Test-Path -LiteralPath $InputDir -PathType Container)) {
        throw "Authoritative JSON directory not found: $InputDir"
    }

    if ($RenderOnly) {
        $arguments = @(
            "run\cli.py", "clustered-render",
            "--run-dir", $OutputDir,
            "--paper2-dir", $Paper2Dir
        )
        if (-not $NoInstall) {
            $arguments += "--install-paper2"
        }
    }
    else {
        $arguments = @(
            "run\cli.py", "clustered-paper",
            "--config", "configs\patient_clustered_source_cap_2000_beta_001.json",
            "--input-dir", $InputDir,
            "--output-dir", $OutputDir,
            "--n-splits", $Splits,
            "--n-bootstrap", $Bootstraps,
            "--paper2-dir", $Paper2Dir
        )
        if ($NoInstall) {
            $arguments += "--no-install"
        }
    }

    Write-Host "Running patient-balanced, patient-clustered Paper2.0 workflow..." -ForegroundColor Cyan
    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Patient-clustered workflow failed with exit code $LASTEXITCODE"
    }
    Write-Host "Done: 11 figures and 6 tables are available under $OutputDir\deliverables\paper2" -ForegroundColor Green
    Write-Host "Check manifest.json: exploratory runs are packaged but are not installed into the manuscript." -ForegroundColor Yellow
}
finally {
    Pop-Location
}
