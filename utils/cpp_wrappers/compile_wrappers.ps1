Param(
    [string]$PythonExecutable = 'python'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$moduleDir = Join-Path $scriptRoot 'cpp_subsampling'

if (-not (Test-Path -LiteralPath $moduleDir)) {
    throw "cpp_subsampling directory was not found at $moduleDir"
}

Push-Location $moduleDir
try {
    Write-Host "Building cpp_subsampling extension with $PythonExecutable" -ForegroundColor White
    & $PythonExecutable 'setup.py' 'build_ext' '--inplace'
    if ($LASTEXITCODE -ne 0) {
        throw "Python build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}