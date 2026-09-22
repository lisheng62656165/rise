param([int[]]$Seeds = @(53403), [int]$Workers = 10, [string]$OutputRoot = 'outputs/full_test')
$ErrorActionPreference = 'Stop'
& python -X utf8 (Join-Path $PSScriptRoot 'run_experiment.py') --seeds @Seeds --workers $Workers --output-dir $OutputRoot
exit $LASTEXITCODE
