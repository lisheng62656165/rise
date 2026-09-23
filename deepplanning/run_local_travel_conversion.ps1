param(
    [Parameter(Mandatory = $true)][string]$ResultDir,
    [Parameter(Mandatory = $true)][ValidateSet("zh", "en")][string]$Language,
    [Parameter(Mandatory = $true)][int]$KeyFromEnd
)

$ErrorActionPreference = "Stop"
$root = "D:\ICLR\experiment\deepplanning_dsr_full_20260828"
$keys = Get-Content "C:\Users\ZhuanZ\Desktop\api\key.txt" | Where-Object { $_.Trim() }
if ($KeyFromEnd -lt 1 -or $KeyFromEnd -gt $keys.Count) {
    throw "Invalid key offset: $KeyFromEnd"
}

$env:NVIDIA_API_KEY = $keys[-$KeyFromEnd].Trim()
$env:DEEPPLANNING_OPENAI_BASE_URL = "https://integrate.api.nvidia.com/v1"
$env:TRAVEL_CONVERSION_MODEL = "nemotron-3.5-lightning-30b-a3b"
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = $env:HTTP_PROXY
$env:ALL_PROXY = $env:HTTP_PROXY
$env:PYTHONPATH = "$root;$root\travelplanning"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

python "$root\run_deepplanning_travel_conversion.py" `
    --travel-root "$root\travelplanning" `
    --result-dir $ResultDir `
    --language $Language `
    --workers 10 `
    --seed 53403 `
    --max-tokens 4096 `
    --max-parse-retries 5 `
    --request-timeout 60 `
    --allow-partial
