$ErrorActionPreference = "Continue"

$ssh = Join-Path $env:WINDIR "System32\OpenSSH\ssh.exe"
$arguments = @(
    "-N",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=10",
    "-o", "ServerAliveCountMax=12",
    "-o", "TCPKeepAlive=yes",
    "-R", "127.0.0.1:17897:127.0.0.1:7897",
    "lisheng@10.77.110.215"
)

while ($true) {
    $process = Start-Process -FilePath $ssh -ArgumentList $arguments -WindowStyle Hidden -PassThru
    $process.WaitForExit()
    Start-Sleep -Seconds 2
}
