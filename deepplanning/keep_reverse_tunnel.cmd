@echo off
setlocal
:reconnect
C:\Windows\System32\OpenSSH\ssh.exe -n -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -R 17897:127.0.0.1:7897 lisheng@10.77.110.215
timeout /t 5 /nobreak >nul
goto reconnect
