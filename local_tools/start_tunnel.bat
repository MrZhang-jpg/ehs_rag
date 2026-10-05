@echo off
chcp 65001 >nul
title EHS RAG - SSH 隧道 (localhost:8000 映射远端 FastAPI)
setlocal

if exist "D:\tools\Anaconda3\python.exe" (
    set "PY=D:\tools\Anaconda3\python.exe"
) else (
    set "PY=python"
)

set "LPORT=%~1"
if "%LPORT%"=="" set "LPORT=8000"
set "RPORT=%~2"
if "%RPORT%"=="" set "RPORT=8000"

netstat -ano | findstr ":%LPORT% " | findstr LISTENING >nul 2>nul
if not errorlevel 1 (
  echo [提示] 本地端口 %LPORT% 已经被占用
  echo        可直接访问 http://127.0.0.1:%LPORT%/
  echo.
)

if "%EHS_SSH_PW%"=="" set /p EHS_SSH_PW=请输入 SSH 密码:
echo.
echo 隧道启动：本地 127.0.0.1:%LPORT% 映射远端 127.0.0.1:%RPORT% ...
echo 访问地址: http://127.0.0.1:%LPORT%/
echo 关闭窗口即可断开隧道
echo.

"%PY%" "%~dp0tunnel.py" %LPORT% %RPORT%
echo.
echo 隧道已断开
pause
