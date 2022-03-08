@echo off
set mydate=%date:~10,4%-%date:~4,2%-%date:~7,2%
echo %mydate%
set /p id=Enter Folder Name: 
echo %id%
mkdir "%mydate% %id%"
echo %USERNAME%
pause
