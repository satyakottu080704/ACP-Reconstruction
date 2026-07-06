@echo off
REM Push ACP-Reconstruction to GitHub — run this from this folder on your PC.
REM Requires: git installed + you signed in to GitHub (git will prompt).

git init -b main
git add -A
git commit -m "first commit: ACP-Reconstruction - sketch-to-floor-plan pipeline (YOLOv11-seg + geometry engine + Acorn-style exporters)"
git remote add origin https://github.com/satyakottu080704/ACP-Reconstruction-.git
git push -u origin main
pause
