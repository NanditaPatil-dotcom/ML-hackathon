@echo off
REM Laptop 2: serve the evaluator UI on http://localhost:5500
REM Then open http://localhost:5500/?api=http://<LAPTOP1_IP>:8000 in a browser.
cd /d %~dp0ui
python -m http.server 5500
