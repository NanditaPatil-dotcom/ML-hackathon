@echo off
REM Laptop 1: start the PathSnap API on all network interfaces, port 8000.
REM Optional overrides (uncomment and edit):
REM set PATHSNAP_DATA_DIR=data\raw
REM set PATHSNAP_MODEL_PATH=models\final_model.joblib
REM set PATHSNAP_EVAL_TRIPS=trip1,trip2
cd /d %~dp0
python -m api.main
