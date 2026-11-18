#!/usr/bin/env bash
# ==============================================================================
# PathSnap Inference Server Launcher (Laptop 1)
# Hosts FastAPI backend, model weights, and student test evaluation trips.
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=========================================================="
echo "      PathSnap Server Startup (Laptop 1: FastAPI)        "
echo "=========================================================="

# Find local IP address to provide to Laptop 2
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")

echo "Project Root: $DIR"
echo "Host IP:      $LOCAL_IP"
echo "API URL:      http://$LOCAL_IP:8000"
echo "Docs URL:     http://$LOCAL_IP:8000/docs"
echo ""
echo "👉 Tell Laptop 2 (Streamlit UI) to connect to:"
echo "   http://$LOCAL_IP:8000"
echo "=========================================================="
echo "Starting Uvicorn ASGI Server on 0.0.0.0:8000..."
echo ""

export PYTHONPATH="$DIR/src:$PYTHONPATH"
python -m uvicorn pathsnap.service.api:app --host 0.0.0.0 --port 8000 --log-level info
