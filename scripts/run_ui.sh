#!/usr/bin/env bash
# ==============================================================================
# PathSnap Evaluator Dashboard Launcher (Laptop 2)
# Hosts Streamlit frontend for evaluator trip inspection.
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=========================================================="
echo "    PathSnap Evaluator Dashboard Startup (Laptop 2)       "
echo "=========================================================="
echo "Project Root: $DIR"
echo "Starting Streamlit..."
echo "=========================================================="

export PYTHONPATH="$DIR/src:$PYTHONPATH"
streamlit run src/pathsnap/ui/app.py --server.port 8501 --server.address 0.0.0.0
