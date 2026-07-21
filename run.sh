#!/usr/bin/env bash
# landscape2portrait — one-command demo.
# Converts a folder of 16:9 clips to 9:16 (resumable) and opens the dashboard.
#
#   ./run.sh /path/to/videos          # convert the folder, then launch the dashboard
#   ./run.sh /path/to/videos --smoke  # only the 10-clip smoke set (fast)
#
# Requires: ffmpeg on PATH, and `pip install -r requirements.txt` in a venv.
# YOLOv8n weights auto-download on first run.
set -euo pipefail
cd "$(dirname "$0")"

VIDEOS="${1:?usage: ./run.sh /path/to/videos [--smoke]}"
PY="${PYTHON:-python}"

if [ "${2:-}" = "--smoke" ]; then
  "$PY" batch.py --folder "$VIDEOS" --list smoke_set.txt --force
else
  "$PY" batch.py --folder "$VIDEOS"
fi

echo "==> launching dashboard at http://localhost:8501 (Ctrl-C to stop)"
"$PY" -m streamlit run app.py
