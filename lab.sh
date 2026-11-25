#!/usr/bin/env bash
# Acoustic Messenger lab launcher (macOS / Linux). Full guide: docs/experiments.md
#
#   ./lab.sh setup                 one-time: virtualenv, packages, tests, list audio devices
#   ./lab.sh tx [options]          Laptop A (speaker):    guided transmit session
#   ./lab.sh rx [options]          Laptop B (microphone): guided record + decode session
#   ./lab.sh analyze               after recording: trials.csv + all experiment plots
#   ./lab.sh devices               list audio devices (use the index with --device)
#
# Options for tx/rx: --experiment exp2_tone_separation  --start 7  --device 0  (rx only: --redo)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

setup() {
  if [ ! -x "$PY" ]; then
    echo "Creating virtual environment .venv ..."
    python3 -m venv .venv
  fi
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
  echo "Running tests ..."
  "$PY" -m pytest -q
  echo "Audio devices (> = default input, < = default output):"
  "$PY" -m sounddevice
  echo "Setup OK."
}

case "${1:-help}" in
  setup)
    setup ;;
  tx|rx)
    role="$1"; shift
    [ -x "$PY" ] || setup
    [ -f data/transmitted/baseline_32bit.wav ] || "$PY" scripts/prepare_experiment_audio.py
    exec "$PY" scripts/lab_session.py "$role" "$@" ;;
  analyze)
    exec "$PY" scripts/run_experiment_analysis.py --manifest data/recorded/manifest.json ;;
  devices)
    exec "$PY" -m sounddevice ;;
  *)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
