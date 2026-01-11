#!/bin/bash
# Quick script to run LFM2.5-Audio WebRTC server
# Uses Python 3.12 environment

PYTHON_312="/Users/rodneyfranklin/miniforge3/envs/liquid-py312/bin/python"
SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting LFM2.5-Audio WebRTC Server..."
echo "📍 Python: $PYTHON_312"
echo "📍 Server dir: $SERVER_DIR"
echo "📍 Endpoint: http://localhost:8000/rtc"
echo ""

$PYTHON_312 "$SERVER_DIR/server.py"
