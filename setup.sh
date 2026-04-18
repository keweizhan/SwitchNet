#!/bin/bash
# Setup script for Audio Transcription Tool
# Automates dependency installation and validation

set -e

echo "========================================="
echo "Audio Transcription Tool - Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed"
    echo "Install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Found: Python $PYTHON_VERSION"

# Check FFmpeg
echo ""
echo "Checking FFmpeg installation..."
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg is not installed"
    echo ""
    echo "Installation instructions:"
    echo "  macOS:   brew install ffmpeg"
    echo "  Linux:   sudo apt install ffmpeg"
    echo "  Windows: choco install ffmpeg"
    echo "  Manual:  https://ffmpeg.org/download.html"
    exit 1
fi

FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | awk '{print $3}')
echo "Found: FFmpeg $FFMPEG_VERSION"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

echo ""
echo "Installation complete."
echo ""
echo "========================================="
echo "Verification"
echo "========================================="
echo ""
echo "To verify installation:"
echo "  $PYTHON_CMD transcribe.py --help"
echo ""
echo "For quick start guide:"
echo "  cat QUICKSTART.md"
echo ""
echo "For detailed documentation:"
echo "  cat README.md"
