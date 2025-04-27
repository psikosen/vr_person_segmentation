#!/bin/bash
# Build script for the 360 VR Person Segmentation Pipeline

# Log file
LOG_FILE="./build_log.txt"

# Ensure the log file exists and is empty
echo "# Build Log - $(date)" > "$LOG_FILE"

echo "Starting build process for 360 VR Person Segmentation Pipeline..." | tee -a "$LOG_FILE"

# Create directories if they don't exist
echo "Creating necessary directories..." | tee -a "$LOG_FILE"
mkdir -p models
mkdir -p temp
mkdir -p tests

# Check Python version
echo "Checking Python version..." | tee -a "$LOG_FILE"
python3 --version >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is required but not found!" | tee -a "$LOG_FILE"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..." | tee -a "$LOG_FILE"
    python3 -m venv venv >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment!" | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "Virtual environment already exists." | tee -a "$LOG_FILE"
fi

# Activate virtual environment
echo "Activating virtual environment..." | tee -a "$LOG_FILE"
source venv/bin/activate >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment!" | tee -a "$LOG_FILE"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..." | tee -a "$LOG_FILE"
pip install -r requirements.txt >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies!" | tee -a "$LOG_FILE"
    exit 1
fi

# Check FFmpeg installation
echo "Checking FFmpeg installation..." | tee -a "$LOG_FILE"
ffmpeg -version >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "Warning: FFmpeg not found. Please install FFmpeg manually." | tee -a "$LOG_FILE"
else
    echo "FFmpeg is installed." | tee -a "$LOG_FILE"
fi

# Check if YOLOv8 model exists, download if not
if [ ! -f "models/yolov8s-seg.pt" ]; then
    echo "Downloading YOLOv8s-seg model..." | tee -a "$LOG_FILE"
    mkdir -p models
    # Use python to download the model
    python -c "from ultralytics import YOLO; YOLO('yolov8s-seg.pt')" >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download YOLOv8s-seg model!" | tee -a "$LOG_FILE"
        echo "Please download it manually and place it in the models directory." | tee -a "$LOG_FILE"
    else
        # Move the model to our models directory
        if [ -f "yolov8s-seg.pt" ]; then
            mv yolov8s-seg.pt models/
            echo "Model downloaded and moved to models directory." | tee -a "$LOG_FILE"
        fi
    fi
else
    echo "YOLOv8s-seg model already exists." | tee -a "$LOG_FILE"
fi

# Run basic tests
echo "Running basic tests..." | tee -a "$LOG_FILE"
python -c "from src import config, video_utils, segmentation, compositor, encoder, pipeline" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Failed to import modules!" | tee -a "$LOG_FILE"
    exit 1
fi

# Make executable scripts executable
echo "Setting execute permissions on scripts..." | tee -a "$LOG_FILE"
chmod +x ./run.sh >> "$LOG_FILE" 2>&1
chmod +x ./src/main.py >> "$LOG_FILE" 2>&1

# Run a quick sanity check on the configuration module
echo "Testing configuration module..." | tee -a "$LOG_FILE"
python -c "from src.config import parse_args; parse_args()" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Configuration module test failed!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Build process completed successfully!" | tee -a "$LOG_FILE"
echo "You can now use ./run.sh to run the pipeline." | tee -a "$LOG_FILE"

# Deactivate virtual environment
deactivate >> "$LOG_FILE" 2>&1

exit 0
