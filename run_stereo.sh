#!/bin/bash
# Stereoscopic VR Person Segmentation Pipeline
# Handles VR videos with side-by-side views for left and right eyes

# Default paths
DEFAULT_INPUT="360.mp4"
DEFAULT_OUTPUT="./output/output_stereo.webm"

# Get input and output paths from arguments
INPUT_VIDEO=${1:-$DEFAULT_INPUT}
OUTPUT_VIDEO=${2:-$DEFAULT_OUTPUT}

echo "Starting Stereoscopic VR Person Segmentation Pipeline..."
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"

# Create temp directory
mkdir -p temp

# Choose best available model
if [ -f "./yolo11x-seg.pt" ]; then
    MODEL_PATH="./yolo11x-seg.pt"
    echo "Using YOLOv11x-seg model"
elif [ -f "./models/yolo11x-seg.pt" ]; then
    MODEL_PATH="./models/yolo11x-seg.pt"
    echo "Using YOLOv11x-seg model"
elif [ -f "./yolov8x-seg