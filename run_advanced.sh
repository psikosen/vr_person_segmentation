#!/bin/bash
# Advanced 360 VR Person Segmentation Pipeline with YOLOv8x-seg model

# Default paths
DEFAULT_INPUT="360.mp4"
DEFAULT_OUTPUT="./output/output_advanced.webm"

# Get input and output paths from arguments
INPUT_VIDEO=${1:-$DEFAULT_INPUT}
OUTPUT_VIDEO=${2:-$DEFAULT_OUTPUT}

echo "Starting Advanced 360 VR Person Segmentation Pipeline..."
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"

# Check for YOLOv8x-seg model and download if needed
MODEL_PATH="models/yolov8x-seg.pt"
ROOT_MODEL_PATH="yolov8x-seg.pt"
mkdir -p models

if [ ! -f "$MODEL_PATH" ]; then
    if [ -f "$ROOT_MODEL_PATH" ]; then
        echo "Found YOLOv8x-seg model in root directory, moving to models directory..."
        mv "$ROOT_MODEL_PATH" "$MODEL_PATH"
    else
        echo "YOLOv8x-seg model not found, downloading..."
        # Activate virtual environment
        source venv/bin/activate
        python -c "from ultralytics import YOLO; YOLO('yolov8x-seg.pt')" || echo "Failed to download model"
        
        if [ -f "$ROOT_MODEL_PATH" ]; then
            echo "Downloaded YOLOv8x-seg model, moving to models directory..."
            mv "$ROOT_MODEL_PATH" "$MODEL_PATH"
        else
            echo "Error: Failed to download YOLOv8x-seg model"
            echo "Falling back to YOLOv8s-seg model..."
            MODEL_PATH="models/yolov8s-seg.pt"
        fi
        
        # Deactivate virtual environment
        deactivate
    fi
fi

# Run the pipeline with YOLOv8x-seg (or fallback) and optimized parameters
./run.sh --input-video "$INPUT_VIDEO" \
         --output-video "$OUTPUT_VIDEO" \
         --model-path "$MODEL_PATH" \
         --confidence 0.2 \
         --viewpoint-hfov 90.0 \
         --viewpoint-vfov 60.0

echo "Pipeline completed. Output saved to $OUTPUT_VIDEO"
