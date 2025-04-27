#!/bin/bash
# Advanced VR Person Segmentation Pipeline using YOLOv11x-seg

# Default paths
DEFAULT_INPUT="360.mp4"
DEFAULT_OUTPUT="./output/output_yolo11.webm"

# Get input and output paths from arguments
INPUT_VIDEO=${1:-$DEFAULT_INPUT}
OUTPUT_VIDEO=${2:-$DEFAULT_OUTPUT}

echo "Starting YOLOv11 Person Segmentation Pipeline..."
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"

# Ensure models directory exists
mkdir -p models

# Check for YOLOv11x-seg model and move it to models if needed
if [ -f "./yolo11x-seg.pt" ] && [ ! -f "./models/yolo11x-seg.pt" ]; then
    echo "Found YOLOv11x-seg model in root directory, copying to models directory..."
    cp "./yolo11x-seg.pt" "./models/yolo11x-seg.pt"
fi

# Set model path
MODEL_PATH="./yolo11x-seg.pt"
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="./models/yolo11x-seg.pt"
fi

# Confirm model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: YOLOv11x-seg model not found"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the direct segmentation script with extra optimized parameters for YOLOv11
python fix_segmentation.py --input-video "$INPUT_VIDEO" \
                          --output-video "$OUTPUT_VIDEO" \
                          --model "$MODEL_PATH" \
                          --confidence 0.18 \
                          --iou 0.3 \
                          --mask-threshold 0.15 \
                          --morph-kernel 11 \
                          --morph-iterations 2 \
                          --overlap-threshold 0.2

# Deactivate virtual environment
deactivate

echo "✅ Processing complete with YOLOv11! Output saved to: $OUTPUT_VIDEO"
