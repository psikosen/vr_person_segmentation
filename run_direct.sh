#!/bin/bash
# Direct VR Person Segmentation using the standalone Python script with YOLOv11

# Default paths
DEFAULT_INPUT="360.mp4"
DEFAULT_OUTPUT="./output/output_direct.webm"

# Get input and output paths from arguments
INPUT_VIDEO=${1:-$DEFAULT_INPUT}
OUTPUT_VIDEO=${2:-$DEFAULT_OUTPUT}

echo "Starting Direct VR Person Segmentation Pipeline with YOLOv11..."
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"

# Check for models in priority order
YOLO11_ROOT="./yolo11x-seg.pt"
YOLO11_MODELS="./models/yolo11x-seg.pt"
YOLO8X_ROOT="./yolov8x-seg.pt"
YOLO8X_MODELS="./models/yolov8x-seg.pt"
YOLO8S_MODELS="./models/yolov8s-seg.pt"

# Choose best available model
if [ -f "$YOLO11_ROOT" ]; then
    MODEL_PATH="$YOLO11_ROOT"
    echo "Using YOLOv11x-seg model from root directory"
elif [ -f "$YOLO11_MODELS" ]; then
    MODEL_PATH="$YOLO11_MODELS"
    echo "Using YOLOv11x-seg model from models directory"
elif [ -f "$YOLO8X_ROOT" ]; then
    MODEL_PATH="$YOLO8X_ROOT"
    echo "Using YOLOv8x-seg model from root directory"
elif [ -f "$YOLO8X_MODELS" ]; then
    MODEL_PATH="$YOLO8X_MODELS"
    echo "Using YOLOv8x-seg model from models directory"
else
    MODEL_PATH="$YOLO8S_MODELS"
    echo "Using YOLOv8s-seg model (newer models not found)"
fi

# Activate virtual environment
source venv/bin/activate

# Run the direct segmentation script
python fix_segmentation.py --input-video "$INPUT_VIDEO" \
                          --output-video "$OUTPUT_VIDEO" \
                          --model "$MODEL_PATH" \
                          --confidence 0.2 \
                          --iou 0.35 \
                          --mask-threshold 0.15 \
                          --morph-kernel 9 \
                          --morph-iterations 3 \
                          --overlap-threshold 0.2

# Deactivate virtual environment
deactivate

echo "Processing complete! Output saved to: $OUTPUT_VIDEO"
