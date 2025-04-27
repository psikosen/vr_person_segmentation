#!/bin/bash
# Optimized 360 VR Person Segmentation Pipeline with improved parameters

# Default paths
DEFAULT_INPUT="360.mp4"
DEFAULT_OUTPUT="./output/output_improved.webm"

# Get input and output paths from arguments
INPUT_VIDEO=${1:-$DEFAULT_INPUT}
OUTPUT_VIDEO=${2:-$DEFAULT_OUTPUT}

echo "Starting Optimized 360 VR Person Segmentation Pipeline..."
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"

# Run the pipeline with optimized parameters
./run.sh --input-video "$INPUT_VIDEO" \
         --output-video "$OUTPUT_VIDEO" \
         --model-path models/yolov8s-seg.pt \
         --confidence 0.25 \
         --viewpoint-hfov 90.0 \
         --viewpoint-vfov 60.0

echo "Pipeline completed. Output saved to $OUTPUT_VIDEO"
