#!/bin/bash
# Universal VR Person Segmentation Pipeline
# Handles both regular 360° videos and stereoscopic (side-by-side) VR videos

# Default settings
DEFAULT_INPUT="360.mp4"
DEFAULT_OUTPUT="./output/output_processed.webm"
MODE="mono"  # Default to mono (regular 360) mode

# Help function
function show_help {
    echo "Universal VR Person Segmentation Script"
    echo "======================================="
    echo "Usage: $0 [options] [input_video] [output_video]"
    echo ""
    echo "Options:"
    echo "  --stereo       Process as stereoscopic VR (side-by-side) video"
    echo "  --mono         Process as regular 360° video (default)"
    echo "  --left-eye     Process only the left eye view (for stereo)"
    echo "  --right-eye    Process only the right eye view (for stereo)"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 my_video.mp4 output.webm                # Process regular 360 video"
    echo "  $0 --stereo my_stereo.mp4 output.webm      # Process stereoscopic video"
    echo "  $0 --left-eye stereo_video.mp4 output.webm # Process left eye only"
    echo ""
}

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stereo)
            MODE="stereo"
            shift
            ;;
        --mono)
            MODE="mono"
            shift
            ;;
        --left-eye)
            MODE="left-eye"
            shift
            ;;
        --right-eye)
            MODE="right-eye"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            # If it's not an option, it must be the input video
            break
            ;;
    esac
done

# Get input and output paths from remaining arguments
INPUT_VIDEO=${1:-$DEFAULT_INPUT}
OUTPUT_VIDEO=${2:-$DEFAULT_OUTPUT}

echo "Starting Universal VR Person Segmentation Pipeline..."
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"
echo "Mode: $MODE"

# Create temp directory
mkdir -p temp

# Choose best available model
if [ -f "./yolo11x-seg.pt" ]; then
    MODEL_PATH="./yolo11x-seg.pt"
    echo "Using YOLOv11x-seg model"
elif [ -f "./models/yolo11x-seg.pt" ]; then
    MODEL_PATH="./models/yolo11x-seg.pt"
    echo "Using YOLOv11x-seg model"
elif [ -f "./yolov8x-seg.pt" ]; then
    MODEL_PATH="./yolov8x-seg.pt" 
    echo "Using YOLOv8x-seg model"
elif [ -f "./models/yolov8x-seg.pt" ]; then
    MODEL_PATH="./models/yolov8x-seg.pt"
    echo "Using YOLOv8x-seg model"
else
    MODEL_PATH="./models/yolov8s-seg.pt"
    echo "Using YOLOv8s-seg model"
fi

# Activate virtual environment
source venv/bin/activate

# Process based on mode
if [ "$MODE" = "stereo" ]; then
    echo "Processing stereoscopic VR video (both eyes)..."
    
    # First extract left and right eye views
    LEFT_EYE_VIDEO="./temp/left_eye.mp4"
    RIGHT_EYE_VIDEO="./temp/right_eye.mp4"
    LEFT_EYE_OUTPUT="./temp/left_eye_output.webm"
    RIGHT_EYE_OUTPUT="./temp/right_eye_output.webm"
    
    # Get video info
    VIDEO_WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=s=x:p=0 "$INPUT_VIDEO")
    VIDEO_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0 "$INPUT_VIDEO")
    HALF_WIDTH=$((VIDEO_WIDTH / 2))
    
    echo "Video dimensions: ${VIDEO_WIDTH}x${VIDEO_HEIGHT}, each eye: ${HALF_WIDTH}x${VIDEO_HEIGHT}"
    
    # Extract left eye (first half of the frame)
    echo "Extracting left eye view..."
    ffmpeg -i "$INPUT_VIDEO" -vf "crop=${HALF_WIDTH}:${VIDEO_HEIGHT}:0:0" -c:v libx264 -preset fast -crf 18 -y "$LEFT_EYE_VIDEO"
    
    # Extract right eye (second half of the frame)
    echo "Extracting right eye view..."
    ffmpeg -i "$INPUT_VIDEO" -vf "crop=${HALF_WIDTH}:${VIDEO_HEIGHT}:${HALF_WIDTH}:0" -c:v libx264 -preset fast -crf 18 -y "$RIGHT_EYE_VIDEO"
    
    # Process left eye
    echo "Processing left eye view..."
    python fix_segmentation.py --input-video "$LEFT_EYE_VIDEO" \
                              --output-video "$LEFT_EYE_OUTPUT" \
                              --model "$MODEL_PATH" \
                              --confidence 0.18 \
                              --iou 0.3 \
                              --mask-threshold 0.15 \
                              --morph-kernel 11 \
                              --morph-iterations 2 \
                              --overlap-threshold 0.2
    
    # Process right eye
    echo "Processing right eye view..."
    python fix_segmentation.py --input-video "$RIGHT_EYE_VIDEO" \
                              --output-video "$RIGHT_EYE_OUTPUT" \
                              --model "$MODEL_PATH" \
                              --confidence 0.18 \
                              --iou 0.3 \
                              --mask-threshold 0.15 \
                              --morph-kernel 11 \
                              --morph-iterations 2 \
                              --overlap-threshold 0.2
    
    # Combine processed videos side by side
    echo "Combining processed views into stereoscopic output..."
    ffmpeg -i "$LEFT_EYE_OUTPUT" -i "$RIGHT_EYE_OUTPUT" \
           -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" \
           -c:v libvpx-vp9 -pix_fmt yuva420p -crf 23 -b:v 0 -deadline good \
           -y "$OUTPUT_VIDEO"
    
elif [ "$MODE" = "left-eye" ]; then
    echo "Processing left eye of stereoscopic video only..."
    
    # Extract left eye (first half of the frame)
    LEFT_EYE_VIDEO="./temp/left_eye.mp4"
    
    # Get video info
    VIDEO_WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=s=x:p=0 "$INPUT_VIDEO")
    VIDEO_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0 "$INPUT_VIDEO")
    HALF_WIDTH=$((VIDEO_WIDTH / 2))
    
    echo "Video dimensions: ${VIDEO_WIDTH}x${VIDEO_HEIGHT}, extracting left half (${HALF_WIDTH}x${VIDEO_HEIGHT})"
    
    # Extract left eye
    ffmpeg -i "$INPUT_VIDEO" -vf "crop=${HALF_WIDTH}:${VIDEO_HEIGHT}:0:0" -c:v libx264 -preset fast -crf 18 -y "$LEFT_EYE_VIDEO"
    
    # Process left eye
    python fix_segmentation.py --input-video "$LEFT_EYE_VIDEO" \
                              --output-video "$OUTPUT_VIDEO" \
                              --model "$MODEL_PATH" \
                              --confidence 0.18 \
                              --iou 0.3 \
                              --mask-threshold 0.15 \
                              --morph-kernel 11 \
                              --morph-iterations 2 \
                              --overlap-threshold 0.2

elif [ "$MODE" = "right-eye" ]; then
    echo "Processing right eye of stereoscopic video only..."
    
    # Extract right eye (second half of the frame)
    RIGHT_EYE_VIDEO="./temp/right_eye.mp4"
    
    # Get video info
    VIDEO_WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=s=x:p=0 "$INPUT_VIDEO")
    VIDEO_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0 "$INPUT_VIDEO")
    HALF_WIDTH=$((VIDEO_WIDTH / 2))
    
    echo "Video dimensions: ${VIDEO_WIDTH}x${VIDEO_HEIGHT}, extracting right half (${HALF_WIDTH}x${VIDEO_HEIGHT})"
    
    # Extract right eye
    ffmpeg -i "$INPUT_VIDEO" -vf "crop=${HALF_WIDTH}:${VIDEO_HEIGHT}:${HALF_WIDTH}:0" -c:v libx264 -preset fast -crf 18 -y "$RIGHT_EYE_VIDEO"
    
    # Process right eye
    python fix_segmentation.py --input-video "$RIGHT_EYE_VIDEO" \
                              --output-video "$OUTPUT_VIDEO" \
                              --model "$MODEL_PATH" \
                              --confidence 0.18 \
                              --iou 0.3 \
                              --mask-threshold 0.15 \
                              --morph-kernel 11 \
                              --morph-iterations 2 \
                              --overlap-threshold 0.2

else  # mono mode (regular 360)
    echo "Processing regular 360° video..."
    
    # Process with standard approach
    python fix_segmentation.py --input-video "$INPUT_VIDEO" \
                              --output-video "$OUTPUT_VIDEO" \
                              --model "$MODEL_PATH" \
                              --confidence 0.18 \
                              --iou 0.3 \
                              --mask-threshold 0.15 \
                              --morph-kernel 11 \
                              --morph-iterations 2 \
                              --overlap-threshold 0.2
fi

# Deactivate virtual environment
deactivate

echo "✅ Processing complete! Output saved to: $OUTPUT_VIDEO"
