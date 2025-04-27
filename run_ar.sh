#!/bin/bash
# AR-Optimized VR Person Segmentation Pipeline
# Processes stereoscopic video for AR use by combining both eye views
# into a single high-quality segmentation

# Default settings
DEFAULT_INPUT="360.mp4"
DEFAULT_OUTPUT="./output/ar_ready.webm"

# Parse arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "AR-Optimized Person Segmentation for Stereoscopic Videos"
    echo "======================================================="
    echo "Usage: $0 [input_video] [output_video]"
    echo ""
    echo "This script processes stereoscopic VR videos for AR use by:"
    echo "1. Extracting both eye views"
    echo "2. Processing each view for person segmentation"
    echo "3. Combining them into a single, high-quality output"
    echo "4. Generating a transparent video suitable for AR overlays"
    echo ""
    echo "Examples:"
    echo "  $0 my_stereo_video.mp4 ./output/ar_people.webm"
    exit 0
fi

# Get input and output paths
INPUT_VIDEO=${1:-$DEFAULT_INPUT}
OUTPUT_VIDEO=${2:-$DEFAULT_OUTPUT}
TEMP_DIR="./temp/ar_processing"

echo "Starting AR-Optimized Person Segmentation..."
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"

# Create temp directories
mkdir -p "$TEMP_DIR"

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

# Extract video information
echo "Analyzing video dimensions..."
VIDEO_WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=s=x:p=0 "$INPUT_VIDEO")
VIDEO_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0 "$INPUT_VIDEO")
VIDEO_FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=s=x:p=0 "$INPUT_VIDEO")
# Convert fraction to decimal if needed
if [[ $VIDEO_FPS == *"/"* ]]; then
    NUM=$(echo $VIDEO_FPS | cut -d'/' -f1)
    DEN=$(echo $VIDEO_FPS | cut -d'/' -f2)
    VIDEO_FPS=$(echo "scale=2; $NUM / $DEN" | bc)
fi

# Determine if this is likely a stereo video by checking aspect ratio
# Stereo videos are typically twice as wide as they are tall
ASPECT_RATIO=$(echo "scale=2; $VIDEO_WIDTH / $VIDEO_HEIGHT" | bc)
IS_LIKELY_STEREO=false
if (( $(echo "$ASPECT_RATIO > 1.9" | bc -l) )); then
    IS_LIKELY_STEREO=true
    echo "Detected stereoscopic video (aspect ratio: $ASPECT_RATIO)"
    HALF_WIDTH=$((VIDEO_WIDTH / 2))
    echo "Each eye view: ${HALF_WIDTH}x${VIDEO_HEIGHT}"
else
    echo "Detected standard video (aspect ratio: $ASPECT_RATIO)"
    echo "Processing as single view"
fi

# Define processing paths
LEFT_EYE_VIDEO="$TEMP_DIR/left_eye.mp4"
RIGHT_EYE_VIDEO="$TEMP_DIR/right_eye.mp4"
LEFT_PROCESSED="$TEMP_DIR/left_processed.webm"
RIGHT_PROCESSED="$TEMP_DIR/right_processed.webm"
COMBINED_PROCESSED="$TEMP_DIR/combined.webm"

# Process based on video type
if [ "$IS_LIKELY_STEREO" = true ]; then
    echo "=== Processing as stereoscopic video for AR ==="
    
    # Extract left eye (first half of frame)
    echo "Extracting left eye view..."
    ffmpeg -i "$INPUT_VIDEO" -vf "crop=${HALF_WIDTH}:${VIDEO_HEIGHT}:0:0" -c:v libx264 -preset fast -crf 18 -y "$LEFT_EYE_VIDEO"
    
    # Extract right eye (second half of frame)
    echo "Extracting right eye view..."
    ffmpeg -i "$INPUT_VIDEO" -vf "crop=${HALF_WIDTH}:${VIDEO_HEIGHT}:${HALF_WIDTH}:0" -c:v libx264 -preset fast -crf 18 -y "$RIGHT_EYE_VIDEO"
    
    # Process left eye with more aggressive parameters
    echo "Processing left eye view..."
    python fix_segmentation.py --input-video "$LEFT_EYE_VIDEO" \
                             --output-video "$LEFT_PROCESSED" \
                             --model "$MODEL_PATH" \
                             --confidence 0.15 \
                             --iou 0.3 \
                             --mask-threshold 0.1 \
                             --morph-kernel 11 \
                             --morph-iterations 2 \
                             --overlap-threshold 0.2
    
    # Process right eye with more aggressive parameters
    echo "Processing right eye view..."
    python fix_segmentation.py --input-video "$RIGHT_EYE_VIDEO" \
                             --output-video "$RIGHT_PROCESSED" \
                             --model "$MODEL_PATH" \
                             --confidence 0.15 \
                             --iou 0.3 \
                             --mask-threshold 0.1 \
                             --morph-kernel 11 \
                             --morph-iterations 2 \
                             --overlap-threshold 0.2
    
    # Now create a Python script to combine the left and right processed videos
    # This merges the alpha channels to get the best segmentation from both views
    MERGE_SCRIPT="$TEMP_DIR/merge_views.py"
    
    echo "Creating merge script for optimal AR view combination..."
    cat > "$MERGE_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Script to merge left and right eye segmentations for optimal AR output.
This takes both processed views and creates a single, high-quality output
by taking the best segmentation from each view.
"""

import cv2
import numpy as np
import sys
import argparse
import os
from subprocess import Popen, PIPE
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def extract_frames(video_path):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def merge_segmentations(left_frames, right_frames):
    """
    Merge two sets of RGBA frames, prioritizing areas with higher alpha values.
    This creates a single best view combining segmentations from both eyes.
    """
    merged_frames = []
    
    for i in range(min(len(left_frames), len(right_frames))):
        # Split into RGB and alpha channels
        left_rgb = left_frames[i][:,:,:3]
        left_alpha = left_frames[i][:,:,3]
        
        right_rgb = right_frames[i][:,:,:3]
        right_alpha = right_frames[i][:,:,3]
        
        # Create a combined alpha mask, taking the maximum value at each pixel
        combined_alpha = np.maximum(left_alpha, right_alpha)
        
        # Create a weight mask based on alpha values
        # This ensures that pixels with higher alpha values contribute more
        left_weight = left_alpha / (left_alpha + right_alpha + 1e-10)
        right_weight = right_alpha / (left_alpha + right_alpha + 1e-10)
        
        # Handle special case where both alphas are zero
        both_zero = (left_alpha == 0) & (right_alpha == 0)
        left_weight = np.where(both_zero, 0, left_weight)
        right_weight = np.where(both_zero, 0, right_weight)
        
        # Weighted blend of RGB channels
        merged_rgb = np.zeros_like(left_rgb)
        for c in range(3):  # RGB channels
            merged_rgb[:,:,c] = (
                left_rgb[:,:,c] * left_weight + 
                right_rgb[:,:,c] * right_weight
            )
        
        # Combine RGB with the maximum alpha channel
        merged_frame = np.dstack((merged_rgb, combined_alpha))
        merged_frames.append(merged_frame)
    
    return merged_frames

def write_video(frames, output_path, fps=30.0):
    """Write frames to a video file with transparency."""
    if not frames:
        logging.error("No frames to write")
        return False
    
    height, width = frames[0].shape[:2]
    
    # Convert frames to bytes
    raw_bytes = b''
    for frame in frames:
        raw_bytes += frame.tobytes()
    
    # Use FFmpeg to encode with transparency
    command = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgba',
        '-r', str(fps),
        '-i', '-',  # Read from stdin
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',
        '-crf', '20',
        '-b:v', '0',
        '-deadline', 'good',
        '-cpu-used', '1',
        output_path
    ]
    
    process = Popen(command, stdin=PIPE)
    process.stdin.write(raw_bytes)
    process.stdin.close()
    process.wait()
    
    return process.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Merge left and right eye segmentations for AR')
    parser.add_argument('--left', required=True, help='Left eye processed video')
    parser.add_argument('--right', required=True, help='Right eye processed video')
    parser.add_argument('--output', required=True, help='Output merged video')
    parser.add_argument('--fps', type=float, default=30.0, help='Frame rate of output video')
    args = parser.parse_args()
    
    logging.info(f"Reading left eye video: {args.left}")
    left_frames = extract_frames(args.left)
    logging.info(f"Reading right eye video: {args.right}")
    right_frames = extract_frames(args.right)
    
    logging.info(f"Merging {len(left_frames)} frames from left eye with {len(right_frames)} frames from right eye")
    merged_frames = merge_segmentations(left_frames, right_frames)
    
    logging.info(f"Writing merged video to: {args.output}")
    success = write_video(merged_frames, args.output, args.fps)
    
    if success:
        logging.info("Successfully merged videos for AR use")
    else:
        logging.error("Failed to write merged video")

if __name__ == "__main__":
    main()
EOF

    # Make the merge script executable
    chmod +x "$MERGE_SCRIPT"
    
    # Run the merge script to combine left and right eye segmentations
    echo "Merging left and right eye segmentations for optimal AR output..."
    python "$MERGE_SCRIPT" --left "$LEFT_PROCESSED" --right "$RIGHT_PROCESSED" \
                         --output "$OUTPUT_VIDEO" --fps "$VIDEO_FPS"
    
else
    # Standard video processing
    echo "=== Processing as standard video for AR ==="
    
    # Process with enhanced parameters
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

# Clean up temp files
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Deactivate virtual environment
deactivate

echo "✅ AR-ready processing complete!"
echo "Output saved to: $OUTPUT_VIDEO"
echo ""
echo "This video contains only the segmented people with transparency"
echo "and is ready to use as an overlay in AR applications."
