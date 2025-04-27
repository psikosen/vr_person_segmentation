#!/usr/bin/env python3
"""
Script to optimize the 360 VR Person Segmentation Pipeline.

This script directly modifies the segmentation parameters to improve full-body shape detection.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_command(command):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Command failed with error: {e.stderr}"

def optimize_segmentation_params():
    """Optimize the segmentation.py file for better full-body shape detection."""
    segmentation_file = Path("src/segmentation.py")
    if not segmentation_file.exists():
        logging.error(f"Could not find {segmentation_file}")
        return False
    
    # Read the file
    with open(segmentation_file, 'r') as f:
        content = f.read()
    
    # Modify the segmentation parameters
    # 1. Modify default parameters in function signature
    modified_content = content.replace(
        "def segment_frame(model, frame: np.ndarray, confidence_threshold: float, \n                 iou_threshold: float = 0.7, \n                 apply_morphology: bool = True",
        "def segment_frame(model, frame: np.ndarray, confidence_threshold: float, \n                 iou_threshold: float = 0.4, \n                 apply_morphology: bool = True"
    )
    
    # 2. Modify the mask threshold
    modified_content = modified_content.replace(
        "binary_mask = (resized_mask > 0.3).astype(np.uint8)",
        "binary_mask = (resized_mask > 0.2).astype(np.uint8)  # Lowered threshold for more complete body shape"
    )
    
    # 3. Modify morphological operations
    modified_content = modified_content.replace(
        "kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))",
        "kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Larger kernel for smoother edges"
    )
    
    # 4. Increase dilation iterations
    modified_content = modified_content.replace(
        "binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)",
        "binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)  # More iterations to better fill gaps"
    )
    
    # 5. Lower overlap threshold
    modified_content = modified_content.replace(
        "if overlap_percentage > 0.3:",
        "if overlap_percentage > 0.25:  # Lower threshold for merging overlapping masks"
    )
    
    # Write the modified file
    with open(segmentation_file, 'w') as f:
        f.write(modified_content)
    
    logging.info("Successfully optimized segmentation parameters")
    return True

def optimize_confidence_threshold():
    """Update the config.py file to use a lower confidence threshold by default."""
    config_file = Path("src/config.py")
    if not config_file.exists():
        logging.error(f"Could not find {config_file}")
        return False
    
    # Read the file
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Modify the default confidence threshold
    modified_content = content.replace(
        "'confidence_threshold': 0.5,",
        "'confidence_threshold': 0.25,  # Lowered for better detection"
    )
    
    # Write the modified file
    with open(config_file, 'w') as f:
        f.write(modified_content)
    
    logging.info("Successfully lowered default confidence threshold")
    return True

def download_yolov8x_seg():
    """Download the YOLOv8x-seg model for better segmentation quality."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    if Path("models/yolov8x-seg.pt").exists():
        logging.info("YOLOv8x-seg model already exists")
        return True
    
    logging.info("Downloading YOLOv8x-seg model...")
    command = "python -c \"from ultralytics import YOLO; YOLO('yolov8x-seg.pt')\""
    success, output = run_command(command)
    
    if success:
        # Move the model to the models directory
        if Path("yolov8x-seg.pt").exists():
            os.rename("yolov8x-seg.pt", "models/yolov8x-seg.pt")
            logging.info("Successfully downloaded and moved YOLOv8x-seg model")
            return True
        else:
            logging.error("Model downloaded but file not found")
            return False
    else:
        logging.error(f"Failed to download model: {output}")
        return False

def create_optimized_scripts():
    """Create optimized scripts for running the segmentation pipeline."""
    # Create a script for the improved version with YOLOv8s-seg
    improved_script = """#!/bin/bash
# Optimized 360 VR Person Segmentation Pipeline with improved parameters

# Set input and output paths
INPUT_VIDEO="360.mp4"
OUTPUT_VIDEO="./output/output_improved.webm"

echo "Starting Optimized 360 VR Person Segmentation Pipeline..."

# Run the pipeline with optimized parameters
./run.sh --input-video "$INPUT_VIDEO" \\
         --output-video "$OUTPUT_VIDEO" \\
         --model-path models/yolov8s-seg.pt \\
         --confidence 0.25 \\
         --viewpoint-hfov 90.0 \\
         --viewpoint-vfov 60.0

echo "Pipeline completed. Output saved to $OUTPUT_VIDEO"
"""
    
    # Create a script for the advanced version with YOLOv8x-seg
    advanced_script = """#!/bin/bash
# Advanced 360 VR Person Segmentation Pipeline with YOLOv8x-seg model

# Set input and output paths
INPUT_VIDEO="360.mp4"
OUTPUT_VIDEO="./output/output_advanced.webm"

echo "Starting Advanced 360 VR Person Segmentation Pipeline..."

# Run the pipeline with YOLOv8x-seg and optimized parameters
./run.sh --input-video "$INPUT_VIDEO" \\
         --output-video "$OUTPUT_VIDEO" \\
         --model-path models/yolov8x-seg.pt \\
         --confidence 0.2 \\
         --viewpoint-hfov 90.0 \\
         --viewpoint-vfov 60.0

echo "Pipeline completed. Output saved to $OUTPUT_VIDEO"
"""
    
    # Write the scripts
    with open("run_improved.sh", 'w') as f:
        f.write(improved_script)
    
    with open("run_advanced.sh", 'w') as f:
        f.write(advanced_script)
    
    # Make them executable
    os.chmod("run_improved.sh", 0o755)
    os.chmod("run_advanced.sh", 0o755)
    
    logging.info("Successfully created optimized scripts")
    return True

def main():
    """Main function to optimize the VR person segmentation pipeline."""
    parser = argparse.ArgumentParser(description='Optimize 360 VR Person Segmentation Pipeline')
    parser.add_argument('--download-model', action='store_true', help='Download YOLOv8x-seg model')
    args = parser.parse_args()
    
    logging.info("Starting optimization of VR person segmentation pipeline")
    
    # Step 1: Optimize segmentation parameters
    if not optimize_segmentation_params():
        logging.error("Failed to optimize segmentation parameters")
        return 1
    
    # Step 2: Optimize confidence threshold
    if not optimize_confidence_threshold():
        logging.error("Failed to optimize confidence threshold")
        return 1
    
    # Step 3: Create optimized scripts
    if not create_optimized_scripts():
        logging.error("Failed to create optimized scripts")
        return 1
    
    # Optional: Download YOLOv8x-seg model
    if args.download_model:
        if not download_yolov8x_seg():
            logging.warning("Failed to download YOLOv8x-seg model, but optimization can still continue")
    
    logging.info("VR person segmentation pipeline optimization completed successfully")
    logging.info("Run './run_improved.sh' for better segmentation results")
    logging.info("Run './run_advanced.sh' for the best segmentation results (requires YOLOv8x-seg model)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
