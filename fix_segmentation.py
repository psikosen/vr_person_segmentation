#!/usr/bin/env python3
"""
Script to fix the person segmentation in the 360 VR video.

This script creates a modified version of the pipeline that runs directly,
bypassing the issues with the generator function.
"""

import os
import sys
import cv2
import logging
import argparse
import numpy as np
from pathlib import Path
import subprocess
from tqdm import tqdm

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

def extract_viewport(input_video, output_path, hfov=90.0, vfov=60.0):
    """Extract the viewport from a 360° video."""
    command = f"ffmpeg -y -i {input_video} -vf v360=equirect:rectilinear:h_fov={hfov}:v_fov={vfov}:yaw=0.0:pitch=0.0:roll=0.0 -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an {output_path}"
    success, output = run_command(command)
    if success:
        logging.info(f"Successfully extracted viewport to: {output_path}")
        return True
    else:
        logging.error(f"Failed to extract viewport: {output}")
        return False

def get_video_properties(video_path):
    """Get video properties using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames
    }

def load_yolo_model(model_path, device=None):
    """Load a YOLOv8 segmentation model."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        logging.info(f"Successfully loaded YOLOv8 model: {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading YOLOv8 model: {e}")
        return None

def segment_frame(model, frame, confidence_threshold=0.25, iou_threshold=0.4, 
                 mask_threshold=0.2, morph_kernel_size=7, morph_iterations=2, overlap_threshold=0.25):
    """Perform segmentation on a single frame with configurable parameters."""
    
    # Perform inference on the frame
    results = model.predict(
        frame,
        classes=[0],  # 0 = person class
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    # Get the result for the first frame
    result = results[0]
    
    # If no masks are detected, return an empty list
    if result.masks is None:
        return []
    
    # Extract mask data
    masks_data = result.masks.data.cpu().numpy()
    
    # Get the original frame dimensions
    height, width = frame.shape[:2]
    
    # Create binary masks
    binary_masks = []
    for mask in masks_data:
        # Resize the mask if needed
        if mask.shape[:2] != (height, width):
            resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            resized_mask = mask
        
        # Threshold to create binary mask
        binary_mask = (resized_mask > mask_threshold).astype(np.uint8)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=morph_iterations)
        
        # Add to the list
        binary_masks.append(binary_mask)
    
    # Merge overlapping masks
    if len(binary_masks) > 1:
        merged_masks = []
        skip_indices = set()
        
        for i in range(len(binary_masks)):
            if i in skip_indices:
                continue
            
            current_mask = binary_masks[i].copy()
            
            # Check for overlap with other masks
            for j in range(i + 1, len(binary_masks)):
                if j in skip_indices:
                    continue
                
                # Calculate overlap
                overlap = np.logical_and(current_mask, binary_masks[j])
                overlap_percentage = np.sum(overlap) / min(np.sum(current_mask), np.sum(binary_masks[j]))
                
                # If significant overlap, merge the masks
                if overlap_percentage > overlap_threshold:
                    current_mask = np.logical_or(current_mask, binary_masks[j]).astype(np.uint8)
                    skip_indices.add(j)
            
            merged_masks.append(current_mask)
        
        binary_masks = merged_masks
    
    return binary_masks

def create_rgba_frame(frame, person_masks, alpha_edge_smooth=True):
    """
    Create an RGBA frame with transparent background and persons visible.
    
    Parameters
    ----------
    frame : numpy.ndarray
        The original RGB frame (H x W x 3, BGR format)
    person_masks : list of numpy.ndarray
        A list of binary masks for detected persons
    alpha_edge_smooth : bool, optional
        Whether to apply slight smoothing to alpha channel edges
        
    Returns
    -------
    numpy.ndarray
        RGBA frame with transparent background (H x W x 4, RGBA format)
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create a composite mask (union of all person masks)
    composite_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in person_masks:
        composite_mask = np.maximum(composite_mask, mask)
    
    # Apply smoothing to the mask edges if requested
    if alpha_edge_smooth and np.any(composite_mask):
        # Slightly blur the mask edges for smoother transitions
        kernel_size = 3
        composite_mask = cv2.GaussianBlur(composite_mask, (kernel_size, kernel_size), 0)
    
    # Create alpha channel from the composite mask
    alpha_channel = composite_mask * 255
    
    # Create RGBA frame by adding alpha channel
    rgba_frame = np.dstack((rgb_frame, alpha_channel))
    
    return rgba_frame

def encode_transparent_video(frames, output_path, fps, dimensions, ffmpeg_path='ffmpeg', crf=23, preset='good'):
    """
    Encode a sequence of RGBA frames as a WebM video with transparency.
    
    Parameters
    ----------
    frames : list of numpy.ndarray
        List of RGBA frames (H x W x 4)
    output_path : str or Path
        Path to save the output WebM video
    fps : float
        Frames per second for the output video
    dimensions : tuple
        (width, height) of the frames
    ffmpeg_path : str, optional
        Path to the FFmpeg executable
    crf : int, optional
        Constant Rate Factor (0-63) for VP9 encoder (lower = higher quality)
    preset : str, optional
        VP9 encoding preset ('good', 'best', 'realtime')
        
    Returns
    -------
    bool
        True if encoding succeeded, False otherwise
    """
    width, height = dimensions
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare FFmpeg command for encoding with transparency
    command = [
        ffmpeg_path,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-pixel_format', 'rgba',
        '-video_size', f'{width}x{height}',
        '-framerate', str(fps),
        '-i', '-',  # Read from stdin
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',
        '-crf', str(crf),
        '-b:v', '0',
        '-deadline', preset,
        '-cpu-used', '1',
        '-auto-alt-ref', '1',
        '-lag-in-frames', '25',
        '-an',  # No audio
        str(output_path)
    ]
    
    try:
        # Start FFmpeg process
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=10**8
        )
        
        # Write frames to FFmpeg's stdin
        for frame in tqdm(frames, desc="Encoding frames", unit="frame"):
            process.stdin.write(frame.tobytes())
        
        # Close the stdin pipe and wait for FFmpeg to finish
        process.stdin.close()
        process.wait()
        
        if process.returncode == 0:
            logging.info(f"Successfully encoded {len(frames)} frames to: {output_path}")
            return True
        else:
            error_output = process.stderr.read().decode('utf-8')
            logging.error(f"FFmpeg encoding failed with code {process.returncode}")
            logging.error(f"FFmpeg stderr: {error_output}")
            return False
            
    except Exception as e:
        logging.error(f"Error during encoding: {e}")
        return False

def main():
    """
    Main function to run the 360 VR Person Segmentation pipeline.
    """
    parser = argparse.ArgumentParser(description="Fix 360 VR Person Segmentation")
    parser.add_argument('--input-video', type=str, default='360.mp4', help='Path to input 360° video file')
    parser.add_argument('--output-video', type=str, default='./output/output_fixed.webm', help='Path to output WebM video file')
    parser.add_argument('--model', type=str, default='models/yolov8s-seg.pt', help='Path to YOLOv8 segmentation model')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.4, help='IoU threshold (0.0-1.0)')
    parser.add_argument('--frame-step', type=int, default=1, help='Process every Nth frame (for speed)')
    parser.add_argument('--hfov', type=float, default=90.0, help='Horizontal field of view (degrees)')
    parser.add_argument('--vfov', type=float, default=60.0, help='Vertical field of view (degrees)')
    parser.add_argument('--mask-threshold', type=float, default=0.2, help='Threshold for binary mask creation (0.0-1.0)')
    parser.add_argument('--morph-kernel', type=int, default=7, help='Size of morphological kernel')
    parser.add_argument('--morph-iterations', type=int, default=2, help='Number of dilation iterations')
    parser.add_argument('--overlap-threshold', type=float, default=0.25, help='Threshold for merging overlapping masks')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create temp directory
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Step 1: Extract viewport from 360° video
    extracted_video_path = temp_dir / "extracted_viewport.mp4"
    logging.info(f"Step 1: Extracting viewport from 360° video...")
    if not extract_viewport(args.input_video, extracted_video_path, args.hfov, args.vfov):
        logging.error("Failed to extract viewport")
        return 1
    
    # Step 2: Get video properties
    logging.info("Step 2: Getting video properties...")
    video_properties = get_video_properties(extracted_video_path)
    if not video_properties:
        logging.error("Failed to get video properties")
        return 1
    
    width = video_properties['width']
    height = video_properties['height']
    fps = video_properties['fps']
    total_frames = video_properties['total_frames']
    
    logging.info(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Step 3: Load YOLOv8 segmentation model
    logging.info("Step 3: Loading YOLOv8 segmentation model...")
    model = load_yolo_model(args.model)
    if not model:
        logging.error("Failed to load YOLOv8 model")
        return 1
    
    # Step 4: Process video frames
    logging.info("Step 4: Processing video frames...")
    
    # Open the video file
    cap = cv2.VideoCapture(str(extracted_video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {extracted_video_path}")
        return 1
    
    # Process frames and create RGBA frames
    rgba_frames = []
    frame_count = 0
    
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while True:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame if frame_step > 1
            if frame_count % args.frame_step != 0:
                frame_count += 1
                pbar.update(1)
                continue
            
            # Perform segmentation with all parameters
            person_masks = segment_frame(
                model, frame, 
                args.confidence, args.iou,
                args.mask_threshold, args.morph_kernel, 
                args.morph_iterations, args.overlap_threshold
            )
            
            # Create RGBA frame
            rgba_frame = create_rgba_frame(frame, person_masks, True)
            
            # Add to list
            rgba_frames.append(rgba_frame)
            
            # Update progress
            frame_count += 1
            pbar.update(1)
            pbar.set_postfix({'persons': len(person_masks)})
    
    # Release the video capture
    cap.release()
    
    # Check if we have any frames
    if not rgba_frames:
        logging.error("No frames were processed")
        return 1
    
    logging.info(f"Processed {len(rgba_frames)} frames from {extracted_video_path}")
    
    # Step 5: Encode transparent video
    logging.info("Step 5: Encoding transparent video...")
    
    encoding_success = encode_transparent_video(
        rgba_frames,
        args.output_video,
        fps / args.frame_step,  # Adjust fps if skipping frames
        (width, height)
    )
    
    if not encoding_success:
        logging.error("Failed to encode transparent video")
        return 1
    
    logging.info(f"Successfully encoded output video: {args.output_video}")
    
    # Step 6: Clean up
    logging.info("Step 6: Cleaning up...")
    try:
        # Remove extracted video file
        if extracted_video_path.exists():
            os.remove(extracted_video_path)
        
        # Try to remove temp directory if empty
        if temp_dir.exists() and not list(temp_dir.iterdir()):
            os.rmdir(temp_dir)
    except Exception as e:
        logging.warning(f"Error during cleanup: {e}")
    
    logging.info("Pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
