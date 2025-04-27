#!/usr/bin/env python3
"""
Main pipeline module for the 360 VR Person Segmentation Pipeline.

This module orchestrates the entire process:
1. Extracting a 2D viewport from a 360° video
2. Segmenting person objects from each frame
3. Compositing persons onto a transparent background
4. Encoding the result as a WebM video with alpha channel
"""

import cv2
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Generator, Any, Optional, Iterator
from tqdm import tqdm

# Adjust imports to work both as a module and as a script
if __name__ == "__main__":
    # Add the parent directory to the path for direct script execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.video_utils import extract_viewport, get_video_properties
    from src.segmentation import load_yolo_model, segment_frame
    from src.compositor import create_rgba_frame
    from src.encoder import encode_transparent_video
else:
    # Use relative imports when imported as a module
    from .video_utils import extract_viewport, get_video_properties
    from .segmentation import load_yolo_model, segment_frame
    from .compositor import create_rgba_frame
    from .encoder import encode_transparent_video


def process_frames(
    video_path: Path,
    model: Any,
    confidence_threshold: float,
    iou_threshold: float = 0.5,
    apply_morphology: bool = True,
    progress_callback: Optional[callable] = None,
    show_progress: bool = True
) -> Generator[tuple[int, cv2.Mat, list], None, None]:
    """
    Generator that processes frames from a video file for segmentation.
    
    Parameters
    ----------
    video_path : Path
        Path to the video file.
    model : YOLO
        Initialized YOLOv8 model for segmentation.
    confidence_threshold : float
        Confidence threshold for person detection.
    iou_threshold : float
        IoU threshold for Non-Maximum Suppression.
    apply_morphology : bool
        Whether to apply morphological operations to improve mask completeness.
    progress_callback : callable, optional
        Function to call with progress updates (frame_number, total_frames).
    show_progress : bool, optional
        Whether to show a progress bar. Default is True.
        
    Yields
    ------
    tuple
        (frame_number, original_frame, person_masks)
    """
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Processing {total_frames} frames from {video_path}")
    
    try:
        # Create a progress bar if requested
        pbar = None
        if show_progress:
            pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
        
        frame_number = 0
        while True:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform segmentation with enhanced parameters
            person_masks = segment_frame(
                model, 
                frame, 
                confidence_threshold,
                iou_threshold=iou_threshold,
                apply_morphology=apply_morphology,
                mask_threshold=mask_threshold,
                morph_kernel_size=morph_kernel_size,
                morph_iterations=morph_iterations,
                overlap_threshold=overlap_threshold
            )
            
            # Update progress
            if progress_callback:
                progress_callback(frame_number, total_frames)
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                # Optionally add more info to the progress bar
                pbar.set_postfix({
                    'persons': len(person_masks),
                    'frame': frame_number
                })
            
            # Yield the frame and masks
            yield frame_number, frame, person_masks
            
            frame_number += 1
            
    finally:
        # Close the progress bar
        if pbar is not None:
            pbar.close()
        
        # Release the video capture
        cap.release()


def run_pipeline(config: Dict[str, Any]) -> bool:
    """
    Execute the complete video processing pipeline.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (as loaded by config.load_config).
        
    Returns
    -------
    bool
        True if the pipeline completed successfully, False otherwise.
    """
    start_time = time.time()
    
    # Extract key configuration values
    input_video_path = Path(config['input_video'])
    output_video_path = Path(config['output_video'])
    model_path = Path(config['model_path'])
    confidence_threshold = config['confidence_threshold']
    ffmpeg_path = config['ffmpeg_path']
    ffprobe_path = config['ffprobe_path']
    device = config['device']
    viewpoint_params = config['viewpoint']
    encoding_params = config['encoding']
    temp_directory = Path(config['temp_directory'])
    cleanup_temp = config['cleanup_temp']
    
    # Get show_progress from config or default to True
    show_progress = config.get('show_progress', True)
    
    # Get optional advanced segmentation parameters with defaults
    iou_threshold = config.get('iou_threshold', 0.5)
    apply_morphology = config.get('apply_morphology', True)
    alpha_edge_smooth = config.get('alpha_edge_smooth', True)
    
    # Get advanced mask parameters with defaults
    mask_threshold = config.get('mask_threshold', 0.3)
    morph_kernel_size = config.get('morph_kernel_size', 5)
    morph_iterations = config.get('morph_iterations', 1)
    overlap_threshold = config.get('overlap_threshold', 0.3)
    
    # Create temporary directory if it doesn't exist
    temp_directory.mkdir(parents=True, exist_ok=True)
    
    # Define paths for intermediate files
    extracted_video_path = temp_directory / "extracted_viewport.mp4"
    
    logging.info(f"Starting 360 VR Person Segmentation Pipeline")
    logging.info(f"Input video: {input_video_path}")
    logging.info(f"Output video: {output_video_path}")
    
    try:
        # Step 1: Extract viewport from 360° video
        logging.info("Step 1: Extracting viewport from 360° video...")
        
        extraction_success = extract_viewport(
            input_video_path,
            extracted_video_path,
            viewpoint_params,
            ffmpeg_path
        )
        
        if not extraction_success:
            logging.error("Viewport extraction failed")
            return False
        
        logging.info(f"Viewport extracted to: {extracted_video_path}")
        
        # Step 2: Get video properties of the extracted video
        logging.info("Step 2: Getting video properties...")
        
        video_properties = get_video_properties(extracted_video_path, ffprobe_path)
        width = video_properties['width']
        height = video_properties['height']
        fps = video_properties['fps']
        
        logging.info(f"Video properties: {width}x{height} @ {fps} fps")
        
        # Step 3: Load YOLOv8 segmentation model
        logging.info("Step 3: Loading YOLOv8 segmentation model...")
        
        model = load_yolo_model(model_path, device)
        logging.info("Model loaded successfully")
        
        # Step 4: Process video frames and encode
        logging.info("Step 4: Processing frames and encoding output...")
        
        # Create frame generator
        def frame_generator() -> Iterator[cv2.Mat]:
            """
            Generator that yields RGBA frames for encoding.
            """
            # We can use a progress callback independent of the progress bar
            # This allows for both log messages and the progress bar
            def progress_callback(frame_number, total_frames):
                if frame_number % 10 == 0 and not show_progress:  # Only log if progress bar is disabled
                    percent = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                    logging.info(f"Processing frame {frame_number}/{total_frames} ({percent:.1f}%)")
            
            for frame_number, frame, person_masks in process_frames(
                extracted_video_path,
                model, 
                confidence_threshold,
                iou_threshold=iou_threshold,
                apply_morphology=apply_morphology, 
                progress_callback=progress_callback, 
                show_progress=show_progress
            ):
                # Create RGBA frame with enhanced alpha channel
                rgba_frame = create_rgba_frame(frame, person_masks, alpha_edge_smooth=alpha_edge_smooth)
                
                # Yield the RGBA frame
                yield rgba_frame
        
        # Get total frames for progress tracking
        total_frames = int(video_properties['duration_seconds'] * fps)
        
        # Encode the processed frames
        encoding_success = encode_transparent_video(
            frame_generator(),
            output_video_path,
            fps,
            (width, height),
            ffmpeg_path,
            encoding_params['crf'],
            encoding_params['preset'],
            show_progress=show_progress,
            total_frames=total_frames
        )
        
        if not encoding_success:
            logging.error("Video encoding failed")
            return False
        
        logging.info(f"Successfully encoded output video: {output_video_path}")
        
        # Step 5: Cleanup
        if cleanup_temp:
            logging.info("Step 5: Cleaning up temporary files...")
            
            try:
                # Remove extracted video file
                if extracted_video_path.exists():
                    os.remove(extracted_video_path)
                
                # Try to remove temp directory if empty
                if temp_directory.exists() and not list(temp_directory.iterdir()):
                    os.rmdir(temp_directory)
            
            except Exception as e:
                logging.warning(f"Error during cleanup: {e}")
        
        # Calculate and log elapsed time
        elapsed_time = time.time() - start_time
        logging.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # This allows for testing the pipeline module independently
    import argparse
    # Import modules when run as script
    from src.config import load_config, parse_args
    
    # Parse command-line arguments
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        # Load configuration
        config = load_config(args, args.config)
        
        # Run the pipeline
        success = run_pipeline(config)
        
        if success:
            print("Pipeline completed successfully")
            exit(0)
        else:
            print("Pipeline failed. Check logs for details.")
            exit(1)
            
    except Exception as e:
        logging.exception("Unhandled error in pipeline")
        print(f"Error: {e}")
        exit(1)
