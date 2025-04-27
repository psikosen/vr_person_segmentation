#!/usr/bin/env python3
"""
Encoder module for the 360 VR Person Segmentation Pipeline.

This module provides functions for encoding sequences of RGBA frames
into transparent WebM video files using FFmpeg.
"""

import logging
import subprocess
from pathlib import Path
from typing import Iterable, Tuple, Optional
import numpy as np
from tqdm import tqdm


def encode_transparent_video(frame_iterable_rgba: Iterable[np.ndarray],
                            output_path: Path,
                            fps: float,
                            dimensions: Tuple[int, int],
                            ffmpeg_path: str = 'ffmpeg',
                            crf: int = 23,
                            preset: str = 'medium',
                            show_progress: bool = True,
                            total_frames: Optional[int] = None) -> bool:
    """
    Encode a sequence of RGBA frames into a WebM video with alpha channel using FFmpeg.
    
    Parameters
    ----------
    frame_iterable_rgba : Iterable[np.ndarray]
        An iterable (e.g., generator) yielding RGBA frames (H x W x 4, uint8).
    output_path : Path
        Path where the output WebM video will be saved.
    fps : float
        Frames per second of the output video.
    dimensions : Tuple[int, int]
        Frame dimensions as (width, height).
    ffmpeg_path : str, optional
        Path to the FFmpeg executable.
    crf : int, optional
        VP9 Constant Rate Factor (0-63). Lower values = higher quality.
    preset : str, optional
        VP9 encoding preset ('fast', 'medium', 'slow').
    show_progress : bool, optional
        Whether to show a progress bar during encoding. Default is True.
    total_frames : int, optional
        Total number of frames to encode. Required if show_progress is True.
        
    Returns
    -------
    bool
        True if encoding was successful, False otherwise.
        
    Raises
    ------
    FileNotFoundError
        If the FFmpeg executable is not found.
    ValueError
        If show_progress is True but total_frames is None.
    """
    # Check if we have the required information for the progress bar
    if show_progress and total_frames is None:
        raise ValueError("If show_progress is True, total_frames must be provided")
    
    # Extract dimensions
    width, height = dimensions
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Construct FFmpeg command for VP9 with alpha
    command = [
        ffmpeg_path,
        '-f', 'rawvideo',
        '-pixel_format', 'rgba',
        '-video_size', f'{width}x{height}',
        '-framerate', str(fps),
        '-i', '-',  # Input from stdin
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',  # Pixel format with alpha
        '-crf', str(crf),
        '-b:v', '0',  # Use CRF only
        '-deadline', 'good',  # VP9 deadline parameter (best, good, realtime)
        '-cpu-used', '2' if preset == 'fast' else '1' if preset == 'medium' else '0',  # CPU usage/quality trade-off
        '-auto-alt-ref', '1',
        '-lag-in-frames', '25',
        '-an',  # No audio
        '-y',  # Overwrite output file if it exists
        str(output_path)
    ]
    
    # Log the command
    logging.info(f"Encoding with command: {' '.join(command)}")
    
    try:
        # Start FFmpeg process with pipe for stdin
        ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE  # Capture stderr for debugging
        )
        
        # Frame counter for progress reporting
        frame_count = 0
        pipe_broken = False
        
        # Create a progress bar if requested
        pbar = None
        if show_progress and total_frames is not None:
            pbar = tqdm(total=total_frames, desc="Encoding frames", unit="frame")
        
        try:
            # Write frames to FFmpeg stdin
            for frame in frame_iterable_rgba:
                if frame.shape[:2] != (height, width) or frame.shape[2] != 4:
                    logging.error(f"Frame dimensions mismatch: expected {height}x{width}x4, got {frame.shape}")
                    continue
                
                try:
                    # Ensure the frame has valid data (at least one non-transparent pixel)
                    if np.any(frame[:, :, 3] > 0):
                        ffmpeg_process.stdin.write(frame.tobytes())
                    else:
                        # For completely transparent frames, still write them but log this occurrence
                        logging.debug("Writing completely transparent frame")
                        ffmpeg_process.stdin.write(frame.tobytes())
                    
                    frame_count += 1
                    
                    # Update progress bar
                    if pbar is not None:
                        pbar.update(1)
                    
                    # Log progress every 100 frames if progress bar is disabled
                    elif frame_count % 100 == 0:
                        logging.info(f"Encoded {frame_count} frames...")
                
                except BrokenPipeError:
                    logging.error("Broken pipe writing to FFmpeg. FFmpeg may have crashed.")
                    pipe_broken = True
                    break
                
                except Exception as e:
                    logging.error(f"Error writing frame to FFmpeg: {e}")
                    logging.exception("Stack trace:")
                    pipe_broken = True
                    break
                
        finally:
            # Close the progress bar
            if pbar is not None:
                pbar.close()
                
            # Close stdin pipe if not already broken
            if not pipe_broken and ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
            
            # Wait for FFmpeg to finish and get return code
            return_code = ffmpeg_process.wait()
            
            # Get stderr output for debugging
            stderr_output = ffmpeg_process.stderr.read().decode('utf-8', errors='replace') if ffmpeg_process.stderr else None
            
            # Check if encoding was successful
            if return_code == 0:
                logging.info(f"Successfully encoded {frame_count} frames to: {output_path}")
                return True
            else:
                logging.error(f"FFmpeg encoding failed with code {return_code}")
                if stderr_output:
                    logging.error(f"FFmpeg stderr: {stderr_output}")
                return False
    
    except FileNotFoundError:
        logging.error(f"FFmpeg executable not found: {ffmpeg_path}")
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg_path}")
    
    except Exception as e:
        logging.error(f"Unexpected error during encoding: {e}")
        return False


def encode_frames_from_files(input_pattern: str,
                           output_path: Path,
                           fps: float,
                           ffmpeg_path: str = 'ffmpeg',
                           crf: int = 23,
                           preset: str = 'medium') -> bool:
    """
    Encode a sequence of PNG files with alpha channel to a WebM video using FFmpeg.
    
    This is an alternative to frame-by-frame encoding that directly uses FFmpeg's
    file input capabilities. Useful for debugging or when frames are already saved to disk.
    
    Parameters
    ----------
    input_pattern : str
        Glob pattern for input PNG files with alpha channel (e.g., "frames/frame_%04d.png").
    output_path : Path
        Path where the output WebM video will be saved.
    fps : float
        Frames per second of the output video.
    ffmpeg_path : str, optional
        Path to the FFmpeg executable.
    crf : int, optional
        VP9 Constant Rate Factor (0-63). Lower values = higher quality.
    preset : str, optional
        VP9 encoding preset ('fast', 'medium', 'slow').
        
    Returns
    -------
    bool
        True if encoding was successful, False otherwise.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Construct FFmpeg command for VP9 with alpha
    command = [
        ffmpeg_path,
        '-framerate', str(fps),
        '-i', input_pattern,  # Input pattern
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',  # Pixel format with alpha
        '-crf', str(crf),
        '-b:v', '0',  # Use CRF only
        '-deadline', 'good',  # VP9 deadline parameter (best, good, realtime)
        '-cpu-used', '2' if preset == 'fast' else '1' if preset == 'medium' else '0',  # CPU usage/quality trade-off
        '-auto-alt-ref', '1',
        '-lag-in-frames', '25',
        '-an',  # No audio
        '-y',  # Overwrite output file if it exists
        str(output_path)
    ]
    
    # Log the command
    logging.info(f"Encoding with command: {' '.join(command)}")
    
    try:
        # Execute FFmpeg command
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit code
        )
        
        # Check if the command was successful
        if completed_process.returncode == 0:
            logging.info(f"Successfully encoded video to: {output_path}")
            return True
        else:
            logging.error(f"FFmpeg error (code {completed_process.returncode}): {completed_process.stderr}")
            return False
            
    except FileNotFoundError:
        logging.error(f"FFmpeg executable not found: {ffmpeg_path}")
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg_path}")


if __name__ == "__main__":
    # This allows for testing the encoder module independently
    import argparse
    import cv2
    import glob
    import os
    
    parser = argparse.ArgumentParser(description="Transparent video encoder")
    parser.add_argument('--input', type=str, help='Input pattern for PNG frames with alpha (e.g., "frames/frame_*.png")')
    parser.add_argument('--output', type=str, required=True, help='Output WebM video path')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second')
    parser.add_argument('--crf', type=int, default=23, help='VP9 CRF value (0-63)')
    parser.add_argument('--preset', type=str, choices=['fast', 'medium', 'slow'], default='medium', help='Encoding preset')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='FFmpeg executable path')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Validate that we can find FFmpeg
    try:
        subprocess.run([args.ffmpeg, '-version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logging.error(f"FFmpeg not found or not working at: {args.ffmpeg}")
        exit(1)
    
    if args.input:
        # Method 1: Use FFmpeg to encode directly from files
        # Replace glob pattern with FFmpeg's number pattern if needed
        input_files = sorted(glob.glob(args.input))
        if not input_files:
            logging.error(f"No input files found matching pattern: {args.input}")
            exit(1)
        
        # Get dimensions from first frame
        first_frame = cv2.imread(input_files[0], cv2.IMREAD_UNCHANGED)
        if first_frame is None or first_frame.shape[2] < 4:
            logging.error(f"Failed to load first frame with alpha channel: {input_files[0]}")
            exit(1)
        
        height, width = first_frame.shape[:2]
        logging.info(f"Input frames dimensions: {width}x{height}")
        
        # Create FFmpeg-style input pattern
        dirname = os.path.dirname(input_files[0])
        basename = os.path.basename(input_files[0])
        name_parts = basename.split('.')
        ext = name_parts[-1]
        name = '.'.join(name_parts[:-1])
        
        # Try to infer numbering pattern, or ask user
        import re
        match = re.search(r'(\d+)$', name)
        if match:
            num_digits = len(match.group(1))
            template = f"{name[:match.start()]}"
            template += f"%0{num_digits}d"
            if ext:
                template += f".{ext}"
            ffmpeg_pattern = os.path.join(dirname, template)
            success = encode_frames_from_files(
                ffmpeg_pattern, 
                Path(args.output), 
                args.fps, 
                args.ffmpeg, 
                args.crf, 
                args.preset
            )
        else:
            logging.error("Couldn't determine frame numbering pattern. Please specify manually.")
            success = False
    
    else:
        logging.error("Input pattern is required")
        exit(1)
    
    if success:
        print(f"Successfully encoded video to: {args.output}")
    else:
        print("Encoding failed. Check logs for details.")
        exit(1)
