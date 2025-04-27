#!/usr/bin/env python3
"""
Video utilities for the 360 VR Person Segmentation Pipeline.

This module provides functions for extracting viewports from 360° videos
and retrieving video properties using FFmpeg.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any


def extract_viewport(input_360_path: Path, 
                    output_2d_path: Path, 
                    viewpoint_params: Dict[str, float], 
                    ffmpeg_path: str = 'ffmpeg') -> bool:
    """
    Extract a 2D viewport from a 360° equirectangular video using FFmpeg's v360 filter.
    
    Parameters
    ----------
    input_360_path : Path
        Path to the input 360° video file.
    output_2d_path : Path
        Path where the extracted 2D video will be saved.
    viewpoint_params : dict
        Dictionary containing viewport parameters:
        - hfov: Horizontal field of view (degrees)
        - vfov: Vertical field of view (degrees)
        - yaw: Yaw angle (degrees)
        - pitch: Pitch angle (degrees)
        - roll: Roll angle (degrees)
    ffmpeg_path : str, optional
        Path to the FFmpeg executable.
        
    Returns
    -------
    bool
        True if the extraction was successful, False otherwise.
        
    Raises
    ------
    FileNotFoundError
        If the input video file or FFmpeg executable does not exist.
    """
    # Validate input path
    if not input_360_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {input_360_path}")
    
    # Ensure output directory exists
    output_2d_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract viewport parameters
    hfov = viewpoint_params.get('hfov', 90.0)
    vfov = viewpoint_params.get('vfov', 60.0)
    yaw = viewpoint_params.get('yaw', 0.0)
    pitch = viewpoint_params.get('pitch', 0.0)
    roll = viewpoint_params.get('roll', 0.0)
    
    # Construct FFmpeg command
    # The v360 filter syntax: v360=input_format:output_format:parameters
    filter_complex = (
        f"v360=equirect:rectilinear:"        # equirectangular to rectilinear projection
        f"h_fov={hfov}:v_fov={vfov}:"        # horizontal and vertical field of view
        f"yaw={yaw}:pitch={pitch}:roll={roll}" # orientation angles
    )
    
    command = [
        ffmpeg_path,
        "-i", str(input_360_path),
        "-vf", filter_complex,
        "-c:v", "libx264",   # H.264 codec for intermediate file
        "-preset", "medium", # Encoding speed/quality balance
        "-crf", "18",        # High quality for intermediate file
        "-pix_fmt", "yuv420p", # Standard pixel format for compatibility
        "-an",               # No audio
        "-y",                # Overwrite output file if it exists
        str(output_2d_path)
    ]
    
    # Log the command
    logging.info(f"Extracting viewport with command: {' '.join(command)}")
    
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
            logging.info(f"Successfully extracted viewport to: {output_2d_path}")
            return True
        else:
            logging.error(f"FFmpeg error (code {completed_process.returncode}): {completed_process.stderr}")
            return False
            
    except FileNotFoundError:
        logging.error(f"FFmpeg executable not found: {ffmpeg_path}")
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg_path}")


def get_video_properties(video_path: Path, ffprobe_path: str = 'ffprobe') -> Dict[str, Any]:
    """
    Get video properties (dimensions, FPS, duration) using FFprobe.
    
    Parameters
    ----------
    video_path : Path
        Path to the video file.
    ffprobe_path : str, optional
        Path to the FFprobe executable.
        
    Returns
    -------
    dict
        Dictionary containing video properties:
        - width: Video width in pixels
        - height: Video height in pixels
        - fps: Frames per second (float)
        - duration_seconds: Duration in seconds (float)
        
    Raises
    ------
    FileNotFoundError
        If the video file or FFprobe executable does not exist.
    subprocess.CalledProcessError
        If FFprobe returns a non-zero exit code.
    json.JSONDecodeError
        If FFprobe's output cannot be parsed as JSON.
    ValueError
        If the required properties cannot be extracted from FFprobe's output.
    """
    # Validate input path
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Construct FFprobe command
    command = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json",
        str(video_path)
    ]
    
    try:
        # Execute FFprobe command
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True  # Raise exception on non-zero exit code
        )
        
        # Parse JSON output
        try:
            data = json.loads(completed_process.stdout)
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing FFprobe output: {e}")
            logging.debug(f"FFprobe output: {completed_process.stdout}")
            raise json.JSONDecodeError(f"Error parsing FFprobe output: {e}",
                                     completed_process.stdout, 0)
        
        # Extract video properties
        try:
            stream_data = data['streams'][0]
            
            # Extract width and height
            width = int(stream_data['width'])
            height = int(stream_data['height'])
            
            # Extract frame rate (usually returned as a fraction)
            fps_fraction = stream_data['r_frame_rate'].split('/')
            fps = float(int(fps_fraction[0]) / int(fps_fraction[1]))
            
            # Extract duration (may not be present in some video files)
            duration_seconds = float(stream_data.get('duration', 0))
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'duration_seconds': duration_seconds
            }
            
        except (KeyError, IndexError, ValueError) as e:
            logging.error(f"Error extracting video properties: {e}")
            logging.debug(f"FFprobe data: {data}")
            raise ValueError(f"Error extracting video properties: {e}")
            
    except FileNotFoundError:
        logging.error(f"FFprobe executable not found: {ffprobe_path}")
        raise FileNotFoundError(f"FFprobe executable not found: {ffprobe_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFprobe error: {e}")
        logging.debug(f"FFprobe stderr: {e.stderr}")
        raise


if __name__ == "__main__":
    # This allows for testing the video_utils module independently
    import argparse
    
    parser = argparse.ArgumentParser(description="Video utilities")
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path for extraction')
    parser.add_argument('--info', action='store_true', help='Get video info only')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='FFmpeg executable path')
    parser.add_argument('--ffprobe', type=str, default='ffprobe', help='FFprobe executable path')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    input_path = Path(args.input)
    
    if args.info or not args.output:
        # Get and display video properties
        try:
            properties = get_video_properties(input_path, args.ffprobe)
            print(f"Video properties for {input_path}:")
            for key, value in properties.items():
                print(f"  {key}: {value}")
        except Exception as e:
            logging.error(f"Error getting video properties: {e}")
    
    if args.output:
        # Extract viewport
        output_path = Path(args.output)
        viewpoint = {
            'hfov': 90.0,
            'vfov': 60.0,
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        }
        
        try:
            success = extract_viewport(input_path, output_path, viewpoint, args.ffmpeg)
            if success:
                print(f"Successfully extracted viewport to: {output_path}")
            else:
                print(f"Failed to extract viewport. Check logs for details.")
        except Exception as e:
            logging.error(f"Error extracting viewport: {e}")
