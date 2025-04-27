#!/usr/bin/env python3
"""
Configuration handling for the 360 VR Person Segmentation Pipeline.

This module defines functions for loading and validating configuration
from command-line arguments and/or a configuration file.
"""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, Union, Any


def load_config(args: Optional[argparse.Namespace] = None, 
               config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Load and validate configuration from defaults, a config file, and/or command-line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace, optional
        Parsed command-line arguments.
    config_path : Path or str, optional
        Path to the configuration file (YAML).
        
    Returns
    -------
    dict
        A dictionary of validated configuration parameters.
        
    Raises
    ------
    FileNotFoundError
        If the specified config_path does not exist.
    yaml.YAMLError
        If the config file could not be parsed as valid YAML.
    ValueError
        If required configuration parameters are missing or invalid.
    """
    # Define default configuration
    config = {
        # Input/Output
        'input_video': None,
        'output_video': None,
        'temp_directory': Path('./temp'),
        
        # Model settings
        'model_path': Path('./models/yolov8s-seg.pt'),
        'confidence_threshold': 0.25,  # Lowered for better detection
        'device': None,  # None = auto-detect
        
        # Viewpoint parameters (for 360° to 2D conversion)
        'viewpoint': {
            'hfov': 90.0,  # Horizontal field of view (degrees)
            'vfov': 60.0,  # Vertical field of view (degrees)
            'yaw': 0.0,    # Yaw angle (degrees)
            'pitch': 0.0,  # Pitch angle (degrees)
            'roll': 0.0,   # Roll angle (degrees)
        },
        
        # Video encoding parameters
        'encoding': {
            'crf': 23,       # Constant Rate Factor (quality)
            'preset': 'medium',  # VP9 encoding preset
        },
        
        # External tools
        'ffmpeg_path': 'ffmpeg',
        'ffprobe_path': 'ffprobe',
        
        # Cleanup temporary files
        'cleanup_temp': True,
        
        # Progress display
        'show_progress': True,  # Show progress bar during processing
    }
    
    # Load configuration from file if specified
    if config_path:
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                # Update config with values from the file
                _update_config_recursive(config, file_config)
                logging.info(f"Loaded configuration from: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file: {e}")
    
    # Override with command-line arguments if provided
    if args:
        arg_dict = vars(args)
        for key, value in arg_dict.items():
            if value is not None:  # Only override if explicitly set
                # Handle special cases: convert strings to Path objects for file paths
                if key in ['input_video', 'output_video', 'model_path']:
                    if value:  # Skip empty strings
                        config[key] = Path(value)
                elif key == 'temp_directory':
                    if value:  # Skip empty strings
                        config[key] = Path(value)
                # Handle nested configurations (viewpoint, encoding)
                elif key.startswith('viewpoint_') and value is not None:
                    param = key.split('_', 1)[1]
                    if param in config['viewpoint']:
                        config['viewpoint'][param] = value
                elif key.startswith('encoding_') and value is not None:
                    param = key.split('_', 1)[1]
                    if param in config['encoding']:
                        config['encoding'][param] = value
                # Regular parameters
                elif key in config:
                    config[key] = value
    
    # Validate configuration
    _validate_config(config)
    
    return config


def _update_config_recursive(target: Dict, source: Dict) -> None:
    """
    Recursively update a nested dictionary with values from another dictionary.
    
    Parameters
    ----------
    target : dict
        The dictionary to update.
    source : dict
        The dictionary with new values.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            # Recursively update nested dictionaries
            _update_config_recursive(target[key], value)
        else:
            # Update the value
            target[key] = value


def _validate_config(config: Dict) -> None:
    """
    Validate the configuration dictionary, raising ValueErrors for invalid settings.
    
    Parameters
    ----------
    config : dict
        The configuration dictionary to validate.
        
    Raises
    ------
    ValueError
        If any configuration settings are invalid.
    """
    # Check required parameters
    if not config['input_video']:
        raise ValueError("Input video path is required")
    
    if not config['output_video']:
        raise ValueError("Output video path is required")
    
    # Validate input file existence
    if not Path(config['input_video']).is_file():
        raise ValueError(f"Input video file not found: {config['input_video']}")
    
    # Validate model file existence
    if not Path(config['model_path']).is_file():
        raise ValueError(f"Model file not found: {config['model_path']}")
    
    # Validate numerical ranges
    if not (0.0 <= config['confidence_threshold'] <= 1.0):
        raise ValueError(f"Confidence threshold must be between 0.0 and 1.0: {config['confidence_threshold']}")
    
    # Validate viewpoint parameters
    for param, value in config['viewpoint'].items():
        if param in ['hfov', 'vfov'] and not (0.0 < value <= 180.0):
            raise ValueError(f"Viewpoint {param} must be between 0 and 180 degrees: {value}")
        elif param in ['yaw', 'pitch', 'roll'] and not (-180.0 <= value <= 180.0):
            raise ValueError(f"Viewpoint {param} must be between -180 and 180 degrees: {value}")
    
    # Validate encoding parameters
    if not (0 <= config['encoding']['crf'] <= 63):  # VP9 CRF range
        raise ValueError(f"CRF value must be between 0 and 63: {config['encoding']['crf']}")
    
    # Ensure output directory exists or can be created
    output_dir = Path(config['output_video']).parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True)
            logging.info(f"Created output directory: {output_dir}")
        except Exception as e:
            raise ValueError(f"Cannot create output directory: {output_dir}. Error: {e}")
    
    # Ensure temp_directory is a Path object
    if isinstance(config['temp_directory'], str):
        config['temp_directory'] = Path(config['temp_directory'])
        
    # Ensure temp directory exists or can be created
    if not config['temp_directory'].exists():
        try:
            config['temp_directory'].mkdir(parents=True)
            logging.info(f"Created temporary directory: {config['temp_directory']}")
        except Exception as e:
            raise ValueError(f"Cannot create temporary directory: {config['temp_directory']}. Error: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the 360 VR Person Segmentation Pipeline.
    
    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="360 VR Person Segmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML)')
    parser.add_argument('--input-video', type=str, help='Path to input 360° video file')
    parser.add_argument('--output-video', type=str, help='Path to output WebM video file')
    parser.add_argument('--model-path', type=str, help='Path to YOLOv8-seg model file')
    parser.add_argument('--confidence', type=float, dest='confidence_threshold',
                        help='Confidence threshold for person detection (0.0-1.0)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                        help='Computation device (cpu, cuda, mps)')
    
    # Viewpoint arguments
    parser.add_argument('--viewpoint-hfov', type=float, help='Horizontal field of view (degrees)')
    parser.add_argument('--viewpoint-vfov', type=float, help='Vertical field of view (degrees)')
    parser.add_argument('--viewpoint-yaw', type=float, help='Yaw angle (degrees)')
    parser.add_argument('--viewpoint-pitch', type=float, help='Pitch angle (degrees)')
    parser.add_argument('--viewpoint-roll', type=float, help='Roll angle (degrees)')
    
    # Encoding arguments
    parser.add_argument('--encoding-crf', type=int, help='VP9 Constant Rate Factor (0-63)')
    parser.add_argument('--encoding-preset', type=str, choices=['fast', 'medium', 'slow'], 
                       help='VP9 encoding preset')
    
    # External tools
    parser.add_argument('--ffmpeg-path', type=str, help='Path to FFmpeg executable')
    parser.add_argument('--ffprobe-path', type=str, help='Path to FFprobe executable')
    
    # Misc arguments
    parser.add_argument('--temp-directory', type=str, help='Directory for temporary files')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup_temp',
                        help='Keep temporary files after processing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-progress', action='store_false', dest='show_progress',
                      help='Disable progress bar during processing')
    
    return parser.parse_args()


if __name__ == "__main__":
    # This allows for testing the config module independently
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        config = load_config(args, args.config)
        print("Configuration loaded successfully:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
