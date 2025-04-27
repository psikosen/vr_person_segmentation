#!/usr/bin/env python3
"""
Main entry point for the 360 VR Person Segmentation Pipeline.

This module handles command-line argument parsing, configuration loading,
and invokes the main pipeline.
"""

import logging
import sys
import time
from pathlib import Path
import os

# Adjust imports to work both as a module and as a script
if __name__ == "__main__":
    # Add the parent directory to the path for direct script execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
    from src.config import load_config, parse_args
    from src.pipeline import run_pipeline
else:
    # Use relative imports when imported as a module
    from .config import load_config, parse_args
    from .pipeline import run_pipeline


def main():
    """
    Main entry point for the application.
    
    Parses command-line arguments, loads configuration, and executes the pipeline.
    """
    # Record start time
    start_time = time.time()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Configure logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    
    # Set up file logging if output directory exists or can be created
    if args.output_video:
        try:
            log_dir = Path(args.output_video).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / "vr_segmentation.log"
            
            # Configure logging to file and console
            logging.basicConfig(
                level=logging.DEBUG if args.verbose else logging.INFO,
                format=log_format,
                datefmt=log_datefmt,
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
            logging.info(f"Logging to: {log_file}")
            
        except Exception as e:
            # Fall back to console-only logging
            logging.basicConfig(
                level=logging.DEBUG if args.verbose else logging.INFO,
                format=log_format,
                datefmt=log_datefmt
            )
            logging.warning(f"Could not set up file logging: {e}")
    else:
        # Console-only logging if no output path specified
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format=log_format,
            datefmt=log_datefmt
        )
    
    # Log the startup information
    logging.info("Starting 360 VR Person Segmentation Pipeline")
    
    try:
        # Load configuration
        logging.info("Loading configuration...")
        config = load_config(args, args.config)
        logging.info("Configuration loaded successfully")
        
        # Run the pipeline
        logging.info("Executing pipeline...")
        success = run_pipeline(config)
        
        # Log completion status
        elapsed_time = time.time() - start_time
        if success:
            logging.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
            sys.exit(0)
        else:
            logging.error(f"Pipeline failed after {elapsed_time:.2f} seconds")
            sys.exit(1)
            
    except Exception as e:
        # Log any unhandled exceptions
        logging.exception("An unhandled error occurred")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logging.error(f"Pipeline failed after {elapsed_time:.2f} seconds")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
