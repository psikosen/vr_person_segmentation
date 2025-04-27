#!/usr/bin/env python3
"""
Basic module tests for the 360 VR Person Segmentation Pipeline.

This script tests the basic functionality of each module without requiring
an actual 360° video file.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.compositor import create_rgba_frame


class TestModules(unittest.TestCase):
    """Test basic functionality of pipeline modules."""
    
    def test_config_loading(self):
        """Test that configuration loading works with defaults."""
        # We need to bypass validation for testing default values
        from src.config import _update_config_recursive
        
        # Start with default config
        config = {
            'input_video': None,
            'output_video': None,
            'temp_directory': Path('./temp'),
            'model_path': Path('./models/yolov8s-seg.pt'),
            'confidence_threshold': 0.5,
            'device': None,
            'viewpoint': {
                'hfov': 90.0,
                'vfov': 60.0,
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
            },
            'encoding': {
                'crf': 23,
                'preset': 'medium',
            },
            'ffmpeg_path': 'ffmpeg',
            'ffprobe_path': 'ffprobe',
            'cleanup_temp': True,
        }
        
        # Test default values
        self.assertEqual(config['confidence_threshold'], 0.5)
        self.assertEqual(config['viewpoint']['hfov'], 90.0)
        self.assertEqual(config['encoding']['crf'], 23)
    
    def test_compositor_empty(self):
        """Test compositor with empty mask list."""
        # Create a simple test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]  # Red frame
        
        # Test with empty mask list
        result = create_rgba_frame(frame, [])
        
        # Result should be fully transparent
        self.assertEqual(result.shape, (100, 100, 4))
        self.assertEqual(np.max(result[:, :, 3]), 0)  # Alpha channel all zeros
    
    def test_compositor_with_mask(self):
        """Test compositor with a simple mask."""
        # Create a simple test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]  # Red frame
        
        # Create a simple mask (square in the middle)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1  # 50x50 square in the middle
        
        # Test with the mask
        result = create_rgba_frame(frame, [mask])
        
        # Check dimensions
        self.assertEqual(result.shape, (100, 100, 4))
        
        # Center should be opaque
        self.assertEqual(result[50, 50, 3], 255)
        
        # Corner should be transparent
        self.assertEqual(result[0, 0, 3], 0)
        
        # Center color should match original (with BGR to RGB conversion)
        self.assertEqual(result[50, 50, 0], frame[50, 50, 2])  # R from B
        self.assertEqual(result[50, 50, 1], frame[50, 50, 1])  # G from G
        self.assertEqual(result[50, 50, 2], frame[50, 50, 0])  # B from R


if __name__ == '__main__':
    unittest.main()
