"""
Tests for the configuration module.
"""

import pytest
import tempfile
from pathlib import Path
import yaml
from src.config import load_config

class TestConfig:
    """Test suite for configuration module."""
    
    def test_load_config_defaults(self):
        """Test loading default configuration."""
        config = load_config()
        
        # Verify default values
        assert config['confidence_threshold'] == 0.5
        assert config['viewpoint']['hfov'] == 90.0
        assert config['encoding']['crf'] == 23
        
    def test_load_config_file(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as temp_file:
            # Write a test configuration
            config_data = {
                'confidence_threshold': 0.7,
                'viewpoint': {
                    'hfov': 120.0,
                    'yaw': 45.0
                },
                'encoding': {
                    'crf': 18
                }
            }
            yaml.dump(config_data, temp_file)
            temp_file.flush()
            
            # Load the configuration
            config = load_config(config_path=temp_file.name)
            
            # Verify values from the file
            assert config['confidence_threshold'] == 0.7
            assert config['viewpoint']['hfov'] == 120.0
            assert config['viewpoint']['yaw'] == 45.0
            assert config['encoding']['crf'] == 18
            
            # Verify other defaults remain
            assert config['viewpoint']['vfov'] == 60.0
            assert config['encoding']['preset'] == 'medium'
    
    def test_config_validation(self):
        """Test validation of configuration values."""
        from argparse import Namespace
        
        # Create test arguments with invalid confidence threshold
        args = Namespace(
            confidence_threshold=1.5,  # Invalid: should be <= 1.0
            input_video='test.mp4',
            output_video='output.webm',
            model_path='model.pt'
        )
        
        # The paths don't exist, so validation should raise a ValueError
        with pytest.raises(ValueError):
            load_config(args)
