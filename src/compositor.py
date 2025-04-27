#!/usr/bin/env python3
"""
Compositor module for the 360 VR Person Segmentation Pipeline.

This module provides functions for compositing segmented person masks
onto a transparent background, creating RGBA frames.
"""

import numpy as np
import cv2
from typing import List


def create_rgba_frame(original_frame: np.ndarray, person_masks: List[np.ndarray], 
                     alpha_edge_smooth: bool = True) -> np.ndarray:
    """
    Create an RGBA frame with segmented persons on a transparent background.
    
    Parameters
    ----------
    original_frame : np.ndarray
        The original BGR frame (H x W x 3, uint8).
    person_masks : List[np.ndarray]
        List of binary masks (H x W, uint8) for person instances.
    alpha_edge_smooth : bool, optional
        Whether to apply slight smoothing to the alpha channel edges for better visual results.
        
    Returns
    -------
    np.ndarray
        RGBA frame (H x W x 4, uint8) with segmented persons on a transparent background.
        The alpha channel (index 3) is 255 for pixels belonging to person instances,
        and 0 for background pixels.
    """
    # Get frame dimensions
    height, width = original_frame.shape[:2]
    
    # Initialize RGBA output with transparent background
    rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)
    
    # If no person masks, return fully transparent frame
    if not person_masks:
        return rgba_frame
    
    # Combine all masks into a single composite mask (logical OR)
    if len(person_masks) == 1:
        composite_mask = person_masks[0]
    else:
        composite_mask = np.maximum.reduce(person_masks)
    
    # Optional: Smooth the mask edges for better visual appearance
    if alpha_edge_smooth:
        # Apply a slight Gaussian blur to create soft edges
        # This helps with the visual appearance at mask boundaries
        composite_mask_float = composite_mask.astype(np.float32)
        composite_mask_float = cv2.GaussianBlur(composite_mask_float, (3, 3), 0)
        
        # Normalize back to range 0-1
        composite_mask_float = np.clip(composite_mask_float, 0, 1)
        
        # Convert to alpha channel (range 0-255)
        alpha_channel = (composite_mask_float * 255).astype(np.uint8)
    else:
        # Use the binary mask directly for the alpha channel
        alpha_channel = composite_mask * 255
    
    # Set alpha channel
    rgba_frame[:, :, 3] = alpha_channel
    
    # Convert BGR to RGB and copy to output where mask is active
    # This avoids unnecessary copying of background pixels
    mask_indices = composite_mask.astype(bool)
    
    # Copy BGR channels to RGB channels where mask is active
    # OpenCV uses BGR, but we want RGB for standard image formats
    rgba_frame[mask_indices, 0] = original_frame[mask_indices, 2]  # R from B
    rgba_frame[mask_indices, 1] = original_frame[mask_indices, 1]  # G from G
    rgba_frame[mask_indices, 2] = original_frame[mask_indices, 0]  # B from R
    
    return rgba_frame


def create_rgba_frame_with_debug(original_frame: np.ndarray, 
                                person_masks: List[np.ndarray], 
                                debug_mode: bool = False) -> np.ndarray:
    """
    Extended version of create_rgba_frame with optional debug visualization.
    
    Parameters
    ----------
    original_frame : np.ndarray
        The original BGR frame (H x W x 3, uint8).
    person_masks : List[np.ndarray]
        List of binary masks (H x W, uint8) for person instances.
    debug_mode : bool, optional
        If True, add a colored border around detected persons.
        
    Returns
    -------
    np.ndarray
        RGBA frame (H x W x 4, uint8) with segmented persons on a transparent background.
    """
    # Create the basic RGBA frame
    rgba_frame = create_rgba_frame(original_frame, person_masks)
    
    # If debug mode is enabled and we have masks, add visualization
    if debug_mode and person_masks:
        # Get composite mask
        if len(person_masks) == 1:
            composite_mask = person_masks[0]
        else:
            composite_mask = np.maximum.reduce(person_masks)
        
        # Create a dilated version of the mask for the border
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(composite_mask, kernel, iterations=2)
        
        # Border is the difference between dilated and original mask
        border_mask = dilated_mask.astype(np.uint8) - composite_mask.astype(np.uint8)
        border_indices = border_mask.astype(bool)
        
        # Set red border in RGB channels
        rgba_frame[border_indices, 0] = 255  # Red
        rgba_frame[border_indices, 1] = 0    # Green
        rgba_frame[border_indices, 2] = 0    # Blue
        rgba_frame[border_indices, 3] = 255  # Fully opaque
    
    return rgba_frame


if __name__ == "__main__":
    # This allows for testing the compositor module independently
    import argparse
    import cv2
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Frame compositor")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mask', type=str, required=True, help='Path to binary mask image')
    parser.add_argument('--output', type=str, required=True, help='Path to output RGBA image (PNG)')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
    args = parser.parse_args()
    
    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to load image: {args.image}")
        exit(1)
    
    # Load the mask
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Failed to load mask: {args.mask}")
        exit(1)
    
    # Binarize the mask if not already binary
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    # Create RGBA frame
    if args.debug:
        rgba_frame = create_rgba_frame_with_debug(image, [binary_mask], True)
    else:
        rgba_frame = create_rgba_frame(image, [binary_mask])
    
    # Save the output image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), rgba_frame)
    print(f"Saved RGBA image to: {output_path}")
