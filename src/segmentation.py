#!/usr/bin/env python3
"""
Segmentation module for the 360 VR Person Segmentation Pipeline.

This module provides functions for loading Ultralytics YOLOv8 segmentation models
and performing instance segmentation on video frames.
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Union
# Note: Importing the YOLO class can be done conditionally to avoid errors if ultralytics is not installed


def load_yolo_model(model_path: Path, device: Optional[str] = None):
    """
    Load a YOLOv8 segmentation model from a .pt file.
    
    Parameters
    ----------
    model_path : Path
        Path to the YOLOv8 model file (.pt).
    device : str, optional
        Device to use for inference ('cpu', 'cuda', 'mps', etc.).
        If None, the device will be auto-detected.
        
    Returns
    -------
    YOLO
        An initialized Ultralytics YOLO model object.
        
    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    ImportError
        If the ultralytics package is not installed.
    RuntimeError
        If there's an error loading the model.
    """
    # Validate model path
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Import here to allow the module to be imported even if ultralytics is not installed
        from ultralytics import YOLO
    except ImportError:
        logging.error("Failed to import ultralytics. Please install with: pip install ultralytics")
        raise ImportError("Required package 'ultralytics' not installed")
    
    try:
        # Load the model
        model = YOLO(str(model_path))
        
        # Log intended device (note: actual device selection happens during inference)
        if device:
            logging.info(f"Requested device for inference: {device}")
        else:
            logging.info("Device will be auto-selected by Ultralytics")
        
        logging.info(f"Successfully loaded YOLOv8 model: {model_path}")
        return model
        
    except Exception as e:
        logging.error(f"Error loading YOLOv8 model: {e}")
        raise RuntimeError(f"Error loading YOLOv8 model: {e}")


def segment_frame(model, frame: np.ndarray, confidence_threshold: float, 
                 iou_threshold: float = 0.4, 
                 apply_morphology: bool = True,
                 mask_threshold: float = 0.2,
                 morph_kernel_size: int = 7,
                 morph_iterations: int = 2,
                 overlap_threshold: float = 0.25) -> List[np.ndarray]:
    """
    Perform instance segmentation on a single video frame, filtering for 'person' class.
    
    Parameters
    ----------
    model : YOLO
        An initialized Ultralytics YOLO model object.
    frame : np.ndarray
        The input video frame (H x W x 3, BGR format).
    confidence_threshold : float
        The minimum detection confidence threshold (0.0-1.0).
    iou_threshold : float, optional
        IoU threshold for NMS (Non-Maximum Suppression) to reduce overlapping detections.
    apply_morphology : bool, optional
        Whether to apply morphological operations to improve mask completeness.
        
    Returns
    -------
    List[np.ndarray]
        A list of binary masks (H x W, uint8) for detected persons.
        May be empty if no persons are detected above the confidence threshold.
    """
    # Perform inference on the frame
    # The person class is index 0 in the COCO dataset
    results = model.predict(
        frame, 
        classes=[0],  # 0 = person class
        conf=confidence_threshold,
        iou=iou_threshold,  # IoU threshold for NMS
        verbose=False
    )
    
    # Get the result for the first (and only) frame
    result = results[0]
    
    # If no masks are detected, return an empty list
    if result.masks is None:
        return []
    
    # Extract mask data (should already be filtered for the person class)
    # Masks are on the same device as the model, so we need to move them to CPU
    masks_data = result.masks.data.cpu().numpy()
    
    # Get the original frame dimensions
    height, width = frame.shape[:2]
    
    # Create binary masks and resize if needed
    binary_masks = []
    for i in range(len(masks_data)):
        # Get the mask
        mask = masks_data[i]
        
        # Check if resize is needed
        if mask.shape[:2] != (height, width):
            # Resize the mask to match the original frame dimensions
            resized_mask = cv2.resize(
                mask, 
                (width, height), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            resized_mask = mask
            
        # Threshold to create binary mask
        # YOLOv8 masks are typically float arrays with values in [0, 1]
        binary_mask = (resized_mask > mask_threshold).astype(np.uint8)  # Customizable threshold to capture more of the person
        
        # Apply morphological operations to improve the mask if requested
        if apply_morphology:
            # Create a kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
            
            # Close small holes in the mask (dilate followed by erode)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # Additional dilation to ensure complete body coverage
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=morph_iterations)
        
        # Add to the list
        binary_masks.append(binary_mask)
    
    # If we have multiple masks, we might want to merge overlapping ones
    if len(binary_masks) > 1:
        # First, find masks that have significant overlap
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
                if overlap_percentage > overlap_threshold:  # Customizable overlap threshold
                    current_mask = np.logical_or(current_mask, binary_masks[j]).astype(np.uint8)
                    skip_indices.add(j)
                    logging.debug(f"Merged overlapping masks with {overlap_percentage:.2f} overlap")
            
            # Log mask size for debugging
            pixel_count = np.sum(current_mask)
            if pixel_count > 0:
                logging.debug(f"Mask size: {pixel_count} pixels, {(pixel_count / (height * width)) * 100:.2f}% of frame")
            
            merged_masks.append(current_mask)
        
        binary_masks = merged_masks
    
    # Count non-empty masks
    valid_masks = [m for m in binary_masks if np.any(m)]
    if valid_masks:
        logging.debug(f"Detected {len(valid_masks)} person instances")
    else:
        logging.debug(f"No valid person masks in this frame")
    
    return binary_masks


if __name__ == "__main__":
    # This allows for testing the segmentation module independently
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8 segmentation")
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output image (with segmentation overlay)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, help='Device for inference (cpu, cuda, mps)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load the model
    model = load_yolo_model(Path(args.model), args.device)
    
    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        logging.error(f"Failed to load image: {args.image}")
        exit(1)
    
    # Perform segmentation
    binary_masks = segment_frame(model, image, args.confidence)
    
    if not binary_masks:
        logging.info("No person instances detected")
    else:
        logging.info(f"Detected {len(binary_masks)} person instances")
        
        # If output path is provided, save a visualization
        if args.output:
            # Create a copy of the image for visualization
            output_image = image.copy()
            
            # Create a composite mask (union of all person masks)
            composite_mask = np.zeros_like(binary_masks[0])
            for mask in binary_masks:
                composite_mask = np.maximum(composite_mask, mask)
            
            # Apply a color overlay to the masked areas
            color_overlay = np.zeros_like(output_image)
            color_overlay[:, :] = [0, 255, 0]  # Green color
            
            # Apply the overlay with alpha blending
            alpha = 0.5
            mask_3ch = np.stack([composite_mask] * 3, axis=2)
            output_image = np.where(
                mask_3ch > 0,
                cv2.addWeighted(output_image, 1-alpha, color_overlay, alpha, 0),
                output_image
            )
            
            # Save the output image
            cv2.imwrite(args.output, output_image)
            logging.info(f"Saved visualization to: {args.output}")
