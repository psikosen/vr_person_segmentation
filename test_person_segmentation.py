#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import logging
from pathlib import Path
from src.segmentation import load_yolo_model, segment_frame
from src.compositor import create_rgba_frame

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test person segmentation with YOLOv8")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="models/yolov8s-seg.pt", help="Path to YOLOv8 model")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--frame-limit", type=int, default=10, help="Maximum frames to process")
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input video not found: {input_path}")
        return 1
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    logging.info(f"Loading YOLOv8 model: {args.model}")
    model = load_yolo_model(Path(args.model))
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video: {input_path}")
        return 1
    
    frame_count = 0
    while cap.isOpened() and (args.frame_limit <= 0 or frame_count < args.frame_limit):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Segment persons
        logging.info(f"Processing frame {frame_count}")
        person_masks = segment_frame(
            model, 
            frame, 
            args.confidence, 
            iou_threshold=0.5,
            apply_morphology=True
        )
        
        # Log number of persons detected
        logging.info(f"Detected {len(person_masks)} persons")
        
        # Create RGBA frame
        rgba_frame = create_rgba_frame(frame, person_masks, alpha_edge_smooth=True)
        
        # Save RGBA frame as PNG
        output_path = output_dir / f"frame_{frame_count:04d}.png"
        cv2.imwrite(str(output_path), rgba_frame)
        logging.info(f"Saved: {output_path}")
        
        # Also save the original frame with mask overlay for comparison
        if person_masks:
            # Create a copy of the original frame
            overlay_frame = frame.copy()
            
            # Create a composite mask (union of all person masks)
            composite_mask = np.zeros_like(person_masks[0])
            for mask in person_masks:
                composite_mask = np.maximum(composite_mask, mask)
            
            # Apply a color overlay to the masked areas
            color_overlay = np.zeros_like(overlay_frame)
            color_overlay[:, :] = [0, 255, 0]  # Green color
            
            # Apply the overlay with alpha blending
            alpha = 0.5
            mask_3ch = np.stack([composite_mask] * 3, axis=2)
            overlay_frame = np.where(
                mask_3ch > 0,
                cv2.addWeighted(overlay_frame, 1-alpha, color_overlay, alpha, 0),
                overlay_frame
            )
            
            # Save the overlay image
            overlay_path = output_dir / f"overlay_{frame_count:04d}.png"
            cv2.imwrite(str(overlay_path), overlay_frame)
            logging.info(f"Saved overlay: {overlay_path}")
        
        frame_count += 1
    
    cap.release()
    logging.info(f"Processed {frame_count} frames")
    return 0

if __name__ == "__main__":
    exit(main())
