# Using YOLOv11 for VR Person Segmentation

This guide explains how to use the YOLOv11x-seg model for optimal person segmentation in VR videos.

## Benefits of YOLOv11

YOLOv11 represents the latest generation in the YOLO family, offering several advantages over previous versions:

- **Improved accuracy**: Better detection of people even in challenging poses or partial occlusion
- **Enhanced segmentation**: More precise outlines of person shapes
- **Better depth handling**: Improved performance in varying depth scenarios common in 360° videos
- **Lower latency**: Faster processing while maintaining higher quality

## Using the YOLOv11 Script

We've created a dedicated script optimized specifically for YOLOv11:

```bash
./run_yolo11.sh [input_video] [output_video]
```

Examples:

```bash
# Process the default video
./run_yolo11.sh

# Process a specific video
./run_yolo11.sh my_360_video.mp4 ./output/result.webm

# Process the main test video
./run_yolo11.sh 360.mp4 ./output/yolo11_result.webm
```

## Optimized Parameters

The YOLOv11 script uses specially tuned parameters:

- **Lower confidence threshold** (0.18): Captures more subtle details in person detection
- **Lower IoU threshold** (0.3): Better handling of overlapping detections
- **Lower mask threshold** (0.15): Creates more complete person silhouettes
- **Larger morphological kernel** (11): Creates smoother outlines
- **Two dilation iterations**: Fills gaps while preserving detail
- **Lower overlap threshold** (0.2): Better merging of fragmented detections

## Advanced Customization

For further tuning, you can modify parameters directly:

```bash
python fix_segmentation.py --input-video 360.mp4 \
                          --output-video ./output/custom.webm \
                          --model ./yolo11x-seg.pt \
                          --confidence 0.18 \
                          --iou 0.3 \
                          --mask-threshold 0.15 \
                          --morph-kernel 11 \
                          --morph-iterations 2 \
                          --overlap-threshold 0.2
```

## Comparing Results

When comparing YOLOv11 results to previous models:

1. **YOLOv11x-seg**: Highest quality, most complete body shapes, best for final output
2. **YOLOv8x-seg**: Good quality, might miss some subtle details
3. **YOLOv8s-seg**: Faster but less accurate, good for testing

For the best full-body shape detection, YOLOv11 with the optimized parameters will produce the most complete and natural-looking results.
