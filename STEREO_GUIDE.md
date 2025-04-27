# Processing Stereoscopic VR Videos

This guide explains how to process stereoscopic (side-by-side) VR videos with the universal script.

## What are Stereoscopic VR Videos?

Stereoscopic VR videos contain two side-by-side views of the same scene - one for each eye. They're common in VR headsets where each eye sees a slightly different perspective to create the 3D effect.

## Using the Universal Script

The universal script supports both regular 360° videos and stereoscopic VR videos through various modes:

```bash
./run_universal.sh [options] [input_video] [output_video]
```

### Options:

- `--stereo`: Process as stereoscopic VR (side-by-side) video
- `--mono`: Process as regular 360° video (default)
- `--left-eye`: Process only the left eye view
- `--right-eye`: Process only the right eye view
- `--help`: Show help message

### Examples:

```bash
# Process a stereoscopic video (both eyes)
./run_universal.sh --stereo my_stereo_video.mp4 ./output/stereo_result.webm

# Process only the left eye view
./run_universal.sh --left-eye my_stereo_video.mp4 ./output/left_eye.webm

# Process only the right eye view
./run_universal.sh --right-eye my_stereo_video.mp4 ./output/right_eye.webm

# Process as regular 360° video (default)
./run_universal.sh my_video.mp4 ./output/result.webm
```

## How Stereoscopic Processing Works

When processing stereoscopic videos, the script:

1. Splits the input video into left and right eye views
2. Processes each view separately with the segmentation pipeline
3. Combines the processed views back into a side-by-side format for VR headsets

This preserves the 3D stereo effect while applying person segmentation to both views.

## Tips for Best Results

1. **Check your input video**: Make sure your stereo video has both views side-by-side in a single frame
2. **Consistent processing**: Use the same model and parameters for both eyes to maintain consistency
3. **Testing**: Try processing just one eye first (with `--left-eye` or `--right-eye`) to check results before processing both
4. **Video dimensions**: Stereoscopic VR videos are typically twice as wide as regular videos (e.g., 3840×1920 instead of 1920×1080)

For high-quality results, the script automatically uses the best available model (YOLOv11 > YOLOv8x > YOLOv8s) and optimal parameters for person segmentation.
