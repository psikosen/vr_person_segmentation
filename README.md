# 360 VR Person Segmentation Pipeline

A command-line tool and backend process that extracts human figures from specified viewpoints in 360° VR videos and outputs a standard 2D video file containing only these figures against a transparent background.

## Project Overview

This tool provides a configurable pipeline that:

1. Ingests a 360° VR video file
2. Extracts a standard 2D video viewpoint based on configuration
3. Applies AI-driven instance segmentation (Ultralytics YOLOv8-seg) to identify and isolate 'person' subjects
4. Generates an output video file (WebM with VP9 codec and alpha channel) where segmented people are opaque, and the original background is transparent

## Key Technologies

- Python
- Ultralytics YOLO (YOLOv8-seg)
- OpenCV
- FFmpeg (for video extraction and encoding)
- WebM with VP9 codec (for alpha channel support)

## Architecture

The pipeline consists of the following components:

- Configuration Manager: Handles user configuration through CLI or config files
- Input Handler: Processes the input 360° video file
- 360° Viewport Extractor: Extracts a 2D viewpoint from the 360° video
- Segmentation Engine: Uses YOLOv8-seg to identify and segment person objects
- Compositor: Creates RGBA frames with transparent backgrounds
- Transparent Video Encoder: Encodes the frames to a WebM video with VP9 codec

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vr_person_segmentation.git
cd vr_person_segmentation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8-seg weights (if not included)
# Instructions will be provided
```

## Usage

```bash
# Basic usage
python src/main.py --input-video input.mp4 --output-video output.webm

# With custom configuration file
python src/main.py --config config.yaml

# With custom parameters
python src/main.py --input-video input.mp4 --output-video output.webm --model-path models/yolov8s-seg.pt --confidence 0.5
```

## Building and Running

The project includes build and run scripts:

```bash
# Build the project
./build_all.sh

# Run the pipeline
./run.sh --input-video input.mp4 --output-video output.webm
```

## Development

See [TODO.md](TODO.md) for development tasks and progress.

## License

[MIT License](LICENSE)
