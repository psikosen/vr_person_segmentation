#!/bin/bash
# Test script to verify the pipeline can be run without actual processing
# This will check if the import structure works correctly

# Set up a temporary directory
TEMP_DIR="/tmp/vr_test_$(date +%s)"
mkdir -p "$TEMP_DIR"

# Create a mock 360° video file (just a small text file)
MOCK_VIDEO="$TEMP_DIR/mock_360.txt"
echo "This is a mock 360 video file" > "$MOCK_VIDEO"

# Run the pipeline with --help to check if it works without actual processing
echo "Testing pipeline with --help..."
./run.sh --help

if [ $? -eq 0 ]; then
    echo "Success: Pipeline help command works correctly!"
else
    echo "Error: Pipeline help command failed!"
    exit 1
fi

# Clean up
rm -rf "$TEMP_DIR"

echo "All tests passed! Import structure works correctly."
exit 0
