#!/bin/bash
# Run script for the 360 VR Person Segmentation Pipeline

# Log file
LOG_FILE="./run_log.txt"

# Create a new log file or clear existing one
echo "# Run Log - $(date)" > "$LOG_FILE"

echo "Starting 360 VR Person Segmentation Pipeline..." | tee -a "$LOG_FILE"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..." | tee -a "$LOG_FILE"
    source venv/bin/activate >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate virtual environment!" | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "Error: Virtual environment not found! Run ./build_all.sh first." | tee -a "$LOG_FILE"
    exit 1
fi

# Run the main program, passing all arguments
echo "Running pipeline with arguments: $@" | tee -a "$LOG_FILE"
cd "$(dirname "$0")"
python -m src.main "$@" 2>&1 | tee -a "$LOG_FILE"
EXIT_STATUS=${PIPESTATUS[0]}

# Deactivate virtual environment
deactivate >> "$LOG_FILE" 2>&1

# Check exit status
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Pipeline completed successfully!" | tee -a "$LOG_FILE"
else
    echo "Pipeline failed with exit status: $EXIT_STATUS" | tee -a "$LOG_FILE"
    echo "Check logs for details." | tee -a "$LOG_FILE"
fi

exit $EXIT_STATUS
