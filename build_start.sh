#!/bin/bash
# Build and validate script for 360 VR Person Segmentation Pipeline
# This script builds the project and checks for syntax errors

# Log file
LOG_FILE="./build_start_log.txt"

# Ensure the log file exists and is empty
echo "# Build Start Log - $(date)" > "$LOG_FILE"

echo "Starting build and validation process..." | tee -a "$LOG_FILE"

# Run the build script first
echo "Running build_all.sh..." | tee -a "$LOG_FILE"
bash ./build_all.sh | tee -a "$LOG_FILE"
BUILD_STATUS=${PIPESTATUS[0]}

if [ $BUILD_STATUS -ne 0 ]; then
    echo "Error: build_all.sh failed with status $BUILD_STATUS" | tee -a "$LOG_FILE"
    exit $BUILD_STATUS
fi

# Activate virtual environment
echo "Activating virtual environment..." | tee -a "$LOG_FILE"
source venv/bin/activate >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment!" | tee -a "$LOG_FILE"
    exit 1
fi

# Check for syntax errors in Python files
echo "Checking Python files for syntax errors..." | tee -a "$LOG_FILE"
SYNTAX_ERROR=0

check_file() {
    local file=$1
    echo "Checking $file..." | tee -a "$LOG_FILE"
    python -m py_compile "$file" >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "  Error: Syntax error in $file" | tee -a "$LOG_FILE"
        SYNTAX_ERROR=1
    else
        echo "  OK: $file" | tee -a "$LOG_FILE"
    fi
}

# Check all Python files in src
for file in src/*.py; do
    check_file "$file"
done

# Check all Python files in tests
for file in tests/*.py; do
    check_file "$file"
done

# Try to import the main module
echo "Trying to import main module..." | tee -a "$LOG_FILE"
python -c "from src import main" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Failed to import main module!" | tee -a "$LOG_FILE"
    SYNTAX_ERROR=1
else
    echo "Main module imported successfully." | tee -a "$LOG_FILE"
fi

# Check for relative import issues
echo "Checking for import issues..." | tee -a "$LOG_FILE"
python -c "from src.pipeline import run_pipeline; from src.config import load_config" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Failed to import pipeline and config modules!" | tee -a "$LOG_FILE"
    SYNTAX_ERROR=1
else
    echo "Pipeline and config modules imported successfully." | tee -a "$LOG_FILE"
fi

# Deactivate virtual environment
deactivate >> "$LOG_FILE" 2>&1

# Final status
if [ $SYNTAX_ERROR -eq 0 ]; then
    echo "Success: No syntax errors found. Project built successfully!" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Syntax errors found. Check build_start_log.txt for details." | tee -a "$LOG_FILE"
    exit 1
fi
