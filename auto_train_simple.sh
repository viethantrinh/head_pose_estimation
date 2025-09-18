#!/bin/bash
# Simple Auto Training Script for Protocol 1
# Chạy training liên tục với protocol 1

echo "==================================="
echo "  AUTO TRAINING PROTOCOL 1"
echo "==================================="
echo "Nhấn Ctrl+C để dừng"
echo ""

# Configuration
CONFIG_FILE="./config/train_config_p1.yaml"
SCRIPT_NAME="main_p1.py"
ITERATION=1

# Check if files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Training script not found: $SCRIPT_NAME"
    exit 1
fi

# Create log directory
mkdir -p ./work_dir/protocol_1/auto_train_logs

# Main loop
while true; do
    echo "=========================================="
    echo "Starting Training Iteration $ITERATION"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Create backup of previous results
    if [ $ITERATION -gt 1 ]; then
        PREV_ITERATION=$((ITERATION-1))
        RESULT_DIR="./work_dir/protocol_1/lan$PREV_ITERATION"
        mkdir -p "$RESULT_DIR"
        
        echo "Saving results of iteration $PREV_ITERATION to lan$PREV_ITERATION..."
        
        # Save important files
        for file in "log.txt" "best_model_test1.pt" "best_model_test2.pt" "config.yaml"; do
            if [ -f "./work_dir/protocol_1/$file" ]; then
                cp "./work_dir/protocol_1/$file" "$RESULT_DIR/"
                echo "Saved $file to lan$PREV_ITERATION"
            fi
        done
        
        # Save all model files
        find "./work_dir/protocol_1" -name "*.pt" -o -name "*.pth" | while read model_file; do
            if [[ "$model_file" != *"/lan"* ]]; then
                cp "$model_file" "$RESULT_DIR/"
                echo "Saved $(basename $model_file) to lan$PREV_ITERATION"
            fi
        done
        
        # Create summary file
        echo "Training Iteration: $PREV_ITERATION" > "$RESULT_DIR/training_summary.txt"
        echo "Completed at: $(date)" >> "$RESULT_DIR/training_summary.txt"
        echo "Work directory: ./work_dir/protocol_1" >> "$RESULT_DIR/training_summary.txt"
        echo "Result directory: $RESULT_DIR" >> "$RESULT_DIR/training_summary.txt"
    fi
    
    # Run training
    echo "Starting training..."
    python "$SCRIPT_NAME" --config "$CONFIG_FILE"
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "Training iteration $ITERATION completed successfully!"
        
        # Save results immediately after successful completion
        RESULT_DIR="./work_dir/protocol_1/lan$ITERATION"
        mkdir -p "$RESULT_DIR"
        
        echo "Saving results to lan$ITERATION..."
        
        # Save important files
        for file in "log.txt" "best_model_test1.pt" "best_model_test2.pt" "config.yaml"; do
            if [ -f "./work_dir/protocol_1/$file" ]; then
                cp "./work_dir/protocol_1/$file" "$RESULT_DIR/"
                echo "Saved $file to lan$ITERATION"
            fi
        done
        
        # Save all model files
        find "./work_dir/protocol_1" -name "*.pt" -o -name "*.pth" | while read model_file; do
            if [[ "$model_file" != *"/lan"* ]]; then
                cp "$model_file" "$RESULT_DIR/"
                echo "Saved $(basename $model_file) to lan$ITERATION"
            fi
        done
        
        # Create summary file
        echo "Training Iteration: $ITERATION" > "$RESULT_DIR/training_summary.txt"
        echo "Completed at: $(date)" >> "$RESULT_DIR/training_summary.txt"
        echo "Work directory: ./work_dir/protocol_1" >> "$RESULT_DIR/training_summary.txt"
        echo "Result directory: $RESULT_DIR" >> "$RESULT_DIR/training_summary.txt"
        
    else
        echo "Training iteration $ITERATION failed with exit code: $TRAIN_EXIT_CODE"
    fi
    
    echo "Iteration $ITERATION finished at: $(date)"
    echo ""
    
    # Wait a bit before next iteration
    echo "Waiting 10 seconds before next iteration..."
    sleep 10
    
    ITERATION=$((ITERATION + 1))
done
