#!/usr/bin/env python3
"""
Auto Training Script for Protocol 1
====================================
Script tự động chạy training liên tục với protocol 1.
Sau khi một lần training hoàn thành, script sẽ tự động bắt đầu training mới.
"""

import os
import sys
import time
import shutil
import logging
import subprocess
import re
from datetime import datetime
from pathlib import Path

# Configuration
CONFIG_FILE = "./config/train_config_p1.yaml"
SCRIPT_NAME = "main_p1.py"
MAX_ITERATIONS = 100  # Số lần training tối đa (set -1 cho vô hạn)
SLEEP_BETWEEN_RUNS = 10  # Thời gian chờ giữa các lần chạy (giây)

# Logging setup


def setup_logging():
    """Setup logging configuration"""
    log_dir = "./work_dir/protocol_1/auto_train_logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"auto_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def save_training_results(iteration):
    """Save training results to lanX folder"""
    work_dir = "./work_dir/protocol_1"
    result_dir = f"{work_dir}/lan{iteration}"

    if os.path.exists(work_dir):
        os.makedirs(result_dir, exist_ok=True)

        # Files to save
        files_to_save = [
            "log.txt",
            "best_model_test1.pt",
            "best_model_test2.pt",
            "config.yaml"
        ]

        # Also save any model files from subdirectories
        for root, dirs, files in os.walk(work_dir):
            for file in files:
                if file.endswith('.pt') or file.endswith('.pth'):
                    src = os.path.join(root, file)
                    # Create relative path structure in result_dir
                    rel_path = os.path.relpath(src, work_dir)
                    dst = os.path.join(result_dir, rel_path)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    logger.info(f"Saved model {rel_path} to lan{iteration}")

        # Save main files
        for file in files_to_save:
            src = os.path.join(work_dir, file)
            if os.path.exists(src):
                dst = os.path.join(result_dir, file)
                shutil.copy2(src, dst)
                logger.info(f"Saved {file} to lan{iteration}")

        # Save training summary
        summary_file = os.path.join(result_dir, "training_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Training Iteration: {iteration}\n")
            f.write(
                f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Work directory: {work_dir}\n")
            f.write(f"Result directory: {result_dir}\n")

        logger.info(f"Training results saved to {result_dir}")


def run_training():
    """Run a single training session"""
    logger.info("Starting training session...")

    try:
        # Run training command
        cmd = ["python", SCRIPT_NAME, "--config", CONFIG_FILE]
        logger.info(f"Executing command: {' '.join(cmd)}")

        # Start the process with direct output to terminal for tqdm
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Track tqdm lines to avoid logging them repeatedly
        last_tqdm_line = ""
        
        # Log output with tqdm filtering
        for line in process.stdout:
            line = line.strip()
            
            # Check if line contains tqdm progress (contains % and |)
            is_tqdm = ('|' in line and '%' in line and 'it/s' in line) or \
                     ('|' in line and '%' in line and 's/it' in line) or \
                     re.search(r'\d+%\|.*\|', line)
            
            if is_tqdm:
                # For tqdm lines, only print to console, don't log every update
                print(f"\r{line}", end="", flush=True)
                # Only log tqdm when it's different from last (major progress updates)
                if line != last_tqdm_line and ('100%' in line or line.count('|') != last_tqdm_line.count('|')):
                    print()  # New line after tqdm
                    logger.info(f"Progress: {line}")
                    last_tqdm_line = line
            else:
                # Normal lines: print and log
                if last_tqdm_line:  # If we had tqdm before, add newline
                    print()
                    last_tqdm_line = ""
                print(line)
                logger.info(line)

        # Ensure we end with a newline if last line was tqdm
        if last_tqdm_line:
            print()

        # Wait for completion
        return_code = process.wait()

        if return_code == 0:
            logger.info("Training completed successfully!")
            return True
        else:
            logger.error(f"Training failed with return code: {return_code}")
            return False

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False


def check_config_file():
    """Check if config file exists"""
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"Config file not found: {CONFIG_FILE}")
        return False

    if not os.path.exists(SCRIPT_NAME):
        logger.error(f"Training script not found: {SCRIPT_NAME}")
        return False

    return True


def print_banner():
    """Print banner information"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║               AUTO TRAINING PROTOCOL 1                  ║
    ║                                                          ║
    ║  Script tự động chạy training liên tục                  ║
    ║  Nhấn Ctrl+C để dừng                                    ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info("Auto Training Script Started")
    logger.info(f"Config file: {CONFIG_FILE}")
    logger.info(f"Training script: {SCRIPT_NAME}")
    logger.info(
        f"Max iterations: {'Unlimited' if MAX_ITERATIONS == -1 else MAX_ITERATIONS}")


def main():
    """Main function"""
    global logger
    logger = setup_logging()

    print_banner()

    # Check prerequisites
    if not check_config_file():
        sys.exit(1)

    iteration = 1
    successful_runs = 0
    failed_runs = 0

    try:
        while MAX_ITERATIONS == -1 or iteration <= MAX_ITERATIONS:
            logger.info(f"{'='*60}")
            logger.info(f"STARTING ITERATION {iteration}")
            logger.info(f"{'='*60}")

            # Save previous results to lanX folder
            if iteration > 1:
                save_training_results(iteration - 1)

            # Run training
            start_time = time.time()
            success = run_training()
            end_time = time.time()

            duration = end_time - start_time
            duration_str = f"{int(duration//3600):02d}:{int((duration%3600)//60):02d}:{int(duration%60):02d}"

            if success:
                successful_runs += 1
                logger.info(
                    f"Iteration {iteration} completed successfully in {duration_str}")
                # Save results immediately after successful completion
                save_training_results(iteration)
            else:
                failed_runs += 1
                logger.error(
                    f"Iteration {iteration} failed after {duration_str}")

            # Print statistics
            logger.info(
                f"Statistics: {successful_runs} successful, {failed_runs} failed runs")

            # Check if we should continue
            if MAX_ITERATIONS != -1 and iteration >= MAX_ITERATIONS:
                logger.info("Reached maximum iterations. Stopping.")
                break

            # Sleep between runs
            if SLEEP_BETWEEN_RUNS > 0:
                logger.info(
                    f"Waiting {SLEEP_BETWEEN_RUNS} seconds before next iteration...")
                time.sleep(SLEEP_BETWEEN_RUNS)

            iteration += 1

    except KeyboardInterrupt:
        logger.info("Auto training stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        logger.info(
            f"Auto training finished. Total: {successful_runs} successful, {failed_runs} failed runs")


if __name__ == "__main__":
    main()
