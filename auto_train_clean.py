#!/usr/bin/env python3
"""
Auto Training Script for Protocol 1 - Clean TQDM
================================================
Script tự động chạy training liên tục với protocol 1.
TQDM sẽ hiển thị bình thường trên một dòng.
"""

import os
import sys
import time
import shutil
import subprocess
from datetime import datetime

# Configuration
CONFIG_FILE = "./config/train_config_p1.yaml"
SCRIPT_NAME = "main_p1.py"
MAX_ITERATIONS = 100  # Số lần training tối đa (set -1 cho vô hạn)
SLEEP_BETWEEN_RUNS = 10  # Thời gian chờ giữa các lần chạy (giây)

def log_message(message, log_file=None):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_line + '\n')

def save_training_results(iteration, log_file):
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

        log_message(f"Đang lưu kết quả vào lan{iteration}...", log_file)

        # Save main files
        for file in files_to_save:
            src = os.path.join(work_dir, file)
            if os.path.exists(src):
                dst = os.path.join(result_dir, file)
                shutil.copy2(src, dst)
                log_message(f"Đã lưu {file} vào lan{iteration}", log_file)

        # Save all model files from subdirectories
        for root, dirs, files in os.walk(work_dir):
            # Skip lan directories to avoid recursion
            if 'lan' in os.path.basename(root):
                continue
            for file in files:
                if file.endswith(('.pt', '.pth')):
                    src = os.path.join(root, file)
                    rel_path = os.path.relpath(src, work_dir)
                    dst = os.path.join(result_dir, rel_path)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    log_message(f"Đã lưu model {rel_path} vào lan{iteration}", log_file)

        # Save training summary
        summary_file = os.path.join(result_dir, "training_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Training Iteration: {iteration}\n")
            f.write(f"Hoàn thành lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Thư mục làm việc: {work_dir}\n")
            f.write(f"Thư mục kết quả: {result_dir}\n")

        log_message(f"Kết quả đã được lưu vào {result_dir}", log_file)

def run_training_direct(log_file):
    """Run training with direct output (tqdm works normally)"""
    log_message("Bắt đầu training...", log_file)
    
    try:
        # Run training command directly in terminal
        cmd = ["python", SCRIPT_NAME, "--config", CONFIG_FILE]
        log_message(f"Executing: {' '.join(cmd)}", log_file)
        
        # Use subprocess.run to let training output directly to terminal
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            log_message("Training hoàn thành thành công!", log_file)
            return True
        else:
            log_message(f"Training thất bại với exit code: {result.returncode}", log_file)
            return False
            
    except KeyboardInterrupt:
        log_message("Training bị dừng bởi người dùng", log_file)
        return False
    except Exception as e:
        log_message(f"Lỗi trong quá trình training: {str(e)}", log_file)
        return False

def check_config_file():
    """Check if config file exists"""
    if not os.path.exists(CONFIG_FILE):
        print(f"Lỗi: Không tìm thấy config file: {CONFIG_FILE}")
        return False

    if not os.path.exists(SCRIPT_NAME):
        print(f"Lỗi: Không tìm thấy training script: {SCRIPT_NAME}")
        return False

    return True

def print_banner():
    """Print banner information"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║               AUTO TRAINING PROTOCOL 1                  ║
    ║                   CLEAN TQDM VERSION                    ║
    ║                                                          ║
    ║  Script tự động chạy training liên tục                  ║
    ║  TQDM sẽ hiển thị bình thường trên một dòng             ║
    ║  Nhấn Ctrl+C để dừng                                    ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main function"""
    print_banner()

    # Setup log file
    log_dir = "./work_dir/protocol_1/auto_train_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"auto_train_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Check prerequisites
    if not check_config_file():
        sys.exit(1)

    log_message("Auto Training Script Started", log_file)
    log_message(f"Config file: {CONFIG_FILE}", log_file)
    log_message(f"Training script: {SCRIPT_NAME}", log_file)
    log_message(f"Max iterations: {'Unlimited' if MAX_ITERATIONS == -1 else MAX_ITERATIONS}", log_file)

    iteration = 1
    successful_runs = 0
    failed_runs = 0

    try:
        while MAX_ITERATIONS == -1 or iteration <= MAX_ITERATIONS:
            log_message("="*60, log_file)
            log_message(f"STARTING ITERATION {iteration}", log_file)
            log_message("="*60, log_file)

            # Run training
            start_time = time.time()
            success = run_training_direct(log_file)
            end_time = time.time()

            duration = end_time - start_time
            duration_str = f"{int(duration//3600):02d}:{int((duration%3600)//60):02d}:{int(duration%60):02d}"

            if success:
                successful_runs += 1
                log_message(f"Iteration {iteration} hoàn thành thành công trong {duration_str}", log_file)
                # Save results immediately after successful completion
                save_training_results(iteration, log_file)
            else:
                failed_runs += 1
                log_message(f"Iteration {iteration} thất bại sau {duration_str}", log_file)

            # Print statistics
            log_message(f"Thống kê: {successful_runs} thành công, {failed_runs} thất bại", log_file)

            # Check if we should continue
            if MAX_ITERATIONS != -1 and iteration >= MAX_ITERATIONS:
                log_message("Đã đạt số iteration tối đa. Dừng.", log_file)
                break

            # Sleep between runs
            if SLEEP_BETWEEN_RUNS > 0:
                log_message(f"Chờ {SLEEP_BETWEEN_RUNS} giây trước iteration tiếp theo...", log_file)
                time.sleep(SLEEP_BETWEEN_RUNS)

            iteration += 1

    except KeyboardInterrupt:
        log_message("Auto training bị dừng bởi người dùng", log_file)
    except Exception as e:
        log_message(f"Lỗi không mong đợi: {str(e)}", log_file)
    finally:
        log_message(f"Auto training kết thúc. Tổng cộng: {successful_runs} thành công, {failed_runs} thất bại", log_file)

if __name__ == "__main__":
    main()
