#!/usr/bin/env python3
"""
Script tổng hợp để phân tích đầy đủ các models trong head pose estimation project
Bao gồm: Model size, Parameters, Inference time
"""

import torch
import time
import sys
import os

# Add the parent directory to path to access model modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def count_parameters(model):
    """Đếm số parameters của model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """Tính model size theo MB"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    return model_size_mb


def measure_inference_time(model, input_data, num_runs=50):
    """Đo inference time"""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Move input to device
    input_data = [x.to(device) for x in input_data]

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            try:
                _ = model(input_data)
            except:
                return None

    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time_ms = ((end_time - start_time) / num_runs) * 1000
    return avg_time_ms


def analyze_model_complete(model_class, model_name):
    """Phân tích đầy đủ một model"""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print(f"{'='*60}")

    try:
        # Tạo model
        model = model_class()

        # 1. Count parameters
        total_params, trainable_params = count_parameters(model)

        # 2. Model size
        model_size_mb = get_model_size_mb(model)

        # 3. Inference time với input shape 32x32
        input_data = [torch.randn(1, 3, 32, 32) for _ in range(4)]
        inference_time = measure_inference_time(model, input_data)

        # In kết quả
        print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print(
            f"Inference Time: {inference_time:.2f} ms" if inference_time else "Inference Time: Failed")

        return {
            'name': model_name,
            'params': total_params,
            'params_M': total_params/1e6,
            'size_mb': model_size_mb,
            'inference_ms': inference_time if inference_time else 0
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def main():
    print("="*80)
    print("HEAD POSE ESTIMATION MODELS ANALYSIS")
    print("="*80)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Input size: 4 streams of (1, 3, 32, 32)")
    print("="*80)

    results = []

    # Models to analyze
    models_to_test = [
        ("model.original_model", "Model", "Original Model"),
        ("model.bi_directional_cross_attention.ca_model_all_stable_light",
         "Model", "CA All Stable Light"),
        ("model.bi_directional_cross_attention.ca_model_neighborhood_stable_light",
         "Model", "CA Neighborhood Stable Light")
    ]

    for module_path, class_name, display_name in models_to_test:
        try:
            # Dynamic import
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)

            result = analyze_model_complete(model_class, display_name)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nCould not analyze {display_name}: {e}")

    # Summary table
    if results:
        print(f"\n{'='*90}")
        print("COMPREHENSIVE COMPARISON TABLE")
        print(f"{'='*90}")
        print(
            f"{'Model':<30} {'Params(M)':<12} {'Size(MB)':<12} {'Time(ms)':<12} {'Efficiency':<12}")
        print(f"{'-'*90}")

        for result in results:
            efficiency = result['params_M'] / \
                result['inference_ms'] if result['inference_ms'] > 0 else 0
            print(f"{result['name']:<30} "
                  f"{result['params_M']:<12.2f} "
                  f"{result['size_mb']:<12.2f} "
                  f"{result['inference_ms']:<12.2f} "
                  f"{efficiency:<12.2f}")

        # Detailed comparison
        if len(results) > 1:
            print(f"\n{'='*90}")
            print("DETAILED COMPARISON (vs Original Model):")
            print(f"{'='*90}")

            base = results[0]  # Original model as base

            for result in results[1:]:
                params_ratio = result['params_M'] / base['params_M']
                size_ratio = result['size_mb'] / base['size_mb']
                time_ratio = result['inference_ms'] / \
                    base['inference_ms'] if base['inference_ms'] > 0 else 1

                print(f"\n{result['name']}:")
                print(
                    f"  Parameters: {params_ratio:.2f}x ({'+' if params_ratio > 1 else ''}{(params_ratio-1)*100:.1f}%)")
                print(
                    f"  Size:       {size_ratio:.2f}x ({'+' if size_ratio > 1 else ''}{(size_ratio-1)*100:.1f}%)")
                print(
                    f"  Time:       {time_ratio:.2f}x ({'slower' if time_ratio > 1 else 'faster'})")

        print(f"\n{'='*90}")
        print("NOTES:")
        print("- Params(M): Số parameters tính theo triệu")
        print("- Size(MB): Kích thước model tính theo MB")
        print("- Time(ms): Thời gian inference trung bình tính theo milliseconds")
        print("- Efficiency: Params(M) / Time(ms) - càng cao càng tốt")
        print("="*90)


if __name__ == "__main__":
    main()
