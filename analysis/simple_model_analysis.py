#!/usr/bin/env python3
"""
Script đơn giản để tính model size và parameters cho head pose estimation models
"""

import torch
import torch.nn as nn
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


def analyze_model_simple(model_class, model_name):
    """Phân tích model đơn giản - chỉ parameters và size"""
    print(f"\n{'='*50}")
    print(f"Analyzing {model_name}")
    print(f"{'='*50}")

    try:
        # Tạo model
        model = model_class()

        # 1. Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(
            f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

        # 2. Model size
        model_size_mb = get_model_size_mb(model)
        print(f"Model Size: {model_size_mb:.2f} MB")

        return {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'params_M': total_params/1e6,
            'model_size_mb': model_size_mb
        }

    except Exception as e:
        print(f"Error analyzing {model_name}: {str(e)}")
        return None


def main():
    print("Simple Model Analysis Script for Head Pose Estimation")
    print("Analyzing model parameters and sizes...")

    results = []

    # Analyze Original Model
    try:
        from model.original_model import Model as OriginalModel
        result = analyze_model_simple(OriginalModel, "Original Model")
        if result:
            results.append(result)
    except Exception as e:
        print(f"Could not analyze Original Model: {e}")

    # Analyze CA Model All Stable Light
    try:
        from model.bi_directional_cross_attention.ca_model_all_stable_light import Model as CAModelAllStableLight
        result = analyze_model_simple(
            CAModelAllStableLight, "CA Model All Stable Light")
        if result:
            results.append(result)
    except Exception as e:
        print(f"Could not analyze CA Model All Stable Light: {e}")

    # Analyze CA Model Neighborhood Stable Light
    try:
        from model.bi_directional_cross_attention.ca_model_neighborhood_stable_light import Model as CAModelNeighborhoodStableLight
        result = analyze_model_simple(
            CAModelNeighborhoodStableLight, "CA Model Neighborhood Stable Light")
        if result:
            results.append(result)
    except Exception as e:
        print(f"Could not analyze CA Model Neighborhood Stable Light: {e}")

    # Summary table
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"{'Model Name':<35} {'Params(M)':<12} {'Size(MB)':<12}")
        print(f"{'-'*70}")

        for result in results:
            print(f"{result['model_name']:<35} "
                  f"{result['params_M']:<12.2f} "
                  f"{result['model_size_mb']:<12.2f}")

        print(f"\n{'='*70}")
        print("COMPARISON:")
        if len(results) > 1:
            base_params = results[0]['params_M']
            base_size = results[0]['model_size_mb']

            for i, result in enumerate(results[1:], 1):
                params_ratio = result['params_M'] / base_params
                size_ratio = result['model_size_mb'] / base_size
                print(f"{result['model_name']} vs {results[0]['model_name']}:")
                print(
                    f"  Parameters: {params_ratio:.2f}x ({'+' if params_ratio > 1 else ''}{(params_ratio-1)*100:.1f}%)")
                print(
                    f"  Size: {size_ratio:.2f}x ({'+' if size_ratio > 1 else ''}{(size_ratio-1)*100:.1f}%)")


if __name__ == "__main__":
    main()
