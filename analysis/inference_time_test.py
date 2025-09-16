#!/usr/bin/env python3
"""
Script để đo inference time cho head pose estimation models
"""

import torch
import time
import sys
import os

# Add the parent directory to path to access model modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def measure_inference_time(model, input_data, num_runs=100, warmup_runs=10):
    """Đo inference time"""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Move input to device
    if isinstance(input_data, list):
        input_data = [x.to(device) for x in input_data]
    else:
        input_data = input_data.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            try:
                _ = model(input_data)
            except:
                pass

    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    successful_runs = 0
    with torch.no_grad():
        for _ in range(num_runs):
            try:
                _ = model(input_data)
                successful_runs += 1
            except Exception as e:
                print(f"Error during inference: {e}")
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    if successful_runs > 0:
        avg_time_ms = ((end_time - start_time) / successful_runs) * 1000
        return avg_time_ms, successful_runs
    else:
        return None, 0


def test_model_inference(model_class, model_name):
    """Test inference time cho một model"""
    print(f"\n{'='*50}")
    print(f"Testing inference: {model_name}")
    print(f"{'='*50}")

    try:
        model = model_class()

        # Thử các input sizes khác nhau
        input_sizes = [
            (1, 3, 32, 32),
            (1, 3, 64, 64),
            (1, 3, 128, 128),
        ]

        for input_shape in input_sizes:
            print(f"\nTesting with input shape: {input_shape}")

            # Tạo input data - 4 streams
            input_data = [torch.randn(input_shape) for _ in range(4)]

            # Đo inference time
            inf_time, successful_runs = measure_inference_time(
                model, input_data, num_runs=50)

            if inf_time is not None:
                print(f"  ✓ Successful runs: {successful_runs}/50")
                print(f"  ✓ Average inference time: {inf_time:.2f} ms")
                return inf_time, input_shape
            else:
                print(f"  ✗ Failed with input shape {input_shape}")

        return None, None

    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")
        return None, None


def main():
    print("Inference Time Testing Script")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*50)

    results = []

    # Test Original Model
    try:
        from model.original_model import Model as OriginalModel
        inf_time, input_shape = test_model_inference(
            OriginalModel, "Original Model")
        if inf_time:
            results.append(("Original Model", inf_time, input_shape))
    except Exception as e:
        print(f"Could not test Original Model: {e}")

    # Test CA Model All Stable Light
    try:
        from model.bi_directional_cross_attention.ca_model_all_stable_light import Model as CAModelAllStableLight
        inf_time, input_shape = test_model_inference(
            CAModelAllStableLight, "CA Model All Stable Light")
        if inf_time:
            results.append(
                ("CA Model All Stable Light", inf_time, input_shape))
    except Exception as e:
        print(f"Could not test CA Model All Stable Light: {e}")

    # Test CA Model Neighborhood Stable Light
    try:
        from model.bi_directional_cross_attention.ca_model_neighborhood_stable_light import Model as CAModelNeighborhoodStableLight
        inf_time, input_shape = test_model_inference(
            CAModelNeighborhoodStableLight, "CA Model Neighborhood Stable Light")
        if inf_time:
            results.append(
                ("CA Model Neighborhood Stable Light", inf_time, input_shape))
    except Exception as e:
        print(f"Could not test CA Model Neighborhood Stable Light: {e}")

    # Summary
    if results:
        print(f"\n{'='*80}")
        print("INFERENCE TIME SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model Name':<35} {'Time(ms)':<12} {'Input Shape':<15}")
        print(f"{'-'*80}")

        for name, time_ms, shape in results:
            print(f"{name:<35} {time_ms:<12.2f} {str(shape):<15}")

        # Speed comparison
        if len(results) > 1:
            print(f"\n{'='*80}")
            print("SPEED COMPARISON:")
            base_time = results[0][1]
            for i, (name, time_ms, _) in enumerate(results[1:], 1):
                speedup = base_time / time_ms
                print(
                    f"{name} vs {results[0][0]}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")


if __name__ == "__main__":
    main()
