import random
from collections import defaultdict
import minitorch
import time
import sys
import numpy as np
#New
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend, size=16) -> None:
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y

##NEW
def plot_results(times):
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not available, skipping plot generation")
        return

    sizes = list(times.keys())
    fast_times = [times[size]["fast"] for size in sizes]
    gpu_times = [times[size]["gpu"] for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, fast_times, 'b-o', label='CPU (Fast)')
    plt.plot(sizes, gpu_times, 'r-o', label='GPU (CUDA)')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance: CPU vs GPU')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig('matmul_benchmark.png')
    except Exception as e:
        print(f"\nError saving plot: {e}")
    plt.close()

def calculate_speedup(times):
    speedups = {}
    for size in times:
        speedups[size] = times[size]["fast"] / times[size]["gpu"]
    return speedups

if __name__ == "__main__":
    # Warmup
    print("Warming up...")
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}

    print("\nRunning benchmarks...")
    for size in [64, 128, 256, 512, 1024]:
        print(f"\nRunning size {size}x{size}")
        times[size] = {}
        fast_times = []
        gpu_times = []

        for trial in range(ntrials):
            print(f"  Trial {trial + 1}/{ntrials}...", end=' ')

            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)
            print(f"CPU: {fast_time:.5f}s, GPU: {gpu_time:.5f}s")

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)

        print(f"  Average - CPU: {times[size]['fast']:.5f}s, GPU: {times[size]['gpu']:.5f}s")

    # Calculate speedups
    speedups = calculate_speedup(times)

    print("\nTiming summary:")
    print("Size".ljust(8) + "CPU (s)".ljust(15) + "GPU (s)".ljust(15) + "Speedup")
    print("-" * 45)
    for size in times:
        cpu_time = times[size]["fast"]
        gpu_time = times[size]["gpu"]
        speedup = speedups[size]
        print(f"{size:<8}{cpu_time:<15.5f}{gpu_time:<15.5f}{speedup:.2f}x")

    # Plot results
    plot_results(times)
    print("\nResults have been plotted to 'matmul_benchmark.png'")