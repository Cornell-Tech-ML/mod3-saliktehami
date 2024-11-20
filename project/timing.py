import minitorch
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # for nice tables

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

def run_matmul(backend, size=16) -> None:
    """Run a single matrix multiplication benchmark."""
    batch_size = 2
    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y

def plot_results(times, save_path='matmul_benchmark.png'):
    """Create a publication-quality plot of benchmark results."""
    sizes = list(times.keys())
    cpu_times = [times[size]["fast"] for size in sizes]
    gpu_times = [times[size]["gpu"] for size in sizes]

    plt.style.use('seaborn')  # Use a nicer style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Regular scale plot
    ax1.plot(sizes, cpu_times, 'b-o', label='CPU', linewidth=2, markersize=8)
    ax1.plot(sizes, gpu_times, 'r-o', label='GPU', linewidth=2, markersize=8)
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Matrix Multiplication Performance\n(Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale plot
    ax2.plot(sizes, cpu_times, 'b-o', label='CPU', linewidth=2, markersize=8)
    ax2.plot(sizes, gpu_times, 'r-o', label='GPU', linewidth=2, markersize=8)
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Matrix Multiplication Performance\n(Log Scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_results_table(times):
    """Print a nicely formatted table of results."""
    # Prepare table data
    headers = ["Matrix Size", "CPU Time (s)", "GPU Time (s)", "Speedup", "Efficiency"]
    rows = []
    for size in times:
        cpu_time = times[size]["fast"]
        gpu_time = times[size]["gpu"]
        speedup = cpu_time / gpu_time
        # Efficiency = speedup / theoretical_max (using matrix size as proxy)
        efficiency = (speedup / (size/64)) * 100  # normalized to smallest size
        rows.append([
            f"{size}x{size}",
            f"{cpu_time:.6f}",
            f"{gpu_time:.6f}",
            f"{speedup:.2f}x",
            f"{efficiency:.1f}%"
        ])

    # Print table
    print("\nBenchmark Results:")
    print(tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".6f"))

def run_benchmarks(sizes=[64, 128, 256, 512, 1024], ntrials=3):
    """Run complete benchmark suite."""
    print("Running benchmarks...")
    times = {}

    for size in sizes:
        print(f"\nSize: {size}x{size}")
        times[size] = {"fast": [], "gpu": []}

        for trial in range(ntrials):
            # CPU timing
            start = time.perf_counter()
            run_matmul(FastTensorBackend, size)
            cpu_time = time.perf_counter() - start
            times[size]["fast"].append(cpu_time)

            # GPU timing
            start = time.perf_counter()
            run_matmul(GPUBackend, size)
            gpu_time = time.perf_counter() - start
            times[size]["gpu"].append(gpu_time)

            print(f"  Trial {trial + 1}: CPU = {cpu_time:.6f}s, GPU = {gpu_time:.6f}s")

        # Calculate averages
        times[size]["fast"] = np.mean(times[size]["fast"])
        times[size]["gpu"] = np.mean(times[size]["gpu"])

    return times

if __name__ == "__main__":
    # Run warmup
    print("Warming up...")
    run_matmul(FastTensorBackend, 32)
    run_matmul(GPUBackend, 32)

    # Run benchmarks
    times = run_benchmarks()

    # Print and plot results
    print_results_table(times)
    plot_results(times)