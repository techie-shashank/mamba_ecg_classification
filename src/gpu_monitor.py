#!/usr/bin/env python3
"""
GPU monitoring utilities for ensuring models are training on GPU.
"""

import torch
import time
import psutil
import threading
from logger import logger


class GPUMonitor:
    """Real-time GPU monitoring during training."""

    def __init__(self, log_interval=30):
        """
        Initialize GPU monitor.
        Args:
            log_interval (int): Interval in seconds between monitoring logs
        """
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start background GPU monitoring."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - GPU monitoring disabled")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"GPU monitoring started (interval: {self.log_interval}s)")

    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("GPU monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop with responsive stopping."""
        while self.monitoring:
            self._log_gpu_stats()

            # Sleep in small chunks to be responsive to stop signals
            sleep_chunks = self.log_interval // 10  # Break into 10 chunks
            chunk_size = self.log_interval / 10

            for _ in range(sleep_chunks):
                if not self.monitoring:  # Check stop signal frequently
                    return
                time.sleep(chunk_size)

            # Handle remaining time if log_interval isn't divisible by 10
            remaining = self.log_interval % 10
            if remaining > 0 and self.monitoring:
                time.sleep(remaining)

    def _log_gpu_stats(self):
        """Log current GPU statistics."""
        if not torch.cuda.is_available():
            return

        # GPU memory
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # GPU utilization (requires nvidia-ml-py if available)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            memory_util = utilization.memory

            logger.info(f"GPU Stats - Memory: {allocated:.2f}/{max_memory:.2f} GB ({allocated/max_memory*100:.1f}%), "
                       f"Cached: {cached:.2f} GB, GPU Util: {gpu_util}%, Memory Util: {memory_util}%")
        except ImportError:
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}/{max_memory:.2f} GB ({allocated/max_memory*100:.1f}%), "
                       f"Cached: {cached:.2f} GB")


def verify_gpu_setup():
    """
    Comprehensive GPU setup verification.
    Returns True if GPU is properly configured, False otherwise.
    """
    logger.info("=== GPU Setup Verification ===")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")

    if not cuda_available:
        logger.error("CUDA is not available! Check your PyTorch installation and GPU drivers.")
        return False

    # GPU information
    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name} - Memory: {props.total_memory / 1024**3:.2f} GB - Compute: {props.major}.{props.minor}")

    # Current device
    current_device = torch.cuda.current_device()
    logger.info(f"Current CUDA device: {current_device}")

    # Test tensor operations on GPU
    try:
        device = torch.device("cuda")
        test_tensor = torch.randn(1000, 1000, device=device)
        result = torch.matmul(test_tensor, test_tensor.T)
        logger.info("GPU tensor operations test: PASSED")

        # Clean up
        del test_tensor, result
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"GPU tensor operations test: FAILED - {str(e)}")
        return False

    # PyTorch CUDA version
    logger.info(f"PyTorch CUDA version: {torch.version.cuda}")

    logger.info("=== GPU Setup Verification Complete ===")
    return True


def force_gpu_usage():
    """
    Force PyTorch to use GPU by setting environment variables.
    Call this before any model initialization.
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info("Forced GPU usage - device set to cuda:0")
    else:
        logger.warning("Cannot force GPU usage - CUDA not available")


def benchmark_gpu_vs_cpu(model_class, input_shape, num_classes, iterations=100):
    """
    Benchmark model performance on GPU vs CPU.
    Args:
        model_class: Model class to benchmark
        input_shape: Input tensor shape (batch_size, seq_len, features)
        num_classes: Number of output classes
        iterations: Number of forward passes to benchmark
    """
    logger.info("=== GPU vs CPU Benchmark ===")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available - skipping GPU benchmark")
        return

    # Create test data
    batch_size, seq_len, features = input_shape
    test_input = torch.randn(batch_size, seq_len, features)

    # CPU benchmark
    logger.info("Benchmarking CPU...")
    model_cpu = model_class(features, num_classes)
    model_cpu.eval()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model_cpu(test_input)
    cpu_time = time.time() - start_time

    # GPU benchmark
    logger.info("Benchmarking GPU...")
    model_gpu = model_class(features, num_classes).cuda()
    test_input_gpu = test_input.cuda()
    model_gpu.eval()

    # Warm up GPU
    with torch.no_grad():
        for _ in range(10):
            _ = model_gpu(test_input_gpu)
    torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model_gpu(test_input_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time

    # Results
    speedup = cpu_time / gpu_time
    logger.info(f"CPU time: {cpu_time:.3f}s")
    logger.info(f"GPU time: {gpu_time:.3f}s")
    logger.info(f"GPU speedup: {speedup:.2f}x")

    # Cleanup
    del model_cpu, model_gpu, test_input, test_input_gpu
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Run GPU verification
    verify_gpu_setup()
