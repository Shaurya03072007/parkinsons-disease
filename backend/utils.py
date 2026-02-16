import torch
import os

def get_device():
    """
    Selects the best available device for training/inference.
    Prioritizes CUDA (GPU), then CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_num_workers():
    """
    Determines the optimal number of workers for DataLoaders.
    User has 56 cores, so we can be generous, but let's leave some for the OS.
    """
    cpu_count = os.cpu_count()
    if cpu_count:
        # specific optimization for user's 56 core machine
        if cpu_count >= 56:
            return 32 # Use 32 workers to be safe and efficient
        return max(1, cpu_count - 2)
    return 2
