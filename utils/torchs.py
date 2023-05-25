import os
import torch
import numpy as np
import pynvml
import random


def get_gpus_meminfo():
    try:
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in range(pynvml.nvmlDeviceGetCount())]
        gpus_free = [pynvml.nvmlDeviceGetMemoryInfo(handle).free for handle in handles]
        gpus_idx = np.argsort(gpus_free)[::-1].tolist()
        gpus_free = [gpus_free[idx] for idx in gpus_idx]
    except Exception:
        gpus_free, gpus_idx = [], []
    return gpus_idx, gpus_free


def get_best_device():
    device_idx = None
    if torch.cuda.is_available():
        gpus, _ = get_gpus_meminfo()
        if gpus:
            device_idx = gpus[0]
    return device_idx


def cuda_is_available():
    return torch.cuda.is_available()


def set_global_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
