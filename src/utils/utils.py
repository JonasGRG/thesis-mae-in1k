import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# Automatically determine the version based on existing log directories
def get_next_version(log_dir, name):
    existing_versions = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d.startswith(name)]
    if existing_versions:
        versions = [int(d.split("_v")[-1]) for d in existing_versions if "_v" in d]
        next_version = max(versions) + 1 if versions else 1
    else:
        next_version = 1
    return next_version
