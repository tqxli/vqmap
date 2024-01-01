import os
import random
import numpy as np
import torch

def set_random_seed(seed: int):
    """
    Fix numpy and torch random seed generation.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True