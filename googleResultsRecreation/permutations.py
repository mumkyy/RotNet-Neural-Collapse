import numpy as np
import torch

def load_permutations(path):
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)

    num_perm = data[0]
    perm_len = data[1]

    perms = data[2:].reshape(num_perm, perm_len)
    perms = perms - 1  # convert from 1-indexed

    return torch.tensor(perms, dtype=torch.long)