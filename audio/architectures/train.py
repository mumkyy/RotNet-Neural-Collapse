import torch
import torch.optim as optim 
import torch.nn as nn 
import torchaudio.transforms as T 

from model import AudioCNN

import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os 

from pathlib import Path

import utils
import datetime
import logging

from pdb import set_trace as breakpoint
from torch.utils.data import Dataset, Dataloader, random


def set_seed(seed: int) ->  None:  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_loop(): 


if __name__ == '__main__': 

    set_seed(42)

