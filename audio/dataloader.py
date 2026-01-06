
import torchaudio
import torch
import math
from torchaudio.datasets import SPEECHCOMMANDS

import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from pathlib import Path

_SPEECH_DATASET_DIR = './datasets/SpeechCommands_dataset'

dataset = torchaudio.datasets.SPEECHCOMMANDS(root=str(_SPEECH_DATASET_DIR), download=True)


def reverse_sample(waveform):

    reversed_waveform = torch.flip(waveform, dims=[1])
    return reversed_waveform

class Dataloader(object): 
    def __init__(self, dataset, batch_size=1, unsupervised=True, epoch_size=None, num_workers=0, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        self.pretext_mode = getattr(self.dataset, 'pretext_mode', 'reverse')
    def get_iterator(self, epoch=0):
        rand_seed = epoch*self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            def _load_function(idx): 
                idx = idx % len(self.dataset)
                waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
                mode = getattr(self.dataset, 'pretex_mode', self.pretext_mode)
                if mode == 'reverse': 
                    r = random.randint(0,1)
                    rev = True if r == 1 else False
                    if rev : 
                        return  r, reverse_sample(waveform = waveform)
                    else : 
                        return r, waveform