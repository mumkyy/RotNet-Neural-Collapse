import torch
import torchaudio
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AudioTransforms:
    """Converts 1D waveform to 2D Mel Spectrogram (Image-like)"""
    def __init__(self, sample_rate=16000):
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=64,       # Height of the 'image'
            n_fft=1024,
            hop_length=512
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform):
        # transform to spectogram
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)
        return spec

class GenericDataset(Dataset):
    def __init__(self, opt):
        self.root_dir = './datasets/SpeechCommands_dataset'
        self.dataset_name = opt.get('dataset_name', 'SPEECHCOMMANDS')
        self.split = opt.get('split', 'train')
        self.pretext_mode = opt.get('pretext_mode', 'reverse')
        self.unsupervised = opt.get('unsupervised', False)
        
        # Load the backend dataset (Train or Test split)
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=self.root_dir, 
            download=True, 
            subset='training' if self.split == 'train' else 'testing'
        )
        
        self.transforms = AudioTransforms()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
        
        # --- Pretext Task Logic ---
        target_label = 0 # Class 0: Original
        
        if self.unsupervised and self.pretext_mode == 'reverse':
            # 50% chance to flip the audio
            if random.random() > 0.5:
                waveform = torch.flip(waveform, dims=[1])
                target_label = 1 # Class 1: Reversed

        # --- 1D to 2D Conversion ---
        # Input: (1, Time) -> Output: (1, Freq, Time)
        spec = self.transforms(waveform)
        
        # Ensure single channel (grayscale-like)
        if spec.shape[0] > 1:
            spec = spec.mean(dim=0, keepdim=True)
            
        # Pad to ensuring consistent size if necessary, or crop
        # For this example, we assume roughly consistent 1-sec clips
        
        return spec, target_label

def create_dataloader(opt):
    dataset = GenericDataset(opt)
    return DataLoader(
        dataset,
        batch_size=opt['batch_size'],
        shuffle=(opt['split'] == 'train'),
        num_workers=opt.get('num_workers', 2),
        drop_last=True
    )